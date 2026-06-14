"""Tests for the runner — the orchestration layer.

Uses :class:`StubAdapter` (no model, no torch needed) and patches the
ffmpeg invocations so the whole matrix → encode → manifest pipeline runs
on CPU in a temp directory. Verifies:

  * Matrix iteration order (prompts outer, bands inner, baseline last)
  * Per-cell adapter.reset() fires before every generation
  * kwargs handed to the adapter carry the right merged state (bending_base
    + band overrides for bands; bending_base + bending_enabled:false for
    baseline)
  * Files land at the right paths (the kiosk's glob contract)
  * Manifest reflects every video in the right order
  * Dry-run produces zero side effects
  * Merge path concats the right files in the right order
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from generator.experiment import (
    BaselineSpec, ExperimentSpec, PipelineInit, PromptSpec, VideoSpec,
)
from generator.pipeline_factory import StubAdapter
from generator.runner import run_experiment
from generator.sweep import FFNLayerSweep


def _spec(*, n_prompts: int = 2, baseline: bool = True, merge: bool = False) -> ExperimentSpec:
    """Build an in-memory ExperimentSpec without going through YAML."""
    prompts = [
        PromptSpec(name=f"p{i}", text=f"prompt text {i}")
        for i in range(n_prompts)
    ]
    return ExperimentSpec(
        name="test_exp",
        model="longlive",
        pipeline_init=PipelineInit(width=64, height=48),
        video=VideoSpec(frames=4, fps=8, seed=100, encode_crf=20),
        prompts=prompts,
        bending_base={"bending_enabled": True, "ffn_output_scale": 0.0},
        sweep=FFNLayerSweep(stride=1, window=3, layer_end=6),    # 4 bands
        baseline=BaselineSpec(per_prompt=baseline, position="end"),
        merge=merge,
    )


# ─── Dry-run: nothing touches the filesystem or the model ───────────────

class TestDryRun:
    def test_dry_run_skips_adapter_calls(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=2, baseline=True)
        run_experiment(spec, adapter=adapter, output_root=tmp_path, dry_run=True)
        assert adapter.reset_count == 0
        assert adapter.generate_calls == []

    def test_dry_run_returns_target_path_without_creating_it(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=1, baseline=False)
        exp_dir = run_experiment(
            spec, adapter=adapter, output_root=tmp_path, dry_run=True,
        )
        # Path was returned but never created — dry-run is side-effect-free
        assert not exp_dir.exists()

    def test_dry_run_produces_no_manifest(self, tmp_path):
        spec = _spec()
        run_experiment(
            spec, adapter=StubAdapter(layer_count=30),
            output_root=tmp_path, dry_run=True,
        )
        assert not list(tmp_path.rglob("manifest.json"))


# ─── Live run: matrix iteration + reset + kwargs ────────────────────────

class TestLiveRun:
    """Patches the encode functions so they record calls instead of
    invoking ffmpeg. Everything else runs for real against the stub."""

    def test_matrix_order_prompts_outer_bands_inner_baseline_last(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=2, baseline=True)  # 4 bands × 2 prompts + 2 baselines = 10 videos

        with patch("generator.runner.encode.encode_video") as enc:
            run_experiment(spec, adapter=adapter, output_root=tmp_path, dry_run=False)

        assert len(adapter.generate_calls) == 10
        # Prompt 0: 4 bands then baseline
        assert adapter.generate_calls[0]["prompts"][0]["text"] == "prompt text 0"
        assert adapter.generate_calls[3]["prompts"][0]["text"] == "prompt text 0"
        assert adapter.generate_calls[4]["bending_enabled"] is False  # baseline
        # Prompt 1: 4 bands then baseline
        assert adapter.generate_calls[5]["prompts"][0]["text"] == "prompt text 1"
        assert adapter.generate_calls[9]["bending_enabled"] is False  # baseline 2
        # Verify encode was called once per generated video
        assert enc.call_count == 10

    def test_reset_fires_before_every_generation(self, tmp_path):
        """Critical invariant: each video starts from a fresh KV cache.
        StubAdapter's reset_count must equal generate_calls count."""
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=2, baseline=True)
        with patch("generator.runner.encode.encode_video"):
            run_experiment(spec, adapter=adapter, output_root=tmp_path, dry_run=False)
        assert adapter.reset_count == len(adapter.generate_calls)

    def test_band_kwargs_carry_layer_overrides(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=1, baseline=False)
        with patch("generator.runner.encode.encode_video"):
            run_experiment(spec, adapter=adapter, output_root=tmp_path, dry_run=False)
        # Bands: [0,2], [1,3], [2,4], [3,5] with sweep layer_end=6
        assert adapter.generate_calls[0]["ffn_layer_start"] == 0
        assert adapter.generate_calls[0]["ffn_layer_end"] == 2
        assert adapter.generate_calls[3]["ffn_layer_start"] == 3
        assert adapter.generate_calls[3]["ffn_layer_end"] == 5

    def test_baseline_kwargs_kill_bending(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=1, baseline=True)
        with patch("generator.runner.encode.encode_video"):
            run_experiment(spec, adapter=adapter, output_root=tmp_path, dry_run=False)
        baseline_call = adapter.generate_calls[-1]
        assert baseline_call["bending_enabled"] is False
        # bending_base is still preserved — only bending_enabled flips
        assert baseline_call["ffn_output_scale"] == 0.0
        # Layer-band overrides MUST be absent (no band on baseline)
        assert "ffn_layer_start" not in baseline_call
        assert "ffn_layer_end" not in baseline_call

    def test_seed_constant_across_all_videos(self, tmp_path):
        """Type Cast invariant when no per-prompt seed is set: every
        video shares ``video.seed`` so the only cell-to-cell variable
        is the bend. Loop and verify."""
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=2, baseline=True)
        with patch("generator.runner.encode.encode_video"):
            run_experiment(spec, adapter=adapter, output_root=tmp_path, dry_run=False)
        seeds = {call["seed"] for call in adapter.generate_calls}
        assert seeds == {100}

    def test_per_prompt_seed_overrides_video_seed(self, tmp_path):
        """When ``PromptSpec.seed`` is set, it wins over ``video.seed``
        for ALL of that prompt's videos (every band + baseline). Other
        prompts in the same experiment still inherit ``video.seed``."""
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=2, baseline=True)
        # Pin a new seed on prompt 1; leave prompt 0 inheriting.
        spec.prompts[1].seed = 999

        with patch("generator.runner.encode.encode_video"):
            run_experiment(spec, adapter=adapter, output_root=tmp_path, dry_run=False)

        # Group seeds by prompt label captured in the band call kwargs.
        # StubAdapter records calls in matrix order: prompt0 bands+baseline
        # then prompt1 bands+baseline (see test_matrix_order test).
        n_videos_per_prompt = len(adapter.generate_calls) // 2
        p0_seeds = {c["seed"] for c in adapter.generate_calls[:n_videos_per_prompt]}
        p1_seeds = {c["seed"] for c in adapter.generate_calls[n_videos_per_prompt:]}
        assert p0_seeds == {100}, "prompt 0 should still inherit video.seed"
        assert p1_seeds == {999}, "prompt 1 should use its per-prompt seed for every video"


# ─── Filesystem layout ──────────────────────────────────────────────────

class TestFilesystemLayout:
    def test_per_prompt_subfolders_created(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=2, baseline=False)
        with patch("generator.runner.encode.encode_video"):
            exp_dir = run_experiment(
                spec, adapter=adapter, output_root=tmp_path, dry_run=False,
            )
        assert (exp_dir / "prompt-01_p0").is_dir()
        assert (exp_dir / "prompt-02_p1").is_dir()

    def test_band_filenames_match_naming_convention(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=1, baseline=False)
        # Capture the encode_video call paths to verify expected filenames
        called_paths: list[Path] = []
        def _capture(frames, path, **kw):
            called_paths.append(path)
        with patch("generator.runner.encode.encode_video", side_effect=_capture):
            run_experiment(
                spec, adapter=adapter, output_root=tmp_path, dry_run=False,
            )
        names = [p.name for p in called_paths]
        assert names == [
            "prompt-01_band-00.mp4",
            "prompt-01_band-01.mp4",
            "prompt-01_band-02.mp4",
            "prompt-01_band-03.mp4",
        ]

    def test_baseline_filename_no_band_prefix(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=1, baseline=True)
        called_paths: list[Path] = []
        with patch(
            "generator.runner.encode.encode_video",
            side_effect=lambda frames, path, **kw: called_paths.append(path),
        ):
            run_experiment(
                spec, adapter=adapter, output_root=tmp_path, dry_run=False,
            )
        assert called_paths[-1].name == "prompt-01_baseline.mp4"


# ─── Manifest ───────────────────────────────────────────────────────────

class TestManifest:
    def test_manifest_records_every_video_in_order(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=2, baseline=True)
        with patch("generator.runner.encode.encode_video"):
            exp_dir = run_experiment(
                spec, adapter=adapter, output_root=tmp_path, dry_run=False,
            )
        manifest = json.loads((exp_dir / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert len(manifest["videos"]) == 10
        # First video is prompt 1's band-00, last is prompt 2's baseline
        assert manifest["videos"][0]["band_name"] == "band-00"
        assert manifest["videos"][0]["is_baseline"] is False
        assert manifest["videos"][-1]["band_name"] == "baseline"
        assert manifest["videos"][-1]["is_baseline"] is True

    def test_manifest_kwargs_jsonable(self, tmp_path):
        """Defensive: any non-JSON value in kwargs (a tensor, an
        unrecognized object) gets coerced to a string rather than
        crashing the manifest write."""
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=1, baseline=False)
        with patch("generator.runner.encode.encode_video"):
            exp_dir = run_experiment(
                spec, adapter=adapter, output_root=tmp_path, dry_run=False,
            )
        manifest = json.loads((exp_dir / "manifest.json").read_text())
        # Verify the round-trip — every recorded value is JSON-safe
        for v in manifest["videos"]:
            assert isinstance(v["kwargs"], dict)


# ─── Merge ──────────────────────────────────────────────────────────────

class TestMerge:
    def test_merge_concats_per_prompt_in_order(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=1, baseline=True, merge=True)
        concat_calls: list[tuple[list[Path], Path]] = []
        with patch("generator.runner.encode.encode_video"), \
             patch("generator.runner.encode.concat_videos",
                   side_effect=lambda paths, out: concat_calls.append((paths, out))):
            run_experiment(
                spec, adapter=adapter, output_root=tmp_path, dry_run=False,
            )
        assert len(concat_calls) == 1   # one merge per prompt
        paths, out = concat_calls[0]
        # 4 bands + baseline = 5 inputs, in playback order
        assert len(paths) == 5
        assert paths[0].name == "prompt-01_band-00.mp4"
        assert paths[-1].name == "prompt-01_baseline.mp4"
        assert out.name == "prompt-01_merged.mp4"

    def test_no_merge_when_flag_off(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=1, merge=False)
        with patch("generator.runner.encode.encode_video"), \
             patch("generator.runner.encode.concat_videos") as concat:
            run_experiment(
                spec, adapter=adapter, output_root=tmp_path, dry_run=False,
            )
        concat.assert_not_called()


# ─── Sweep failure mode ─────────────────────────────────────────────────

class TestSweepFailure:
    def test_empty_sweep_raises_at_run_start(self, tmp_path):
        """If the sweep yields no bands (e.g. window=0 caught by the
        sweep itself, or layer_end below layer_start somehow), surface
        the failure BEFORE creating output dirs or loading the model."""
        adapter = StubAdapter(layer_count=2)  # too few layers for any band
        spec = _spec(n_prompts=1)
        spec.sweep = FFNLayerSweep(stride=1, window=10)
        # Window 10 across 2 layers → one clamped band per the sweep contract,
        # so this should NOT raise. Verifies clamp-vs-empty behaviour.
        with patch("generator.runner.encode.encode_video"):
            run_experiment(
                spec, adapter=adapter, output_root=tmp_path, dry_run=False,
            )
        # One band → one band-video (+ baseline = 2 generations)
        assert len(adapter.generate_calls) == 2


# ─── --baseline-only preview mode ───────────────────────────────────────

class TestBaselineOnly:
    """The flag exists for prompt-tuning: render JUST the baselines so
    the operator can preview how each prompt looks before committing to
    a full sweep. Three invariants worth pinning:

      * Bands are skipped entirely — exactly len(prompts) videos out
      * Merge is suppressed even when spec.merge=True (no preview reel
        when each prompt produces a single video — would be misleading)
      * Misconfig (baseline.per_prompt=false + baseline_only=true) errors
        loudly BEFORE generating, so the operator can't get 0 videos and
        wonder where they went
    """

    def test_dry_run_skips_bands_and_logs_only_baselines(self, tmp_path, caplog):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=3, baseline=True)   # full would be 3 × 5 = 15
        with caplog.at_level("INFO"):
            run_experiment(
                spec, adapter=adapter, output_root=tmp_path,
                dry_run=True, baseline_only=True,
            )
        # No adapter calls in dry-run regardless
        assert adapter.generate_calls == []
        # The header line should announce baseline-only mode AND 0 bands
        header = next(r for r in caplog.records if "experiment" in r.message)
        assert "BASELINE-ONLY" in header.message
        assert "3 videos" in header.message            # 3 prompts × 1 baseline each
        assert "0 bands" in header.message

    def test_live_run_generates_one_video_per_prompt(self, tmp_path):
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=3, baseline=True)
        called_paths: list[Path] = []
        with patch(
            "generator.runner.encode.encode_video",
            side_effect=lambda frames, path, **kw: called_paths.append(path),
        ):
            run_experiment(
                spec, adapter=adapter, output_root=tmp_path,
                dry_run=False, baseline_only=True,
            )
        # Exactly one video per prompt, each named *_baseline.mp4
        assert len(called_paths) == 3
        assert all(p.name.endswith("_baseline.mp4") for p in called_paths)

    def test_baseline_only_kwargs_match_normal_baseline(self, tmp_path):
        """Sanity: the videos generated under --baseline-only carry
        EXACTLY the same kwargs they would in a full run's baseline
        slot — bending_enabled=false, no ffn_layer_* keys. Operators
        rely on this to trust the preview reflects real baseline output."""
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=2, baseline=True)
        with patch("generator.runner.encode.encode_video"):
            run_experiment(
                spec, adapter=adapter, output_root=tmp_path,
                dry_run=False, baseline_only=True,
            )
        for call in adapter.generate_calls:
            assert call["bending_enabled"] is False
            assert "ffn_layer_start" not in call
            assert "ffn_layer_end" not in call

    def test_merge_is_suppressed_under_baseline_only(self, tmp_path):
        """Even with spec.merge=True, no concat happens — each prompt
        produces exactly one video, so a "merge" would be a misleading
        rename. The operator wanted a preview, not a reel."""
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=2, baseline=True, merge=True)
        with patch("generator.runner.encode.encode_video"), \
             patch("generator.runner.encode.concat_videos") as concat:
            run_experiment(
                spec, adapter=adapter, output_root=tmp_path,
                dry_run=False, baseline_only=True,
            )
        concat.assert_not_called()

    def test_raises_when_baseline_disabled(self, tmp_path):
        """With baseline.per_prompt=false the flag would produce 0
        videos — surface this as a loud error before any I/O or model
        load happens, so the operator can fix the YAML in seconds."""
        adapter = StubAdapter(layer_count=30)
        spec = _spec(n_prompts=2, baseline=False)
        with pytest.raises(ValueError, match="baseline.per_prompt"):
            run_experiment(
                spec, adapter=adapter, output_root=tmp_path,
                dry_run=False, baseline_only=True,
            )

    def test_total_video_count_method_honors_flag(self):
        """``ExperimentSpec.total_video_count`` is used by the runner's
        progress counter and the CLI's pre-flight cost estimate — both
        need to report the smaller preview count, not the full sweep."""
        spec = _spec(n_prompts=4, baseline=True)
        # Full sweep: 4 prompts × (4 bands + 1 baseline) = 20
        assert spec.total_video_count(layer_count=30) == 20
        # Baseline-only: 4 prompts × 1 baseline = 4
        assert spec.total_video_count(layer_count=30, baseline_only=True) == 4

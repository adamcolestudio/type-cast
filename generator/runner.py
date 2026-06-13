"""Experiment runner — iterate the matrix, drive the adapter, encode.

The runner is the only module that knows the full pipeline shape:

    for each prompt:
        for each sweep band:
            reset adapter → generate → encode → record
        if baseline:
            reset adapter → generate (no bend) → encode → record
        if merge:
            concat-copy all per-prompt videos into one

Pipeline is model-agnostic through the :class:`VideoGenerator` protocol —
the runner doesn't know whether it's driving LongLive AR chunks or LTX
single-shot calls.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import encode, naming
from .experiment import ExperimentSpec
from .manifest import Manifest
from .pipeline_factory import VideoGenerator

logger = logging.getLogger(__name__)


def run_experiment(
    spec: ExperimentSpec,
    *,
    adapter: VideoGenerator,
    output_root: Path,
    dry_run: bool = False,
) -> Path:
    """Execute one experiment end-to-end. Returns the experiment-run
    folder path so the caller can print it / open it / rsync it.

    Args:
        spec: The loaded experiment.
        adapter: A constructed :class:`VideoGenerator`. The factory
            (:func:`pipeline_factory.build_adapter`) builds the right
            kind; the runner doesn't care which model.
        output_root: Parent directory under which the ``<name>_<stamp>/``
            experiment folder is created.
        dry_run: When True, no model is invoked and no files are written.
            The runner walks the matrix and logs every operation it
            WOULD perform. Use to validate YAML + naming + folder layout
            without burning a GPU minute.

    Returns:
        Path to the created (or, in dry-run, would-be-created) experiment
        folder.
    """
    layer_count = adapter.model_layer_count
    bands = list(spec.sweep.iter_bands(layer_count=layer_count))
    if not bands:
        raise ValueError(
            f"sweep produced 0 bands (layer_count={layer_count}, "
            f"sweep={spec.sweep}). Check sweep.window vs sweep.layer_end."
        )

    exp_dir = output_root / naming.experiment_dir_name(spec.name)
    total = spec.total_video_count(layer_count=layer_count)
    logger.info(
        "[type-cast] experiment '%s' → %s (%d videos: %d prompts × (%d bands + %d baselines))",
        spec.name, exp_dir, total, len(spec.prompts), len(bands),
        1 if spec.baseline.per_prompt else 0,
    )

    if not dry_run:
        exp_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(spec, exp_dir) if not dry_run else None
    counter = {"done": 0, "total": total}

    for idx, prompt in enumerate(spec.prompts, start=1):
        prompt_dir = exp_dir / naming.prompt_dir_name(idx, prompt.name)
        if not dry_run:
            prompt_dir.mkdir(parents=True, exist_ok=True)
        per_prompt_files: list[Path] = []

        # --- sweep bands ---
        for band in bands:
            video_path = prompt_dir / naming.band_filename(idx, band.name)
            kwargs = _build_call_kwargs(spec, prompt, band_overrides=band.overrides)
            _generate_one(
                adapter, kwargs=kwargs, video_path=video_path,
                spec=spec, dry_run=dry_run, counter=counter,
            )
            per_prompt_files.append(video_path)
            if manifest is not None:
                manifest.add(
                    prompt_idx=idx, prompt_name=prompt.name,
                    band_name=band.name, kwargs=kwargs, video_path=video_path,
                )

        # --- baseline at END (per design — loops naturally into band-00) ---
        if spec.baseline.per_prompt:
            video_path = prompt_dir / naming.baseline_filename(idx)
            kwargs = _build_call_kwargs(spec, prompt, band_overrides=None)
            _generate_one(
                adapter, kwargs=kwargs, video_path=video_path,
                spec=spec, dry_run=dry_run, counter=counter,
            )
            per_prompt_files.append(video_path)
            if manifest is not None:
                manifest.add(
                    prompt_idx=idx, prompt_name=prompt.name,
                    band_name="baseline", kwargs=kwargs,
                    video_path=video_path, is_baseline=True,
                )

        # --- per-prompt concat-merge ---
        if spec.merge:
            merged_path = prompt_dir / naming.merged_filename(idx)
            if dry_run:
                logger.info("[dry-run] would concat-merge %d files → %s",
                            len(per_prompt_files), merged_path)
            else:
                encode.concat_videos(per_prompt_files, merged_path)
                logger.info("[type-cast] merged → %s", merged_path)

    if manifest is not None:
        manifest.finalize(status="complete")
    logger.info("[type-cast] DONE — %s", exp_dir)
    return exp_dir


# ── Internals ────────────────────────────────────────────────────────────

def _generate_one(
    adapter: VideoGenerator,
    *,
    kwargs: dict,
    video_path: Path,
    spec: ExperimentSpec,
    dry_run: bool,
    counter: dict,
) -> None:
    """Reset adapter → generate → encode → optional PNG dump. Single
    helper so the band + baseline paths share exactly the same shape."""
    counter["done"] += 1
    n, total = counter["done"], counter["total"]
    if dry_run:
        logger.info(
            "[dry-run] [%d/%d] would generate %s (ffn=%s-%s)",
            n, total, video_path,
            kwargs.get("ffn_layer_start"), kwargs.get("ffn_layer_end"),
        )
        return

    logger.info(
        "[type-cast] [%d/%d] generating %s (ffn=%s-%s)",
        n, total, video_path,
        kwargs.get("ffn_layer_start"), kwargs.get("ffn_layer_end"),
    )

    adapter.reset()
    frames = adapter.generate(
        kwargs=kwargs,
        target_frames=spec.video.frames,
        on_chunk=_chunk_progress,
    )
    encode.encode_video(
        frames, video_path,
        fps=spec.video.fps,
        crf=spec.video.encode_crf,
    )
    if spec.video.keep_frames:
        frames_dir = naming.frames_dir_name(video_path)
        encode.write_frames(frames, frames_dir)
        logger.info("[type-cast] kept frames → %s", frames_dir)


def _chunk_progress(call_idx: int, chunk_n: int, accumulated: int) -> None:
    """Best-effort progress reporter for autoregressive chunks."""
    logger.debug("  chunk %d (+%d frames, %d accumulated)", call_idx, chunk_n, accumulated)


def _build_call_kwargs(
    spec: ExperimentSpec, prompt, *, band_overrides: dict | None,
) -> dict[str, Any]:
    """Merge the constant bending_base with the per-cell band overrides
    (None = baseline = no overrides). Always pins the seed so the runner
    fully owns reproducibility — the operator can't accidentally rely on
    a session-default seed."""
    kwargs: dict[str, Any] = {
        # LongLive expects ``prompts`` (plural, weighted) per scope's
        # test.py reference. For text mode the weight is the standard
        # 100. Future adapters may translate.
        "prompts": [{"text": prompt.text, "weight": 100}],
        "seed": spec.video.seed,
        # bending_base is the constant scaffolding — e.g.
        # ``{"bending_enabled": true, "ffn_output_scale": 0.0}``.
        # For the baseline we still want bending_base, but override
        # ``bending_enabled: False`` so the bend has no effect.
        **dict(spec.bending_base),
    }
    if band_overrides is None:
        # Baseline: kill the bend wholesale. Keeping bending_base in the
        # kwargs (rather than dropping it) means the model still sees the
        # same kwargs shape — only the master switch differs. Cleaner
        # diff in the manifest, easier visual comparison.
        kwargs["bending_enabled"] = False
    else:
        kwargs.update(band_overrides)
    return kwargs

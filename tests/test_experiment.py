"""Tests for the experiment YAML loader.

The YAML is the operator's only interface to the runner — typos there
silently produce wrong videos (e.g. ``promts:`` instead of ``prompts:``
yields a 0-prompt experiment). Validation catches these at load time
with a path-and-line error.

Schema must be locked: any field rename here means experiment YAMLs in
the wild become invalid, so changes need an intentional migration.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from generator.experiment import (
    ExperimentLoadError,
    ExperimentSpec,
    load_experiment,
)
from generator.sweep import FFNLayerSweep


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "exp.yaml"
    p.write_text(body)
    return p


# ─── Minimal valid YAML ──────────────────────────────────────────────────

MINIMAL = """
name: t
model: longlive
prompts:
  - { name: a, text: A }
sweep:
  kind: ffn_layer
  stride: 1
  window: 3
"""


class TestMinimal:
    def test_loads_with_defaults(self, tmp_path):
        spec = load_experiment(_write(tmp_path, MINIMAL))
        assert isinstance(spec, ExperimentSpec)
        assert spec.name == "t"
        assert spec.model == "longlive"
        # Defaults for unspecified sections
        assert spec.pipeline_init.width == 480
        assert spec.pipeline_init.height == 800
        assert spec.video.fps == 8
        assert spec.video.frames == 161
        assert spec.video.seed == 100
        assert spec.video.encode_crf == 16
        assert spec.video.keep_frames is False
        assert spec.baseline.per_prompt is True
        assert spec.baseline.position == "end"
        assert spec.merge is False

    def test_sweep_parses_to_concrete_dataclass(self, tmp_path):
        spec = load_experiment(_write(tmp_path, MINIMAL))
        assert isinstance(spec.sweep, FFNLayerSweep)
        assert spec.sweep.stride == 1
        assert spec.sweep.window == 3

    def test_bending_base_defaults_to_empty(self, tmp_path):
        spec = load_experiment(_write(tmp_path, MINIMAL))
        assert spec.bending_base == {}

    def test_output_dir_defaults_to_none(self, tmp_path):
        """Null default means the CLI falls back to its sibling-of-yaml
        convention — unchanged from the original behavior."""
        spec = load_experiment(_write(tmp_path, MINIMAL))
        assert spec.output_dir is None


# ─── output_dir field ────────────────────────────────────────────────────

class TestOutputDir:
    """Pins the YAML field that lets an experiment redirect its outputs
    (e.g. to an external SSD) without a CLI flag."""

    def test_absolute_path_parses(self, tmp_path):
        body = MINIMAL + "\noutput_dir: /mnt/ssd/typecast_runs\n"
        spec = load_experiment(_write(tmp_path, body))
        assert spec.output_dir == "/mnt/ssd/typecast_runs"

    def test_tilde_path_parses(self, tmp_path):
        """``~/outputs/`` is a common operator shorthand. The parser
        keeps the literal string; the CLI does the ``expanduser`` at
        resolve time so the manifest records the as-typed value."""
        body = MINIMAL + "\noutput_dir: ~/typecast_runs\n"
        spec = load_experiment(_write(tmp_path, body))
        assert spec.output_dir == "~/typecast_runs"

    def test_empty_string_maps_to_none(self, tmp_path):
        """An operator might write ``output_dir: ""`` to "unset" the
        field; that should fall back to the CLI default, not pass an
        empty string downstream."""
        body = MINIMAL + '\noutput_dir: ""\n'
        spec = load_experiment(_write(tmp_path, body))
        assert spec.output_dir is None


# ─── per-prompt seed override ───────────────────────────────────────────

class TestPerPromptSeed:
    """Per-prompt ``seed:`` lets one prompt pin a different noise seed
    than the experiment-wide default, useful when one prompt rolls a
    poor noise pattern at the shared seed. None = inherit ``video.seed``."""

    def test_seed_defaults_to_none(self, tmp_path):
        spec = load_experiment(_write(tmp_path, MINIMAL))
        assert all(p.seed is None for p in spec.prompts)

    def test_per_prompt_seed_parses(self, tmp_path):
        body = MINIMAL.replace(
            "prompts:\n  - { name: a, text: A }",
            "prompts:\n  - { name: a, text: A, seed: 201 }\n  - { name: b, text: B }",
        )
        spec = load_experiment(_write(tmp_path, body))
        # Override on the first prompt, inherit on the second.
        assert spec.prompts[0].seed == 201
        assert spec.prompts[1].seed is None


# ─── Required field validation ───────────────────────────────────────────

class TestRequiredFields:
    def test_missing_name_raises(self, tmp_path):
        body = MINIMAL.replace("name: t", "")
        with pytest.raises(ExperimentLoadError, match="name"):
            load_experiment(_write(tmp_path, body))

    def test_missing_model_raises(self, tmp_path):
        body = MINIMAL.replace("model: longlive", "")
        with pytest.raises(ExperimentLoadError, match="model"):
            load_experiment(_write(tmp_path, body))

    def test_unsupported_model_raises(self, tmp_path):
        body = MINIMAL.replace("longlive", "stable-diffusion-xl")
        with pytest.raises(ExperimentLoadError, match="unsupported model"):
            load_experiment(_write(tmp_path, body))

    def test_missing_prompts_raises(self, tmp_path):
        body = """
name: t
model: longlive
sweep:
  kind: ffn_layer
"""
        with pytest.raises(ExperimentLoadError, match="prompts"):
            load_experiment(_write(tmp_path, body))

    def test_empty_prompts_list_raises(self, tmp_path):
        body = """
name: t
model: longlive
prompts: []
sweep:
  kind: ffn_layer
"""
        with pytest.raises(ExperimentLoadError, match="non-empty"):
            load_experiment(_write(tmp_path, body))

    def test_prompt_missing_text_raises(self, tmp_path):
        body = """
name: t
model: longlive
prompts:
  - { name: a }
sweep:
  kind: ffn_layer
"""
        with pytest.raises(ExperimentLoadError, match="prompts\\[0\\].*name.*text"):
            load_experiment(_write(tmp_path, body))


# ─── Sweep validation ────────────────────────────────────────────────────

class TestSweepValidation:
    def test_missing_sweep_raises(self, tmp_path):
        body = """
name: t
model: longlive
prompts:
  - { name: a, text: A }
"""
        with pytest.raises(ExperimentLoadError, match="sweep"):
            load_experiment(_write(tmp_path, body))

    def test_unknown_sweep_kind_raises(self, tmp_path):
        body = MINIMAL.replace("kind: ffn_layer", "kind: galactic_brain")
        with pytest.raises(ExperimentLoadError, match="unsupported sweep kind"):
            load_experiment(_write(tmp_path, body))

    def test_sweep_inherits_kind_defaults(self, tmp_path):
        body = """
name: t
model: longlive
prompts:
  - { name: a, text: A }
sweep:
  kind: ffn_layer
"""
        spec = load_experiment(_write(tmp_path, body))
        assert spec.sweep.stride == 1
        assert spec.sweep.window == 3
        assert spec.sweep.layer_start == 0
        assert spec.sweep.layer_end is None


# ─── File-level error reporting ──────────────────────────────────────────

class TestFileErrors:
    def test_missing_file_raises_with_path(self, tmp_path):
        with pytest.raises(ExperimentLoadError, match="not found"):
            load_experiment(tmp_path / "does-not-exist.yaml")

    def test_malformed_yaml_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("name: t\nmodel: : :\n")   # invalid YAML
        with pytest.raises(ExperimentLoadError, match="invalid YAML"):
            load_experiment(p)

    def test_top_level_must_be_mapping(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("- foo\n- bar\n")
        with pytest.raises(ExperimentLoadError, match="must be a mapping"):
            load_experiment(p)


# ─── Total count calculation (used by CLI progress estimate) ─────────────

class TestTotalCount:
    def test_matches_handcomputed(self, tmp_path):
        # 2 prompts, 28 bands (30-layer model, stride 1, window 3), +baseline
        # → 2 × (28 + 1) = 58
        spec = load_experiment(_write(tmp_path, MINIMAL.replace(
            "prompts:\n  - { name: a, text: A }",
            "prompts:\n  - { name: a, text: A }\n  - { name: b, text: B }",
        )))
        assert spec.total_video_count(layer_count=30) == 2 * (28 + 1)

    def test_no_baseline_subtracts(self, tmp_path):
        body = MINIMAL.replace(
            "prompts:\n  - { name: a, text: A }",
            "prompts:\n  - { name: a, text: A }\n  - { name: b, text: B }",
        ) + "\nbaseline:\n  per_prompt: false\n"
        spec = load_experiment(_write(tmp_path, body))
        assert spec.total_video_count(layer_count=30) == 2 * 28


# ─── Real-world experiment file ──────────────────────────────────────────

class TestRealExperiment:
    """Load the actual experiments/ YAML to lock its expected shape.
    Catches drift between the canonical YAML and the loader's schema."""

    def test_real_yaml_loads(self):
        path = Path(__file__).parent.parent / "experiments" \
            / "cowboy_femme_ffn_output_scale_0.yaml"
        spec = load_experiment(path)
        assert spec.name == "type_cast_v1_ffn_output_scale_0"
        assert spec.model == "longlive"
        assert spec.pipeline_init.width == 800
        assert spec.pipeline_init.height == 480
        assert spec.video.frames == 161
        assert spec.video.fps == 16
        assert spec.video.seed == 102
        # Canonical YAML expanded into a full preview palette; pin only
        # the first two slots (the original pair) so the test survives
        # further prompt additions without becoming a churn magnet.
        assert len(spec.prompts) >= 2
        assert spec.prompts[0].name == "cowboy"
        assert spec.prompts[1].name == "femme"
        assert spec.bending_base["ffn_output_scale"] == 0.0
        assert spec.merge is True
        assert spec.baseline.per_prompt is True
        # Sanity-check the matrix size: the canonical sweep produces
        # 28 sliding 3-wide bands over 30 FFN layers, plus 1 baseline
        # per prompt. Total = n_prompts × (28 + 1).
        n = len(spec.prompts)
        assert spec.total_video_count(layer_count=30) == n * (28 + 1)

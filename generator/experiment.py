"""Experiment specification — the parsed-and-validated YAML model.

The YAML is the source of truth for an experiment run. This module turns
it into typed dataclasses, validates required fields, and surfaces clear
errors when the operator mistypes a key (typos in deeply-nested YAML are
otherwise silent: the runner just sees ``None`` and produces wrong videos).

Schema lives in code (these dataclasses) rather than a separate JSON
Schema file because (a) the schema is small enough to read end-to-end
and (b) we want validation, defaults, and post-processing in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sweep import FFNLayerSweep, Sweep


# ─── Sub-specs ───────────────────────────────────────────────────────────────

@dataclass
class PipelineInit:
    """Kwargs forwarded to the bender pipeline at construction time. The
    pipeline is built ONCE per experiment run (model load is the dominant
    cost); per-call kwargs come from :class:`VideoSpec` + bending state."""
    width: int = 480
    height: int = 800
    # Optional LongLive autoregressive knobs — None = inherit scope's
    # model.yaml defaults. Surfaced here so an experiment can override
    # them without touching the bender's WebUI (which doesn't expose
    # them — these aren't real-time-tweakable knobs anyway).
    num_frame_per_block: int | None = None
    local_attn_size: int | None = None
    sink_size: int | None = None


@dataclass
class VideoSpec:
    """Per-video output properties. Same for every video in the matrix —
    the only thing that varies cell-to-cell is the bending band."""
    frames: int = 161           # target frame count per video
    fps: int = 8                # encoder + playback frame rate
    seed: int = 100             # held constant across the whole experiment
    encode_crf: int = 16        # libx264 CRF — your SETUP.md baseline
    keep_frames: bool = False   # also write PNG sequence alongside mp4 (archive)


@dataclass
class PromptSpec:
    name: str                    # short slug for filenames (e.g. "cowboy")
    text: str                    # the actual prompt fed to the encoder
    # Optional per-prompt seed override. None = inherit ``VideoSpec.seed``
    # (the experiment-wide default). Use when one prompt happens to roll
    # an unlucky noise pattern at the shared seed and you want to pin a
    # better one without re-seeding the whole sweep. The resolved seed
    # (per-prompt OR inherited) lands in the manifest as ``video.seed``
    # of each generated video so reproducibility is preserved.
    seed: int | None = None


@dataclass
class BaselineSpec:
    """Per-experiment baseline (no-bend) policy. Per your design: one
    baseline per prompt, written at end of the band sequence so it loops
    naturally back into band-00 on the kiosk."""
    per_prompt: bool = True
    position: str = "end"        # "end" only for v1; "start" reserved for future


@dataclass
class ExperimentSpec:
    name: str
    model: str                   # "longlive" v1; "ltx2.3" / others later
    pipeline_init: PipelineInit
    video: VideoSpec
    prompts: list[PromptSpec]
    bending_base: dict[str, Any] # constant scaffolding (e.g. bending_enabled, ffn_output_scale)
    sweep: Sweep                 # what varies cell-to-cell
    baseline: BaselineSpec
    merge: bool = False          # if true, concat all per-prompt videos into one mp4
    # Where to write the experiment-run folder. None = the CLI default
    # (``<yaml-dir>/../output/``). Set to an absolute path (e.g. an
    # external SSD mount) when local disk is tight or you want runs to
    # live alongside other archival assets. Relative paths resolve
    # against the YAML file's parent directory so the experiment stays
    # self-contained when moved. CLI ``--output`` overrides this.
    output_dir: str | None = None

    def total_video_count(self, *, layer_count: int, baseline_only: bool = False) -> int:
        """How many videos this experiment will produce — useful for the
        runner's progress bar and the CLI's pre-flight cost estimate.

        ``baseline_only`` mirrors the runner's CLI flag: when True, the
        sweep contributes zero bands and only the per-prompt baseline
        counts. (Caller must still validate that ``baseline.per_prompt``
        is True under this mode — otherwise the count is 0.)"""
        bands = 0 if baseline_only else sum(
            1 for _ in self.sweep.iter_bands(layer_count=layer_count)
        )
        per_prompt = bands + (1 if self.baseline.per_prompt else 0)
        return per_prompt * len(self.prompts)


# ─── YAML loading + validation ──────────────────────────────────────────────

class ExperimentLoadError(ValueError):
    """Raised when an experiment YAML is malformed or missing required fields.
    Carries the source path + the YAML key path for easy debugging."""


def load_experiment(path: str | Path) -> ExperimentSpec:
    """Parse + validate an experiment YAML. Errors are raised with a path
    that points at the offending key (best-effort) so the operator can
    fix without grepping."""
    import yaml
    path = Path(path)
    if not path.exists():
        raise ExperimentLoadError(f"experiment YAML not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ExperimentLoadError(f"{path}: invalid YAML — {e}") from e

    if not isinstance(raw, dict):
        raise ExperimentLoadError(f"{path}: top-level must be a mapping, got {type(raw).__name__}")

    return _parse_experiment(raw, source=path)


def _parse_experiment(raw: dict, *, source: Path) -> ExperimentSpec:
    def _require(d: dict, key: str, where: str):
        if key not in d:
            raise ExperimentLoadError(f"{source}: missing required field '{where}.{key}'")
        return d[key]

    name = _require(raw, "name", "")
    model = _require(raw, "model", "")
    if model not in ("longlive",):       # v1 — extend list as adapters land
        raise ExperimentLoadError(
            f"{source}: unsupported model '{model}'. Supported in v1: longlive"
        )

    pipeline_init = _parse_pipeline_init(raw.get("pipeline_init") or {}, source)
    video = _parse_video(raw.get("video") or {}, source)
    prompts = _parse_prompts(_require(raw, "prompts", ""), source)
    bending_base = raw.get("bending_base") or {}
    if not isinstance(bending_base, dict):
        raise ExperimentLoadError(f"{source}: 'bending_base' must be a mapping")
    sweep = _parse_sweep(_require(raw, "sweep", ""), source)
    baseline = _parse_baseline(raw.get("baseline") or {}, source)
    merge = bool(raw.get("merge", False))
    output_dir = _optional_str(raw.get("output_dir"))

    return ExperimentSpec(
        name=name, model=model, pipeline_init=pipeline_init, video=video,
        prompts=prompts, bending_base=bending_base, sweep=sweep,
        baseline=baseline, merge=merge, output_dir=output_dir,
    )


def _parse_pipeline_init(d: dict, source: Path) -> PipelineInit:
    return PipelineInit(
        width=int(d.get("width", 480)),
        height=int(d.get("height", 800)),
        num_frame_per_block=_optional_int(d.get("num_frame_per_block")),
        local_attn_size=_optional_int(d.get("local_attn_size")),
        sink_size=_optional_int(d.get("sink_size")),
    )


def _parse_video(d: dict, source: Path) -> VideoSpec:
    return VideoSpec(
        frames=int(d.get("frames", 161)),
        fps=int(d.get("fps", 8)),
        seed=int(d.get("seed", 100)),
        encode_crf=int(d.get("encode_crf", 16)),
        keep_frames=bool(d.get("keep_frames", False)),
    )


def _parse_prompts(raw: list, source: Path) -> list[PromptSpec]:
    if not isinstance(raw, list) or not raw:
        raise ExperimentLoadError(f"{source}: 'prompts' must be a non-empty list")
    out: list[PromptSpec] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ExperimentLoadError(f"{source}: prompts[{i}] must be a mapping")
        if "name" not in item or "text" not in item:
            raise ExperimentLoadError(
                f"{source}: prompts[{i}] needs both 'name' and 'text' keys"
            )
        out.append(PromptSpec(
            name=str(item["name"]),
            text=str(item["text"]),
            seed=_optional_int(item.get("seed")),
        ))
    return out


def _parse_sweep(raw: dict, source: Path) -> Sweep:
    if not isinstance(raw, dict):
        raise ExperimentLoadError(f"{source}: 'sweep' must be a mapping")
    kind = raw.get("kind")
    if kind == "ffn_layer":
        return FFNLayerSweep(
            stride=int(raw.get("stride", 1)),
            window=int(raw.get("window", 3)),
            layer_start=int(raw.get("layer_start", 0)),
            layer_end=_optional_int(raw.get("layer_end")),
        )
    # When you add ``kind: linear`` (generic two-anchor interpolation),
    # dispatch here. Other planned kinds: ``neuron_stride``, ``param_range``.
    raise ExperimentLoadError(
        f"{source}: unsupported sweep kind '{kind}'. Supported in v1: ffn_layer"
    )


def _parse_baseline(d: dict, source: Path) -> BaselineSpec:
    position = str(d.get("position", "end"))
    if position not in ("end",):
        raise ExperimentLoadError(
            f"{source}: baseline.position must be 'end' (v1 only supports end)"
        )
    return BaselineSpec(
        per_prompt=bool(d.get("per_prompt", True)),
        position=position,
    )


def _optional_int(v) -> int | None:
    if v is None:
        return None
    return int(v)


def _optional_str(v) -> str | None:
    """``None`` and empty-string both map to None — operators sometimes
    write ``device: ""`` in YAML to mean "default" which would otherwise
    pass through as a non-None empty string and break the precedence."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None

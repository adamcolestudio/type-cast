"""Adapter layer between Type Cast's runner and the bender's pipelines.

The runner doesn't know whether the model is autoregressive (LongLive) or
single-shot (LTX2.3 in the future). It calls one protocol:

    adapter.reset()
    frames = adapter.generate(kwargs=..., target_frames=...)

Each adapter wraps the model-specific details (chunk loop, init_cache
semantics, single-call slicing) and exposes the same interface.

V1 ships:
  * :class:`LongLiveAdapter` — wraps the bender's LongLive pipeline + the
    autoregressive driver shipped in scope-attention-bender.
  * :class:`StubAdapter` — synthetic frames for dry-runs + unit tests,
    zero GPU/torch/scope dependencies.

LTX2.3 / Wan2.1 adapters are a future addition — they fit the same
protocol with a different ``generate()`` body (slice the single-shot
output to target_frames, no reset needed beyond the bender's own state).
"""
from __future__ import annotations

from typing import Any, Callable, Protocol


class VideoGenerator(Protocol):
    """The minimum surface the runner needs from an adapter."""

    @property
    def model_layer_count(self) -> int:
        """Number of FFN layers in the model — the sweep needs this to
        default ``layer_end`` when the YAML left it null."""
        ...

    def reset(self) -> None:
        """Make the adapter ready to start a new video. For AR pipelines
        this clears the KV cache + frame counter. For single-shot it's
        a no-op."""
        ...

    def generate(
        self,
        *,
        kwargs: dict,
        target_frames: int,
        on_chunk: Callable[[int, int, int], None] | None = None,
    ) -> Any:
        """Produce one full video as a ``[F, H, W, 3]`` tensor (uint8 or
        float). ``on_chunk(call_idx, chunk_frames, accumulated)`` may
        fire per internal chunk for progress reporting; single-shot
        adapters can ignore it."""
        ...


# ─── LongLive — production adapter ────────────────────────────────────────

class LongLiveAdapter:
    """Drives the bender's LongLive pipeline via the autoregressive driver.

    Construction loads the model — expensive (~20 GB VRAM, multi-second
    init). Reused across the entire experiment matrix (every video in
    a run shares the same pipeline instance). Construct once, reset
    before each video, generate.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        device: str | None = None,
        num_frame_per_block: int | None = None,
        local_attn_size: int | None = None,
        sink_size: int | None = None,
        seed: int = 42,
    ):
        # Heavy imports are deferred so this module loads without
        # scope/torch present — supports dry-run on the dev machine.
        from scope_attention_bender.pipelines.longlive_pipeline import (
            AttentionBenderLongLivePipeline,
        )

        init_kwargs: dict = {
            "width": width,
            "height": height,
            "base_seed": seed,
            # Type Cast runs standalone; the bender's companion WebUI
            # would just fight Scope for a port if both are up. Off.
            "start_webui": False,
        }
        # Only pass overrides that are actually set — None means
        # "inherit scope's model.yaml default".
        if num_frame_per_block is not None:
            init_kwargs["num_frame_per_block"] = num_frame_per_block
        if local_attn_size is not None:
            init_kwargs["local_attn_size"] = local_attn_size
        if sink_size is not None:
            init_kwargs["sink_size"] = sink_size
        if device is not None:
            import torch
            init_kwargs["device"] = torch.device(device)

        self._pipeline = AttentionBenderLongLivePipeline(**init_kwargs)

    @property
    def model_layer_count(self) -> int:
        """Read from the bender's _MODEL_LIMITS dict — populated by the
        LongLive wrapper's ``__init__`` after the FFN patching pass
        runs. For LongLive (Wan 1.3B), this is 30."""
        from scope_attention_bender.core.bender import _MODEL_LIMITS
        n = int(_MODEL_LIMITS.get("ffn_layers", 0))
        if n == 0:
            # The patching pass hasn't run, or returned zero — something
            # is wrong with the pipeline init. Fail loud rather than
            # silently sweeping a zero-layer range.
            raise RuntimeError(
                "LongLiveAdapter: model_layer_count is 0 — the FFN patching "
                "pass didn't find any layers. Check the pipeline init logs."
            )
        return n

    def reset(self) -> None:
        """Per-video reset — KV cache, frame counter, mode tracking. See
        ``scope_attention_bender.pipelines.longlive_pipeline:reset_for_new_video``."""
        self._pipeline.reset_for_new_video()

    def generate(self, *, kwargs, target_frames, on_chunk=None):
        from scope_attention_bender.orchestrator.autoregressive_driver import (
            generate_autoregressive_video,
        )
        return generate_autoregressive_video(
            self._pipeline,
            kwargs=kwargs,
            target_frames=target_frames,
            on_chunk=on_chunk,
        )


# ─── Stub — for dry-runs and tests ───────────────────────────────────────

class StubAdapter:
    """Synthetic frames, no model load. Used by:

      * ``type-cast run --dry-run`` (with ``--no-encode`` to skip ffmpeg)
        for fast iteration on YAML / sweep / folder layout
      * ``tests/test_runner.py`` to exercise the matrix iteration without
        a GPU
      * future Type Cast development on machines without scope installed
    """

    def __init__(
        self,
        *,
        layer_count: int = 30,
        height: int = 480,
        width: int = 800,
    ):
        self._layer_count = layer_count
        self._height = height
        self._width = width
        # Public state for tests to inspect.
        self.reset_count = 0
        self.generate_calls: list[dict] = []

    @property
    def model_layer_count(self) -> int:
        return self._layer_count

    def reset(self) -> None:
        self.reset_count += 1

    def generate(self, *, kwargs, target_frames, on_chunk=None):
        import numpy as np
        self.generate_calls.append(dict(kwargs))
        # Stamp each video with its call-order index so tests can verify
        # iteration order. Wraps at 256 to fit uint8 — fine for stub use.
        idx = (len(self.generate_calls) - 1) % 256
        return np.full(
            (target_frames, self._height, self._width, 3),
            fill_value=idx, dtype=np.uint8,
        )


# ─── Dispatch ────────────────────────────────────────────────────────────

def build_adapter(spec, *, device: str | None = None, stub: bool = False) -> VideoGenerator:
    """Construct the right adapter for an :class:`ExperimentSpec`.

    Args:
        spec: A loaded :class:`generator.experiment.ExperimentSpec`.
        device: Optional torch device override (e.g. ``"cuda:1"`` to
            target a specific GPU when multi-GPU). None lets torch pick.
        stub: When True, returns :class:`StubAdapter` regardless of
            ``spec.model``. Used by ``--dry-run``.
    """
    if stub:
        return StubAdapter(
            height=spec.pipeline_init.height,
            width=spec.pipeline_init.width,
        )

    if spec.model == "longlive":
        return LongLiveAdapter(
            width=spec.pipeline_init.width,
            height=spec.pipeline_init.height,
            device=device,
            num_frame_per_block=spec.pipeline_init.num_frame_per_block,
            local_attn_size=spec.pipeline_init.local_attn_size,
            sink_size=spec.pipeline_init.sink_size,
            seed=spec.video.seed,
        )

    raise ValueError(f"unsupported model '{spec.model}' (v1: longlive only)")

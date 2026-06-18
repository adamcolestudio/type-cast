"""Sweep specifications — what varies cell-to-cell in an experiment matrix.

A :class:`Sweep` produces an ordered sequence of :class:`BandSpec` instances.
Each band names a single video in the experiment and carries the per-call
bending overrides that the runner merges with the constant ``bending_base``.

V1 ships :class:`FFNLayerSweep` only. The protocol exists so future sweep
kinds (generic two-anchor linear interpolation, neuron-stride sweeps,
arbitrary parameter ranges) drop in without touching the runner — add a
new dataclass + a single dispatch line in ``experiment._parse_sweep``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol


@dataclass
class BandSpec:
    """One row in the matrix: a name (for filenames + manifest) plus the
    kwargs the runner overlays onto bending_base for this specific video."""
    name: str
    overrides: dict


class Sweep(Protocol):
    """Anything that produces ordered bands. The runner only sees this
    protocol — it doesn't care whether the sweep is layer-windowed,
    neuron-strided, or generic-anchor-interpolated."""

    def iter_bands(self, *, layer_count: int) -> Iterator[BandSpec]:
        """Yield each band in playback order. ``layer_count`` lets the
        sweep adapt to the model's actual layer count when ``layer_end``
        is left null in the YAML (defaults to "sweep the whole model")."""
        ...


@dataclass
class FFNLayerSweep:
    """Slide a ``window``-wide layer band across the model's FFN layers
    with stride ``stride``. The canonical Type Cast sweep — your "bottom
    to top" deconstruction visualization.

    For LongLive (30 FFN layers) with ``stride=1, window=3``:
    bands are ``[0,3), [1,4), …, [27,30)`` — 28 bands.

    Layer indices map to the bender's ``ffn_layer_start`` /
    ``ffn_layer_end`` kwargs, which are **inclusive** on both ends in
    the bender's config (see ``_FFN_BENDING_CONFIG`` defaults). So a
    "window of 3" with start=0 means layers 0, 1, 2 are bent — we emit
    ``ffn_layer_start=0, ffn_layer_end=2``.

    Future kinds will adopt the same protocol — see
    :class:`generator.sweep.Sweep`.
    """

    stride: int = 1
    window: int = 3
    layer_start: int = 0
    layer_end: int | None = None    # None → use model's actual layer count

    def __post_init__(self):
        if self.stride < 1:
            raise ValueError(f"stride must be >=1, got {self.stride}")
        if self.window < 1:
            raise ValueError(f"window must be >=1, got {self.window}")
        if self.layer_start < 0:
            raise ValueError(f"layer_start must be >=0, got {self.layer_start}")
        if self.layer_end is not None and self.layer_end <= self.layer_start:
            raise ValueError(
                f"layer_end ({self.layer_end}) must exceed layer_start "
                f"({self.layer_start})"
            )

    def iter_bands(self, *, layer_count: int) -> Iterator[BandSpec]:
        end = self.layer_end if self.layer_end is not None else layer_count
        if end > layer_count:
            # Defensive — silently cap rather than emit windows past the
            # end of the model (which would produce no bending effect
            # and waste a generation).
            end = layer_count
        # We need ``start + window - 1`` to be a valid layer index,
        # i.e. ``start + window <= end``. Stop the range accordingly.
        last_start = end - self.window
        if last_start < self.layer_start:
            # Window larger than the requested range — produce one band
            # at the start clamped to the available range. Not the
            # typical case but better than an empty sweep.
            yield BandSpec(
                name=f"band-{self.layer_start:02d}",
                overrides={
                    "ffn_layer_start": self.layer_start,
                    "ffn_layer_end":   end - 1,
                },
            )
            return
        for start in range(self.layer_start, last_start + 1, self.stride):
            yield BandSpec(
                name=f"band-{start:02d}",
                overrides={
                    "ffn_layer_start": start,
                    "ffn_layer_end":   start + self.window - 1,
                },
            )


@dataclass
class SeedSweep:
    """Vary the RNG seed cell-to-cell, holding bending constant — the
    "which seed renders this prompt best?" tool.

    Pair with ``bending_base: {bending_enabled: false}`` for a pure
    baseline seed hunt (no bend; only the noise seed changes), or leave a
    bend enabled to find the best seed for that specific bend. Either way
    each seed becomes one video named ``seed-<value>``, so the filename
    (``prompt-NN_seed-<value>.mp4``) and the manifest both decode straight
    back to the exact seed — pick the winner and pin it in ``video.seed``
    (or a per-prompt ``seed:``) for the real run.

    The seed lands in ``overrides`` and the runner merges band overrides
    LAST, so it wins over ``video.seed`` / ``prompt.seed`` — exactly what
    a hunt wants. ``layer_count`` is ignored (seeds don't depend on model
    shape) but kept in the signature for :class:`Sweep` protocol parity.
    """

    seeds: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.seeds:
            raise ValueError("seed sweep needs at least one seed")

    def iter_bands(self, *, layer_count: int) -> Iterator[BandSpec]:
        for s in self.seeds:
            yield BandSpec(name=f"seed-{s}", overrides={"seed": s})

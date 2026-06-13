"""Tests for the sweep generators.

The sweep is the engine that turns "stride 1, window 3" into 28 distinct
``ffn_layer_start`` / ``ffn_layer_end`` pairs. Off-by-one here means
either burning compute on a layer band that doesn't reach the model edge
OR walking off the end of the network — both are silent-data-quality
bugs the visual inspection of the resulting videos wouldn't catch right
away. Pin the math here.
"""
from __future__ import annotations

import pytest

from generator.sweep import BandSpec, FFNLayerSweep


# ─── Canonical Type Cast v1 sweep ────────────────────────────────────────

class TestCanonicalSweep:
    """The v1 sweep — 30-layer model, stride=1, window=3 → 28 bands."""

    def test_count_matches_design(self):
        sweep = FFNLayerSweep(stride=1, window=3)
        bands = list(sweep.iter_bands(layer_count=30))
        assert len(bands) == 28   # last band starts at 27 (covers 27,28,29)

    def test_first_band_starts_at_layer_zero(self):
        sweep = FFNLayerSweep(stride=1, window=3)
        bands = list(sweep.iter_bands(layer_count=30))
        assert bands[0].overrides == {"ffn_layer_start": 0, "ffn_layer_end": 2}
        assert bands[0].name == "band-00"

    def test_last_band_touches_last_layer(self):
        """Critical: the sweep must INCLUDE layer 29 (the last). An
        off-by-one that stops at layer 28 leaves the deepest layer
        unbent in every video — the Type Cast 'top of the network'
        observation point is missing."""
        sweep = FFNLayerSweep(stride=1, window=3)
        bands = list(sweep.iter_bands(layer_count=30))
        assert bands[-1].overrides == {"ffn_layer_start": 27, "ffn_layer_end": 29}
        assert bands[-1].name == "band-27"

    def test_band_names_zero_padded(self):
        """Filenames sort lexically; without padding band-2.mp4 sorts
        after band-10.mp4 and the kiosk's directory-listing playback
        would jump around."""
        sweep = FFNLayerSweep(stride=1, window=3)
        bands = list(sweep.iter_bands(layer_count=30))
        for band in bands:
            assert band.name.startswith("band-")
            n = band.name.removeprefix("band-")
            assert len(n) == 2 and n.isdigit()


# ─── Stride and window variations ────────────────────────────────────────

class TestStrideWindow:
    def test_stride_2_halves_band_count(self):
        """Half as many bands when stride doubles. With stride=2,
        window=3, layers=30: bands at 0,2,4,…,26 → 14 bands."""
        sweep = FFNLayerSweep(stride=2, window=3)
        bands = list(sweep.iter_bands(layer_count=30))
        starts = [b.overrides["ffn_layer_start"] for b in bands]
        assert starts == list(range(0, 28, 2))

    def test_window_larger_than_stride_overlaps(self):
        """stride=1, window=4 → adjacent bands overlap by 3 layers.
        This is the canonical 'sliding window' Type Cast pattern."""
        sweep = FFNLayerSweep(stride=1, window=4)
        bands = list(sweep.iter_bands(layer_count=30))
        # 30 - 4 + 1 = 27 bands
        assert len(bands) == 27
        # First two bands overlap on layers 1,2,3
        assert bands[0].overrides["ffn_layer_end"] == 3
        assert bands[1].overrides["ffn_layer_start"] == 1

    def test_window_equals_stride_non_overlapping(self):
        """stride=3, window=3 → bands tile the model without overlap."""
        sweep = FFNLayerSweep(stride=3, window=3)
        bands = list(sweep.iter_bands(layer_count=30))
        # 30 / 3 = 10 bands: [0,2], [3,5], …, [27,29]
        assert len(bands) == 10
        starts = [b.overrides["ffn_layer_start"] for b in bands]
        ends = [b.overrides["ffn_layer_end"] for b in bands]
        for i in range(len(bands) - 1):
            assert starts[i + 1] == ends[i] + 1   # tile exactly


# ─── Sweep windowing within a sub-range ──────────────────────────────────

class TestSubRange:
    def test_layer_start_skips_early_layers(self):
        """Operator can choose to sweep only the upper half of the
        network. Useful for ablation: 'do shallow layers even matter?'"""
        sweep = FFNLayerSweep(stride=1, window=3, layer_start=15)
        bands = list(sweep.iter_bands(layer_count=30))
        assert bands[0].overrides["ffn_layer_start"] == 15
        assert len(bands) == 30 - 15 - 3 + 1   # 13

    def test_layer_end_caps_terminal_layers(self):
        """Counterpart: sweep only the bottom half."""
        sweep = FFNLayerSweep(stride=1, window=3, layer_end=15)
        bands = list(sweep.iter_bands(layer_count=30))
        # Bands [0,2] … [12,14] → 13 bands
        assert len(bands) == 13
        assert bands[-1].overrides["ffn_layer_end"] == 14

    def test_layer_end_silently_caps_to_model_count(self):
        """Operator types layer_end: 99 by mistake but the model only
        has 30 layers. Cap silently so the sweep still produces sane
        videos rather than emitting empty windows."""
        sweep = FFNLayerSweep(stride=1, window=3, layer_end=99)
        bands = list(sweep.iter_bands(layer_count=30))
        assert bands[-1].overrides["ffn_layer_end"] == 29


# ─── Validation + edge cases ─────────────────────────────────────────────

class TestValidation:
    def test_stride_must_be_positive(self):
        with pytest.raises(ValueError, match="stride"):
            FFNLayerSweep(stride=0, window=3)
        with pytest.raises(ValueError, match="stride"):
            FFNLayerSweep(stride=-1, window=3)

    def test_window_must_be_positive(self):
        with pytest.raises(ValueError, match="window"):
            FFNLayerSweep(stride=1, window=0)

    def test_layer_start_must_be_non_negative(self):
        with pytest.raises(ValueError, match="layer_start"):
            FFNLayerSweep(stride=1, window=3, layer_start=-1)

    def test_layer_end_must_exceed_start(self):
        with pytest.raises(ValueError, match="layer_end"):
            FFNLayerSweep(stride=1, window=3, layer_start=5, layer_end=5)

    def test_window_larger_than_range_emits_clamped_band(self):
        """Edge case: operator asks for window=10 across only 5 layers.
        Rather than yielding 0 bands (silent empty experiment), produce
        one clamped band so something gets generated and the operator
        sees their YAML didn't match the model size."""
        sweep = FFNLayerSweep(stride=1, window=10, layer_start=0, layer_end=5)
        bands = list(sweep.iter_bands(layer_count=30))
        assert len(bands) == 1
        # Clamps to the requested layer range, not the model count
        assert bands[0].overrides == {"ffn_layer_start": 0, "ffn_layer_end": 4}

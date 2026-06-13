"""Tests for the naming convention.

The kiosk on the Pi will glob for these files literally — any drift
breaks playback. Pin every filename pattern + the directory layout.
"""
from __future__ import annotations

import re
from pathlib import Path

from generator import naming


class TestSlug:
    def test_lowercases_and_dashes(self):
        assert naming.slug("Cowboy McLawman") == "cowboy-mclawman"

    def test_strips_special_chars(self):
        assert naming.slug("café d'épôque #1!") == "caf-d-p-que-1"

    def test_handles_empty_with_fallback(self):
        assert naming.slug("") == "untitled"
        assert naming.slug("!!!") == "untitled"
        assert naming.slug("", fallback="x") == "x"

    def test_caps_length(self):
        s = naming.slug("a" * 100)
        assert len(s) <= 48


class TestStamp:
    def test_format_is_iso_compact(self):
        s = naming.stamp()
        assert re.match(r"^\d{8}-\d{6}$", s), s


class TestDirAndFilenames:
    def test_experiment_dir_includes_stamp(self):
        d = naming.experiment_dir_name("cowboy ffn 0")
        # `<slug>_<YYYYMMDD-HHMMSS>` — the trailing 15 chars are the stamp
        assert d.endswith(re.search(r"\d{8}-\d{6}$", d).group())
        assert d.startswith("cowboy-ffn-0_")

    def test_prompt_dir(self):
        assert naming.prompt_dir_name(1, "cowboy") == "prompt-01_cowboy"
        assert naming.prompt_dir_name(8, "Type Three") == "prompt-08_type-three"

    def test_band_filename(self):
        assert naming.band_filename(1, "band-00") == "prompt-01_band-00.mp4"
        assert naming.band_filename(8, "band-27") == "prompt-08_band-27.mp4"

    def test_baseline_filename(self):
        # Baseline at end of sweep — Type Cast naming has no "band-" prefix
        # so the kiosk can distinguish baseline from sweep bands by name alone
        assert naming.baseline_filename(1) == "prompt-01_baseline.mp4"

    def test_merged_filename(self):
        assert naming.merged_filename(1) == "prompt-01_merged.mp4"

    def test_frames_dir_sibling_to_mp4(self):
        """When keep_frames is on, PNGs sit next to the mp4 with the
        same stem — so an operator inspecting the folder can match them
        at a glance."""
        mp4 = Path("/tmp/exp/prompt-01_band-00.mp4")
        frames_dir = naming.frames_dir_name(mp4)
        assert frames_dir.parent == mp4.parent
        assert frames_dir.name == "prompt-01_band-00_frames"

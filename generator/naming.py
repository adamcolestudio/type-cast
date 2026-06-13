"""Output naming convention — single source of truth.

Filenames + folder layout are pinned here so:
  * the kiosk-side webui knows where to look (no string drift)
  * regenerating an experiment lands files at predictable paths the
    Pi rsync workflow can resume against
  * an external inspector can decode an experiment from filenames alone

Layout per experiment run::

    <output_root>/<experiment_name>_<timestamp>/
        manifest.json
        prompt-01_<promptname>/
            prompt-01_band-00.mp4
            prompt-01_band-01.mp4
            ...
            prompt-01_baseline.mp4
            prompt-01_merged.mp4         # only when merge: true
        prompt-02_<promptname>/
            ...
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path


def stamp() -> str:
    """``YYYYMMDD-HHMMSS`` for run folder names. Date.now() is fine here
    — we're not inside the bender's workflow recorder where deterministic
    timestamps matter."""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def slug(s: str, fallback: str = "untitled") -> str:
    """Filesystem-safe slug: lowercase, alphanumeric + dashes, capped at
    48 chars. Avoid forcing the operator to write filesystem-safe names
    in the YAML — they can write 'Cowboy McLawman' and it'll land at
    'cowboy-mclawman'."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    s = s[:48].rstrip("-")
    return s or fallback


def experiment_dir_name(experiment_name: str) -> str:
    """``<slug>_<timestamp>`` — the per-run folder. The timestamp prevents
    re-runs of the same experiment from colliding so the operator never
    accidentally overwrites a previous output."""
    return f"{slug(experiment_name, fallback='experiment')}_{stamp()}"


def prompt_dir_name(prompt_idx: int, prompt_name: str) -> str:
    """``prompt-NN_<slug>/`` — the per-prompt subfolder. 1-indexed to
    match how operators discuss prompts ("prompt 1" not "prompt 0").
    Two-digit padding scales to 99 prompts; bump to three if you ever
    push past."""
    return f"prompt-{prompt_idx:02d}_{slug(prompt_name)}"


def band_filename(prompt_idx: int, band_name: str) -> str:
    """``prompt-NN_<bandname>.mp4`` — one video per matrix cell. ``band_name``
    comes from the sweep so the runner doesn't dictate its format (different
    sweep kinds use different naming patterns)."""
    return f"prompt-{prompt_idx:02d}_{band_name}.mp4"


def baseline_filename(prompt_idx: int) -> str:
    """``prompt-NN_baseline.mp4`` — sits at the END of the sweep per
    your spec, so when the kiosk loops back to band-00 the cut from
    baseline → fully-bent feels like a fresh start."""
    return f"prompt-{prompt_idx:02d}_baseline.mp4"


def merged_filename(prompt_idx: int) -> str:
    """``prompt-NN_merged.mp4`` — the lossless concat of all per-prompt
    videos in playback order. Only produced when ``merge: true``."""
    return f"prompt-{prompt_idx:02d}_merged.mp4"


def frames_dir_name(mp4_path: Path) -> Path:
    """When ``keep_frames: true``, the PNG sequence lives next to the mp4
    with the same stem and a ``_frames`` suffix — so the operator can
    distinguish at a glance which mp4 each PNG dump backs."""
    return mp4_path.with_name(mp4_path.stem + "_frames")

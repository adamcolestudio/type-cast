"""Per-experiment manifest.json — full record of what was generated.

Drops alongside the videos so you can reconstruct an experiment from
disk months later without needing the original YAML. Includes:

  * The full ExperimentSpec (so the YAML schema doesn't have to round-trip)
  * Per-video: the exact kwargs the pipeline was called with (the
    deformation band, prompt, seed) and the relative file path.
  * The bender's git SHA (best-effort) so a re-run on a different
    bender version is detectable.
  * The timestamp the run started.

Written incrementally — each video appends + flushes to disk so a
crashed run still has a partial manifest you can inspect.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any


class Manifest:
    """Builder + writer for the per-experiment manifest file."""

    def __init__(self, spec, exp_dir: Path):
        self._spec = spec
        self._exp_dir = exp_dir
        self._path = exp_dir / "manifest.json"
        self._data: dict[str, Any] = {
            "experiment": _spec_to_dict(spec),
            "bender_sha": _try_bender_sha(),
            "started_at": _now_iso(),
            "videos": [],
        }

    def add(
        self,
        *,
        prompt_idx: int,
        prompt_name: str,
        band_name: str,
        kwargs: dict,
        video_path: Path,
        is_baseline: bool = False,
    ) -> None:
        """Record one generated video. ``video_path`` is converted to
        relative-to-exp_dir for portability — the manifest+folder pair
        can be moved without breaking links."""
        try:
            rel = video_path.relative_to(self._exp_dir)
        except ValueError:
            rel = video_path
        self._data["videos"].append({
            "prompt_idx": prompt_idx,
            "prompt_name": prompt_name,
            "band_name": band_name,
            "is_baseline": is_baseline,
            # Surfaced top-level (it's also inside kwargs) so a seed hunt's
            # chosen output maps back to a number at a glance.
            "seed": kwargs.get("seed"),
            "file": str(rel),
            "kwargs": _jsonable(kwargs),
        })
        self.write()

    def finalize(self, *, status: str = "complete") -> None:
        """Add the terminal state + finish timestamp. Call at the end of
        a successful run; on failure call with status='failed' so the
        manifest reflects reality."""
        self._data["finished_at"] = _now_iso()
        self._data["status"] = status
        self.write()

    def write(self) -> None:
        """Atomic-ish write: dump to a temp sibling then rename. Prevents
        a partial file if the process is killed mid-write."""
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2, default=str))
        os.replace(tmp, self._path)


# ── Helpers ──────────────────────────────────────────────────────────────

def _spec_to_dict(spec) -> dict:
    """Serialize the ExperimentSpec dataclass tree. The Sweep field is a
    Protocol so dataclasses.asdict alone doesn't handle it — substitute
    the actual dataclass at that key."""
    raw = asdict(spec) if dataclasses.is_dataclass(spec) else dict(spec)
    if "sweep" in raw and dataclasses.is_dataclass(spec.sweep):
        raw["sweep"] = asdict(spec.sweep)
        raw["sweep"]["kind"] = type(spec.sweep).__name__
    return raw


def _jsonable(d: dict) -> dict:
    """Strip non-JSON values (tensors, etc.) from kwargs before dumping.
    The bender's kwargs are mostly primitives but may carry image
    tensors in i2v mode — skip those."""
    out = {}
    for k, v in d.items():
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [
                x if isinstance(x, (bool, int, float, str, dict)) else str(x)
                for x in v
            ]
        elif isinstance(v, dict):
            out[k] = _jsonable(v)
        else:
            out[k] = str(v)
    return out


def _try_bender_sha() -> str | None:
    """Best-effort: query the scope-attention-bender repo's git HEAD.
    Returns None if the package isn't installed editable, or git isn't
    on PATH, or the package is shipped via wheel. Manifest gets a tag
    field either way."""
    try:
        import scope_attention_bender as _ab
        pkg_dir = Path(_ab.__file__).resolve().parent.parent
        result = subprocess.run(
            ["git", "-C", str(pkg_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:                                          # noqa: BLE001
        pass
    return None


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")

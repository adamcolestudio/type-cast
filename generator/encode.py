"""ffmpeg invocations — encode raw frames to Pi-optimal MP4 + concat.

Two operations:

1. :func:`encode_video` — write a ``[F, H, W, C]`` uint8 frame tensor
   to an MP4. Pipes raw rgb24 frames to ffmpeg via stdin (no PNG
   intermediate by default — saves ~10× the disk IO during a run).
   Encoder flags mirror your SETUP.md exactly:

       -c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p
       -crf 16 -preset veryslow -tune grain
       -g 32 -an -movflags +faststart

2. :func:`concat_videos` — stream-copy concat per prompt (no re-encode,
   no quality loss) using ffmpeg's concat demuxer.

The encoder process is invoked via subprocess so failures bubble up as
:class:`EncodeError` with the full ffmpeg stderr — the operator can
diagnose codec/resolution issues without re-running the whole experiment.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any


class EncodeError(RuntimeError):
    """ffmpeg returned non-zero. Carries the cmdline + stderr tail so the
    operator sees enough context to diagnose without re-running."""


def encode_video(
    frames: Any,
    out_path: Path,
    *,
    fps: int,
    crf: int = 16,
    preset: str = "veryslow",
    tune: str = "grain",
) -> None:
    """Encode a ``[F, H, W, 3]`` uint8 RGB tensor to MP4 with the
    Pi-3B+-on-Waveshare-5" baseline flags from your SETUP.md.

    Frames are piped to ffmpeg via stdin as rgb24 raw bytes — no PNG
    intermediate. If you want the PNGs alongside (archive use), call
    :func:`write_frames` separately before :func:`encode_video`.

    Args:
        frames: Frame tensor, ``[F, H, W, 3]``, uint8. Numpy arrays and
            torch tensors both work; the function calls ``.numpy()`` for
            torch via duck-typing. C-contiguous required for the raw
            stdin pipe (the function asserts it before invoking ffmpeg).
        out_path: Destination ``.mp4`` path. Parent must exist.
        fps: Encoder + playback frame rate. Your default in v1 is 8 (a
            low-decode-load starting point for the Pi 3B+).
        crf: libx264 constant rate factor. 16 = your SETUP.md baseline.
        preset: x264 speed/quality preset. ``veryslow`` matches SETUP.md
            — slow encode, smallest file for a given quality. Drop to
            ``slow`` or ``medium`` for faster iteration during testing.
        tune: x264 tune profile. ``grain`` preserves the high-frequency
            texture in the bender's deformed output (per SETUP.md).
            Note SETUP.md flags this as experimental: confirm it still
            plays smoothly on the Waveshare; drop or try ``film`` if not.

    Raises:
        EncodeError: ffmpeg failed (non-zero exit). Stderr is in the
            exception message.
    """
    arr = _to_contiguous_uint8_rgb(frames)
    f, h, w, c = arr.shape
    if c != 3:
        raise EncodeError(f"expected 3 channels (RGB), got shape {arr.shape}")

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",                                # stdin
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level", "4.0",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
        "-tune", tune,
        "-g", "32",
        "-an",
        "-movflags", "+faststart",
        str(out_path),
    ]

    proc = subprocess.run(
        cmd, input=arr.tobytes(), capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise EncodeError(
            f"ffmpeg failed (rc={proc.returncode}) on {out_path}:\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr tail:\n{proc.stderr.decode('utf-8', 'replace')[-2000:]}"
        )


def write_frames(frames: Any, out_dir: Path) -> None:
    """Dump a ``[F, H, W, 3]`` uint8 RGB tensor as ``frame_NNNNN.png``
    files in ``out_dir``. Used when ``keep_frames: true`` to archive
    raw frames alongside the encoded mp4 — lets you re-encode later
    with different flags without re-running the model.

    Uses Pillow to write — adds a small dep cost on the Type Cast env,
    but Pillow's PNG writer is reliable and matches the SETUP.md command
    that expects ``frame_%05d.png`` as input."""
    from PIL import Image
    arr = _to_contiguous_uint8_rgb(frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i]).save(out_dir / f"frame_{i:05d}.png")


def concat_videos(video_paths: list[Path], out_path: Path) -> None:
    """Stream-copy concat using ffmpeg's concat demuxer. No re-encode —
    output is bit-for-bit a concatenation of the input streams, so
    quality is unaffected.

    Per-prompt concat order (in your spec): all band videos in sweep
    order, then the baseline. The kiosk loops the concat naturally back
    to band-00 after baseline plays — feels like a reset between cycles.

    Args:
        video_paths: Ordered list of MP4 paths to concat. All must
            share codec / resolution / fps; this is automatically the
            case when they're all produced by :func:`encode_video`
            with the same VideoSpec.
        out_path: Destination MP4. Parent must exist.

    Raises:
        EncodeError: ffmpeg failed.
    """
    if not video_paths:
        raise EncodeError("concat_videos called with empty list")

    # ffmpeg's concat demuxer reads a list file: one ``file '<path>'``
    # line per input. Use a temp file so we can guarantee cleanup even
    # if ffmpeg crashes mid-run.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False,
    ) as listfile:
        for p in video_paths:
            # Escape single quotes in paths for the concat demuxer's
            # quote-and-escape syntax: '\'' inside single-quoted string.
            escaped = str(p).replace("'", r"'\''")
            listfile.write(f"file '{escaped}'\n")
        listpath = listfile.name

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", listpath,
        "-c", "copy",                            # stream copy — no re-encode
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            raise EncodeError(
                f"ffmpeg concat failed (rc={proc.returncode}) on {out_path}:\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stderr tail:\n{proc.stderr.decode('utf-8', 'replace')[-2000:]}"
            )
    finally:
        Path(listpath).unlink(missing_ok=True)


# ── Internals ────────────────────────────────────────────────────────────

def _to_contiguous_uint8_rgb(frames: Any):
    """Coerce torch tensor / numpy array to a contiguous uint8 numpy array.
    The raw rgb24 stdin pipe needs the byte layout to be exactly
    ``[F,H,W,3] uint8 C-contiguous``."""
    import numpy as np

    # Torch tensor: .cpu().numpy()
    if hasattr(frames, "detach") and hasattr(frames, "cpu"):
        frames = frames.detach().cpu().numpy()

    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        # Floats in [0,1] (a common pipeline output) need scaling.
        # Anything else is a programming error — surface it.
        if arr.dtype.kind == "f":
            arr = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            raise EncodeError(
                f"expected uint8 or float frames, got dtype {arr.dtype}"
            )
    if arr.ndim != 4:
        raise EncodeError(f"expected [F,H,W,C] 4D tensor, got {arr.ndim}D shape {arr.shape}")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr

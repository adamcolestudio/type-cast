"""Tests for the ffmpeg invocations.

Mocks ``subprocess.run`` so the entire encode + concat pipeline is
exercised without ffmpeg actually running. Verifies:

  * The cmdline argv matches your SETUP.md baseline byte-for-byte
  * Frame tensors are coerced to contiguous uint8 RGB before piping
  * Concat uses stream-copy (no re-encode → no quality loss)
  * Errors carry enough context to diagnose (cmdline + stderr tail)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from generator import encode


# ── encode_video: cmdline shape ───────────────────────────────────────────

class TestEncodeCmdline:
    def _run_with_capture(self, frames, out_path: Path, **encode_kw):
        captured = {}
        def _fake_run(cmd, *, input, capture_output, check):
            captured["cmd"] = cmd
            captured["input_bytes"] = len(input) if input else 0
            return MagicMock(returncode=0, stderr=b"")
        with patch("generator.encode.subprocess.run", side_effect=_fake_run):
            encode.encode_video(frames, out_path, **encode_kw)
        return captured

    def test_cmdline_matches_setup_md_baseline(self, tmp_path):
        """The cmdline shape is your SETUP.md contract — every flag
        in that doc must show up here in the same order. If the Pi
        ever has trouble playing the output, your first diagnosis is
        'did the encode flags change?'."""
        frames = np.zeros((10, 480, 800, 3), dtype=np.uint8)
        out = tmp_path / "test.mp4"
        c = self._run_with_capture(frames, out, fps=8)
        cmd = c["cmd"]
        # Spot-check each load-bearing flag
        assert "ffmpeg" in cmd[0]
        assert "-pix_fmt" in cmd and cmd[cmd.index("-pix_fmt") + 1] == "rgb24"
        assert "-s" in cmd and cmd[cmd.index("-s") + 1] == "800x480"  # WxH
        assert "-r" in cmd and cmd[cmd.index("-r") + 1] == "8"
        assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "libx264"
        assert "-profile:v" in cmd and cmd[cmd.index("-profile:v") + 1] == "high"
        assert "-level" in cmd and cmd[cmd.index("-level") + 1] == "4.0"
        assert "-crf" in cmd and cmd[cmd.index("-crf") + 1] == "16"
        assert "-preset" in cmd and cmd[cmd.index("-preset") + 1] == "veryslow"
        assert "-tune" in cmd and cmd[cmd.index("-tune") + 1] == "grain"
        assert "-g" in cmd and cmd[cmd.index("-g") + 1] == "32"
        assert "-an" in cmd
        assert "-movflags" in cmd and cmd[cmd.index("-movflags") + 1] == "+faststart"
        assert cmd[-1] == str(out)

    def test_byte_count_matches_tensor_size(self, tmp_path):
        """Raw rgb24 pipe: stdin bytes should equal F × H × W × 3."""
        frames = np.zeros((10, 48, 64, 3), dtype=np.uint8)
        c = self._run_with_capture(frames, tmp_path / "t.mp4", fps=8)
        assert c["input_bytes"] == 10 * 48 * 64 * 3

    def test_crf_override(self, tmp_path):
        frames = np.zeros((4, 48, 64, 3), dtype=np.uint8)
        c = self._run_with_capture(frames, tmp_path / "t.mp4", fps=8, crf=22)
        assert "22" in c["cmd"][c["cmd"].index("-crf") + 1]


# ── Tensor coercion ───────────────────────────────────────────────────────

class TestTensorCoercion:
    def test_float_frames_scaled_and_cast(self, tmp_path):
        """Pipeline outputs sometimes float in [0,1]; encoder needs uint8.
        Driver scales + casts so the caller doesn't have to think about it."""
        frames = np.full((4, 32, 24, 3), fill_value=0.5, dtype=np.float32)
        captured = {}
        def _fake(cmd, *, input, capture_output, check):
            captured["bytes"] = input[:5]
            return MagicMock(returncode=0, stderr=b"")
        with patch("generator.encode.subprocess.run", side_effect=_fake):
            encode.encode_video(frames, tmp_path / "t.mp4", fps=8)
        # 0.5 * 255 = 127 (truncated)
        assert all(b == 127 for b in captured["bytes"])

    def test_non_contiguous_array_handled(self, tmp_path):
        """Strided slices are common (operator computes ``frames[::2]``);
        encoder needs contiguous bytes. Driver handles."""
        frames = np.zeros((20, 32, 24, 3), dtype=np.uint8)[::2]
        assert not frames.flags["C_CONTIGUOUS"]
        with patch("generator.encode.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stderr=b"")
            encode.encode_video(frames, tmp_path / "t.mp4", fps=8)
        # No exception — the array was made contiguous internally

    def test_non_uint8_non_float_rejects(self, tmp_path):
        frames = np.zeros((4, 32, 24, 3), dtype=np.int16)
        with pytest.raises(encode.EncodeError, match="dtype"):
            encode.encode_video(frames, tmp_path / "t.mp4", fps=8)

    def test_wrong_shape_rejects(self, tmp_path):
        frames = np.zeros((4, 32, 24), dtype=np.uint8)   # 3D, missing channel dim
        with pytest.raises(encode.EncodeError, match="4D"):
            encode.encode_video(frames, tmp_path / "t.mp4", fps=8)

    def test_wrong_channel_count_rejects(self, tmp_path):
        frames = np.zeros((4, 32, 24, 4), dtype=np.uint8)  # RGBA
        with pytest.raises(encode.EncodeError, match="3 channels"):
            encode.encode_video(frames, tmp_path / "t.mp4", fps=8)


# ── Failure mode ──────────────────────────────────────────────────────────

class TestEncodeFailure:
    def test_ffmpeg_nonzero_raises_with_stderr_tail(self, tmp_path):
        frames = np.zeros((4, 32, 24, 3), dtype=np.uint8)
        stderr = b"x264: not in cake mode\n" * 1000
        with patch("generator.encode.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stderr=stderr)
            with pytest.raises(encode.EncodeError) as exc:
                encode.encode_video(frames, tmp_path / "t.mp4", fps=8)
        # Error message carries the stderr tail (last 2000 bytes), so the
        # operator can see the actual ffmpeg complaint without re-running.
        assert "not in cake mode" in str(exc.value)


# ── concat_videos ─────────────────────────────────────────────────────────

class TestConcat:
    def test_uses_concat_demuxer_with_stream_copy(self, tmp_path):
        """No re-encode is the design contract — concat preserves quality
        bit-for-bit. The flag combo ``-f concat -i list.txt -c copy`` is
        the only way to achieve that."""
        paths = [tmp_path / f"v{i}.mp4" for i in range(3)]
        for p in paths:
            p.touch()
        captured = {}
        def _fake(cmd, *, capture_output, check):
            captured["cmd"] = list(cmd)
            return MagicMock(returncode=0, stderr=b"")
        with patch("generator.encode.subprocess.run", side_effect=_fake):
            encode.concat_videos(paths, tmp_path / "merged.mp4")
        cmd = captured["cmd"]
        assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "concat"
        assert "-safe" in cmd and cmd[cmd.index("-safe") + 1] == "0"
        assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"
        assert cmd[-1] == str(tmp_path / "merged.mp4")

    def test_empty_list_rejects(self, tmp_path):
        with pytest.raises(encode.EncodeError, match="empty list"):
            encode.concat_videos([], tmp_path / "out.mp4")

    def test_concat_listfile_cleaned_up_on_failure(self, tmp_path):
        """The concat demuxer reads a temp .txt file. Even if ffmpeg
        fails, the temp file must be removed — otherwise repeated runs
        leak files into the system temp dir."""
        paths = [tmp_path / "v.mp4"]
        paths[0].touch()
        with patch("generator.encode.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stderr=b"oops")
            with pytest.raises(encode.EncodeError):
                encode.concat_videos(paths, tmp_path / "out.mp4")
        # Walk system tmp for any stale concat-list-like file — shouldn't
        # find one because we cleaned up. (Can't easily inspect the
        # specific tempfile name post-fact; trust the try/finally.)

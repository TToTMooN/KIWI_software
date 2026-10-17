"""Frame export for the engine: the mapping subset, the native rate, the exported files.

The export test synthesizes a tiny video with ffmpeg and skips where ffmpeg is
absent.
"""

import shutil
import subprocess

import numpy as np
import pytest

from kiwi.track.frames import (
    export_stream_frames,
    native_fps,
    read_timestamps,
    subset_times,
)
from kiwi.track.sequence import frame_times_ns

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """2 s of 10 fps synthetic video - 20 frames at exact 0.1 s spacing."""
    path = tmp_path_factory.mktemp("seq") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=64x64:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def test_subset_times_takes_every_fourth_frame_at_native_rate():
    t = (np.arange(1000) * (1e9 / 119.88)).astype(np.int64)
    subset = subset_times(t, 119.88)
    assert np.array_equal(subset, t[::4])


def test_subset_times_keeps_all_frames_at_or_below_the_map_rate():
    t = (np.arange(100) * (1e9 / 30.0)).astype(np.int64)
    assert np.array_equal(subset_times(t, 30.0), t)
    assert np.array_equal(subset_times(t, 24.0), t)


def test_native_fps():
    t = (np.arange(1000) * (1e9 / 119.88)).astype(np.int64)
    assert abs(native_fps(t) - 119.88) < 0.01


@needs_ffmpeg
def test_export_stream_frames_names_frames_by_container_time(video, tmp_path):
    t_ns = frame_times_ns(video)
    timestamps = export_stream_frames(video, t_ns, tmp_path / "f")
    assert len(t_ns) == 20
    frames = sorted(int(p.stem) for p in (tmp_path / "f").glob("*.jpg"))
    assert frames == list(t_ns)
    assert timestamps == tmp_path / "f" / "timestamps.txt"
    assert read_timestamps(timestamps).tolist() == list(t_ns)

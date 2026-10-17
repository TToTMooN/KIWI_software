"""Container frame times of a video stream, read from its own packets.

The ffmpeg-driven tests synthesize a tiny video first; they skip where
ffmpeg is absent (it is a hard runtime dependency of the convert stage,
so CI and any real deployment have it).
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from kiwi.track.sequence import frame_times_ns, video_time_base

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


@needs_ffmpeg
def test_frame_times_are_the_container_times(video):
    t = np.asarray(frame_times_ns(video), dtype=np.int64)
    assert len(t) == 20
    assert np.all(t[1:] > t[:-1])
    assert t[0] == 0
    # 10 fps -> 100 ms spacing, exact in container ticks
    assert np.all(np.diff(t) == 100_000_000)


@needs_ffmpeg
def test_time_base_is_probed_not_assumed(video):
    tb = video_time_base(video)
    assert 0 < float(tb) < 1

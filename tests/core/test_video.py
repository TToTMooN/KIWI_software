"""Unit tests for kiwi.core.video (uniform capture open / metadata / iterate / release).

Run: pixi run test
Writes a tiny real mp4 to tmp_path via cv2.VideoWriter - no fixtures on disk.
"""

from __future__ import annotations

import numpy as np
import pytest

# cv2 is a runtime dep here, but keep the guard so the pure-python tests stay
# runnable in a minimal env: a missing cv2 skips this module (kiwi.core.video
# imports it transitively) instead of erroring at collection.
cv2 = pytest.importorskip("cv2")

from kiwi.core.video import (  # noqa: E402
    VideoInfo,
    iter_frames,
    open_capture,
    video_capture,
)

W, H, N, FPS = 32, 24, 10, 20.0


@pytest.fixture
def clip(tmp_path):
    """A 10-frame WxH mp4 whose frame i is a solid gray of value i*10 (so each
    frame is identifiable by its mean)."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert writer.isOpened()
    for i in range(N):
        writer.write(np.full((H, W, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


def _frame_value(bgr: np.ndarray) -> int:
    return int(round(bgr.mean() / 10.0)) * 10


# --- open_capture ----------------------------------------------------------


def test_open_capture_ok(clip):
    cap = open_capture(clip)
    try:
        assert cap.isOpened()
    finally:
        cap.release()


def test_open_capture_missing_raises(tmp_path):
    with pytest.raises(OSError, match="could not open video"):
        open_capture(tmp_path / "nope.mp4")


# --- VideoInfo -------------------------------------------------------------


def test_video_info(clip):
    info = VideoInfo.read(clip)
    assert info.width == W and info.height == H
    assert info.size == (W, H)
    assert info.fps == pytest.approx(FPS, abs=0.5)
    assert info.n_frames == N


def test_fps_fallback_to_30():
    class _FakeCap:
        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 100,
                cv2.CAP_PROP_FRAME_HEIGHT: 50,
                cv2.CAP_PROP_FPS: 0.0,  # container reports unknown
                cv2.CAP_PROP_FRAME_COUNT: 5,
            }[prop]

    assert VideoInfo.from_capture(_FakeCap()).fps == 30.0


# --- video_capture context manager -----------------------------------------


def test_video_capture_releases_even_on_error(clip):
    cap_ref = {}
    with pytest.raises(RuntimeError):
        with video_capture(clip) as (cap, info):
            cap_ref["cap"] = cap
            assert info.n_frames == N
            raise RuntimeError("boom")
    assert not cap_ref["cap"].isOpened()  # released despite the exception


# --- iter_frames -----------------------------------------------------------


def test_iter_frames_all(clip):
    frames = list(iter_frames(clip))
    assert [i for i, _ in frames] == list(range(N))
    assert [_frame_value(f) for _, f in frames] == [i * 10 for i in range(N)]


def test_iter_frames_stride(clip):
    idxs = [i for i, _ in iter_frames(clip, stride=3)]
    assert idxs == [0, 3, 6, 9]  # true frame indices, counting from start


def test_iter_frames_start_and_stride(clip):
    got = list(iter_frames(clip, start=2, stride=2))
    assert [i for i, _ in got] == [2, 4, 6, 8]
    assert _frame_value(got[0][1]) == 20  # frame 2's value


def test_iter_frames_bad_stride(clip):
    with pytest.raises(ValueError, match="stride must be >= 1"):
        list(iter_frames(clip, stride=0))


def test_iter_frames_on_live_capture_does_not_release(clip):
    # Passing a live capture: caller keeps ownership.
    with video_capture(clip) as (cap, _info):
        first = [i for i, _ in iter_frames(cap, stride=5)]
        assert first == [0, 5]
        assert cap.isOpened()  # iter_frames must NOT release a borrowed capture

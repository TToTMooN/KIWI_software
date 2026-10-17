"""`VideoSource` seeks to its first frame; the frames must not change because of it.

The risk this guards is specific and silent: a seek that lands somewhere other
than it reports pairs the wrong video frame with a state sample, producing a
dataset that loads fine and trains on a lie. So the check here is PIXEL
EQUALITY against a sequential decode from frame 0, never "close enough".

Synthesised fixtures: CI has no `data/`, and a generated clip with a distinct
frame index burned into every frame localises a one-frame error.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from kiwi.export.video_source import VideoSource  # noqa: E402

W = H = 64
N_FRAMES = 90


def _frame(i: int) -> np.ndarray:
    """A BGR frame uniquely identified by ``i``, robust to lossy compression.

    Large flat blocks, well separated in value: a neighbouring frame differs
    by a wide margin, so "did we land on the right frame" survives H.264
    quantising while an exact-equality check between the two decode paths
    stays meaningful.
    """
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = ((i * 2) % 256, (255 - i * 2) % 256, (i * 3) % 256)
    img[: H // 2, : W // 2] = (i % 2) * 255
    return img


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("vs") / "clip.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    if not vw.isOpened():
        pytest.skip("no mp4v encoder available in this build of cv2")
    for i in range(N_FRAMES):
        vw.write(_frame(i))
    vw.release()
    if not p.exists() or p.stat().st_size == 0:
        pytest.skip("cv2 could not write a test clip")
    return p


def _sequential(path, indices):
    """Ground truth: a plain sequential decode from frame 0, no seek."""
    out, cur, frame = [], -1, None
    cap = cv2.VideoCapture(str(path))
    try:
        for target in indices:
            while cur < target:
                ok, bgr = cap.read()
                if not ok:
                    break
                cur += 1
                frame = bgr
            assert cur >= target, f"fixture too short for index {target}"
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return out


@pytest.mark.parametrize(
    "indices",
    [
        [0, 1, 2, 3],  # starts at 0 -> no seek at all
        [70, 71, 72],  # deep start -> the case the seek exists for
        [40, 40, 41, 41, 42],  # repeats, as when the output grid oversamples
        [5],  # single frame
        [N_FRAMES - 1],  # last frame
    ],
    ids=["from-zero", "deep-start", "repeats", "single", "last"],
)
def test_seek_yields_pixel_identical_frames(clip, indices):
    got = list(VideoSource(clip, np.array(indices)))
    want = _sequential(clip, indices)
    assert len(got) == len(want)
    for k, (a, b) in enumerate(zip(got, want)):
        assert np.array_equal(a, b), (
            f"index {indices[k]} differs between seek and sequential"
        )


def test_seek_lands_on_the_requested_frame(clip):
    """Content check independent of the sequential path.

    Distinguishes "both paths are wrong the same way" from "both are right" by
    comparing against the frame that was actually written for that index.
    """
    idx = 60
    (got,) = list(VideoSource(clip, np.array([idx])))
    expect = cv2.cvtColor(_frame(idx), cv2.COLOR_BGR2RGB)
    # Lossy codec: compare the dominant colour, which the fixture separates widely.
    assert np.abs(got.mean(axis=(0, 1)) - expect.mean(axis=(0, 1))).max() < 24


class _CapProxy:
    """Delegates to a real ``cv2.VideoCapture``, counting reads and optionally
    lying about the post-seek position.

    ``cv2.VideoCapture`` is an immutable extension type, so its methods cannot
    be monkeypatched; wrapping the capture the module is handed is the way in.
    """

    def __init__(self, cap, *, lie_pos=None, counter=None):
        self._cap, self._lie_pos, self._counter = cap, lie_pos, counter

    def read(self, *a, **kw):
        if self._counter is not None:
            self._counter["n"] += 1
        return self._cap.read(*a, **kw)

    def set(self, *a, **kw):
        return self._cap.set(*a, **kw)

    def get(self, prop):
        if self._lie_pos is not None and prop == cv2.CAP_PROP_POS_FRAMES:
            return self._lie_pos
        return self._cap.get(prop)


def _patch_capture(monkeypatch, **proxy_kw):
    import contextlib

    from kiwi.export import video_source as mod

    real = mod.video_capture

    @contextlib.contextmanager
    def wrapped(path):
        with real(path) as (cap, info):
            yield _CapProxy(cap, **proxy_kw), info

    monkeypatch.setattr(mod, "video_capture", wrapped)


@pytest.mark.parametrize(
    "lie_pos",
    [999999.0, 12.5, -1.0],
    ids=["past-target", "non-integer", "negative"],
)
def test_untrusted_seek_falls_back_instead_of_yielding_a_wrong_frame(
    clip, monkeypatch, lie_pos
):
    """A backend that reports an unusable landing must degrade to the slow path."""
    indices = [70, 71, 72]
    want = _sequential(clip, indices)
    _patch_capture(monkeypatch, lie_pos=lie_pos)
    got = list(VideoSource(clip, np.array(indices)))
    assert len(got) == len(want)
    for k, (a, b) in enumerate(zip(got, want)):
        assert np.array_equal(a, b), f"index {indices[k]} wrong after an untrusted seek"


def test_seek_actually_avoids_the_preamble(clip, monkeypatch):
    """The point of the seek: a deep start must not read every earlier frame."""
    counter = {"n": 0}
    _patch_capture(monkeypatch, counter=counter)
    list(VideoSource(clip, np.array([80, 81])))
    # Decoding from 0 would take 82 reads; a keyframe-anchored seek takes fewer.
    assert counter["n"] < 82, (
        f"{counter['n']} reads suggests the preamble was still decoded"
    )


def test_fallback_path_still_reads_the_preamble(clip, monkeypatch):
    """Guards the guard: if the fallback silently seeked too, the test above is vacuous."""
    counter = {"n": 0}
    _patch_capture(monkeypatch, lie_pos=999999.0, counter=counter)
    list(VideoSource(clip, np.array([80, 81])))
    assert counter["n"] >= 82, f"{counter['n']} reads: fallback did not decode from 0"


def test_odd_resize_dimension_is_refused(clip):
    """yuv420p needs even dims; an odd resize must fail at construction, not
    at encode time deep into a long export."""
    with pytest.raises(ValueError, match="odd dimension"):
        VideoSource(clip, np.array([0]), resize=(63, 64))

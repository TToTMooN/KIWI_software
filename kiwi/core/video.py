"""Uniform video I/O: open a capture, read its metadata, iterate frames, release.

  - :func:`open_capture` - open or raise one uniform ``OSError``.
  - :class:`VideoInfo` - width / height / fps (30.0 fallback) / n_frames.
  - :func:`video_capture` - context manager yielding ``(cap, info)``, release
    guaranteed even on exception.
  - :func:`iter_frames` - ``(frame_idx, bgr)`` read sequentially, optional
    ``stride`` / ``start``; accepts a path (opens+releases) or a live capture.

Sequential batch/processing sites use :func:`video_capture` /
:func:`iter_frames`. Interactive players that hold a capture open across a
playback and seek around keep their own read loop but still open via
:func:`open_capture` and read metadata via :class:`VideoInfo`.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "VideoInfo",
    "open_capture",
    "video_capture",
    "iter_frames",
]


def open_capture(path: str | Path) -> cv2.VideoCapture:
    """Open ``path`` as a ``cv2.VideoCapture`` or raise ``OSError``.

    The single place the "could not open" check lives. Caller owns the
    returned capture (use :func:`video_capture` to have release handled).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"could not open video: {path}")
    return cap


@dataclasses.dataclass(frozen=True)
class VideoInfo:
    """Capture metadata, read once. ``fps`` falls back to 30.0 when the
    container reports 0/unknown."""

    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` - matches the ``cv2`` resize/writer arg order."""
        return (self.width, self.height)

    @classmethod
    def from_capture(cls, cap: cv2.VideoCapture) -> "VideoInfo":
        return cls(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    @classmethod
    def read(cls, path: str | Path) -> "VideoInfo":
        """Open ``path`` just long enough to read its metadata, then release."""
        with video_capture(path) as (_cap, info):
            return info


@contextlib.contextmanager
def video_capture(path: str | Path) -> Iterator[tuple[cv2.VideoCapture, VideoInfo]]:
    """Context-managed capture: yields ``(cap, info)``; releases on exit.

    ``with video_capture(path) as (cap, info): ...`` - the release happens
    even if the body raises.
    """
    cap = open_capture(path)
    try:
        yield cap, VideoInfo.from_capture(cap)
    finally:
        cap.release()


def iter_frames(
    source: str | Path | cv2.VideoCapture,
    *,
    stride: int = 1,
    start: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_idx, bgr)`` reading sequentially from ``start``.

    ``stride`` yields every Nth frame counting from ``start`` (e.g. ``start=0,
    stride=6`` -> 0, 6, 12, ...). ``frame_idx`` is the true frame index, not
    the yield count. Reads sequentially and skips.

    ``source`` may be a path (opened and released here) or a live capture
    (caller keeps ownership; iteration starts wherever the capture is, unless
    ``start`` is given, and the capture is NOT released).
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    owns = not isinstance(source, cv2.VideoCapture)
    cap = open_capture(source) if owns else source
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if (idx - start) % stride == 0:
                yield idx, bgr
            idx += 1
    finally:
        if owns:
            cap.release()

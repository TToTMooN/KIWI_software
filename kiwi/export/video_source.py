"""A :class:`~kiwi.core.observation_episode.FrameSource` over an mp4 file.

Given a per-timestamp list of source frame indices (length ``N``,
non-decreasing, repeats allowed when the grid oversamples),
:class:`VideoSource` seeks ONCE to the first requested index and then decodes
*sequentially* from there, yielding the frame for each requested index in
order. Frames come out as ``(H, W, 3)`` ``uint8`` **RGB** (cv2 decodes BGR),
optionally resized. This is the only cv2-touching module in the export
package.

``_seek_to`` accepts a landing only when the backend reports an exact integer
position at or before the target, and otherwise rewinds and decodes forward
from frame 0.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from kiwi.core.video import VideoInfo, video_capture

__all__ = ["VideoSource"]


@dataclasses.dataclass
class VideoSource:
    """Lazy RGB frame provider for one camera over a fixed index schedule."""

    path: Path
    frame_indices: np.ndarray  # (N,) int, non-decreasing
    resize: tuple[int, int] | None = None  # (width, height); even dims for yuv420p

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        info = VideoInfo.read(self.path)
        self._src_size = (info.width, info.height)
        if self.resize is not None:
            self.width, self.height = int(self.resize[0]), int(self.resize[1])
        else:
            self.width, self.height = info.width, info.height
        if self.width % 2 or self.height % 2:
            raise ValueError(
                f"frame size {self.width}x{self.height} has an odd dimension; "
                f"yuv420p video needs even width and height (adjust --resize)"
            )

    def __len__(self) -> int:
        return len(self.frame_indices)

    def _seek_to(self, cap: "cv2.VideoCapture", target: int) -> int:
        """Jump toward ``target``; return the index of the next frame ``cap`` will read.

        Returns 0 (i.e. "no jump happened") whenever the seek cannot be
        TRUSTED, so the caller falls back to decoding forward from the start.
        """
        if target <= 0:
            return 0
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
            return 0
        landed = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Only honour a landing that is exact-integer and AT OR BEFORE the
        # target: at-or-before is still correct (we decode the remainder
        # forward), past it is not recoverable without re-opening.
        if landed != int(landed) or not 0 <= int(landed) <= target:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return 0
        return int(landed)

    def __iter__(self) -> Iterator[np.ndarray]:
        with video_capture(self.path) as (cap, _info):
            # Skip straight to the first frame we actually want. Only this ONE
            # jump is a seek; the rest of the non-decreasing schedule is
            # served by sequential reads.
            first = int(self.frame_indices[0]) if len(self.frame_indices) else 0
            cur = self._seek_to(cap, first) - 1
            frame = None  # last decoded BGR
            for target in self.frame_indices:
                target = int(target)
                while cur < target:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    cur += 1
                    frame = bgr
                if cur < target:
                    raise OSError(
                        f"{self.path}: could not read frame {target} "
                        f"(decoding stopped after {cur + 1} frames)"
                    )
                yield self._to_rgb(frame)

    def _to_rgb(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (self.width, self.height) != self._src_size:
            rgb = cv2.resize(
                rgb, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
        return rgb

"""Running per-feature statistics for ``meta/stats.json``.

LeRobot normalizes inputs with per-feature ``min`` / ``max`` / ``mean`` /
``std``. Tabular features get exact stats over every row; image features get
per-channel stats (shape ``(3, 1, 1)``, values in ``[0, 1]``) accumulated over
a sample of frames.
Accumulation is in float64 (sum / sum-of-squares / min / max) and finalized
once at the end.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from kiwi.export.schema import (
    CAMERA_HEIGHT_KEY_PREFIX,
    GROUND_PLANE_KEY,
    HEAD_MARKER_POSE_KEY,
    IMAGE_KEY_PREFIX,
)

__all__ = ["RunningStats"]


@dataclasses.dataclass
class _Accum:
    """Streaming sum / sum-sq / min / max over an arbitrary trailing shape."""

    count: int = 0
    _sum: np.ndarray | None = None
    _sumsq: np.ndarray | None = None
    _min: np.ndarray | None = None
    _max: np.ndarray | None = None

    def add(self, rows: np.ndarray) -> None:
        """Accumulate ``rows`` shaped ``(n, *feat)`` (reduce over axis 0)."""
        rows = np.asarray(rows, dtype=np.float64)
        n = rows.shape[0]
        if n == 0:
            return
        s = rows.sum(axis=0)
        sq = np.square(rows).sum(axis=0)
        lo = rows.min(axis=0)
        hi = rows.max(axis=0)
        if self._sum is None:
            self._sum, self._sumsq, self._min, self._max = s, sq, lo, hi
        else:
            self._sum += s
            self._sumsq += sq
            self._min = np.minimum(self._min, lo)
            self._max = np.maximum(self._max, hi)
        self.count += n

    def result(self) -> dict:
        mean = self._sum / self.count
        var = np.maximum(self._sumsq / self.count - np.square(mean), 0.0)
        std = np.sqrt(var)
        return {
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "count": [self.count],
        }


class RunningStats:
    """Accumulate per-feature stats across episodes, then emit ``meta/stats.json``."""

    def __init__(self) -> None:
        self._acc: dict[str, _Accum] = {}
        self._img_frames: dict[str, int] = {}  # image keys report frame count

    def _get(self, key: str) -> _Accum:
        return self._acc.setdefault(key, _Accum())

    def add_tabular(self, key: str, rows: np.ndarray) -> None:
        """``rows`` is ``(n, D)`` (a vector feature) or ``(n,)`` (a scalar column)."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows[:, None]
        if (
            key == HEAD_MARKER_POSE_KEY
            or key == GROUND_PLANE_KEY
            or key.startswith(CAMERA_HEIGHT_KEY_PREFIX)
        ):
            # Optional columns are NaN where their source is absent; an
            # episode without a floor contributes nothing, not zeros.
            rows = rows[np.isfinite(rows).all(axis=tuple(range(1, rows.ndim)))]
        if len(rows):
            self._get(key).add(rows)

    def add_image(self, key: str, frame_rgb: np.ndarray) -> None:
        """Accumulate one ``(H, W, 3)`` uint8 RGB frame's pixels as per-channel stats [0,1].

        Updates only the mean/std/min/max accumulator (over a sampled subset of
        frames). The reported ``count`` is the frame count, set separately via
        :meth:`add_image_frames`.
        """
        f = np.asarray(frame_rgb, dtype=np.float64) / 255.0
        # Per-channel: reduce H,W into one "row" of 3 channel-values per pixel.
        pixels = f.reshape(-1, f.shape[-1])  # (H*W, 3)
        self._get(key).add(pixels)
        self._img_frames.setdefault(key, 0)

    def add_image_frames(self, key: str, n_frames: int) -> None:
        """Record an image feature's frame count (what LeRobot reports as ``count``)."""
        self._img_frames[key] = self._img_frames.get(key, 0) + int(n_frames)

    def result(self) -> dict:
        """Final stats dict. Image keys are reshaped to LeRobot's ``(C, 1, 1)``.

        For image keys ``count`` reports the number of frames (matching
        LeRobot), not the pixel count the accumulator used to compute
        per-channel mean/std.
        """
        out: dict = {}
        for key, acc in self._acc.items():
            r = acc.result()
            if key.startswith(IMAGE_KEY_PREFIX):
                # Channel stats live as (C, 1, 1) nested lists in LeRobot.
                for k in ("min", "max", "mean", "std"):
                    r[k] = [[[v]] for v in r[k]]
                r["count"] = [self._img_frames.get(key, 0)]
            out[key] = r
        return out

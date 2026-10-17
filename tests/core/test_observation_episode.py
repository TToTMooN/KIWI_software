"""Unit tests for the pure ObservationEpisode datatype (no cv2 / no I/O)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np
import pytest

from kiwi.core.observation_episode import VIDEO_DTYPE, FeatureColumn, ObservationEpisode


@dataclasses.dataclass
class _FakeFrames:
    n: int
    height: int = 4
    width: int = 6

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.n):
            yield np.full((self.height, self.width, 3), i % 255, dtype=np.uint8)


def _episode(n: int = 5) -> ObservationEpisode:
    state = np.zeros((n, 8), dtype=np.float32)
    feats = {
        "observation.state": FeatureColumn(
            "observation.state", "float32", (8,), ["a"] * 8, state
        ),
        "observation.images.head": FeatureColumn(
            "observation.images.head",
            VIDEO_DTYPE,
            (4, 6, 3),
            ["height", "width", "channels"],
            _FakeFrames(n),
        ),
    }
    return ObservationEpisode(
        fps=30.0,
        timestamps=np.arange(n) / 30.0,
        task="t",
        frame="marker",
        features=feats,
    )


def test_len_and_splits():
    ep = _episode(5)
    assert len(ep) == ep.num_frames == 5
    assert set(ep.tabular_features()) == {"observation.state"}
    assert set(ep.video_features()) == {"observation.images.head"}


def test_validate_ok():
    _episode(7).validate()  # no raise


def test_validate_row_mismatch():
    ep = _episode(5)
    ep.features["observation.state"].data = np.zeros((4, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="rows"):
        ep.validate()


def test_validate_shape_mismatch():
    ep = _episode(5)
    ep.features["observation.state"] = FeatureColumn(
        "observation.state", "float32", (8,), None, np.zeros((5, 7), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="per-frame shape"):
        ep.validate()


def test_video_shape_mismatch():
    ep = _episode(5)
    ep.features["observation.images.head"].shape = (8, 8, 3)
    with pytest.raises(ValueError, match="video feature"):
        ep.validate()

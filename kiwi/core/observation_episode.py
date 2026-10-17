"""The canonical, format-agnostic representation of one recorded episode.

``ObservationEpisode`` is the hub datatype: a finished demonstration is loaded
into it exactly once, and every export target (LeRobot v2.1 / v3 / ...) is
written *from* it. It lives in ``core`` (next to
:class:`kiwi.core.trajectory.Trajectory`): numpy arrays + a small schema, no
cv2 / no robot / no dataset-format knowledge.

**Observation-only.** It holds what was *measured*: per-arm end-effector pose,
gripper, camera video, optionally joints. It does NOT carry an ``action``
channel; the export writer synthesizes ``action`` from ``observation.state``.

Tabular features carry their per-frame data inline as ``(N, *shape)`` arrays.
Video features carry a :class:`FrameSource` instead - a lazy, ordered provider of
exactly ``N`` frames aligned to ``timestamps`` - so a multi-camera episode never
holds all its decoded frames in memory at once.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import numpy as np

VIDEO_DTYPE = "video"


@runtime_checkable
class FrameSource(Protocol):
    """A lazy, ordered source of exactly ``len(self)`` RGB frames.

    Iterating yields ``(H, W, 3)`` ``uint8`` RGB frames in timeline order, one per
    episode timestamp. Implementations (``kiwi.export``'s video source) decode on
    demand so frames are never all held in RAM. Kept here as a Protocol:
    ``core`` knows the contract, not the decoder.
    """

    height: int
    width: int

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[np.ndarray]: ...


@dataclasses.dataclass
class FeatureColumn:
    """One per-frame channel of an episode, plus how it maps to a dataset feature.

    ``dtype`` is a LeRobot-style dtype string: ``"float32"`` / ``"int64"`` for
    tabular data, or :data:`VIDEO_DTYPE` (``"video"``) for camera frames. ``shape``
    is the per-frame shape (e.g. ``(16,)`` for a state vector, ``(480, 640, 3)`` for
    an image). ``names`` labels the components of a tabular vector (or the
    ``["height", "width", "channels"]`` axes of a video). ``data`` is an
    ``(N, *shape)`` array for tabular features, or a :class:`FrameSource` for video.
    """

    key: str
    dtype: str
    shape: tuple[int, ...]
    names: list[str] | None = None
    data: Any = None  # np.ndarray (N, *shape) for tabular; FrameSource for video

    @property
    def is_video(self) -> bool:
        return self.dtype == VIDEO_DTYPE

    def __len__(self) -> int:
        return len(self.data)


@dataclasses.dataclass
class ObservationEpisode:
    """One episode of measured observations on a single uniform clock.

    ``timestamps`` is the shared, monotonically increasing clock (seconds, starting
    at 0) every feature is sampled on at ``fps``. ``frame`` is an OPAQUE name
    copied verbatim from the world trajectory's own header - nothing here
    interprets it. Known values: ``"shared_rear_metric_gravity_aligned"`` (a
    scene's map frame) and ``"marker"`` (the ArUco environment anchor).
    ``features`` maps a LeRobot-style key (``"observation.state"``,
    ``"observation.images.left"``, ...) to its :class:`FeatureColumn`. ``meta``
    carries provenance (source paths, robot type, units, clock offsets) -
    free-form, not part of the per-frame schema.
    """

    fps: float
    timestamps: np.ndarray  # (N,) seconds, uniform at fps, starts at 0
    task: str
    frame: str  # opaque; e.g. shared_rear_metric_gravity_aligned | marker
    features: dict[str, FeatureColumn]
    meta: dict = dataclasses.field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def num_frames(self) -> int:
        return len(self.timestamps)

    def tabular_features(self) -> dict[str, FeatureColumn]:
        """Features whose data is an in-memory ``(N, *shape)`` array."""
        return {k: f for k, f in self.features.items() if not f.is_video}

    def video_features(self) -> dict[str, FeatureColumn]:
        """Features backed by a :class:`FrameSource` (camera streams)."""
        return {k: f for k, f in self.features.items() if f.is_video}

    def validate(self) -> None:
        """Check every feature has exactly ``num_frames`` rows and the right shape.

        Raises ``ValueError`` on the first mismatch. Cheap structural sanity - call
        it before handing the episode to a writer.
        """
        n = self.num_frames
        if self.timestamps.ndim != 1:
            raise ValueError(
                f"timestamps must be 1-D, got shape {self.timestamps.shape}"
            )
        for key, f in self.features.items():
            if len(f) != n:
                raise ValueError(
                    f"feature {key!r} has {len(f)} rows but episode has {n} frames"
                )
            if f.is_video:
                if tuple(f.shape) != (f.data.height, f.data.width, 3):
                    raise ValueError(
                        f"video feature {key!r} shape {tuple(f.shape)} != "
                        f"({f.data.height}, {f.data.width}, 3)"
                    )
            else:
                arr = np.asarray(f.data)
                if arr.shape[1:] != tuple(f.shape):
                    raise ValueError(
                        f"feature {key!r} per-frame shape {arr.shape[1:]} != {tuple(f.shape)}"
                    )

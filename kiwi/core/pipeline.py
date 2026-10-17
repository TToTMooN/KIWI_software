"""Shared frame data and the tool-state estimator contract.

EEF batch processing and the interactive tuner use :class:`Detections`
and :class:`GripperState` to exchange per-frame inputs and results.
:class:`GripperEstimator` defines the structural interface implemented by
each tool's estimator; it requires no inheritance.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np


@dataclasses.dataclass
class Detections:
    """Marker corners + IDs found in a single frame.

    ``corners[i]`` is the (4, 2) float32 pixel quad for the marker with
    ID ``ids[i]``. Empty arrays when nothing was detected.
    """

    corners: np.ndarray  # (N, 4, 2) float32
    ids: np.ndarray  # (N,) int32


@dataclasses.dataclass
class GripperState:
    """One frame's tool-state estimate, tool-agnostic.

    ``value`` is the scalar tool state - the hinge open angle for Chopsticks.
    ``None`` means the frame produced no measurement (a dropout); the eef
    artifact stores it as NaN. The unit is a property of the estimator
    (declared on its ``kiwi.eef`` registry bundle), never per frame.

    Validity is ``value is not None`` and nothing else. A gate-rejected fit
    keeps its diagnostics - ``error`` carries the reason string and
    ``rejected_value`` the value the gate refused, for diagnosis only, never
    for use - but a whole-series post-pass may also revoke a value after the
    fact, so the absence of ``error`` proves nothing.

    ``H_plane_to_img`` / ``H_img_to_plane`` are set when the estimator anchors
    a measurement plane (Chopsticks' marker plane). ``n_detected`` / ``n_kept``
    count feature candidates before / after the estimator's spatial filters.
    Everything tool-specific - fit mode, per-side line fits, asymmetry, kept
    dot positions - lives in ``detail``; generic consumers never read it.
    """

    value: float | None = None
    error: str | None = None
    rejected_value: float | None = None
    has_pose: bool = False
    n_detected: int = 0
    n_kept: int = 0
    H_plane_to_img: np.ndarray | None = None
    H_img_to_plane: np.ndarray | None = None
    detail: dict[str, object] = dataclasses.field(default_factory=dict)


@runtime_checkable
class GripperEstimator(Protocol):
    """Per-frame tool-state estimator.

    Consumes the rectified frame (needs raw pixels for dot detection),
    the marker :class:`Detections` from the tool's detector, and the
    rectified-camera intrinsics. Returns a ``GripperState``. A missing pose
    or rejected fit has ``value=None``; ``error`` carries a reason where
    recorded and may be None when no pose was available.

    Implementations own the choice of *which* marker anchors the tool
    plane - they filter ``Detections`` for their configured marker ID, and
    a fixed-pose implementation may ignore ``Detections`` entirely.
    """

    def __call__(
        self,
        rectified: np.ndarray,
        detections: Detections,
        K: np.ndarray,
    ) -> GripperState: ...

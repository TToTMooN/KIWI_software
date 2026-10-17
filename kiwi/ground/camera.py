"""Inputs: a Camera = video + poses of its optical centre + kb4 intrinsics.

Pose contract: t [s], position and R_world_from_camera of the camera OPTICAL
CENTRE in a METRIC frame whose z axis is gravity-up; row i of the pose table
IS frame i of the video (no resampling). Camera frame is OpenCV (x right, y
down, z forward). Frames must be square: one isotropic scale applies to the
kb4 parameters when frames are decoded at a smaller size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Camera:
    name: str
    video: Path
    t: np.ndarray
    """(N,) seconds on the episode clock."""
    pos: np.ndarray
    """(N, 3) optical centre."""
    R: np.ndarray
    """(N, 3, 3) R_world_from_camera."""
    kb4: dict
    """fx fy cx cy k1..k4 at image_size."""
    image_size: int
    """Native square frame size the kb4 parameters refer to."""

    @property
    def off_nadir_deg(self) -> np.ndarray:
        return np.degrees(np.arccos(np.clip(-self.R[:, 2, 2], -1, 1)))

    @property
    def azimuth(self) -> np.ndarray:
        return np.arctan2(self.R[:, 1, 2], self.R[:, 0, 2])

    @property
    def fps(self) -> float:
        return float(1 / np.median(np.diff(self.t)))


def kb4_scaled(kb4: dict, scale: float) -> dict:
    """The kb4 parameters for a frame resized by ``scale`` (isotropic)."""
    return {
        **kb4,
        "fx": kb4["fx"] * scale,
        "fy": kb4["fy"] * scale,
        "cx": kb4["cx"] * scale,
        "cy": kb4["cy"] * scale,
    }


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TUM ``t tx ty tz qx qy qz qw`` -> (t, pos, R_world_from_camera).

    t in seconds and strictly increasing, one row per frame.
    """
    d = np.loadtxt(path, comments="#", ndmin=2)
    t = d[:, 0]
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError(f"{path}: the time column must be strictly increasing")
    if np.median(dt) > 1:
        raise ValueError(
            f"{path}: TUM time must be in seconds (median frame spacing "
            f"{np.median(dt):.3g})"
        )
    return t, d[:, 1:4], Rotation.from_quat(d[:, 4:8]).as_matrix()


def square_size(kb4_doc: dict, source: str) -> int:
    """The native frame size a kb4 document was fitted at; square frames only."""
    res = kb4_doc.get("resolution")
    if not res or len(res) != 2:
        raise ValueError(f"{source}: kb4 document needs 'resolution': [W, H]")
    if res[0] != res[1]:
        raise ValueError(
            f"{source}: the ground stage needs square frames (W == H), got {res}"
        )
    return int(res[0])

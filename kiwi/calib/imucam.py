"""Camera<->IMU extrinsics: the Kalibr solve's inputs and the physical cross-check.

The solve itself runs in the container (:mod:`kiwi.calib.kalibr.docker`);
what lives here is the front/back acceptance test. The two lenses of an X5
share one IMU, so the front and rear solves are not independent: composing
them must recover the camera's physical back-to-back geometry,

    T_back_front = T_camB_imu @ inv(T_camF_imu)   ->  ~180 deg, ~30 mm

The check needs BOTH lenses. A unit with one calibrated lens has no such
verification available - report that as "unverifiable", never as a pass.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "ANGLE_TOL_DEG",
    "DIST_MAX_MM",
    "DIST_MIN_MM",
    "FrontBackCheck",
    "front_back_consistency",
]

# Physical bounds for an X5's two lenses: back-to-back (180 deg) with the
# optical centres separated by the body depth.
ANGLE_TOL_DEG = 1.5
DIST_MIN_MM, DIST_MAX_MM = 20.0, 40.0


@dataclasses.dataclass(frozen=True)
class FrontBackCheck:
    """One unit's composition result, with the reasons a bound failed."""

    angle_deg: float  # rotation magnitude of T_back_front (expect ~180)
    dist_mm: float  # translation magnitude (expect the body depth, ~30 mm)
    reasons: tuple[str, ...]  # empty = PASS

    @property
    def ok(self) -> bool:
        return not self.reasons


def front_back_consistency(
    T_front_cam_imu: np.ndarray, T_back_cam_imu: np.ndarray
) -> FrontBackCheck:
    """Compose the two solves and judge them against the physical geometry.

    Both inputs are ``cam0.T_cam_imu`` matrices (``p_cam = T @ p_imu``) as
    read by :func:`kiwi.core.camera.extrinsics.load_T_cam_imu` - the
    top-level key is ``cam0`` in BOTH files; the artifact name says which
    lens. A FAIL means the two solves disagree about the shared IMU:
    suspect, in order, the rear run consuming the wrong lens/topic or the
    wrong ``ds_*``, too few views, or a badly locked board pose - and
    re-run the solve.
    """
    T_bf = np.asarray(T_back_cam_imu) @ np.linalg.inv(np.asarray(T_front_cam_imu))
    angle = float(
        np.degrees(np.linalg.norm(Rotation.from_matrix(T_bf[:3, :3]).as_rotvec()))
    )
    dist_mm = float(np.linalg.norm(T_bf[:3, 3]) * 1000.0)
    reasons: list[str] = []
    if abs(angle - 180.0) > ANGLE_TOL_DEG:
        reasons.append(f"angle off 180 by {abs(angle - 180.0):.2f} deg")
    if not (DIST_MIN_MM <= dist_mm <= DIST_MAX_MM):
        reasons.append(f"dist outside {DIST_MIN_MM:.0f}-{DIST_MAX_MM:.0f} mm")
    return FrontBackCheck(angle_deg=angle, dist_mm=dist_mm, reasons=tuple(reasons))

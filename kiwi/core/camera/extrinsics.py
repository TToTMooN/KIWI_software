"""Readers for the per-camera extrinsic calibration artifacts.

The rule this module exists to enforce: **artifact readers live in
``kiwi.core``; artifact writers live in ``kiwi.calib``.** ``calib`` publishes
``extrinsic-imu*.yaml`` into the camera registry; state estimation, scene
mapping and ``real2sim`` consume them. A component may not import another
component (see ``tests/test_boundaries.py``), so the one parser both sides
call lives here, keyed by the same artifact names as
:data:`kiwi.core.camera.registry.ARTIFACT_NAMES`.

Every consumer reads ``cam0.T_cam_imu`` through this one reader, so all of
them get the same checks: the ``%YAML:1.0`` directive OpenCV-style chains
open with is stripped, and the matrix is shape-checked.

Frame conventions (stated once, relied on everywhere):

* ``T_cam_imu``: ``p_cam = T_cam_imu @ p_imu`` (Kalibr's ``R_ItoC | p_IinC``).
  The top-level key is ``cam0`` in BOTH the front and rear files - that is
  Kalibr's single-camera output convention, not a claim about which lens; the
  *artifact name* (``extrinsic_imu`` vs ``extrinsic_imu_back``) says which.
  A chain spelling the INVERSE (``T_imu_cam``) is refused with instructions,
  never silently inverted here.
* The clock equation is frozen as ``t_imu = t_camera + dt_imu_from_camera``,
  and ``dt_imu_from_camera`` IS Kalibr's ``timeshift_cam_imu`` (the same
  camera-to-IMU time-shift convention: adding it to a camera
  timestamp lands on the IMU clock). Consumers use
  :attr:`CamImuExtrinsic.dt_imu_from_camera_s`.
* The hand-eye (``T_flange_cam``, the ``camera_lens0`` joint) and the tool tip
  are NOT camera-registry artifacts: both live in the descriptions tree, and
  export composes camera->tip from the rendered URDF through
  :func:`kiwi.core.descriptions.fixed_frame_T`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from kiwi.core.opencv_yaml import load_opencv_yaml

__all__ = [
    "CamImuExtrinsic",
    "load_T_cam_imu",
    "read_cam_imu",
]


@dataclasses.dataclass(frozen=True)
class CamImuExtrinsic:
    """One lens's Kalibr cam-IMU solve: the transform plus its time offset.

    ``timeshift_s`` is Kalibr's ``timeshift_cam_imu`` (seconds), ``None``
    when the chain omits it, as a hand-built one may - consumers that need
    it must check, not default.
    """

    T_cam_imu: np.ndarray  # (4, 4): p_cam = T_cam_imu @ p_imu
    timeshift_s: float | None

    @property
    def dt_imu_from_camera_s(self) -> float | None:
        """The frozen clock equation: ``t_imu = t_camera + this``.

        Kalibr's ``timeshift_cam_imu`` already has that direction (adding it
        to a camera timestamp lands on the IMU clock).
        """
        return self.timeshift_s


def read_cam_imu(path: Path) -> CamImuExtrinsic:
    """Read ``cam0.T_cam_imu`` (+ optional timeshift) from a camchain YAML.

    Accepts both plain-YAML registry files (``extrinsic-imu*.yaml``) and
    OpenCV-style chains with a ``%YAML:1.0`` directive, read through
    :func:`kiwi.core.opencv_yaml.load_opencv_yaml`. Raises ``ValueError``
    when the file has no ``cam0.T_cam_imu`` or the matrix is not 4x4.
    """
    path = Path(path)
    cfg = load_opencv_yaml(path.read_text())
    cam0 = cfg.get("cam0")
    if isinstance(cam0, dict) and "T_cam_imu" not in cam0 and "T_imu_cam" in cam0:
        raise ValueError(
            f"{path}: cam0 spells the INVERSE transform (T_imu_cam). The "
            f"registry stores T_cam_imu - invert it and republish the "
            f"artifact."
        )
    if not isinstance(cam0, dict) or "T_cam_imu" not in cam0:
        raise ValueError(f"{path}: no cam0.T_cam_imu (not a Kalibr camchain?)")
    T = np.asarray(cam0["T_cam_imu"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"cam0.T_cam_imu in {path} is {T.shape}, expected (4, 4)")
    ts = cam0.get("timeshift_cam_imu")
    return CamImuExtrinsic(T_cam_imu=T, timeshift_s=None if ts is None else float(ts))


def load_T_cam_imu(path: Path) -> np.ndarray:
    """``cam0.T_cam_imu`` (``p_cam = T @ p_imu``) from a Kalibr camchain YAML."""
    return read_cam_imu(path).T_cam_imu

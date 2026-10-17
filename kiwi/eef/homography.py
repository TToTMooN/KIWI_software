"""Fixed camera<->marker pose calibration (the "tool homography").

The camera is rigidly mounted to the tool, so the marker-plane pose is
constant. This recovers it once from a clip - detect the marker, ``solvePnP``
over well-conditioned frames, take the robust central pose (chordal-mean
rotation + per-component median translation) - and writes it as
``build.fixed_pose`` into the gripper unit's config
(``config/gripper/<tool>/units/<unit>.yaml``, ``chopstick_unit_v1``),
together with
the tag identity it was calibrated against (``build.marker.id`` / ``.size``):
the pose is bound to the tag's placement and linear in its size, so the three
are one coherent build record.

Downstream, the estimator rebuilds the plane<->image homography from this
pose with the live rectifier K, so the per-frame path can skip PnP when the
marker is blurred, occluded, or out of frame. Angle estimation still needs
usable jaw dots.
``rvec``/``tvec`` are the physical marker pose in the rectified-pinhole
frame, so the ``fov_deg`` / ``out_size`` / ``lens`` used here are recorded
alongside; reuse the same rectifier settings at runtime (and recalibrate if
the camera is remounted).

:func:`write_fixed_pose` updates the unit YAML in place and refuses to replace
an existing calibrated ``fixed_pose`` without ``force``. Calibration returns
the pose spread in :class:`FixedPoseResult`; the CLI prints its summary so the
operator can assess whether the mounted pose is sufficiently rigid.

This module owns tool-mounted marker-pose calibration and uses the same
rectification and marker geometry as the EEF estimator.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from kiwi.core.markers import (
    detect_aruco_markers,
    marker_object_points,
    resolve_aruco_dict,
)
from kiwi.core.tool_config import GRIPPER_UNIT_SCHEMA
from kiwi.core.video import VideoInfo, iter_frames

__all__ = [
    "RIGIDITY_ROT_P90_WARN_DEG",
    "RIGIDITY_TRANS_P90_WARN_MM",
    "FixedPoseResult",
    "aggregate_pose",
    "calibrate_fixed_pose",
    "write_fixed_pose",
]

# For a rigid camera<->marker mount the per-frame pose cluster is tight;
# beyond these the pose is not actually constant (flex / wrong marker /
# marker moved) and warrants inspection before reusing the fixed pose.
RIGIDITY_ROT_P90_WARN_DEG = 3.0
RIGIDITY_TRANS_P90_WARN_MM = 10.0


def aggregate_pose(rvecs, tvecs) -> dict:
    """Robust central pose from per-frame ``(rvec, tvec)`` samples.

    Rotation: the chordal mean (``Rotation.mean``); translation:
    per-component median (robust to a few bad frames). Also returns spread
    diagnostics - for a rigid camera<->marker mount the cluster is tight; a
    large spread means the pose is not actually constant and the fixed pose
    should not be trusted.
    """
    rvecs = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
    tvecs = np.asarray(tvecs, dtype=np.float64).reshape(-1, 3)
    rots = Rotation.from_rotvec(rvecs)
    mean_rot = rots.mean()
    tvec_med = np.median(tvecs, axis=0)
    ang_dev_deg = np.degrees((rots * mean_rot.inv()).magnitude())
    t_dev_mm = np.linalg.norm(tvecs - tvec_med, axis=1) * 1000.0
    return {
        "rvec": mean_rot.as_rotvec(),
        "tvec": tvec_med,
        "rot_dev_p50": float(np.median(ang_dev_deg)),
        "rot_dev_p90": float(np.percentile(ang_dev_deg, 90)),
        "t_dev_p50_mm": float(np.median(t_dev_mm)),
        "t_dev_p90_mm": float(np.percentile(t_dev_mm, 90)),
    }


@dataclasses.dataclass(frozen=True)
class FixedPoseResult:
    """One calibration run's pose + the evidence to judge it by."""

    rvec: np.ndarray
    tvec: np.ndarray
    fov_deg: float
    out_size: tuple[int, int]
    lens: str
    n_used: int
    reproj_med_px: float
    rot_dev_p50_deg: float
    rot_dev_p90_deg: float
    t_dev_p50_mm: float
    t_dev_p90_mm: float

    @property
    def warnings(self) -> list[str]:
        if (
            self.rot_dev_p90_deg > RIGIDITY_ROT_P90_WARN_DEG
            or self.t_dev_p90_mm > RIGIDITY_TRANS_P90_WARN_MM
        ):
            return [
                "large pose spread - the camera<->marker pose may not be "
                "rigid (flex / wrong marker / marker moved). Inspect before "
                "trusting this fixed_pose."
            ]
        return []

    def fixed_pose_block(self) -> dict:
        """The ``fixed_pose:`` mapping exactly as the estimator consumes it."""
        return {
            "rvec": [float(x) for x in self.rvec],
            "tvec": [float(x) for x in self.tvec],
            "fov_deg": float(self.fov_deg),
            "out_size": [int(self.out_size[0]), int(self.out_size[1])],
            "lens": self.lens,
        }

    def summary(self) -> str:
        lines = [
            f"fixed_pose from {self.n_used} frames: "
            f"rvec={np.round(self.rvec, 4).tolist()} "
            f"tvec={np.round(self.tvec, 4).tolist()} "
            f"(dist={np.linalg.norm(self.tvec) * 1000:.1f}mm, "
            f"reproj med={self.reproj_med_px:.2f}px)",
            f"pose spread (rigidity check): rot p50={self.rot_dev_p50_deg:.2f}deg "
            f"p90={self.rot_dev_p90_deg:.2f}deg; "
            f"trans p50={self.t_dev_p50_mm:.1f}mm p90={self.t_dev_p90_mm:.1f}mm",
            *(f"WARNING: {w}" for w in self.warnings),
        ]
        return "\n".join(lines)


def calibrate_fixed_pose(
    lens0_mp4: Path,
    intrinsics: Path,
    *,
    marker_id: int,
    marker_size_m: float,
    dictionary: str = "DICT_4X4_50",
    fov_deg: float = 160.0,
    out_size: tuple[int, int] = (1920, 1920),
    lens: str = "front",
    max_frames: int = 300,
    min_frames: int = 20,
    max_reproj_px: float = 2.0,
) -> FixedPoseResult:
    """Detect the marker + solvePnP over strided frames; aggregate robustly.

    ``marker_size_m`` sets the METRIC SCALE of the pose being calibrated -
    solvePnP translation is linear in it, so a stale size bakes a
    proportionally wrong ``tvec`` into the config without any error. Frames
    whose PnP reprojection error exceeds ``max_reproj_px`` are dropped;
    fewer than ``min_frames`` good detections raises.
    """
    from kiwi.core.rectify import load_rectifier_auto

    if not marker_size_m > 0:
        raise ValueError(f"marker_size_m must be positive, got {marker_size_m!r}")
    info = VideoInfo.read(lens0_mp4)
    rect = load_rectifier_auto(
        intrinsics,
        (info.width, info.height),
        lens=lens,
        out_size=out_size,
        fov_deg=fov_deg,
    )
    K = rect.K_pinhole
    dist0 = np.zeros(5, dtype=np.float64)
    obj_pts = marker_object_points(marker_size_m)
    dict_id = resolve_aruco_dict(dictionary)
    stride = max(1, info.n_frames // max_frames) if info.n_frames > 0 else 1

    rvecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []
    reproj: list[float] = []
    for _idx, bgr in iter_frames(lens0_mp4, stride=stride):
        undistorted = rect.apply(bgr)
        corners_list, _ids = detect_aruco_markers(
            undistorted, marker_dict=dict_id, allowed_ids={marker_id}
        )
        if not corners_list:
            continue
        corners = corners_list[0].reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, corners, K, dist0, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist0)
        err = float(np.sqrt(((proj.reshape(-1, 2) - corners) ** 2).sum(axis=1)).mean())
        if err > max_reproj_px:
            continue
        rvecs.append(rvec.reshape(3))
        tvecs.append(tvec.reshape(3))
        reproj.append(err)

    if len(rvecs) < min_frames:
        raise RuntimeError(
            f"only {len(rvecs)} good detections of marker {marker_id} "
            f"(< min_frames={min_frames}) in {lens0_mp4.name}; check the "
            f"marker id, the intrinsics, and the clip quality."
        )
    agg = aggregate_pose(rvecs, tvecs)
    return FixedPoseResult(
        rvec=agg["rvec"],
        tvec=agg["tvec"],
        fov_deg=float(fov_deg),
        out_size=(int(rect.out_size[0]), int(rect.out_size[1])),
        lens=lens,
        n_used=len(rvecs),
        reproj_med_px=float(np.median(reproj)),
        rot_dev_p50_deg=agg["rot_dev_p50"],
        rot_dev_p90_deg=agg["rot_dev_p90"],
        t_dev_p50_mm=agg["t_dev_p50_mm"],
        t_dev_p90_mm=agg["t_dev_p90_mm"],
    )


def write_fixed_pose(
    unit_path: Path,
    result: FixedPoseResult,
    *,
    tool: str | None = None,
    marker_id: int | None = None,
    marker_size_m: float | None = None,
    force: bool = False,
) -> Path:
    """Write ``result`` into the unit's ``build:`` block, preserving other keys.

    Creates a ``chopstick_unit_v1`` file when absent (recording ``tool`` - the
    unit->design link). The tag identity actually calibrated against -
    ``marker_id`` + ``marker_size_m`` - is written into ``build.marker``
    alongside the pose, REPLACING what the file said, so recalibrating after
    a re-sticker updates all three in one write.

    An existing calibrated ``build.fixed_pose`` is refused without
    ``force``. Note the rewrite goes through the YAML dumper, so hand-written
    comments in the unit file do not survive - review the git diff before
    committing.
    """
    unit_path = Path(unit_path)
    existing = yaml.safe_load(unit_path.read_text()) if unit_path.exists() else {}
    existing = existing or {}
    if (
        not force
        and ((existing.get("build") or {}).get("fixed_pose") or {}).get("rvec")
        is not None
    ):
        raise RuntimeError(
            f"{unit_path} already carries a calibrated build.fixed_pose; pass "
            f"--force to replace it (the git diff is the review record)."
        )
    existing.setdefault("schema", GRIPPER_UNIT_SCHEMA)
    if tool is not None:
        existing.setdefault("tool", tool)
    build = existing.setdefault("build", {})
    marker = build.setdefault("marker", {})
    if marker_id is not None:
        marker["id"] = int(marker_id)
    if marker_size_m is not None:
        marker["size"] = float(marker_size_m)
    build["fixed_pose"] = result.fixed_pose_block()
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(yaml.safe_dump(existing, sort_keys=False))
    return unit_path

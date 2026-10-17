"""Head-marker observations from the two fused arm states.

The head rig carries one ArUco marker. Each wrist rear camera sees it now and then, and
its pose follows from the arm's fused front-camera trajectory and the calibrated lens
geometry (``T_scene_rear = T_scene_front @ inv(T_rear_front)``), so every accepted
detection becomes a scene-frame marker pose ``T_scene_marker = T_scene_rear(t) @
T_rear_marker``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.state.geometry import interpolate_poses, se3, se3_inv

# The rear video is sampled at about 30 fps and each sampled frame is reduced to a 960 px
# edge before detection.
EFFECTIVE_FPS = 30.0
DETECTION_MAX_EDGE_PX = 960

# A detection is kept when the better planar-PnP solution fits clearly better than the
# other one and reprojects within this bound (pixels of the detection image).
AMBIGUITY_RATIO = 1.5
MAX_REPROJECTION_PX = 3.0

ARUCO_DICTS = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------


def marker_object_points(size_m: float) -> np.ndarray:
    h = size_m / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def _pose_matrix(rvec, tvec) -> np.ndarray:
    return se3(Rotation.from_rotvec(np.asarray(rvec).reshape(3)).as_matrix(), tvec)


def solve_marker_pose(
    corners_px: np.ndarray, size_m: float, K: np.ndarray, dist_kb4: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """IPPE-square PnP on KB4-undistorted corners.

    Returns ``(T_cam_marker, reprojection_rms_px, ambiguity_ratio)`` of the better of the
    two solutions. The reprojection error is measured in the raw distorted image; the
    ratio is second-best over best error.
    """
    obj = marker_object_points(size_m)
    pts = np.asarray(corners_px, np.float64).reshape(-1, 1, 2)
    pinhole = cv2.fisheye.undistortPoints(pts, K, dist_kb4, P=K).reshape(-1, 2)
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj, pinhole, K, np.zeros(4), flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        proj, _ = cv2.fisheye.projectPoints(
            obj.reshape(-1, 1, 3), rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist_kb4
        )
        delta = proj.reshape(-1, 2) - np.asarray(corners_px).reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
        candidates.append((_pose_matrix(rvec, tvec), err))
    candidates.sort(key=lambda c: c[1])
    return candidates[0][0], candidates[0][1], candidates[1][1] / candidates[0][1]


def detect_in_image(
    image: np.ndarray,
    detector,
    marker_id: int,
    size_m: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray | None:
    """``T_cam_marker`` of the marker with ``marker_id``, or None when it is not accepted."""
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return None
    hits = [c for c, i in zip(corners, ids.ravel(), strict=True) if int(i) == marker_id]
    if not hits:
        return None
    T, err, ratio = solve_marker_pose(
        np.asarray(hits[0], dtype=float).reshape(4, 2), size_m, K, dist
    )
    if ratio < AMBIGUITY_RATIO or err > MAX_REPROJECTION_PX:
        return None
    return T


def detect_marker_poses(
    video: Path,
    t_ns: np.ndarray,
    kb4: np.ndarray,
    resolution: tuple[int, int],
    clock_offset_s: float,
    marker: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect the head marker in one rear video at about ``EFFECTIVE_FPS``.

    Returns the detection times (frame time plus ``clock_offset_s``) and
    ``T_rear_marker`` (N, 4, 4).
    """
    offset = clock_offset_s
    source_fps = 1e9 / float(np.median(np.diff(t_ns)))
    stride = int(round(source_fps / EFFECTIVE_FPS))

    w0, h0 = resolution
    scale = DETECTION_MAX_EDGE_PX / max(w0, h0)
    size = (int(round(w0 * scale)), int(round(h0 * scale)))
    K = np.array(
        [
            [kb4[0] * size[0] / w0, 0.0, kb4[2] * size[0] / w0],
            [0.0, kb4[1] * size[1] / h0, kb4[3] * size[1] / h0],
            [0.0, 0.0, 1.0],
        ]
    )
    dist = np.asarray(kb4[4:8], dtype=float)

    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = 1
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[marker["dictionary"]]), params
    )

    times, poses = [], []
    cap = cv2.VideoCapture(str(video))
    index = 0
    while cap.grab():
        if index % stride == 0:
            _, image = cap.retrieve()
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            T = detect_in_image(
                image, detector, int(marker["id"]), float(marker["size"]), K, dist
            )
            if T is not None:
                times.append(float(t_ns[index]) * 1e-9 + offset)
                poses.append(T)
        index += 1
    cap.release()
    return np.asarray(times, dtype=float), np.asarray(poses)


# ----------------------------------------------------------------------------
# Composition with the fused front-camera trajectory
# ----------------------------------------------------------------------------


def compose_scene_marker(
    t_det: np.ndarray,
    T_rear_marker: np.ndarray,
    t_front: np.ndarray,
    T_scene_front: np.ndarray,
    T_rear_front: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``T_scene_marker = T_scene_front(t) @ inv(T_rear_front) @ T_rear_marker`` for the
    detections inside the span of the front trajectory."""
    inside = (t_det >= t_front[0]) & (t_det <= t_front[-1])
    T_scene_rear = interpolate_poses(t_front, T_scene_front, t_det[inside]) @ se3_inv(
        T_rear_front
    )
    return t_det[inside], T_scene_rear @ T_rear_marker[inside]

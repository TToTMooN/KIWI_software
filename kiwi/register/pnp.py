"""Single-tag planar PnP + SE(3) averaging.

Everything single-marker anchoring needs: per-marker detection + PnP with the
reflection ambiguity resolved, the reprojection RMS that weights every
downstream average, and the weighted SE(3) mean.

Quaternions here are scipy ``xyzw`` throughout - one convention, stated once
(``kiwi.core.trajectory``).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.core.geometry import rvec_tvec_to_T
from kiwi.core.markers import MarkerConfig, detect_aruco_markers, marker_object_points

__all__ = [
    "avg_T",
    "detect_per_marker_pnp",
    "reproj_rms_px",
    "solve_single_tag_facing",
]


def solve_single_tag_facing(
    obj: np.ndarray,
    img_pts: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Single-tag planar PnP that resolves the reflection ambiguity.

    ``SOLVEPNP_IPPE_SQUARE`` returns one of two reflection-symmetric
    solutions (which one varies frame-to-frame for near-head-on views).
    Using ``solvePnPGeneric`` we get both and pick the one whose marker
    face normal points *back at* the camera: the marker's local +Z is
    ``R[:, 2]`` in the camera frame, and OpenCV's camera looks down +Z,
    so a marker facing the camera has ``R[2, 2] < 0``.

    ``IPPE_SQUARE`` is only valid on the canonical centred square from
    :func:`kiwi.core.markers.marker_object_points` - fed world-frame corners
    of an off-origin marker it silently returns a pose tens of degrees off
    with a small reprojection error. Solve in the marker's local frame and
    compose.

    Returns ``(rvec, tvec)`` or ``None`` if PnP failed.
    """
    n_sols, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj,
        img_pts,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if n_sols == 0:
        return None
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        if R[2, 2] < 0:
            return rvec, tvec
    # No solution faces the camera - fall back to the lowest-reproj-err one
    # (solvePnPGeneric sorts by reprojection error).
    return rvecs[0], tvecs[0]


def reproj_rms_px(
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None,
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
) -> float:
    """Reprojection rms (px) of an SE(3) + intrinsics fit against its corners.

    This is the number the anchor average weights by (``1 / (rms**2 + eps)``),
    so sharp head-on observations dominate over oblique/blurry ones.
    """
    projected, _ = cv2.projectPoints(
        obj_pts.astype(np.float64),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        K.astype(np.float64),
        np.asarray(dist_coeffs).astype(np.float64) if dist_coeffs is not None else None,
    )
    diff = projected.reshape(-1, 2) - img_pts.reshape(-1, 2).astype(np.float64)
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def avg_T(
    Ts: list[np.ndarray],
    weights: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Mean of 4x4 transforms: weighted arithmetic mean translation, weighted
    Markley-2007 quaternion mean for rotation (dominant eigenvector of
    ``sum(w_i * q_i q_i.T)``).

    Hemisphere alignment first - ``q`` and ``-q`` are the same rotation.
    ``weights=None`` (and an all-zero weight vector) falls back to a uniform
    mean rather than dividing by zero.
    """
    if len(Ts) == 1:
        return Ts[0].copy()
    ts = np.stack([T[:3, 3] for T in Ts])
    qs = np.stack([Rotation.from_matrix(T[:3, :3]).as_quat() for T in Ts])
    for i in range(1, len(qs)):
        if np.dot(qs[0], qs[i]) < 0:
            qs[i] = -qs[i]
    if weights is None:
        M = qs.T @ qs
        t_mean = ts.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != len(Ts):
            raise ValueError(f"weights length {w.shape[0]} != Ts length {len(Ts)}")
        M = (qs * w[:, None]).T @ qs
        w_sum = w.sum()
        if w_sum <= 0:
            M = qs.T @ qs
            t_mean = ts.mean(axis=0)
        else:
            t_mean = (ts * w[:, None]).sum(axis=0) / w_sum
    _, V = np.linalg.eigh(M)
    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = Rotation.from_quat(V[:, -1]).as_matrix()
    T_avg[:3, 3] = t_mean
    return T_avg


def detect_per_marker_pnp(
    img: np.ndarray,
    marker_configs: dict[int, MarkerConfig],
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray, float]], list[np.ndarray], list[int]
]:
    """Detect every configured marker and run single-tag PnP per marker.

    Returns ``(obs, all_corners, all_ids)`` where ``obs`` is
    ``{id: (T_camera_marker (4x4), img_pts (4, 2) float32, reproj_rms_px)}``
    for markers whose PnP succeeded, and ``all_corners`` / ``all_ids`` carry
    every detected marker (even ones where PnP failed) so a caller can draw a
    complete overlay. Standalone tags decode at the default ``border_bits=1``
    (the anchor marker is one; a Kalibr AprilGrid is NOT - see
    :func:`kiwi.core.markers.detect_aruco_markers`).
    """
    by_dict: dict[int, list[int]] = {}
    for cfg in marker_configs.values():
        by_dict.setdefault(cfg.cv2_dictionary, []).append(cfg.id)

    obs: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    all_corners: list[np.ndarray] = []
    all_ids: list[int] = []
    for dict_id, ids in by_dict.items():
        corners, det_ids = detect_aruco_markers(
            img,
            marker_dict=dict_id,
            allowed_ids=set(ids),
        )
        if det_ids is None:
            continue
        for c, tid in zip(corners, det_ids.flatten()):
            tid = int(tid)
            cfg = marker_configs[tid]
            img_pts = c.reshape(4, 2).astype(np.float32)
            obj = marker_object_points(cfg.size)
            sol = solve_single_tag_facing(obj, img_pts, K, dist_coeffs)
            if sol is not None:
                rvec, tvec = sol
                err = reproj_rms_px(rvec, tvec, K, dist_coeffs, obj, img_pts)
                obs[tid] = (rvec_tvec_to_T(rvec, tvec), img_pts, err)
            all_corners.append(c)
            all_ids.append(tid)
    return obs, all_corners, all_ids

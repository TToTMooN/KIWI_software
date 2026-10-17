"""Planar geometry helpers for marker-anchored pipelines.

``homography_from_aruco_pose`` builds the plane <-> image homographies for
a target plane parallel to (and optionally offset from) the marker face,
given OpenCV's ArUco PnP outputs. ``rvec_tvec_to_T`` lifts the same PnP
outputs to a homogeneous transform. ``apply_homography_batch`` applies
a 3x3 homography to an array of 2D points.
These are pure NumPy / OpenCV math with no I/O, so any
component can import them.
"""

from __future__ import annotations

import cv2
import numpy as np


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Compose a 4x4 homogeneous transform from ``solvePnP`` outputs."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return T


def homography_from_aruco_pose(K, rvec, tvec, z_offset_m: float = 0.0):
    """Build plane-to-image and image-to-plane homographies from an ArUco pose.

    OpenCV ArUco pose convention:
        P_camera = R * P_marker + t

    Target plane (parallel to the marker face, optionally offset along
    the marker's local Z):
        Z_marker = z_offset_m

    Homography:
        s [u, v, 1]^T = K [r1 r2 (z_offset_m * r3 + t)] [X, Y, 1]^T

    A non-zero ``z_offset_m`` lifts (or lowers) the homography to a
    plane parallel to the marker - useful when the features of
    interest (e.g. the chopstick reference dots) don't lie exactly on
    the marker face. Pass the physical offset in metres; positive is
    the marker's own +Z (out of the marker face), negative is into it.

    Args:
        K:    3x3 camera intrinsic matrix
        rvec: 3x1 rotation vector from ArUco pose estimation
        tvec: 3x1 translation vector from ArUco pose estimation
        z_offset_m: target-plane offset along marker +Z (metres)

    Returns:
        H_plane_to_img: 3x3 homography from target-plane coords (m) to pixels
        H_img_to_plane: 3x3 homography from pixels to target-plane coords (m)
    """

    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    r1 = R[:, 0:1]
    r2 = R[:, 1:2]
    r3 = R[:, 2:3]

    t_shifted = t + z_offset_m * r3
    H_plane_to_img = K @ np.hstack([r1, r2, t_shifted])
    H_img_to_plane = np.linalg.inv(H_plane_to_img)

    return H_plane_to_img, H_img_to_plane


def apply_homography_batch(H, points):
    """Apply a 3x3 homography to an (N, 2) array of points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    q = (H @ h.T).T
    return q[:, :2] / q[:, 2:3]

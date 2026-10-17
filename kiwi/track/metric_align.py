"""Metric scale and gravity direction of a monocular map from raw IMU.

This is the inertial-only initialisation of visual-inertial ORB-SLAM run offline on a
finished map's localized poses. For consecutive nodes i<j<k with IMU preintegrals
(alpha, beta) and known body attitudes R::

    P_j - P_i = v_i dt1 + 1/2 g dt1^2 + R_i (alpha_ij + Ja_ij b)
    v_j       = v_i + g dt1 + R_i (beta_ij + Jb_ij b)
    P_k - P_j = v_j dt2 + 1/2 g dt2^2 + R_j (alpha_jk + Ja_jk b)

with ``P = s p_cam + R_mc t_cb`` the metric body position. Eliminating the velocities
gives three linear equations per triple in ``x = [s, g, b]``; a Huber polish then fixes
``|g| = 9.81``. Gyro bias is ignored (the node intervals are fractions of a second).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

GRAVITY_M_S2 = 9.81
NODE_DT_S = 0.25
HUBER_F_SCALE = 0.05


@dataclass(frozen=True)
class MetricAlignment:
    scale: float  # metres per map unit
    R_wg: (
        np.ndarray
    )  # smallest rotation taking the fitted gravity to (0, 0, -9.81); yaw free


def preintegrate(t, gyro, acc, i0: int, i1: int):
    """Preintegrate samples [i0, i1): (dR, alpha, beta, Ja, Jb) in the body frame at i0."""
    dR = np.eye(3)
    alpha = np.zeros(3)
    beta = np.zeros(3)
    Ja = np.zeros((3, 3))
    Jb = np.zeros((3, 3))
    for k in range(i0, i1):
        dt = t[k + 1] - t[k]
        a = acc[k]
        alpha += beta * dt + 0.5 * dR @ a * dt * dt
        Ja += Jb * dt - 0.5 * dR * dt * dt
        beta += dR @ a * dt
        Jb += -dR * dt
        dR = dR @ Rotation.from_rotvec(gyro[k] * dt).as_matrix()
    return dR, alpha, beta, Ja, Jb


def estimate_metric_alignment(
    t_cam_s: np.ndarray,
    R_map_cam: np.ndarray,
    p_map_cam: np.ndarray,
    T_cam_imu: np.ndarray,
    imu_t: np.ndarray,
    imu_gyro: np.ndarray,
    imu_acc: np.ndarray,
) -> MetricAlignment:
    """Fit scale and gravity to tracked camera poses and raw IMU on one clock.

    ``t_cam_s`` already carries the camera-IMU time shift. ``R_map_cam`` is (N,3,3),
    ``p_map_cam`` (N,3) in map units. ``T_cam_imu`` maps IMU coordinates to the camera.
    """
    t_cam = np.asarray(t_cam_s, dtype=float)
    p_cam = np.asarray(p_map_cam, dtype=float)
    R_mc = np.asarray(R_map_cam, dtype=float)
    t_imu = np.asarray(imu_t, dtype=float)
    gyro = np.asarray(imu_gyro, dtype=float)
    acc = np.asarray(imu_acc, dtype=float)
    R_cb = T_cam_imu[:3, :3]
    t_cb = T_cam_imu[:3, 3]

    sel = [0]
    for i in range(1, len(t_cam)):
        if t_cam[i] - t_cam[sel[-1]] >= NODE_DT_S - 1e-6:
            sel.append(i)
    sel = np.asarray(sel)
    node_t = t_cam[sel]
    node_p = p_cam[sel]
    node_R = np.einsum("nij,jk->nik", R_mc[sel], R_cb)
    node_lever = np.einsum("nij,j->ni", R_mc[sel], t_cb)

    idx = np.searchsorted(t_imu, node_t)
    pre = [
        preintegrate(t_imu, gyro, acc, idx[i], idx[i + 1]) for i in range(len(sel) - 1)
    ]

    rows_a, rows_b = [], []
    for i in range(len(node_t) - 2):
        dt1 = node_t[i + 1] - node_t[i]
        dt2 = node_t[i + 2] - node_t[i + 1]
        _, a1, b1, ja1, jb1 = pre[i]
        _, a2, _, ja2, _ = pre[i + 1]
        ri, rj = node_R[i], node_R[i + 1]
        d1 = node_p[i + 1] - node_p[i]
        d2 = node_p[i + 2] - node_p[i + 1]
        l1 = node_lever[i + 1] - node_lever[i]
        l2 = node_lever[i + 2] - node_lever[i + 1]
        r = dt2 / dt1
        col_s = (d2 - r * d1).reshape(3, 1)
        col_g = -0.5 * dt2 * (dt1 + dt2) * np.eye(3)
        col_b = -(ri @ jb1 * dt2 - r * (ri @ ja1) + rj @ ja2)
        rows_a.append(np.hstack([col_s, col_g, col_b]))
        rows_b.append(ri @ b1 * dt2 - r * (ri @ a1) + rj @ a2 - (l2 - r * l1))
    A = np.vstack(rows_a)
    b = np.hstack(rows_b)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)

    def unpack(q):
        g = GRAVITY_M_S2 * np.array(
            [np.sin(q[1]) * np.cos(q[2]), np.sin(q[1]) * np.sin(q[2]), np.cos(q[1])]
        )
        return q[0], g, q[3:6]

    g0 = x[1:4]
    th = np.arctan2(np.hypot(g0[0], g0[1]), g0[2])
    ph = np.arctan2(g0[1], g0[0])
    q0 = np.concatenate([[x[0], th, ph], x[4:7]])

    def resid(q):
        s, g, ba = unpack(q)
        return A @ np.concatenate([[s], g, ba]) - b

    sol = least_squares(resid, q0, loss="huber", f_scale=HUBER_F_SCALE)
    s, g, _ = unpack(sol.x)

    a_vec = g / np.linalg.norm(g)
    b_vec = np.array([0.0, 0.0, -1.0])
    v = np.cross(a_vec, b_vec)
    R_wg = Rotation.from_rotvec(
        v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), a_vec @ b_vec)
    ).as_matrix()
    return MetricAlignment(scale=float(s), R_wg=R_wg)

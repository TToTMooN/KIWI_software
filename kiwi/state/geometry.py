"""A pose is a 4x4 ``T_AB`` that maps coordinates in frame B to frame A
(``p_A = T_AB @ p_B``); quaternions are Hamilton ``[x, y, z, w]`` as in TUM files.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return se3(R.T, -R.T @ t)


def quats_continuous(q_xyzw: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so consecutive rows stay in the same hemisphere."""
    q = np.array(q_xyzw, dtype=float, copy=True)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def interpolate_poses(
    t_track: np.ndarray, T_track: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Poses (N, 4, 4) of a track at the times ``t``: Slerp rotation, linear position.

    ``t_track`` is strictly increasing and every ``t`` lies in ``[t_track[0], t_track[-1]]``.
    """
    t_track = np.asarray(t_track, dtype=float)
    T_track = np.asarray(T_track, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3] = Slerp(t_track, Rotation.from_matrix(T_track[:, :3, :3]))(
        t
    ).as_matrix()
    for k in range(3):
        T[:, k, 3] = np.interp(t, t_track, T_track[:, k, 3])
    return T

"""Robot-agnostic pose / quaternion helpers shared across the replay stack.

No dependency on any particular robot or scene - just MuJoCo's quaternion
math and the wrist / workspace-transform conventions the demonstration
trajectories are expressed in. ``kiwi.replay.solve`` and the viewer both
import from here.
"""

from __future__ import annotations

import mujoco
import numpy as np

_X_AXIS = np.array([1.0, 0.0, 0.0])
_Y_AXIS = np.array([0.0, 1.0, 0.0])
_Z_AXIS = np.array([0.0, 0.0, 1.0])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a wxyz quaternion from axis-angle."""
    out = np.zeros(4)
    mujoco.mju_axisAngle2Quat(out, axis, angle)
    return out


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, v, q)
    return out


def _wrist_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """Compose the wrist hinge chain into a single wxyz quat.

    The replay qpos orientation columns are intrinsic-XYZ Euler (see
    ``kiwi.replay.state``), so the wrist rotation in world is Rx . Ry . Rz.
    """
    qx = _axis_angle_quat(_X_AXIS, rx)
    qy = _axis_angle_quat(_Y_AXIS, ry)
    qz = _axis_angle_quat(_Z_AXIS, rz)
    return _quat_mul(qx, _quat_mul(qy, qz))


def _wrist_root_pose(qpos_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One replay qpos frame -> world (pos, quat_wxyz) of the wrist."""
    pos = qpos_frame[0:3].copy()
    quat = _wrist_quat(float(qpos_frame[3]), float(qpos_frame[4]), float(qpos_frame[5]))
    return pos, quat


def _workspace_rot_quat(
    yaw_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0
) -> np.ndarray:
    """World-frame workspace rotation RotZ(yaw) . RotY(pitch) . RotX(roll).

    Returned as a wxyz quaternion. With pitch=roll=0 this is a pure yaw.
    """
    q_yaw = _axis_angle_quat(_Z_AXIS, np.deg2rad(yaw_deg))
    q_pitch = _axis_angle_quat(_Y_AXIS, np.deg2rad(pitch_deg))
    q_roll = _axis_angle_quat(_X_AXIS, np.deg2rad(roll_deg))
    return _quat_mul(q_yaw, _quat_mul(q_pitch, q_roll))


def dashed_segments(
    points: np.ndarray, dash: float = 0.012, gap: float = 0.012
) -> np.ndarray:
    """Resample a polyline into evenly spaced dashes for a dotted-line look.

    Walks ``points`` (shape ``(N, 3)``) by world-space arc length, emitting a
    short straight segment every ``dash`` metres followed by a ``gap`` metre
    hole. Returns an ``(M, 2, 3)`` float32 array suitable for viser's
    ``add_line_segments``; returns an empty ``(0, 2, 3)`` array when the path
    is too short (or zero length) to dash.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    seg_vec = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])  # arc length at each vertex
    total = float(s[-1])
    if total <= 0.0:
        return np.empty((0, 2, 3), dtype=np.float32)

    def point_at(dist: float) -> np.ndarray:
        """Linear interpolation of the polyline at arc-length ``dist``."""
        i = int(np.searchsorted(s, dist, side="right") - 1)
        i = min(max(i, 0), len(seg_len) - 1)
        if seg_len[i] <= 0.0:
            return pts[i]
        return pts[i] + (dist - s[i]) / seg_len[i] * seg_vec[i]

    starts = np.arange(0.0, total, dash + gap)
    out = np.empty((len(starts), 2, 3), dtype=np.float32)
    for k, d0 in enumerate(starts):
        out[k, 0] = point_at(d0)
        out[k, 1] = point_at(min(d0 + dash, total))
    return out


def _workspace_transform(
    wrist_pos: np.ndarray,
    wrist_quat: np.ndarray,
    *,
    offset: np.ndarray,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply T_workspace = Translate(offset) . RotZ(yaw).RotY(pitch).RotX(roll)
    to a wrist pose."""
    q_rot = _workspace_rot_quat(yaw_deg, pitch_deg, roll_deg)
    new_pos = _quat_rotate(q_rot, wrist_pos) + offset
    new_quat = _quat_mul(q_rot, wrist_quat)
    return new_pos, new_quat

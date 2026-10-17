"""The canonical replay-trajectory contract + the one retarget transform.

The episode source (``kiwi.replay.episodes``) converts each arm's EE stream
into the same (N, 7) ``qpos`` layout via the single ``convert_ee_pose`` here.

Column layout (one row per frame, one array per arm):
    [0:3] wrist xyz (m, world frame - the replay re-anchors it per base)
    [3:6] wrist Rx,Ry,Rz (intrinsic-XYZ Euler, rad; matches pose._wrist_quat)
    [6]   gripper hinge (rad; the source's gripper mapped onto the chopsticks
          range)

The hinge values use the SA CAD travel limits.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.replay.gripper import OPEN as GRIPPER_OPEN, CLOSED as GRIPPER_CLOSED


def convert_ee_pose(
    pos, rot: Rotation, grip01, *, rot_offset_deg, traj_yaw_deg: float
) -> np.ndarray:
    """Map one arm's raw EE stream into the (N, 7) replay layout.

    ``pos`` (N, 3) raw world-frame positions; ``rot`` a scipy ``Rotation`` of
    length N (the raw EE orientations, however the source represents them);
    ``grip01`` (N,) gripper openness in [0, 1] (1 = fully open).
    ``traj_yaw_deg`` is a world-frame yaw (about Z) baked into positions +
    orientations to match the dual-Franka layout; ``rot_offset_deg`` is an
    intrinsic-XYZ body-frame rotation right-multiplied onto each orientation.
    Returns (N, 7) float64.

    Single source of truth for the retarget transform: the episode adapter
    builds ``(pos, rot, grip01)`` from its format and delegates here.
    """
    pos = np.asarray(pos, dtype=np.float64)
    R_yaw = Rotation.from_euler("Z", traj_yaw_deg, degrees=True)
    R_align = Rotation.from_euler("XYZ", rot_offset_deg, degrees=True)
    pos_w = R_yaw.apply(pos)
    rpy = (R_yaw * rot * R_align).as_euler("XYZ")  # world yaw . source . body-roll
    hinge = GRIPPER_CLOSED + np.clip(grip01, 0.0, 1.0) * (GRIPPER_OPEN - GRIPPER_CLOSED)
    return np.concatenate([pos_w, rpy, hinge[:, None]], axis=1).astype(np.float64)

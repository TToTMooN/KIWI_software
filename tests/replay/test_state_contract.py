"""The (N, 7) replay contract: convert_ee_pose golden values."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kiwi.replay.state import GRIPPER_CLOSED, GRIPPER_OPEN, convert_ee_pose


def test_identity_transform_passes_positions_through():
    pos = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    rot = Rotation.identity(2)
    grip = np.array([0.0, 1.0])
    q = convert_ee_pose(pos, rot, grip, rot_offset_deg=(0, 0, 0), traj_yaw_deg=0.0)
    assert q.shape == (2, 7)
    np.testing.assert_allclose(q[:, :3], pos)
    np.testing.assert_allclose(q[:, 3:6], 0.0, atol=1e-12)


def test_gripper_maps_01_onto_the_hinge_range():
    pos = np.zeros((3, 3))
    rot = Rotation.identity(3)
    q = convert_ee_pose(
        pos,
        rot,
        np.array([0.0, 0.5, 1.0]),
        rot_offset_deg=(0, 0, 0),
        traj_yaw_deg=0.0,
    )
    np.testing.assert_allclose(
        q[:, 6],
        [GRIPPER_CLOSED, (GRIPPER_CLOSED + GRIPPER_OPEN) / 2, GRIPPER_OPEN],
    )
    # Out-of-range openness clamps instead of extrapolating.
    q2 = convert_ee_pose(
        pos[:1],
        Rotation.identity(1),
        np.array([1.5]),
        rot_offset_deg=(0, 0, 0),
        traj_yaw_deg=0.0,
    )
    assert q2[0, 6] == pytest.approx(GRIPPER_OPEN)


def test_traj_yaw_rotates_the_world_frame():
    pos = np.array([[1.0, 0.0, 0.5]])
    rot = Rotation.identity(1)
    q = convert_ee_pose(
        pos, rot, np.array([1.0]), rot_offset_deg=(0, 0, 0), traj_yaw_deg=90.0
    )
    # Rz(90): x -> y.
    np.testing.assert_allclose(q[0, :3], [0.0, 1.0, 0.5], atol=1e-12)
    # The orientation picks up the same yaw, in intrinsic-XYZ Euler columns.
    got = Rotation.from_euler("XYZ", q[0, 3:6])
    want = Rotation.from_euler("Z", 90, degrees=True)
    assert (got.inv() * want).magnitude() == pytest.approx(0.0, abs=1e-12)


def test_rot_offset_is_body_frame_right_multiplied():
    base = Rotation.from_euler("Z", [[45], [-30]], degrees=True)
    q = convert_ee_pose(
        np.zeros((2, 3)),
        base,
        np.array([1.0, 1.0]),
        rot_offset_deg=(0, 0, 180),
        traj_yaw_deg=0.0,
    )
    got = Rotation.from_euler("XYZ", q[:, 3:6])
    want = base * Rotation.from_euler("XYZ", [0, 0, 180], degrees=True)
    assert (got.inv() * want).magnitude() == pytest.approx([0.0, 0.0], abs=1e-12)

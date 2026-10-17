"""The world-frame re-seating baked into an export (`--world-offset` / `--world-yaw`).

The contract these pin down: a replay yaws position AND orientation about
the world origin, then adds its workspace translation in the already-rotated
frame. `world_transform` has to compose the two the same way, or a dataset
baked here lands somewhere else than the launch flags it replaced.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kiwi.export.build import world_transform


def _replay_compose(pos, rot, yaw_deg, workspace):
    """What a replay does with the same two numbers as launch flags.

    Mirrors the replay's world-yaw contract (kiwi.replay.state.convert_ee_pose):
    `R_yaw.apply(pos)` for position and `R_yaw * rot` for orientation, with the
    workspace translation added afterwards in rig coordinates.
    """
    r_yaw = Rotation.from_euler("Z", yaw_deg, degrees=True)
    return r_yaw.apply(pos) + np.asarray(workspace), r_yaw * rot


def test_both_unset_is_none():
    assert world_transform(None, None) is None


def test_offset_only_leaves_orientation_untouched():
    """With no yaw the transform is a pure translation, bit for bit."""
    t = world_transform((0.8, 0.0, 0.0), None)
    assert np.array_equal(t[:3, :3], np.eye(3))
    assert np.array_equal(t[:3, 3], [0.8, 0.0, 0.0])


def test_yaw_only_rotates_about_the_origin():
    t = world_transform(None, -90.0)
    # A point on +y goes to +x under Rz(-90).
    got = t @ np.array([0.0, 1.0, 0.0, 1.0])
    assert np.allclose(got[:3], [1.0, 0.0, 0.0], atol=1e-12)
    assert np.array_equal(t[:3, 3], np.zeros(3))


def test_rotation_is_applied_before_translation():
    """`Translate(offset) @ Rz(yaw)`, NOT the other way round.

    The two do not commute, so this is the whole contract: the offset is a
    coordinate in the ROTATED frame.
    """
    yaw, off = -90.0, (0.87, 0.175, 0.0107)
    t = world_transform(off, yaw)
    p = np.array([0.3, -0.2, 0.05])
    got = (t @ np.append(p, 1.0))[:3]
    expect = Rotation.from_euler("Z", yaw, degrees=True).apply(p) + np.asarray(off)
    assert np.allclose(got, expect, atol=1e-12)
    # ...and the reversed order really is a different answer, so the test bites.
    wrong = Rotation.from_euler("Z", yaw, degrees=True).apply(p + np.asarray(off))
    assert not np.allclose(got, wrong, atol=1e-6)


@pytest.mark.parametrize("yaw", [-90.0, 0.0, 45.0, 180.0])
def test_baking_matches_the_replay_it_replaces(yaw):
    """A baked export replayed at identity == the raw export replayed with flags.

    The bake moves the transform from launch flags into the data without
    moving the data.
    """
    rng = np.random.default_rng(0)
    workspace = (0.87, 0.175, 0.0107)
    pos = rng.normal(size=(16, 3))
    rot = Rotation.from_quat(rng.normal(size=(16, 4)))

    # Baked: left-multiply the world transform onto each source pose.
    t = world_transform(workspace, yaw)
    poses = np.tile(np.eye(4), (16, 1, 1))
    poses[:, :3, :3] = rot.as_matrix()
    poses[:, :3, 3] = pos
    baked = t @ poses

    ref_pos, ref_rot = _replay_compose(pos, rot, yaw, workspace)
    assert np.allclose(baked[:, :3, 3], ref_pos, atol=1e-12)
    assert np.allclose(baked[:, :3, :3], ref_rot.as_matrix(), atol=1e-12)


def test_pre_rotated_offset_with_replay_yaw_equals_the_one_step_bake():
    """A pre-rotated offset with a replay-time yaw equals the one-step bake.

    Baking `Rz(+90) @ workspace` as a pure translation and then replaying with
    a -90 deg world yaw must land in the same place as baking yaw -90 plus the
    plain workspace offset.
    """
    yaw, workspace = -90.0, np.array([0.87, 0.175, 0.0107])
    p = np.array([0.3, -0.2, 0.05])

    pre_rotated_offset = Rotation.from_euler("Z", -yaw, degrees=True).apply(workspace)
    t_offset = world_transform(tuple(pre_rotated_offset), None)
    translated = (t_offset @ np.append(p, 1.0))[:3]
    two_step_final = Rotation.from_euler("Z", yaw, degrees=True).apply(translated)

    baked_final = (world_transform(tuple(workspace), yaw) @ np.append(p, 1.0))[:3]
    assert np.allclose(two_step_final, baked_final, atol=1e-12)

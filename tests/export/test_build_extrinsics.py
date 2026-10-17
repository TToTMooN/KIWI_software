"""The export's camera->tip constant and its provenance serialization.

There is no body-frame vocabulary to pin: the exported body is always the
tool tip, and the single constant applied to every pose is

    T_camera_tcp = inv(T_mount_cam) @ T_mount_tcp

with both legs expressed in the GRIPPER MOUNT frame - taken from the robot
description's URDF by fixed-joint FK, or from a gripper unit's override block.
Expressing both in the mount frame is what makes the tip rigid to the camera:
the mount cancels, and the robot flange never enters the chain. These tests are
unit-level (no episode fixtures): the direction of the composition, and the
explicit rejection of uncalibrated geometry and synthetic calibration overrides.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kiwi.core.descriptions import fixed_frame_T, load_robot_description
from kiwi.core.tool_config import load_T_mount_cam, load_T_mount_tcp
from kiwi.export.build import _matrix


def _block(key: str, t: np.ndarray, **kw) -> dict:
    """One `build.mount_*:` block, chopstick_unit_v1 shape: matrix + `measured`."""
    return {key: [[float(v) for v in row] for row in t], **kw}


def _se3(rotvec, xyz) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    t[:3, 3] = xyz
    return t


def test_matrix_roundtrips_se3():
    rng = np.random.default_rng(0)
    t = np.eye(4)
    t[:3, 3] = rng.normal(size=3)
    out = _matrix(t)
    assert isinstance(out, list) and len(out) == 4 and all(len(r) == 4 for r in out)
    assert all(isinstance(v, float) for r in out for v in r)
    np.testing.assert_array_equal(np.asarray(out), t)  # full precision, no rounding


def test_two_override_blocks_compose_through_the_mount_frame():
    """``inv(T_mount_cam) @ T_mount_tcp``, with the mount cancelling.

    Both sides are computed explicitly: the composition as the builder does it,
    and the same quantity derived by walking a tip-frame point up to the mount
    and back down into the camera. That second route is what pins the DIRECTION
    - an inverted leg would still be a valid SE(3) and still ship poses.
    """
    t_mount_cam = _se3([0.1, -0.2, 0.3], [0.01, 0.055, 0.056])
    t_mount_tcp = _se3([0.0, 0.0, 0.4], [0.0, 0.0, 0.195])
    cam_block = _block("T_mount_cam", t_mount_cam)
    tcp_block = _block("T_mount_tcp", t_mount_tcp)

    composed = np.linalg.inv(load_T_mount_cam(cam_block)) @ load_T_mount_tcp(tcp_block)

    # A point 20 mm out along the tip's own +z, routed the long way round.
    p_tcp = np.array([0.0, 0.0, 0.02, 1.0])
    p_mount = t_mount_tcp @ p_tcp
    p_cam = np.linalg.inv(t_mount_cam) @ p_mount
    np.testing.assert_allclose(composed @ p_tcp, p_cam, atol=1e-12)
    # ...and the constant's own translation IS the tip origin in camera frame.
    np.testing.assert_allclose(
        composed[:3, 3], (np.linalg.inv(t_mount_cam) @ t_mount_tcp)[:3, 3], atol=1e-12
    )


def test_a_mis_pasted_leg_is_refused_by_the_closed_key_set():
    """The wrong leg's matrix key in a block is an unknown key, not a silent no-op.

    A `mount_tcp:` block holding the CAMERA leg's matrix would otherwise
    compose the tip constant out of two camera poses; the block's closed key
    set catches it.
    """
    t = _se3([0.0, 0.0, 0.0], [0.0, 0.0, 0.195])
    with pytest.raises(ValueError, match="unknown key"):
        load_T_mount_tcp(_block("T_mount_cam", t))


def test_dc_composes_the_pinned_constant_and_accepts_overrides():
    """The DC camera leg composes the pinned camera->tip constant.

    The composite `inv(T_mount_cam) @ T_mount_tcp` against this model's own
    `hand_tcp` is pinned here: a regenerated URDF that moves the frame changes
    every exported TCP track, silently, and this is what catches it.
    """
    from kiwi.export.build import _mount_transforms

    desc = load_robot_description("chopstick_dc")
    cam, tcp = _mount_transforms(desc)
    np.testing.assert_allclose(
        (np.linalg.inv(cam) @ tcp)[:3, 3],
        [-0.002587151, 0.030566119, 0.146386477],
        atol=1e-9,
    )
    # Synthetic calibration for composition testing only; never a shipped default.
    expected_cam = _se3([0.1, -0.2, 0.3], [0.01, 0.055, 0.056])
    cam, tcp = _mount_transforms(
        desc, _block("T_mount_cam", expected_cam, measured=True)
    )
    np.testing.assert_array_equal(cam, expected_cam)
    np.testing.assert_allclose(
        tcp, fixed_frame_T(desc.urdf_path, "chopstick_dc_base", "hand_tcp")
    )
    expected_tcp = _se3([0.2, 0.1, 0], [0.19, 0.01, 0])
    cam, tcp = _mount_transforms(
        desc, _block("T_mount_cam", expected_cam), _block("T_mount_tcp", expected_tcp)
    )
    np.testing.assert_allclose(
        np.linalg.inv(cam) @ tcp, np.linalg.inv(expected_cam) @ expected_tcp
    )

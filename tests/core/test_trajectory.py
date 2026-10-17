"""Unit tests for kiwi.core.trajectory (the tracking-output contract + machinery).

Pure numpy/scipy - no hardware, no ROS. The Trajectory contract itself is also
pinned by tests/test_scaffold.py; these cover the machinery around it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kiwi.core.trajectory import (
    PoseTrack,
    Trajectory,
    load_tum,
    se3,
    se3_stack,
)

RNG = np.random.default_rng(7)


def _random_tpq(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.sort(RNG.uniform(0.0, 10.0, n))
    p = RNG.normal(size=(n, 3))
    q = Rotation.random(n, random_state=3).as_quat()
    return t, p, q


# --- se3 helpers -----------------------------------------------------------


def test_se3_roundtrip():
    rot = Rotation.from_euler("xyz", [0.3, -0.2, 1.1])
    pos = np.array([1.0, -2.0, 0.5])
    T = se3(pos, rot)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, 3], pos)
    np.testing.assert_allclose(T[:3, :3], rot.as_matrix())
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_se3_stack_matches_scalar():
    _, p, q = _random_tpq(5)
    T = se3_stack(p, q)
    assert T.shape == (5, 4, 4)
    for k in range(5):
        np.testing.assert_allclose(T[k], se3(p[k], Rotation.from_quat(q[k])))


# --- Trajectory <-> SE(3) conversion ---------------------------------------


def test_poses_from_poses_roundtrip():
    t, p, q = _random_tpq(8)
    traj = Trajectory(t, p, q, frame="map", body="camera").validate()
    back = Trajectory.from_poses(t, traj.poses(), frame="map", body="camera").validate()
    np.testing.assert_allclose(back.positions, p, atol=1e-12)
    # Quaternion double-cover: q and -q are the same rotation.
    dots = np.abs(np.sum(back.quaternions * q, axis=1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-12)


def test_poses_are_t_frame_body():
    # One pose: camera at (1,0,0) with a 90deg yaw - a body-frame point on the
    # optical axis must land where the frame expects it.
    rot = Rotation.from_euler("z", np.pi / 2)
    traj = Trajectory(
        np.array([0.0]),
        np.array([[1.0, 0.0, 0.0]]),
        rot.as_quat()[None, :],
        frame="map",
        body="camera",
    )
    p_body = np.array([0.0, 0.0, 2.0, 1.0])  # 2m along camera z
    p_frame = traj.poses()[0] @ p_body
    np.testing.assert_allclose(p_frame[:3], [1.0, 0.0, 2.0], atol=1e-12)


def test_from_poses_shape_mismatch():
    t, p, q = _random_tpq(4)
    with pytest.raises(ValueError, match="poses must be"):
        Trajectory.from_poses(t[:3], se3_stack(p, q), frame="x", body="camera")


def test_empty_trajectory_ok():
    traj = Trajectory.from_poses(
        np.zeros(0), np.zeros((0, 4, 4)), frame="x", body="camera"
    )
    assert len(traj.validate()) == 0


# --- PoseTrack -------------------------------------------------------------


def _track(n=5):
    t = np.linspace(0.0, 2.0, n)
    rots = Rotation.from_euler(
        "z", np.linspace(0.0, 90.0, n).reshape(-1, 1), degrees=True
    )
    pos = np.column_stack([np.linspace(0.0, 1.0, n), np.zeros(n), np.zeros(n)])
    return PoseTrack(t=t, rotations=rots, positions=pos), t, rots, pos


def test_pose_track_at_endpoints_and_midpoint():
    track, t, rots, pos = _track()
    np.testing.assert_allclose(track.at(t[0])[:3, 3], pos[0], atol=1e-12)
    np.testing.assert_allclose(
        track.at(t[-1])[:3, :3], rots[-1].as_matrix(), atol=1e-12
    )
    # Midpoint between two samples: linear position, slerped rotation.
    tm = 0.5 * (t[0] + t[1])
    T = track.at(tm)
    np.testing.assert_allclose(T[:3, 3], 0.5 * (pos[0] + pos[1]), atol=1e-12)
    expect_ang = np.degrees(Rotation.from_matrix(T[:3, :3]).magnitude())
    np.testing.assert_allclose(
        expect_ang, 11.25, atol=1e-9
    )  # half of the 22.5 deg step


def test_pose_track_ang_speed_gating():
    track, t, _, _ = _track()
    # Constant 45 deg/s by construction (90 deg over 2 s).
    assert track.ang_speed_at(1.0) == pytest.approx(45.0, abs=1e-6)
    assert track.gap_at(t[1]) == pytest.approx(0.0)
    assert track.gap_at(t[1] + 0.1) == pytest.approx(0.1)


def test_pose_track_validation():
    rots = Rotation.random(3, random_state=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        PoseTrack(
            t=np.array([0.0, 2.0, 1.0]), rotations=rots, positions=np.zeros((3, 3))
        )
    with pytest.raises(ValueError, match=">= 2 samples"):
        PoseTrack(t=np.array([0.0]), rotations=rots[:1], positions=np.zeros((1, 3)))


# --- loaders ---------------------------------------------------------------


def _write_tum(path, n=4, t0=0.0, scale=1.0):
    t, p, q = _random_tpq(n)
    rows = np.column_stack([(t + t0) * scale, p, q])
    path.write_text("\n".join(" ".join(f"{v:.9f}" for v in r) for r in rows) + "\n")
    return t + t0, p, q


def test_load_tum_roundtrip(tmp_path):
    t, p, q = _write_tum(tmp_path / "a.tum")
    lt, lp, lq, grip = load_tum(tmp_path / "a.tum")
    np.testing.assert_allclose(lt, t, atol=1e-9)
    np.testing.assert_allclose(lp, p, atol=1e-9)
    np.testing.assert_allclose(lq, q, atol=1e-9)
    assert grip is None


def test_load_tum_tolerates_bare_header(tmp_path):
    f = tmp_path / "h.tum"
    t, p, q = _random_tpq(3)
    rows = np.column_stack([t, p, q])
    body = "\n".join(" ".join(f"{v:.9f}" for v in r) for r in rows)
    f.write_text("t tx ty tz qx qy qz qw\n" + body + "\n")
    lt, _, _, _ = load_tum(f)
    np.testing.assert_allclose(lt, t, atol=1e-9)


def test_load_tum_rejects_narrow(tmp_path):
    f = tmp_path / "bad.tum"
    f.write_text("0.0 1.0 2.0\n")
    with pytest.raises(ValueError, match=">=8 cols"):
        load_tum(f)


def test_nanosecond_timestamps_autoscale(tmp_path):
    # EuRoC-derived tools emit ns; the loader must hand back seconds.
    t, _, _ = _write_tum(tmp_path / "ns.tum", t0=1.4e9, scale=1e9)
    lt, _, _, _ = load_tum(tmp_path / "ns.tum")
    np.testing.assert_allclose(lt, t, rtol=1e-7)


def test_epoch_second_timestamps_stay_seconds(tmp_path):
    # Canonical TUM RGB-D groundtruth stamps are Unix-epoch SECONDS (~1.3e9).
    # They must pass through unscaled - misreading them as ns collapses the
    # timeline by 1e9 while staying monotonic, i.e. silently.
    t, _, _ = _write_tum(tmp_path / "epoch.tum", t0=1.305031102e9)
    lt, _, _, _ = load_tum(tmp_path / "epoch.tum")
    np.testing.assert_allclose(lt, t, rtol=0, atol=1e-4)
    assert np.all(np.diff(lt) > 1e-4)  # real inter-frame gaps survive


def test_corrupt_first_row_raises_not_dropped(tmp_path):
    # A numeric-but-malformed first row must surface as an error - the bare-
    # header fallback may only ever discard an actual textual header.
    f = tmp_path / "bad.tum"
    t, p, q = _random_tpq(3)
    rows = np.column_stack([t, p, q])
    body = "\n".join(" ".join(f"{v:.9f}" for v in r) for r in rows)
    f.write_text("0.0 1.0 2.0\n" + body + "\n")  # first row: 3 cols, rest: 8
    with pytest.raises(ValueError):
        load_tum(f)

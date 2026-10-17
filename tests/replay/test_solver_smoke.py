"""End-to-end IK smoke test: a reachable stationary trajectory solves clean."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def rig():
    pytest.importorskip("mink")
    from kiwi.replay.batch import _apply_run_setup, _build_rig
    from kiwi.replay.rig import FR3_CHOPSTICK_SA

    r = _build_rig(
        FR3_CHOPSTICK_SA,
        0.805,
        None,
        None,  # home from the model's keyframe
        combined=True,
        lm_damping=1e-3,
        avoid_collisions=False,
        collision_margin=0.004,
        count_collisions=True,
    )
    base = {
        "dist": 0.805,
        "z": 0.0,
        "roll_left": 0.0,
        "roll_right": 0.0,
        "pitch_left": 0.0,
        "pitch_right": 0.0,
        "yaw_left": 0.0,
        "yaw_right": 0.0,
    }
    _apply_run_setup(r, None, base, 0.5, 1.0, 1e-3)
    return r


def _home_tcp_trajs(rig, n=5):
    """A trajectory pair that parks each TCP exactly where home puts it."""
    import mujoco
    from scipy.spatial.transform import Rotation

    from kiwi.replay.solve import SIDES

    rig.ik.reset()
    data = rig.ik.configuration.data
    mujoco.mj_kinematics(rig.model, data)
    trajs = {}
    for s in SIDES:
        site = data.site(rig.idx[s]["site_id"])
        rpy = Rotation.from_matrix(site.xmat.reshape(3, 3)).as_euler("XYZ")
        frame = np.concatenate([site.xpos, rpy, [0.0]])
        trajs[s] = np.tile(frame, (n, 1))
    return trajs


def test_home_pose_trajectory_solves_to_sub_millimeter(rig):
    from kiwi.replay.batch import _solve_episode

    ws = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    result = _solve_episode(
        rig,
        _home_tcp_trajs(rig),
        ws=ws,
        base_dist=0.805,
        ik_iters=120,
        start_frame=0,
        offset6=None,
    )
    m = result["metrics"]
    assert m.n == 5
    assert m.qp_failures == 0
    assert m.worst_pos < 1e-3  # meters
    assert m.cross_f == 0 and m.grip_f == 0
    for s in ("left", "right"):
        assert result["solved"][s].shape == (5, 7)
        # Solved joints stay near the ready pose that already achieves the
        # target - the solve refines, it does not wander.
        home_arm = rig.home_qpos[rig.idx[s]["arm_qadr"]]
        assert np.abs(result["solved"][s][-1] - home_arm).max() < 0.2


def test_an_offset_target_moves_the_solution(rig):
    from kiwi.replay.batch import _solve_episode

    ws = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    trajs = _home_tcp_trajs(rig)
    off = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0])  # 5 cm down
    result = _solve_episode(
        rig,
        trajs,
        ws=ws,
        base_dist=0.805,
        ik_iters=120,
        start_frame=0,
        offset6=off,
    )
    m = result["metrics"]
    assert m.worst_pos < 5e-3  # still reachable
    # The commanded target actually moved down by the offset.
    assert result["target"]["left"][0, 2] == pytest.approx(
        trajs["left"][0, 2] - 0.05, abs=1e-6
    )


def test_collision_avoidance_toggles_live(rig):
    from kiwi.replay.batch import _solve_episode
    from kiwi.replay.ik import dual_arm_collision_pairs

    ik = rig.ik
    assert ik.collision_limit is None and len(ik._limits) == 1
    pairs = dual_arm_collision_pairs(rig.model)
    ws = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    try:
        ik.set_collision_avoidance(pairs, margin=0.004)
        assert ik.collision_limit is not None and len(ik._limits) == 2
        # The home pose keeps the arms well clear, so the constrained solve
        # still tracks cleanly.
        result = _solve_episode(
            rig,
            _home_tcp_trajs(rig, 3),
            ws=ws,
            base_dist=0.805,
            ik_iters=120,
            start_frame=0,
            offset6=None,
        )
        assert result["metrics"].worst_pos < 5e-3
    finally:
        ik.set_collision_avoidance(None)
    assert ik.collision_limit is None and len(ik._limits) == 1


def test_batch_solver_avoidance_setter_drops_the_cached_rig():
    pytest.importorskip("mink")
    from kiwi.replay.batch import BatchSolver
    from kiwi.replay.rig import FR3_CHOPSTICK_SA

    bs = BatchSolver(FR3_CHOPSTICK_SA, 0.805, None)
    bs._rig = object()  # stand-in for a built rig
    bs.set_avoid_collisions(True, 0.006)
    assert bs._rig is None
    assert bs._avoid_collisions is True
    assert bs._collision_margin == 0.006

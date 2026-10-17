import json

import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.state.geometry import se3
from kiwi.track.scene_frame import (
    MapToScene,
    read_scene_frame,
    solve_front_to_scene,
    write_scene_frame,
)


def _trajectory(t):
    R = Rotation.from_rotvec(
        np.column_stack([0.5 * np.sin(0.4 * t), 0.3 * np.cos(0.6 * t), 0.2 * t * 0.3])
    )
    p = np.column_stack([np.sin(0.5 * t), 0.5 * np.cos(0.8 * t), 0.2 * t])
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3] = R.as_matrix()
    T[:, :3, 3] = p
    return T


def test_placement_recovers_known_G():
    rng = np.random.default_rng(1)
    t_rear = np.arange(0.0, 30.0, 1 / 120)
    T_rear_scene = _trajectory(t_rear)  # rear camera in the scene frame
    X = se3(
        Rotation.from_euler("xyz", [0.1, 3.0, 0.05]).as_matrix(), [0.03, 0.01, -0.02]
    )  # T_rear_front
    G = se3(Rotation.from_euler("xyz", [0.3, -0.2, 1.4]).as_matrix(), [1.0, -2.0, 0.5])
    t_front = t_rear[::2] + 0.0041  # different cadence and phase
    T_rear_at = _trajectory(t_front)
    T_front_scene = T_rear_at @ X  # T_scene_front = T_scene_rear @ T_rear_front
    T_front_map = np.linalg.inv(G) @ T_front_scene
    # localization noise
    T_front_map[:, :3, 3] += rng.normal(0, 0.001, (len(t_front), 3))
    run_id = np.zeros(len(t_rear), int)
    G_hat = solve_front_to_scene(t_front, T_front_map, t_rear, T_rear_scene, X, run_id)
    dR = Rotation.from_matrix(G_hat[:3, :3] @ G[:3, :3].T).magnitude()
    assert np.degrees(dR) < 0.05
    assert np.linalg.norm(G_hat[:3, 3] - G[:3, 3]) < 1e-3


def test_map_to_scene_roundtrip(tmp_path):
    m = MapToScene(
        2.5,
        Rotation.from_euler("xyz", [0.2, 0.1, -0.3]).as_matrix(),
        np.array([0.1, 0.2, 0.3]),
    )
    T = _trajectory(np.linspace(0, 1, 5))
    out = m.apply(T)
    assert out.shape == (5, 4, 4)
    assert np.allclose(out[:, :3, 3], 2.5 * T[:, :3, 3] @ m.rotation.T + m.translation)
    path = write_scene_frame(
        tmp_path / "scene_frame.json",
        {"front_left": m, "shared_rear": MapToScene(1.0, np.eye(3), np.zeros(3))},
        "ep",
    )
    back = read_scene_frame(path)
    assert (
        np.allclose(back["front_left"].rotation, m.rotation)
        and back["front_left"].scale == 2.5
    )
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema"] == "scene_frame"
    assert doc["frame"] == "shared_rear_metric_gravity_aligned"

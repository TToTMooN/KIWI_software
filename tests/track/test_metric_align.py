import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.track.metric_align import GRAVITY_M_S2, estimate_metric_alignment


def synthetic_motion(duration=40.0, imu_hz=1000.0, cam_hz=120.0, seed=0):
    """Smooth 3-D motion with rotation; IMU in the body frame, camera poses in a scaled map."""
    rng = np.random.default_rng(seed)
    g_world = np.array([0.0, 0.0, -GRAVITY_M_S2])
    bias = np.array([0.05, -0.03, 0.08])
    T_cam_imu = np.eye(4)
    T_cam_imu[:3, :3] = Rotation.from_euler("xyz", [0.3, -1.2, 2.0]).as_matrix()
    T_cam_imu[:3, 3] = [0.02, -0.01, 0.03]
    t_imu = np.arange(0.0, duration, 1.0 / imu_hz)

    def pos(t):
        return np.column_stack(
            [0.8 * np.sin(0.7 * t), 0.5 * np.cos(0.9 * t), 0.3 * np.sin(1.3 * t + 0.4)]
        )

    def vel(t):
        return np.column_stack(
            [
                0.56 * np.cos(0.7 * t),
                -0.45 * np.sin(0.9 * t),
                0.39 * np.cos(1.3 * t + 0.4),
            ]
        )

    def acc(t):
        return np.column_stack(
            [
                -0.392 * np.sin(0.7 * t),
                -0.405 * np.cos(0.9 * t),
                -0.507 * np.sin(1.3 * t + 0.4),
            ]
        )

    def rot(t):
        return Rotation.from_rotvec(
            np.column_stack(
                [0.4 * np.sin(0.5 * t), 0.3 * t * 0.1, 0.5 * np.cos(0.3 * t)]
            )
        )

    R = rot(t_imu)
    # body angular velocity by finite differences of the rotation
    dR = (R[:-1].inv() * R[1:]).as_rotvec() * imu_hz
    gyro = np.vstack([dR, dR[-1:]])
    a_body = R.inv().apply(acc(t_imu) - g_world) + bias
    gyro += rng.normal(0, 1e-4, gyro.shape)
    a_body += rng.normal(0, 2e-3, a_body.shape)

    t_cam = np.arange(0.5, duration - 0.5, 1.0 / cam_hz)
    T_w_imu = np.tile(np.eye(4), (len(t_cam), 1, 1))
    T_w_imu[:, :3, :3] = rot(t_cam).as_matrix()
    T_w_imu[:, :3, 3] = pos(t_cam)
    T_w_cam = T_w_imu @ np.linalg.inv(T_cam_imu)
    scale = 2.7
    R_mw = Rotation.from_euler("xyz", [0.4, -0.7, 1.1]).as_matrix()
    R_map_cam = np.einsum("ij,njk->nik", R_mw, T_w_cam[:, :3, :3])
    p_map_cam = (T_w_cam[:, :3, 3] @ R_mw.T) / scale
    g_map = R_mw @ g_world
    return dict(
        t_cam=t_cam,
        R_map_cam=R_map_cam,
        p_map_cam=p_map_cam,
        T_cam_imu=T_cam_imu,
        t_imu=t_imu,
        gyro=gyro,
        acc=a_body,
        scale=scale,
        g_map=g_map,
        bias=bias,
        vel=vel,
    )


def test_recovers_scale_and_gravity():
    s = synthetic_motion()
    fit = estimate_metric_alignment(
        s["t_cam"],
        s["R_map_cam"],
        s["p_map_cam"],
        s["T_cam_imu"],
        s["t_imu"],
        s["gyro"],
        s["acc"],
    )
    assert abs(fit.scale / s["scale"] - 1) < 0.01
    assert np.allclose(fit.R_wg @ fit.R_wg.T, np.eye(3), atol=1e-12)
    down = fit.R_wg @ s["g_map"]
    assert np.degrees(np.arccos(-down[2] / GRAVITY_M_S2)) < 0.5

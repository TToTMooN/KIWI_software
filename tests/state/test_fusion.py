"""Synthetic check of the visual-inertial fusion: analytic IMU, noisy pose observations."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kiwi.state.fusion import fuse
from kiwi.state.geometry import se3, se3_inv

pytest.importorskip("gtsam")

AXIS = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
G = np.array([0.0, 0.0, -9.81])
BIAS = np.array([0.10, -0.05, 0.08])


def position(t):
    t = np.asarray(t)
    return np.column_stack(
        [0.3 * np.sin(0.7 * t), 0.25 * np.cos(0.5 * t), 0.2 * np.sin(0.9 * t) + 1.0]
    )


def acceleration(t):
    t = np.asarray(t)
    return np.column_stack(
        [
            -0.3 * 0.7**2 * np.sin(0.7 * t),
            -0.25 * 0.5**2 * np.cos(0.5 * t),
            -0.2 * 0.9**2 * np.sin(0.9 * t),
        ]
    )


def angle(t):
    return 0.5 * np.sin(0.4 * np.asarray(t))


def rate(t):
    return 0.5 * 0.4 * np.cos(0.4 * np.asarray(t))


def rotation(t):
    return Rotation.from_rotvec(np.outer(angle(t), AXIS))


def body_poses(t):
    R = rotation(t).as_matrix()
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = position(t)
    return T


def noisy(T, rng, sigma_m=0.002, sigma_deg=0.3):
    out = T.copy()
    dR = Rotation.from_rotvec(
        rng.normal(0, np.radians(sigma_deg), (len(T), 3))
    ).as_matrix()
    out[:, :3, :3] = dR @ T[:, :3, :3]
    out[:, :3, 3] += rng.normal(0, sigma_m, (len(T), 3))
    return out


def test_fuse_recovers_the_trajectory():
    rng = np.random.default_rng(0)
    T_front_imu = se3(
        Rotation.from_euler("xyz", [90, 0, 90], degrees=True).as_matrix(),
        [0.01, 0.02, -0.01],
    )
    T_rear_imu = se3(
        Rotation.from_euler("xyz", [-90, 0, 90], degrees=True).as_matrix(),
        [-0.015, 0.02, 0.01],
    )

    imu_t = np.arange(-0.5, 20.5, 0.001)
    R_imu = rotation(imu_t)
    acc_body = R_imu.inv().apply(acceleration(imu_t) - G) + BIAS
    gyro_body = np.outer(rate(imu_t), AXIS)  # fixed axis: body rate equals world rate

    t_nodes = np.arange(0, 20, 1 / 120.0)
    X_true = body_poses(t_nodes)
    front = (
        t_nodes,
        noisy(X_true @ se3_inv(T_front_imu), rng),
        np.zeros(len(t_nodes), int),
    )

    t_rear = np.arange(0.003, 20, 1 / 60.0)
    rear = (
        t_rear,
        noisy(body_poses(t_rear) @ se3_inv(T_rear_imu), rng),
        np.zeros(len(t_rear), int),
    )

    noise = {
        "accelerometer_noise_density": 4e-4,
        "accelerometer_random_walk": 4e-5,
        "gyroscope_noise_density": 7e-5,
        "gyroscope_random_walk": 8e-6,
    }
    T = fuse(
        t_nodes, imu_t, gyro_body, acc_body, front, rear, T_front_imu, T_rear_imu, noise
    )

    err = T[:, :3, 3] - X_true[:, :3, 3]
    rms = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    assert rms < 0.01, rms

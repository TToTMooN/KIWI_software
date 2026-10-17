"""Per-arm visual-inertial batch fusion.

One factor graph per wrist. Every native front frame is a node carrying the
IMU body pose ``X_i = T_scene_imu``, the scene-frame velocity ``v_i`` and the
IMU biases ``b_i``. Consecutive nodes are linked by GTSAM's ``ImuFactor`` and a
``BetweenFactor`` on the biases; tracked front and rear localizations become
robust unary pose factors on the same body pose after the fixed camera-IMU
transform. The two arms share no factors: both sets of observations live in
the one scene frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

IMU_NOISE_SCALE = 10.0
VISUAL_INTERP_MAX_GAP_S = 0.050
VISUAL_FACTOR_PERIOD_S = 0.100
FRONT_SIGMA_ROT_DEG = 1.0
FRONT_SIGMA_TRANS_M = 0.015
REAR_SIGMA_ROT_DEG = 1.0
REAR_SIGMA_TRANS_M = 0.020
POSE_HUBER_K = 2.5
VELOCITY_PRIOR_SIGMA_M_S = 1.0
ACCEL_BIAS_PRIOR_SIGMA_M_S2 = 0.20
GYRO_BIAS_PRIOR_SIGMA_RAD_S = 0.05
INTEGRATION_COV = 1e-8
MAX_ITERATIONS = 100
GRAVITY = (0.0, 0.0, -9.81)


@dataclass
class BodyTrack:
    """Tracked body poses ``T_scene_imu`` on the IMU-aligned clock, with run ids."""

    t: np.ndarray
    R: Rotation
    p: np.ndarray
    run_id: np.ndarray


@dataclass
class Scheduled:
    node: int
    R: Rotation
    p: np.ndarray
    run_id: int


# ----------------------------------------------------------------------------
# Observation handling
# ----------------------------------------------------------------------------


def body_track(obs: tuple, T_cam_imu: np.ndarray) -> BodyTrack:
    """Camera observations ``(t, T_scene_cam, run_id)`` -> body track ``T_scene_imu``."""
    t, T_cam, run_id = obs
    T_body = np.asarray(T_cam) @ T_cam_imu
    return BodyTrack(
        np.asarray(t, dtype=float),
        Rotation.from_matrix(T_body[:, :3, :3]),
        T_body[:, :3, 3],
        np.asarray(run_id),
    )


def schedule(track: BodyTrack, t_nodes: np.ndarray, exact_only: bool) -> list:
    """Admit observations at nodes: exact matches always, interpolation only for the rear.

    Interpolation needs two bracketing rows at most ``VISUAL_INTERP_MAX_GAP_S`` apart
    from the same contiguous tracking run; lost frames are never bridged.
    """
    out: list[Scheduled] = []
    for node, tn in enumerate(t_nodes):
        hi = int(np.searchsorted(track.t, tn, side="left"))
        if hi < len(track.t) and track.t[hi] == tn:
            out.append(Scheduled(node, track.R[hi], track.p[hi], int(track.run_id[hi])))
            continue
        if exact_only or hi == 0 or hi == len(track.t):
            continue
        lo = hi - 1
        bracket = track.t[hi] - track.t[lo]
        if bracket > VISUAL_INTERP_MAX_GAP_S or track.run_id[lo] != track.run_id[hi]:
            continue
        f = (tn - track.t[lo]) / bracket
        rel = track.R[lo].inv() * track.R[hi]
        R = track.R[lo] * Rotation.from_rotvec(f * rel.as_rotvec())
        p = (1 - f) * track.p[lo] + f * track.p[hi]
        out.append(Scheduled(node, R, p, int(track.run_id[lo])))
    return out


def thin(scheduled: list, t_nodes: np.ndarray) -> list:
    """Keep about one factor per ``VISUAL_FACTOR_PERIOD_S`` within each run, plus every
    run's endpoints."""
    if not scheduled:
        return []
    runs: list[list[Scheduled]] = [[scheduled[0]]]
    for m in scheduled[1:]:
        prev = runs[-1][-1]
        if (
            m.run_id != prev.run_id
            or t_nodes[m.node] - t_nodes[prev.node] > VISUAL_INTERP_MAX_GAP_S
        ):
            runs.append([])
        runs[-1].append(m)
    selected: list[Scheduled] = []
    for run in runs:
        selected.append(run[0])
        last_t = t_nodes[run[0].node]
        for m in run[1:-1]:
            if t_nodes[m.node] - last_t >= VISUAL_FACTOR_PERIOD_S:
                selected.append(m)
                last_t = t_nodes[m.node]
        if run[-1].node != selected[-1].node:
            selected.append(run[-1])
    return selected


def run_ids_from_tracked(tracked: np.ndarray) -> np.ndarray:
    """Contiguous runs of tracked rows get one id each (untracked rows get -1)."""
    ids = np.full(len(tracked), -1, dtype=int)
    rid = -1
    prev = False
    for i, ok in enumerate(tracked):
        if ok and not prev:
            rid += 1
        if ok:
            ids[i] = rid
        prev = ok
    return ids


# ----------------------------------------------------------------------------
# IMU handling
# ----------------------------------------------------------------------------


def integrate_interval(pim, imu_t, gyro, acc, start: float, end: float) -> None:
    """Zero-order-hold integration of ``[start, end)``, split exactly at node boundaries."""
    k = max(0, int(np.searchsorted(imu_t, start, side="right")) - 1)
    cursor = float(start)
    while cursor < end - 1e-12:
        boundary = end
        if k + 1 < len(imu_t):
            boundary = min(boundary, float(imu_t[k + 1]))
        dt = boundary - cursor
        if dt <= 0:
            k += 1
            continue
        pim.integrateMeasurement(acc[k], gyro[k], dt)
        cursor = boundary
        if k + 1 < len(imu_t) and cursor >= imu_t[k + 1] - 1e-12:
            k += 1


# ----------------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------------


def fuse(
    t_nodes: np.ndarray,
    imu_t: np.ndarray,
    imu_gyro: np.ndarray,
    imu_acc: np.ndarray,
    front_obs: tuple,
    rear_obs: tuple,
    T_front_imu: np.ndarray,
    T_rear_imu: np.ndarray,
    noise: dict,
) -> np.ndarray:
    """Solve one arm's graph; returns ``T_scene_imu`` (N, 4, 4) at the nodes.

    Observations are ``(t_cam_s, T_scene_cam (N,4,4), run_id)`` of tracked rows on the
    IMU-aligned clock; ``t_nodes`` are the native front frame times on that clock."""
    import gtsam
    from gtsam.symbol_shorthand import B, V, X

    t_nodes = np.asarray(t_nodes, dtype=float)
    n = len(t_nodes)
    imu_t = np.asarray(imu_t, dtype=float)

    tracks = {
        "front": body_track(front_obs, T_front_imu),
        "rear": body_track(rear_obs, T_rear_imu),
    }
    scheduled = {
        lens: thin(schedule(track, t_nodes, exact_only=(lens == "front")), t_nodes)
        for lens, track in tracks.items()
    }

    # Initial values from the densest track; never factors.
    densest = max(tracks.values(), key=lambda tr: len(tr.t))
    if len(densest.t) == 1:
        R0 = Rotation.from_quat(np.repeat(densest.R.as_quat()[None], n, axis=0))
        p0 = np.repeat(densest.p, n, axis=0)
    else:
        tq = np.clip(t_nodes, densest.t[0], densest.t[-1])
        R0 = Slerp(densest.t, densest.R)(tq)
        p0 = np.column_stack(
            [np.interp(tq, densest.t, densest.p[:, k]) for k in range(3)]
        )
    v0 = np.gradient(p0, t_nodes, axis=0, edge_order=1)

    s = IMU_NOISE_SCALE
    params = gtsam.PreintegrationParams(np.asarray(GRAVITY, dtype=float))
    params.setAccelerometerCovariance(
        np.eye(3) * (noise["accelerometer_noise_density"] * s) ** 2
    )
    params.setGyroscopeCovariance(
        np.eye(3) * (noise["gyroscope_noise_density"] * s) ** 2
    )
    params.setIntegrationCovariance(np.eye(3) * INTEGRATION_COV)
    # Bias random walk: sigma grows with the square root of the interval between nodes.
    walk = np.r_[
        [noise["accelerometer_random_walk"] * s] * 3,
        [noise["gyroscope_random_walk"] * s] * 3,
    ]

    bias0 = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    for i in range(n):
        values.insert(X(i), gtsam.Pose3(gtsam.Rot3(R0[i].as_matrix()), p0[i]))
        values.insert(V(i), v0[i])
        values.insert(B(i), bias0)
    graph.add(
        gtsam.PriorFactorVector(
            V(0), v0[0], gtsam.noiseModel.Isotropic.Sigma(3, VELOCITY_PRIOR_SIGMA_M_S)
        )
    )
    graph.add(
        gtsam.PriorFactorConstantBias(
            B(0),
            gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
            gtsam.noiseModel.Diagonal.Sigmas(
                np.r_[
                    [ACCEL_BIAS_PRIOR_SIGMA_M_S2] * 3, [GYRO_BIAS_PRIOR_SIGMA_RAD_S] * 3
                ]
            ),
        )
    )
    for i in range(n - 1):
        pim = gtsam.PreintegratedImuMeasurements(params, bias0)
        integrate_interval(pim, imu_t, imu_gyro, imu_acc, t_nodes[i], t_nodes[i + 1])
        graph.add(gtsam.ImuFactor(X(i), V(i), X(i + 1), V(i + 1), B(i), pim))
        dt = float(t_nodes[i + 1] - t_nodes[i])
        graph.add(
            gtsam.BetweenFactorConstantBias(
                B(i),
                B(i + 1),
                gtsam.imuBias.ConstantBias(),
                gtsam.noiseModel.Diagonal.Sigmas(walk * np.sqrt(dt)),
            )
        )

    def pose_noise(rot_deg, trans_m):
        base = gtsam.noiseModel.Diagonal.Sigmas(
            np.r_[[np.radians(rot_deg)] * 3, [trans_m] * 3]
        )
        return gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(POSE_HUBER_K), base
        )

    noises = {
        "front": pose_noise(FRONT_SIGMA_ROT_DEG, FRONT_SIGMA_TRANS_M),
        "rear": pose_noise(REAR_SIGMA_ROT_DEG, REAR_SIGMA_TRANS_M),
    }
    for lens, items in scheduled.items():
        for m in items:
            graph.add(
                gtsam.PriorFactorPose3(
                    X(m.node),
                    gtsam.Pose3(gtsam.Rot3(m.R.as_matrix()), m.p),
                    noises[lens],
                )
            )

    lm = gtsam.LevenbergMarquardtParams()
    lm.setMaxIterations(MAX_ITERATIONS)
    lm.setOrderingType("METIS")
    result = gtsam.LevenbergMarquardtOptimizer(graph, values, lm).optimize()

    T = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        pose = result.atPose3(X(i))
        T[i, :3, :3] = pose.rotation().matrix()
        T[i, :3, 3] = pose.translation()
    return T

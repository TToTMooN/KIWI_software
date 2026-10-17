"""The shared scene frame: each map's frozen placement and the front-map placement solve.

The scene frame is the shared rear map made metric and gravity aligned. A front map is
placed into it by a constant ``G`` in SE(3) that makes the front trajectory agree with the
rear trajectory composed with the arm's fixed rear-front lever ``X``::

    G T_F(t) ~= T_R(t) X

The lever and the per-map scales are fixed inputs; only ``G`` is solved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from kiwi.track.metric_align import MetricAlignment

SCENE_FRAME_NAME = "shared_rear_metric_gravity_aligned"
PAIR_GAP_MAX_S = 0.05
SIGMA_ROT_RAD = np.radians(0.5)
SIGMA_POS_M = 0.003


@dataclass(frozen=True)
class MapToScene:
    """``p_scene = scale * (rotation @ p_map) + translation``;
    ``R_scene_cam = rotation @ R_map_cam``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, T_map_cam: np.ndarray) -> np.ndarray:
        T = np.asarray(T_map_cam, dtype=float)
        out = np.tile(np.eye(4), (len(T), 1, 1))
        out[:, :3, :3] = np.einsum("ij,njk->nik", self.rotation, T[:, :3, :3])
        out[:, :3, 3] = (
            self.scale * np.einsum("ij,nj->ni", self.rotation, T[:, :3, 3])
            + self.translation
        )
        return out

    @classmethod
    def from_metric_alignment(cls, a: MetricAlignment) -> MapToScene:
        return cls(
            scale=float(a.scale),
            rotation=np.asarray(a.R_wg, float),
            translation=np.zeros(3),
        )

    def to_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "rotation": [[float(v) for v in row] for row in self.rotation],
            "translation": [float(v) for v in self.translation],
        }

    @classmethod
    def from_dict(cls, d: dict) -> MapToScene:
        return cls(
            float(d["scale"]),
            np.asarray(d["rotation"], float),
            np.asarray(d["translation"], float),
        )


def _pairs(t_front, t_rear, T_rear, rear_run_id):
    """Interpolate the rear track onto the front times (slerp + linear).

    A front time is kept when its rear bracket is two rows of one tracking run at most
    ``PAIR_GAP_MAX_S`` apart.
    """
    t_f = np.asarray(t_front, float)
    t_r = np.asarray(t_rear, float)
    T_r = np.asarray(T_rear, float)
    rid = np.asarray(rear_run_id)
    j = np.clip(np.searchsorted(t_r, t_f), 1, len(t_r) - 1)
    keep = (t_f >= t_r[0]) & (t_f <= t_r[-1])
    keep &= t_r[j] - t_r[j - 1] <= PAIR_GAP_MAX_S
    keep &= rid[j] == rid[j - 1]
    tq = t_f[keep]
    R_at = Slerp(t_r, Rotation.from_matrix(T_r[:, :3, :3]))(tq)
    p_at = np.column_stack([np.interp(tq, t_r, T_r[:, k, 3]) for k in range(3)])
    return keep, R_at, p_at


def solve_front_to_scene(
    t_front: np.ndarray,
    T_front_metric: np.ndarray,
    t_rear: np.ndarray,
    T_rear_metric: np.ndarray,
    T_rear_front: np.ndarray,
    rear_run_id: np.ndarray,
) -> np.ndarray:
    """Solve the constant ``G`` (front metric frame -> scene frame) with the lever fixed.

    Both tracks are metric camera poses: the front one in its own gravity-aligned map
    frame, the rear one in the scene frame, times on the arm's IMU clock. Returns the 4x4
    ``G``.
    """
    keep, R_R, p_R = _pairs(t_front, t_rear, T_rear_metric, rear_run_id)
    T_F = np.asarray(T_front_metric)[keep]
    R_F = Rotation.from_matrix(T_F[:, :3, :3])
    p_F = T_F[:, :3, 3]
    X = np.asarray(T_rear_front, float)
    RX = Rotation.from_matrix(X[:3, :3])
    tX = X[:3, 3]

    # Initial value: the per-pair transforms T_R X inv(T_F), averaged.
    T_R = np.tile(np.eye(4), (len(T_F), 1, 1))
    T_R[:, :3, :3] = R_R.as_matrix()
    T_R[:, :3, 3] = p_R
    cand = np.einsum("nij,jk,nkl->nil", T_R, X, np.linalg.inv(T_F))
    R0 = Rotation.from_matrix(cand[:, :3, :3]).mean()
    t0 = np.median(cand[:, :3, 3], axis=0)

    def fun(q):
        RG = Rotation.from_rotvec(q[:3]) * R0
        r_rot = ((RG * R_F).inv() * (R_R * RX)).as_rotvec()
        r_pos = (RG.apply(p_F) + q[3:6]) - (p_R + R_R.apply(tX))
        return np.concatenate(
            [(r_rot / SIGMA_ROT_RAD).ravel(), (r_pos / SIGMA_POS_M).ravel()]
        )

    sol = least_squares(fun, np.concatenate([[0, 0, 0], t0]), loss="huber", f_scale=1.0)
    G = np.eye(4)
    G[:3, :3] = (Rotation.from_rotvec(sol.x[:3]) * R0).as_matrix()
    G[:3, 3] = sol.x[3:6]
    return G


def write_scene_frame(path: Path, maps: dict[str, MapToScene], episode: str) -> Path:
    doc = {
        "schema": "scene_frame",
        "frame": SCENE_FRAME_NAME,
        "episode": episode,
        "maps": {role: maps[role].to_dict() for role in sorted(maps)},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def read_scene_frame(path: Path) -> dict[str, MapToScene]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {role: MapToScene.from_dict(entry) for role, entry in doc["maps"].items()}

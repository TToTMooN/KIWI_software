"""Per-frame engine and scene-frame observation rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "LocalizationRows",
    "read_localization_rows",
    "TRACKED",
    "Observations",
    "read_observations",
    "write_observations",
]

TRACKED = 2  # ORB-SLAM3 Tracking::OK


@dataclass
class Observations:
    """Per-frame engine or scene-frame poses: ``t_ns state tx ty tz qx qy qz qw``."""

    t_ns: np.ndarray  # int64 (N,)
    state: np.ndarray  # int (N,)
    T: np.ndarray  # (N, 4, 4) camera pose; identity where untracked

    @property
    def tracked(self) -> np.ndarray:
        return self.state == TRACKED

    @property
    def t_s(self) -> np.ndarray:
        return self.t_ns.astype(float) * 1e-9

    def __len__(self) -> int:
        return len(self.t_ns)


def read_observations(path: Path) -> Observations:
    rows = np.loadtxt(path, ndmin=2)
    state = rows[:, 1].astype(int)
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    ok = state == TRACKED
    T[ok, :3, :3] = Rotation.from_quat(rows[ok, 5:9]).as_matrix()
    T[ok, :3, 3] = rows[ok, 2:5]
    return Observations(t_ns=rows[:, 0].astype(np.int64), state=state, T=T)


def write_observations(path: Path, obs: Observations) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = len(obs)
    q = np.zeros((n, 4))
    p = np.zeros((n, 3))
    ok = obs.tracked
    q[ok] = Rotation.from_matrix(obs.T[ok, :3, :3]).as_quat()
    p[ok] = obs.T[ok, :3, 3]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            if ok[i]:
                f.write(
                    f"{int(obs.t_ns[i])} {int(obs.state[i])} "
                    f"{p[i, 0]:.9f} {p[i, 1]:.9f} {p[i, 2]:.9f} "
                    f"{q[i, 0]:.9f} {q[i, 1]:.9f} {q[i, 2]:.9f} {q[i, 3]:.9f}\n"
                )
            else:
                f.write(f"{int(obs.t_ns[i])} {int(obs.state[i])} 0 0 0 0 0 0 0\n")


@dataclass
class LocalizationRows:
    """The columns of an observation file as plain arrays, one entry per row."""

    t_ns: np.ndarray  # (N,) int64
    state: np.ndarray  # (N,) int64 engine tracking state
    p: np.ndarray  # (N, 3) camera position, zeros where untracked
    q_xyzw: np.ndarray  # (N, 4) Hamilton xyzw, zeros where untracked

    def __len__(self) -> int:
        return len(self.t_ns)

    @property
    def tracked(self) -> np.ndarray:
        return self.state == TRACKED


def read_localization_rows(path: Path) -> LocalizationRows:
    """Read the first nine columns of an observation file without converting poses."""
    rows = np.loadtxt(path, ndmin=2)
    if rows.size == 0 or rows.shape[1] < 9:
        raise ValueError(f"{path}: no rows")
    return LocalizationRows(
        rows[:, 0].astype(np.int64),
        rows[:, 1].astype(np.int64),
        rows[:, 2:5],
        rows[:, 5:9],
    )

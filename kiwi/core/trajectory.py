"""The metric trajectory contract used by state, registration and export.

State publishes the front camera in the shared scene frame; registration
can re-express that trajectory in a marker frame. Viz, export and replay
consume these declared poses. Timestamps are float64 seconds, increasing
strictly. Published episode trajectories declare the episode global clock.
Positions are metres; rotations use Hamilton quaternions in scipy xyzw
order. Frame names the reference frame and body names the posed object.

Native estimator text is not interchangeable with a normalized camera TUM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

__all__ = [
    "PoseTrack",
    "Trajectory",
    "TumHeader",
    "load_tum",
    "read_tum_header",
    "se3",
    "se3_stack",
    "write_tum",
]


# --- SE(3) helpers ---------------------------------------------------------


def se3(position: np.ndarray, rot: Rotation) -> np.ndarray:
    """4x4 SE(3) from a position + scipy Rotation."""
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = position
    return T


def se3_stack(position: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """(N,4,4) SE(3) from N positions + N (qx,qy,qz,qw) quaternions."""
    position = np.asarray(position, dtype=np.float64)
    T = np.tile(np.eye(4), (len(position), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:, :3, 3] = position
    return T


# --- the contract ----------------------------------------------------------


@dataclass
class Trajectory:
    """A timestamped 6-DoF pose track. See module docstring for conventions."""

    timestamps: np.ndarray  # (N,) float64, seconds, global clock, increasing
    positions: np.ndarray  # (N, 3) float64, meters
    quaternions: np.ndarray  # (N, 4) float64, Hamilton, xyzw (scipy order)
    frame: str
    body: str

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    def validate(self) -> "Trajectory":
        """Cheap shape/monotonicity checks; raises ValueError on violation."""
        t, p, q = self.timestamps, self.positions, self.quaternions
        if t.ndim != 1 or p.shape != (len(t), 3) or q.shape != (len(t), 4):
            raise ValueError(f"shape mismatch: t{t.shape} p{p.shape} q{q.shape}")
        if len(t) > 1 and not np.all(np.diff(t) > 0):
            raise ValueError("timestamps must be strictly increasing")
        return self

    def poses(self) -> np.ndarray:
        """(N,4,4) SE(3) matrices ``T_frame_body`` equivalent to p/q storage."""
        return se3_stack(self.positions, self.quaternions)

    @classmethod
    def from_poses(
        cls,
        timestamps: np.ndarray,
        poses: np.ndarray,
        frame: str,
        body: str,
    ) -> "Trajectory":
        """Build from (N,4,4) SE(3) matrices ``T_frame_body``."""
        timestamps = np.asarray(timestamps, dtype=np.float64)
        poses = np.asarray(poses, dtype=np.float64)
        if poses.shape != (len(timestamps), 4, 4):
            raise ValueError(
                f"poses must be ({len(timestamps)},4,4) to match timestamps, got {poses.shape}"
            )
        if len(timestamps):
            quaternions = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
        else:
            quaternions = np.zeros((0, 4))
        return cls(
            timestamps=timestamps,
            positions=poses[:, :3, 3].copy(),
            quaternions=quaternions,
            frame=frame,
            body=body,
        )


# --- interpolation ---------------------------------------------------------


@dataclass
class PoseTrack:
    """An interpolatable SE(3) track: Slerp rotation + linear translation.

    For joining an asynchronously-sampled pose stream (e.g. a robot EE log)
    to timestamps from another sensor - sampling T(t) at a camera frame's
    time. Also carries a per-sample angular speed so callers can gate out
    fast-motion samples (where a small sync-offset error would bias the
    interpolated pose the most).
    """

    t: np.ndarray  # (N,) seconds, strictly increasing (Slerp requires it)
    rotations: Rotation  # N rotations
    positions: np.ndarray  # (N,3)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        if len(self.t) < 2:
            raise ValueError("PoseTrack needs >= 2 samples to interpolate")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("PoseTrack t must be strictly increasing")
        if self.positions.shape != (len(self.t), 3):
            raise ValueError(
                f"positions must be ({len(self.t)},3), got {self.positions.shape}"
            )
        self._slerp = Slerp(self.t, self.rotations)
        # Per-sample angular speed (deg/s), midpoint differences.
        rel = np.degrees((self.rotations[:-1].inv() * self.rotations[1:]).magnitude())
        speed_mid = rel / np.clip(np.diff(self.t), 1e-6, None)
        ang = np.empty_like(self.t)
        ang[1:-1] = 0.5 * (speed_mid[:-1] + speed_mid[1:])
        ang[0], ang[-1] = speed_mid[0], speed_mid[-1]
        self.ang_speed = ang

    @property
    def t0(self) -> float:
        return float(self.t[0])

    @property
    def t1(self) -> float:
        return float(self.t[-1])

    def at(self, time: float) -> np.ndarray:
        """4x4 pose at ``time`` (Slerp rotation + linear translation)."""
        T = np.eye(4)
        T[:3, :3] = self._slerp(time).as_matrix()
        T[:3, 3] = [np.interp(time, self.t, self.positions[:, k]) for k in range(3)]
        return T

    def ang_speed_at(self, time: float) -> float:
        """Interpolated angular speed (deg/s) at ``time`` - for motion gating."""
        return float(np.interp(time, self.t, self.ang_speed))

    def gap_at(self, time: float) -> float:
        """Distance (s) to the nearest raw sample - for sync-gap gating."""
        return float(np.min(np.abs(self.t - time)))


# --- loaders ---------------------------------------------------------------
# ``load_tum`` returns the 4-tuple ``(t, p, q_xyzw, grip)``:
#
#   - ``t``       (N,)    seconds - nanosecond timestamps are auto-detected/scaled.
#   - ``p``       (N, 3)  position.
#   - ``q_xyzw``  (N, 4)  orientation, scipy-native order.
#   - ``grip``    always ``None``: a TUM file carries no tool value. The slot
#                 exists so every caller unpacks the same shape.

# |t| above this is nanoseconds, not seconds. The discriminator only has to
# separate epoch-seconds (~1e9, the canonical TUM RGB-D stamps) from
# epoch-nanoseconds (~1e18, what EuRoC-derived tools emit); 1e12 (~31,700
# years in seconds, ~17 minutes in ns) splits them unambiguously.
_NS_THRESHOLD = 1e12


def _load_text(path: Path) -> np.ndarray:
    """np.loadtxt that tolerates a non-#-prefixed header line (the *.tum header)."""
    try:
        return np.loadtxt(path, comments="#", ndmin=2)
    except ValueError:
        # Some TUM writers emit a bare "t tx ty tz ..." header (no '#').
        # Only drop line 1 when it actually is such a header - a numeric
        # first row means the file is malformed (e.g. inconsistent column
        # count further down), and silently discarding a pose would mask it.
        with open(path, encoding="utf-8") as f:
            first = f.readline().split()
        try:
            [float(tok) for tok in first]
        except ValueError:
            return np.loadtxt(path, comments="#", ndmin=2, skiprows=1)
        raise


def _maybe_ns_to_s(t: np.ndarray) -> np.ndarray:
    """Scale nanosecond timestamps to seconds (EuRoC-derived tools emit ns)."""
    t = t.astype(np.float64)
    return t * 1e-9 if np.median(np.abs(t)) > _NS_THRESHOLD else t


def load_tum(path: Path):
    """TUM: ``t tx ty tz qx qy qz qw`` -> (t, p, q_xyzw, None)."""
    d = _load_text(path)
    if d.shape[1] < 8:
        raise ValueError(
            f"{path}: TUM needs >=8 cols (t tx ty tz qx qy qz qw), got {d.shape[1]}"
        )
    return _maybe_ns_to_s(d[:, 0]), d[:, 1:4], d[:, 4:8], None


def write_tum(traj: Trajectory, path: Path, *, clock: str, source: str = "") -> Path:
    """Write a trajectory as TUM text, with its clock DECLARED in the header.

    Readers such as ``kiwi.export.build.read_world_tum_header`` refuse a
    ``.tum`` whose header does not say which clock its timestamps are on,
    rather than guess. ``clock`` is that declaration; the episode stages pass
    ``"episode global clock (origin=<label>)"``.
    """
    path = Path(path)
    header = (
        f"t tx ty tz qx qy qz qw  (frame={traj.frame} body={traj.body}; t on {clock}"
    )
    header += f"; source={source})" if source else ")"
    rows = np.column_stack([traj.timestamps, traj.positions, traj.quaternions])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, fmt="%.9f", header=header)
    return path


# Anchored on the FINAL ')' so a clock that itself contains parentheses -
# "episode global clock (origin=right)" - parses whole; the optional
# "; source=..." tail is split off afterwards.
_TUM_HEADER_RE = re.compile(r"\(frame=(\S+)\s+body=(\S+);\s*t on (.+)\)\s*$")


@dataclass(frozen=True)
class TumHeader:
    """What a :func:`write_tum` header declares: frame, body, clock."""

    frame: str
    body: str
    clock: str


def read_tum_header(path: Path) -> TumHeader | None:
    """The frame/body/clock a ``.tum`` declares, or ``None`` when it doesn't.

    ``None`` covers both a missing header line and one in another dialect
    (e.g. a bare column list) - the caller decides whether an undeclared file
    is refusable there. What to refuse is the caller's contract:
    ``kiwi.export.build.read_world_tum_header`` keeps its own, stricter reader.
    """
    with Path(path).open(encoding="utf-8") as fh:
        header = fh.readline()
    if not header.startswith("#"):
        return None
    m = _TUM_HEADER_RE.search(header)
    if m is None:
        return None
    clock = m.group(3).split("; source=")[0].strip()
    return TumHeader(frame=m.group(1), body=m.group(2), clock=clock)

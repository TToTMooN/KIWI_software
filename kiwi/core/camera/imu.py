"""Extract Insta360 IMU telemetry for scene mapping and state estimation.

CSV columns are t [s], gyro [rad/s], acceleration [m/s²], in recording IMU
axes. A nonnegative pre-roll shift is recorded in ImuMeta; scene consumers
undo it before applying the calibrated camera-to-IMU time shift.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

__all__ = [
    "ImuMeta",
    "compute_clock_offset_ns",
    "extract_imu_to_csv",
]

_RAD = np.pi / 180.0


@dataclasses.dataclass(frozen=True)
class ImuMeta:
    """Clock metadata the camera side needs to stay time-aligned with the IMU."""

    n_samples: int
    clock_offset_s: float  # added to every stream to keep time non-negative
    t_first_s: float  # first IMU time AFTER the offset (~= 0.0)
    t_last_s: float  # last IMU time AFTER the offset


def compute_clock_offset_ns(first_ts_ms: float) -> int:
    """``clock_offset_ns = -min(0, round(first_ts_ms * 1e6))`` (the pre-roll)."""
    first_t_ns = int(round(first_ts_ms * 1_000_000))
    return -min(0, first_t_ns)


def extract_imu_to_csv(insv: Path, out_csv: Path) -> ImuMeta:
    """Write ``t gx gy gz ax ay az`` to ``out_csv``; return :class:`ImuMeta`.

    Timestamps are shifted by ``clock_offset_ns`` using integer-nanosecond
    rounding, gyro deg/s -> rad/s, accel m/s^2 unchanged - the exact recipe
    the scene IMU cache expects. ``telemetry_parser`` is imported
    lazily.
    """
    import telemetry_parser

    tp = telemetry_parser.Parser(str(insv))
    samples = tp.normalized_imu()
    if not samples:
        raise RuntimeError(
            f"{insv}: telemetry_parser.normalized_imu() returned no samples - "
            f"not an Insta360 recording, or a truncated copy?"
        )

    clock_offset_ns = compute_clock_offset_ns(samples[0]["timestamp_ms"])

    rows = np.empty((len(samples), 7), dtype=np.float64)
    for i, s in enumerate(samples):
        ts_ns = int(round(s["timestamp_ms"] * 1_000_000)) + clock_offset_ns
        g = s["gyro"]
        a = s["accl"]
        rows[i, 0] = ts_ns * 1e-9
        rows[i, 1] = g[0] * _RAD
        rows[i, 2] = g[1] * _RAD
        rows[i, 3] = g[2] * _RAD
        rows[i, 4] = a[0]
        rows[i, 5] = a[1]
        rows[i, 6] = a[2]

    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_csv,
        rows,
        fmt=["%.9f", "%.9f", "%.9f", "%.9f", "%.9f", "%.9f", "%.9f"],
        header="t gx gy gz ax ay az",
        comments="",
    )
    return ImuMeta(
        n_samples=len(rows),
        clock_offset_s=clock_offset_ns * 1e-9,
        t_first_s=float(rows[0, 0]),
        t_last_s=float(rows[-1, 0]),
    )

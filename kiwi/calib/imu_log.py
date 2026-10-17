"""Compact on-disk IMU logs extracted from Insta360 ``.insv`` recordings.

The ``.npz`` keeps the 1 kHz IMU telemetry of a stationary Allan-calibration
video; after extraction the ``.insv`` files can be deleted. This module owns
the ``imu_raw_v1`` ``.npz`` format:

  * per-part arrays ``t_000``/``imu_000``, ``t_001``/``imu_001``, ... - the
    telemetry of each auto-split ``.insv`` part file, verbatim: float64
    seconds on that part's own restarting clock, and an ``(N, 6)`` float64
    block whose columns are gyro xyz [rad/s] then accel xyz [m/s^2]
  * ``meta_json`` - a JSON string with ``schema``, ``columns``, source-file
    provenance and camera identity

Parts are stored raw (not pre-stitched), duplicated boundary samples
included: ``allan.stitch_segments`` reassembles - and cross-checks - the
continuous series from the ``.npz`` alone.

``parse_insv_imu`` imports ``telemetry_parser`` lazily; everything else is
pure numpy + json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "imu_raw_v1"
COLUMNS = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
UNITS = {"t": "s", "gyro": "rad/s", "accel": "m/s^2"}


def parse_insv_imu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (t [s], values [N, 6]) from one ``.insv`` file.

    Unit conventions:
    telemetry_parser's normalized_imu() gives gyro in deg/s (converted here
    to rad/s) and accel already in m/s^2.
    """
    import telemetry_parser

    samples = telemetry_parser.Parser(str(path)).normalized_imu()
    t = np.array([s["timestamp_ms"] for s in samples]) * 1e-3
    gyro = np.array([s["gyro"] for s in samples]) * (np.pi / 180.0)
    accel = np.array([s["accl"] for s in samples])
    return t, np.hstack([gyro, accel])


def save_imu_npz(
    path: Path,
    ts: list[np.ndarray],
    values: list[np.ndarray],
    meta: dict | None = None,
) -> None:
    """Write per-part IMU series to an ``imu_raw_v1`` ``.npz`` (compressed)."""
    if len(ts) != len(values):
        raise ValueError(
            f"got {len(ts)} timestamp arrays but {len(values)} value arrays"
        )
    if not ts:
        raise ValueError("no segments given")
    full_meta = {
        "schema": SCHEMA,
        "columns": COLUMNS,
        "units": UNITS,
        "n_segments": len(ts),
        "clock_note": "t restarts per segment; stitch with kiwi.calib.allan.stitch_segments",
        **(meta or {}),
    }
    arrays: dict[str, np.ndarray] = {}
    for i, (t, v) in enumerate(zip(ts, values)):
        t = np.ascontiguousarray(t, dtype=np.float64)
        v = np.ascontiguousarray(v, dtype=np.float64)
        if t.ndim != 1 or v.shape != (len(t), 6):
            raise ValueError(
                f"segment {i}: expected t (N,) and values (N, 6), got {t.shape} / {v.shape}"
            )
        arrays[f"t_{i:03d}"] = t
        arrays[f"imu_{i:03d}"] = v
    np.savez_compressed(path, meta_json=json.dumps(full_meta), **arrays)


def load_imu_npz(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Read an ``imu_raw_v1`` ``.npz`` back into (ts, values, meta)."""
    with np.load(path) as z:
        if "meta_json" not in z:
            raise ValueError(f"{path}: not an {SCHEMA} file (no meta_json)")
        meta = json.loads(str(z["meta_json"].item()))
        if meta.get("schema") != SCHEMA:
            raise ValueError(
                f"{path}: schema {meta.get('schema')!r}, expected {SCHEMA!r}"
            )
        n = int(meta["n_segments"])
        ts = [z[f"t_{i:03d}"] for i in range(n)]
        values = [z[f"imu_{i:03d}"] for i in range(n)]
    return ts, values, meta


def extract_npz(paths: list[Path], out: Path, meta: dict | None = None) -> dict:
    """Parse ``.insv`` part files into a verified ``imu_raw_v1`` ``.npz``.

    Stores each part verbatim, then verifies the write: (1) the ``.npz``
    round-trips bit-exactly, and (2) multi-part runs still stitch into one
    continuous series from the ``.npz`` alone. A verification failure raises
    ``RuntimeError`` - the caller must treat the output as unusable.

    Returns stats for reporting: ``{"n_parts", "n_samples", "insv_bytes",
    "npz_bytes", "stitched_samples" (multi-part only)}``. The source videos
    are not deleted here.
    """
    from kiwi.calib.allan import stitch_segments

    ts, vals = [], []
    for p in paths:
        t, v = parse_insv_imu(p)
        ts.append(t)
        vals.append(v)
    save_imu_npz(out, ts, vals, meta)

    stats: dict = {
        "n_parts": len(paths),
        "n_samples": int(sum(len(t) for t in ts)),
        "insv_bytes": int(sum(p.stat().st_size for p in paths)),
        "npz_bytes": int(Path(out).stat().st_size),
    }
    # Verify 1: the .npz round-trips bit-exactly.
    ts2, vals2, _ = load_imu_npz(out)
    for i, (t, t2, v, v2) in enumerate(zip(ts, ts2, vals, vals2)):
        if not (np.array_equal(t, t2) and np.array_equal(v, v2)):
            raise RuntimeError(f"{out}: segment {i} does not round-trip bit-exactly")
    # Verify 2: the saved parts still assemble into one continuous series.
    if len(ts2) > 1:
        t_all, _ = stitch_segments(ts2, vals2)
        stats["stitched_samples"] = int(len(t_all))
    return stats

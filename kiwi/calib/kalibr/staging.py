"""Stage a recording into the folder ``kalibr_bagcreater`` eats.

Two writers, one format contract (:mod:`kiwi.calib.kalibr.layout`):

* :func:`stage_frames` - a demuxed per-lens mp4 -> ``frames/camN/*.png``.
  The host performs Kalibr's own grayscale conversion
  (``cvtColor(COLOR_BGR2GRAY)``, the exact call its bgr8 ingest path makes)
  and writes 1-channel PNGs, so the array the container's solver sees is
  bit-identical to what Kalibr's own ingest makes of a bgr8 bag. The
  pre-filters decide which PNGs exist.
* :func:`read_imu_rows_ns` - ``.insv`` telemetry -> ``(ts_ns, 6 values)``
  rows for :func:`kiwi.calib.kalibr.layout.write_imu_csv`. The gyro
  deg/s -> rad/s conversion happens HERE and nowhere else (bagcreater and
  Kalibr both pass values through untouched), and the ns stamps follow the
  fixed formula ``int(round(timestamp_ms * 1e6))`` (see
  :mod:`kiwi.calib.kalibr.layout`).

Timestamps: images stamp at ``frame_idx * period_ns + offset_ns``; the
caller derives ``offset_ns`` from the first IMU row
(:func:`kiwi.calib.kalibr.layout.imu_clock_offset_ns`) so pre-roll lands at
non-negative time, and applies the same shift to both streams.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np

from kiwi.calib.kalibr import layout
from kiwi.core.video import VideoInfo, iter_frames

__all__ = [
    "StagedFrames",
    "read_imu_rows_ns",
    "stage_frames",
    "write_gray_frame",
]


def write_gray_frame(path: Path, bgr: np.ndarray) -> None:
    """One staged frame: Kalibr's exact grayscale, 1-channel PNG, write-checked.

    ``cv2.imwrite`` returning False (disk full, bad path) raises ``OSError``.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # == ImageDatasetReader.py:178
    ok = cv2.imwrite(str(path), gray, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise OSError(f"could not write {path}")


@dataclasses.dataclass(frozen=True)
class StagedFrames:
    """What one lens's staging pass produced (feeds the run manifest)."""

    n_read: int
    n_written: int
    n_sparse_dropped: int  # by the min-corners extract filter
    image_size: tuple[int, int]  # (W, H)
    fps: float
    period_ns: int
    first_ts_ns: int | None
    last_ts_ns: int | None


def stage_frames(
    mp4: Path,
    dst_dir: Path,
    *,
    min_dt_ms: float = 0.0,
    frame_stride: int = 1,
    offset_ns: int = 0,
    extract=None,
) -> StagedFrames:
    """Write a lens's frames into ``dst_dir`` as stamped gray PNGs.

    Three filters select frames: ``frame_stride`` keeps every Nth true frame
    index (the imucam path's knob - scale it with fps to ~6 Hz),
    ``min_dt_ms`` drops frames closer than that to the last kept one (the
    intrinsics path's time-budget knob), and ``extract`` (from
    :func:`kiwi.calib.frames.build_extractor`) drops frames the AprilGrid
    detector finds too sparse.
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    info = VideoInfo.read(mp4)
    period = layout.period_ns(info.fps)
    min_dt_ns = int(min_dt_ms * 1_000_000)
    last_kept_ns: int | None = None
    first_ts: int | None = None
    n_read = written = sparse = 0
    size: tuple[int, int] | None = None
    for frame_idx, bgr in iter_frames(mp4):
        n_read = frame_idx + 1
        if frame_idx % frame_stride != 0:
            continue
        ts_ns = frame_idx * period + offset_ns
        if last_kept_ns is not None and ts_ns - last_kept_ns < min_dt_ns:
            continue
        if extract is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if extract(rgb) is None:
                sparse += 1
                continue
        if size is None:
            h, w = bgr.shape[:2]
            size = (w, h)
        write_gray_frame(dst_dir / (layout.stamp(ts_ns) + layout.IMAGE_SUFFIX), bgr)
        if first_ts is None:
            first_ts = ts_ns
        last_kept_ns = ts_ns
        written += 1
    return StagedFrames(
        n_read=n_read,
        n_written=written,
        n_sparse_dropped=sparse,
        image_size=size if size is not None else (info.width, info.height),
        fps=info.fps,
        period_ns=period,
        first_ts_ns=first_ts,
        last_ts_ns=last_kept_ns,
    )


def read_imu_rows_ns(insv: Path) -> list[tuple[int, list[float]]]:
    """``.insv`` telemetry -> unshifted ``(ts_ns, [wx..az])`` rows.

    The one place gyro converts deg/s -> rad/s (accel is already m/s^2);
    stamps are ``int(round(timestamp_ms * 1e6))``. The caller shifts by
    :func:`kiwi.calib.kalibr.layout.imu_clock_offset_ns` before writing.
    """
    import telemetry_parser  # deferred: keep this module importable without it

    rad = float(np.pi) / 180.0
    rows: list[tuple[int, list[float]]] = []
    for s in telemetry_parser.Parser(str(insv)).normalized_imu():
        ts_ns = int(round(s["timestamp_ms"] * 1_000_000))
        g, a = s["gyro"], s["accl"]
        rows.append(
            (
                ts_ns,
                [
                    float(g[0]) * rad,
                    float(g[1]) * rad,
                    float(g[2]) * rad,
                    float(a[0]),
                    float(a[1]),
                    float(a[2]),
                ],
            )
        )
    return rows

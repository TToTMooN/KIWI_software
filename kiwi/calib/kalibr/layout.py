"""The staged-input format ``kalibr_bagcreater`` consumes - every byte pinned.

KIWI never writes a ROS bag. It stages a plain folder into the Kalibr work
dir::

    <work_dir>/frames/cam0/<19-digit ts_ns>.png   (camN/ for more cameras)
    <work_dir>/frames/imu0.csv

and the container's own ``kalibr_bagcreater`` (installed in the ``kalibr``
image) assembles the ROS1 bag from it. Every ROS-shaped dependency stays
behind the Docker boundary.

This module owns the ``kalibr_bagcreater`` format contract, each rule
pinned to the vendored script's line
(``third_party/kalibr/aslam_offline_calibration/kalibr/python/kalibr_bagcreater``):

* the filename stem is integer **nanoseconds**, sliced as
  ``int(stem[0:-9])`` s / ``int(stem[-9:])`` ns (``:72``, ``:87`` for the
  CSV) - a stem under 10 chars makes ``int("")`` raise, and staged clocks
  start near 0, so stamps are zero-padded to :data:`STAMP_DIGITS`;
* files are sorted as **strings** (``:43``) - the fixed width makes
  lexicographic order equal numeric order;
* the extension match is case-sensitive and ``.jpeg`` is absent (``:39``):
  lowercase ``.png`` only (lossless);
* images are read ``IMREAD_GRAYSCALE`` and bagged ``mono8`` (``:69``,
  ``:80``). Kalibr's own ``bgr8`` ingest path is
  ``cvtColor(COLOR_BGR2GRAY)`` (``kalibr_common/ImageDatasetReader.py:178``),
  so the host performs that exact conversion and writes **1-channel** PNGs;
* the IMU CSV **must** carry a header row (``:120`` consumes row 1
  unconditionally - without one the first sample is silently dropped), with
  column order ``timestamp, omega_xyz, alpha_xyz`` (``:122``) and gyro
  already in **rad/s** (bagcreater passes values through untouched);
* the CSV filename stem *is* the topic (``:117``) and image topics derive
  from the folder name (``:112``), so :func:`cam_topic` / :data:`IMU_TOPIC`
  are the single source the generated camchain/imu YAMLs must agree with;
* ``--folder`` is ``os.walk``'d recursively and any ``imu*.csv`` anywhere
  under it becomes a phantom topic (``:51``, ``:61``): the staged tree
  lives in its own ``frames/`` subdir, never the work dir itself.

Timestamp *values* follow fixed formulas (images at
``frame_idx * period_ns + offset``, IMU at ``round(ms * 1e6) + offset``,
``offset = -min(0, first_imu_ns)``). A positive offset skews the
``--bag-from-to`` window (:func:`kiwi.calib.kalibr.docker.imucam_run`).

Pure stdlib - no cv2, no numpy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "FRAMES_DIR",
    "IMAGE_SUFFIX",
    "IMU_CSV_HEADER",
    "IMU_CSV_NAME",
    "IMU_TOPIC",
    "STAMP_DIGITS",
    "cam_dir",
    "cam_topic",
    "imu_clock_offset_ns",
    "parse_stamp",
    "period_ns",
    "stamp",
    "write_imu_csv",
]

# 19 digits is the EuRoC/ASL ns-since-epoch width; >= 10 satisfies the
# bagcreater stem slice, and the fixed width makes string sort numeric.
STAMP_DIGITS = 19
FRAMES_DIR = "frames"  # the only thing --folder ever points at
IMAGE_SUFFIX = ".png"  # lowercase; bagcreater:39 is case-sensitive
IMU_CSV_NAME = "imu0.csv"  # stem == topic (bagcreater:117)
IMU_TOPIC = "/imu0"
IMU_CSV_HEADER = "timestamp,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z"


def stamp(ts_ns: int) -> str:
    """A timestamp filename stem / CSV field: 19-digit zero-padded ns."""
    ts_ns = int(ts_ns)
    if not 0 <= ts_ns < 10**STAMP_DIGITS:
        raise ValueError(f"ts_ns {ts_ns} outside [0, 10^{STAMP_DIGITS})")
    return f"{ts_ns:0{STAMP_DIGITS}d}"


def parse_stamp(stem: str) -> int:
    """Recover ns from a stem *exactly the way bagcreater does* (:72/:87)."""
    return int(stem[0:-9]) * 10**9 + int(stem[-9:])


def cam_dir(index: int) -> str:
    """Folder name for camera ``index`` under ``frames/`` -> topic ``/camN/image_raw``."""
    return f"cam{index}"


def cam_topic(index: int) -> str:
    """The rostopic bagcreater derives from :func:`cam_dir` (bagcreater:112)."""
    return f"/{cam_dir(index)}/image_raw"


def period_ns(fps: float) -> int:
    """Frame period in integer ns: ``int(round(1e9/fps))``."""
    return int(round(1e9 / fps))


def imu_clock_offset_ns(first_imu_ns: int) -> int:
    """The shift putting IMU pre-roll at non-negative time.

    Telemetry timestamps can start below zero (pre-roll before video t=0);
    both streams shift together by ``-min(0, first_imu_ns)`` so every staged
    stamp satisfies :func:`stamp`'s non-negative range.
    """
    return -min(0, int(first_imu_ns))


def write_imu_csv(
    path: Path, rows: Iterable[tuple[int, Sequence[float]]]
) -> tuple[Path, int]:
    """Write ``frames/imu0.csv``: ``(ts_ns, [wx, wy, wz, ax, ay, az])`` rows.

    Gyro must already be rad/s and accel m/s^2 - bagcreater passes the
    values through untouched, so unit conversion upstream of this writer is
    the only place it can happen. Floats are written as ``repr`` (shortest
    round-trip form), making the bagged IMU stream bit-identical to the
    in-memory float64s. Rows must arrive sorted ascending; the writer emits
    exactly one header, ``\\n`` endings, and one trailing newline.

    Returns ``(path, n_rows)`` for the expected-message-count assertion.
    """
    path = Path(path)
    n = 0
    last_ts: int | None = None
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(IMU_CSV_HEADER + "\n")
        for ts_ns, values in rows:
            if len(values) != 6:
                raise ValueError(
                    f"row {n}: expected 6 values (gyro+accel), got {len(values)}"
                )
            if last_ts is not None and ts_ns < last_ts:
                raise ValueError(
                    f"row {n}: timestamps not ascending ({ts_ns} < {last_ts})"
                )
            last_ts = ts_ns
            f.write(
                stamp(ts_ns) + "," + ",".join(repr(float(v)) for v in values) + "\n"
            )
            n += 1
    return path, n

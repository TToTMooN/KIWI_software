"""The single ``docker run`` per calibration: bagcreater, assert, solve.

One container invocation chains three steps under ``set -e``:

1. ``kalibr_bagcreater --folder /data/frames --output-bag /data/<bag>`` -
   the ROS1 bag is built inside the container, where ROS1 lives;
2. an exact message-count assertion - the host knows exactly how many
   messages it staged, so the container checks that the bag carries all of
   them before any solving starts;
3. the Kalibr solver (``kalibr_calibrate_cameras`` /
   ``kalibr_calibrate_imu_camera``).

The container mechanics: host uid/gid mapping so outputs are user-owned,
``HOME=/tmp`` (``/root`` is unwritable for a non-root uid), one bind mount
at ``/data``, ``--entrypoint /bin/bash``. ``set -u`` is absent - ROS1's
``setup.bash`` references unbound variables.

Building is pure argv/string assembly; at execution the error contract is:
docker missing -> ``RuntimeError`` naming the fix; a non-zero exit -> the
integer rc. The CAMERAS path continues into the log-salvage path on a
diverged solve (:mod:`kiwi.calib.intrinsics`) while the imu-cam path treats
it as fatal - the caller owns that decision, not this module.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path

from kiwi.calib.kalibr import layout
from kiwi.calib.kalibr.yamls import APRILGRID_NAME

__all__ = [
    "FOCAL_INIT_NAME",
    "IMAGE_ENV",
    "KalibrRun",
    "cameras_run",
    "default_image",
    "execute",
    "imucam_run",
    "write_focal_init",
]

IMAGE_ENV = "KIWI_KALIBR_IMAGE"
_DEFAULT_IMAGE = "kalibr:latest"

CAMERAS_BAG = "calib.bag"  # bagtag "calib" -> calib-camchain.yaml etc.
IMUCAM_BAG = "imucam_calib.bag"  # -> imucam_calib-camchain-imucam.yaml etc.
LOG_NAME = "kalibr.log"
FOCAL_INIT_NAME = "focal-initialization.txt"  # stdin for Kalibr's manual focal init


def default_image() -> str:
    """The container image: ``$KIWI_KALIBR_IMAGE``, else ``kalibr:latest``.

    Built once by hand from the submodule's ``Dockerfile_ros1_20_04`` (see
    ``docs/hardware.md``); the only pre-flight check possible on the host
    is that docker itself exists.
    """
    return os.environ.get(IMAGE_ENV, _DEFAULT_IMAGE)


@dataclasses.dataclass(frozen=True)
class KalibrRun:
    """One prepared container invocation: inspectable before it runs."""

    work_dir: Path
    argv: tuple[str, ...]
    inner: str  # the bash -c payload, for inspection/tests
    log_path: Path


def _count_assert(bag_name: str, expected_messages: int) -> str:
    """The in-container bag integrity check (exits 3 on mismatch)."""
    return (
        'python3 -c "'
        "import rosbag, sys\n"
        f"n = rosbag.Bag('/data/{bag_name}').get_message_count()\n"
        f"exp = {expected_messages}\n"
        "print('[kiwi] bag messages: %d (expected %d)' % (n, exp))\n"
        'sys.exit(0 if n == exp else 3)"'
    )


def _docker_argv(
    work_dir: Path, inner: str, image: str, shm_size: str | None = None
) -> tuple[str, ...]:
    argv = [
        "docker",
        "run",
        "--rm",
        *(("--shm-size", shm_size) if shm_size else ()),
        # Host uid/gid so kalibr's outputs (camchain yaml, report pdf) end up
        # owned by the invoking user, not root.
        "-u",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-v",
        f"{Path(work_dir).resolve()}:/data",
        "-w",
        "/data",
        "--entrypoint",
        "/bin/bash",
        image,
        "-c",
        inner,
    ]
    return tuple(argv)


def write_focal_init(work_dir: Path, focal_px: float, lines: int = 20) -> Path:
    """Write the manual focal-init answers ``cameras_run`` redirects to stdin.

    Kalibr's ``KALIBR_MANUAL_FOCAL_LENGTH_INIT`` option
    (``PinholeProjection.hpp``) reads ONE value from stdin each time a
    pinhole camera's automatic focal init fails; repeating the value covers
    every camera and retry in the run without counting them.
    """
    out = Path(work_dir) / FOCAL_INIT_NAME
    out.write_text((str(float(focal_px)) + "\n") * lines)
    return out


def cameras_run(
    work_dir: Path,
    *,
    topics: list[str],
    models: list[str],
    expected_messages: int,
    bag_name: str = CAMERAS_BAG,
    target_name: str = APRILGRID_NAME,
    manual_focal_init: bool = False,
) -> KalibrRun:
    """Prepare the intrinsics invocation: bagcreater -> assert -> calibrate_cameras.

    ``topics`` and ``models`` are parallel, one per camera; a mismatch is a
    caller bug (``ValueError``). ``--folder`` points at ``/data/frames`` and
    never ``/data`` - bagcreater walks its argument recursively and any
    ``imu*.csv`` under it becomes a phantom topic.

    ``manual_focal_init`` arms Kalibr's ``KALIBR_MANUAL_FOCAL_LENGTH_INIT``
    option and feeds the solver :data:`FOCAL_INIT_NAME` (the caller writes it
    via :func:`write_focal_init`). Stdin is only ever read when a pinhole
    camera's automatic init fails; DS models never read it at all. With it
    off, the inner script carries neither the export nor the stdin redirect.
    """
    if len(topics) != len(models):
        raise ValueError(
            f"topics ({len(topics)}) and models ({len(models)}) must pair up"
        )
    inner = (
        "set -e\n"
        "source /catkin_ws/devel/setup.bash\n"
        + ("export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1\n" if manual_focal_init else "")
        + f"rosrun kalibr kalibr_bagcreater --folder /data/{layout.FRAMES_DIR} "
        f"--output-bag /data/{bag_name}\n"
        f"{_count_assert(bag_name, expected_messages)}\n"
        "rosrun kalibr kalibr_calibrate_cameras "
        f"--bag /data/{bag_name} "
        f"--topics {' '.join(topics)} "
        f"--models {' '.join(models)} "
        f"--target /data/{target_name} "
        "--dont-show-report"
        + (f" < /data/{FOCAL_INIT_NAME}" if manual_focal_init else "")
        + "\n"
    )
    work_dir = Path(work_dir)
    return KalibrRun(
        work_dir=work_dir,
        argv=_docker_argv(work_dir, inner, default_image()),
        inner=inner,
        log_path=work_dir / LOG_NAME,
    )


def imucam_run(
    work_dir: Path,
    *,
    expected_messages: int,
    bag_name: str = IMUCAM_BAG,
    cam_yaml: str = "camchain.yaml",
    imu_yaml: str = "imu.yaml",
    target_name: str = APRILGRID_NAME,
    bag_from_to: tuple[float, float] | None = None,
    timeoffset_padding: float = 0.1,
    shm_size: str = "8g",
) -> KalibrRun:
    """Prepare the imu-cam invocation: bagcreater -> assert -> calibrate_imu_camera.

    ``--shm-size``: Kalibr's multiprocessing corner extractor uses
    ``/dev/shm``. ``bag_from_to`` is the window the board is actually
    visible in (seconds) - Kalibr resolves it per topic against that topic's
    own min stamp, so a positive clock offset skews the two windows. The
    skew is not corrected.
    """
    from_to = f" --bag-from-to {bag_from_to[0]} {bag_from_to[1]}" if bag_from_to else ""
    inner = (
        "set -e\n"
        "source /catkin_ws/devel/setup.bash\n"
        f"rosrun kalibr kalibr_bagcreater --folder /data/{layout.FRAMES_DIR} "
        f"--output-bag /data/{bag_name}\n"
        f"{_count_assert(bag_name, expected_messages)}\n"
        "rosrun kalibr kalibr_calibrate_imu_camera "
        f"--bag /data/{bag_name} "
        f"--cam /data/{cam_yaml} "
        f"--imu /data/{imu_yaml} "
        f"--target /data/{target_name} "
        f"--timeoffset-padding {timeoffset_padding}"
        f"{from_to} "
        "--dont-show-report\n"
    )
    work_dir = Path(work_dir)
    return KalibrRun(
        work_dir=work_dir,
        argv=_docker_argv(work_dir, inner, default_image(), shm_size=shm_size),
        inner=inner,
        log_path=work_dir / LOG_NAME,
    )


def execute(run: KalibrRun) -> int:
    """Run the container, streaming output to the terminal AND the log.

    The log matters beyond diagnostics: the intrinsics salvage path parses
    per-batch results out of it when the global bundle adjustment diverges.
    Docker being absent raises ``RuntimeError`` (the component error
    contract); a non-zero container exit returns as the rc for the caller
    to judge - salvageable for cameras, fatal for imu-cam.
    """
    if shutil.which("docker") is None:
        raise RuntimeError(
            "docker not found on PATH - the Kalibr solve runs in the "
            f"{default_image()} container (see docs/hardware.md to build it)"
        )
    with run.log_path.open("w") as logf:
        proc = subprocess.Popen(
            list(run.argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        return proc.wait()

"""One-time calibration workflows (component: imports ``kiwi.core`` only).

Calibration is per-device, not per-episode: every workflow here runs once for
a camera unit (or once after a remount) and publishes a dated artifact into
``config/cameras/<unit>/``. Nothing in this package is a pipeline stage - the
front end is the ``kiwi calib`` verb tree, and ``kiwi run`` never calls in.

Owns:

* **fisheye intrinsics** via Kalibr from a camera original (``.insv``, or a
  single-lens ``.mp4``): DoubleSphere fit -> the registry ``intrinsics.npz``,
  plus the KB4/equidistant export for scene mapping and ground geometry
* **camera<->IMU extrinsics** (``T_cam_imu`` per lens) via Kalibr, plus the
  front/back composition cross-check (the two lenses' solves composed against
  the camera's physical back-to-back geometry)
* **IMU noise densities** (:mod:`kiwi.calib.allan` / :mod:`kiwi.calib.imu_noise`):
  overlapping Allan deviation with slope-band N/K fits -> the per-unit
  ``imu_noise_v1`` artifact that state estimation reads
* **the rolling-QR clock rig** and the per-recording-mode A/V delay
  measurement (the ``av_sync_v1`` artifact :mod:`kiwi.sync` refuses to run
  mixed-mode pairs without)

Contract notes:

* **Artifact readers live in ``kiwi.core``; writers live here.** ``state``
  reads the camchains calib writes via
  :mod:`kiwi.core.camera.extrinsics`; neither component imports the other.
  Schema strings are never spelled locally - writer and reader both take
  them from :data:`kiwi.core.camera.registry.ARTIFACT_SCHEMAS`.
* The registry convention is **dated file + explicit ``camera.yaml`` pointer
  key**, resolved only through
  :meth:`kiwi.core.camera.registry.CameraUnit.artifact`.
  Publication = drop the dated file + move the pointer, one reviewable
  commit; this package writes dated files and never moves pointers itself.
  Recordings match cameras by ``serial:``, never by directory name.
* There is **no hand-eye solver** here. The deployed hand-eye is not a
  camera-registry artifact - it is the ``camera_lens0`` joint
  (``T_flange_cam``) in the descriptions tree.
* **Kalibr runs in Docker, invoked as a process.** The ROS1 bag
  Kalibr requires is built *inside* the container
  by its own ``kalibr_bagcreater`` from a staged frames/CSV folder, so no
  ROS-shaped dependency (rosbags included) exists on the host.

Carve-outs:

* Per-tool homography calibration (a gripper unit's ``build.fixed_pose``)
  belongs to ``kiwi.eef`` (``kiwi eef calibrate``).
* There is no hat-tag rig calibration (tag <-> GO3 extrinsic).
* There is no live-capture calibration path (such as a live-USB
  dual-fisheye stream): KIWI calibrates from recorded camera originals
  (``.insv``, or a single-lens ``.mp4``).
"""

from kiwi.calib.allan import (
    SLOPE_TOL,
    fit_random_walk,
    fit_white_noise,
    overlapping_allan,
    stitch_segments,
)
from kiwi.calib.imu_log import (
    extract_npz,
    load_imu_npz,
    parse_insv_imu,
    save_imu_npz,
)
from kiwi.calib.imu_noise import (
    AXES,
    IMU_NOISE_SCHEMA,
    ImuNoise,
    ImuNoiseResult,
    assemble_segments,
    noise_from_series,
    plot_allan_curves,
    read_imu_noise,
    write_imu_noise,
)

__all__ = [
    "AXES",
    "IMU_NOISE_SCHEMA",
    "SLOPE_TOL",
    "ImuNoise",
    "ImuNoiseResult",
    "assemble_segments",
    "extract_npz",
    "fit_random_walk",
    "fit_white_noise",
    "load_imu_npz",
    "noise_from_series",
    "overlapping_allan",
    "parse_insv_imu",
    "plot_allan_curves",
    "read_imu_noise",
    "save_imu_npz",
    "stitch_segments",
    "write_imu_noise",
]

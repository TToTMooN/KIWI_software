"""The YAMLs Kalibr consumes: aprilgrid target, camchain, imu noise.

One parser owns the board translation; the camchain renderer is the only
place the ``rostopic:`` field is spelled, and it takes the topic from
:mod:`kiwi.calib.kalibr.layout`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from kiwi.calib.kalibr import layout
from kiwi.core.markers import AprilGridConfig, load_apriltag_grid_configs

__all__ = [
    "APRILGRID_NAME",
    "KALIBR_IMU_PRIORS",
    "resolve_grid",
    "write_aprilgrid",
    "write_camchain",
    "write_imu_yaml",
]

APRILGRID_NAME = "aprilgrid.yaml"  # conventional filename inside the work dir


def resolve_grid(
    marker_config: Path, name: str | None = None
) -> tuple[str, AprilGridConfig]:
    """The named AprilGrid from a project marker YAML, or its sole entry.

    Raises ``ValueError`` when the file has no ``apriltag_grid:`` section,
    the requested name is absent, or no name was given and the choice is
    ambiguous.
    """
    grids = load_apriltag_grid_configs(marker_config)
    if not grids:
        raise ValueError(f"{marker_config} has no apriltag_grid section")
    if name is not None:
        if name not in grids:
            raise ValueError(
                f"{marker_config} has no apriltag_grid entry {name!r} "
                f"(has: {', '.join(sorted(grids))})"
            )
        return name, grids[name]
    if len(grids) != 1:
        raise ValueError(
            f"{marker_config} has {len(grids)} apriltag_grid entries; "
            "pass --target-name to pick one"
        )
    return next(iter(grids.items()))


def write_aprilgrid(grid: AprilGridConfig, dst: Path) -> Path:
    """Translate a project AprilGrid config into Kalibr's target format.

    Kalibr's ``aprilgrid`` uses camelCase fields; the project-specific keys
    (dictionary, start id, border bits) are dropped because Kalibr's
    detector is hard-coded for AprilTag 36h11 with its own 2-cell border
    convention (``blackTagBorder=2`` - the same convention ``border_bits``
    records for the host-side detector).
    """
    out = {
        "target_type": "aprilgrid",
        "tagCols": int(grid.tag_cols),
        "tagRows": int(grid.tag_rows),
        "tagSize": float(grid.tag_size),
        "tagSpacing": float(grid.tag_spacing),
    }
    dst = Path(dst)
    dst.write_text(yaml.safe_dump(out, sort_keys=False))
    return dst


def write_camchain(
    intrinsics_npz: Path,
    lens: str,  # "front" | "back"
    out: Path,
    *,
    unit: str | None = None,
    source_video: str | None = None,
) -> Path:
    """Render the single-camera camchain Kalibr's imu-cam solve consumes.

    Dispatches on the registry ``intrinsics.npz`` keys - ``ds_<lens>`` (Double
    Sphere, the X4/X5 currency) or ``K_<lens>``/``D_<lens>`` (equidistant
    fisheye, the single-lens GO3 currency; see
    :func:`kiwi.calib.intrinsics.save_intrinsics_npz` for both schemas) - and
    pins the ``rostopic`` to what bagcreater will derive from the staged
    folder. The staged topic is ``/cam0/image_raw`` for BOTH lenses (topics
    come from folder names), so a provenance comment carries the lens
    identity the topic cannot - Kalibr ``yaml.safe_load``s the file, so
    comments are safe.

    Raises ``ValueError`` when the npz has no entry for this lens or the
    entry is malformed - run the intrinsics calibration first.
    """
    d = np.load(str(intrinsics_npz))
    ds_key, k_key, d_key = f"ds_{lens}", f"K_{lens}", f"D_{lens}"
    resolution = [int(v) for v in d["image_size"]]
    topic = layout.cam_topic(0)
    if ds_key in d.files:
        ds = [float(x) for x in d[ds_key]]
        if len(ds) != 6:
            raise ValueError(
                f"{intrinsics_npz}: {ds_key} has length {len(ds)}, expected 6"
            )
        cam0 = {
            "camera_model": "ds",
            "intrinsics": ds,  # [xi, alpha, fx, fy, cx, cy]
            "distortion_model": "none",
            "distortion_coeffs": [],
            "resolution": resolution,
            "rostopic": topic,
        }
    elif k_key in d.files and d_key in d.files:
        K = np.asarray(d[k_key], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(
                f"{intrinsics_npz}: {k_key} has shape {K.shape}, expected (3, 3)"
            )
        dist = [float(x) for x in np.asarray(d[d_key]).ravel()]
        if len(dist) != 4:
            raise ValueError(
                f"{intrinsics_npz}: {d_key} has {len(dist)} coeffs, expected 4 "
                "(kalibr pinhole-equi / cv2.fisheye)"
            )
        cam0 = {
            "camera_model": "pinhole",
            "intrinsics": [
                float(K[0, 0]),
                float(K[1, 1]),
                float(K[0, 2]),
                float(K[1, 2]),
            ],
            "distortion_model": "equidistant",
            "distortion_coeffs": dist,
            "resolution": resolution,
            "rostopic": topic,
        }
    else:
        raise ValueError(
            f"{intrinsics_npz} has no {ds_key!r} and no {k_key!r}/{d_key!r} "
            f"entry (has: {', '.join(sorted(d.files))}); run the intrinsics "
            "calibration for this lens first"
        )
    header = (
        f"# kiwi calib: unit={unit or 'unknown'}  lens={lens}  cam_id=cam0\n"
        + (f"# source: {source_video}\n" if source_video else "")
        + f"# rostopic is {topic} for BOTH lenses: kalibr_bagcreater derives the\n"
        "# topic from the frames/ folder name. The lens lives here and in\n"
        "# kiwi_calib.json, never in the topic.\n"
    )
    out = Path(out)
    out.write_text(header + yaml.safe_dump({"cam0": cam0}, sort_keys=False))
    return out


# The IMU noise priors every published extrinsic was solved with. These
# are conservative solve priors, NOT the unit's measured Allan densities.
# Solving with the measured imu_noise_v1 artifact instead is an explicit
# operator choice, never a silent default.
KALIBR_IMU_PRIORS: dict[str, float] = {
    "accelerometer_noise_density": 2.0e-3,
    "accelerometer_random_walk": 3.0e-3,
    "gyroscope_noise_density": 1.7e-4,
    "gyroscope_random_walk": 1.9e-5,
}


def write_imu_yaml(
    out: Path,
    *,
    noise: dict[str, float] | None = None,
    update_rate: float = 1000.0,
) -> Path:
    """Render the ``imu.yaml`` Kalibr's imu-cam solve consumes.

    ``noise`` maps the four Kalibr keys (see :data:`KALIBR_IMU_PRIORS`, the
    default); the rostopic pins to :data:`kiwi.calib.kalibr.layout.IMU_TOPIC`,
    the staged CSV's derived topic.
    """
    n = dict(KALIBR_IMU_PRIORS if noise is None else noise)
    missing = set(KALIBR_IMU_PRIORS) - set(n)
    if missing:
        raise ValueError(f"imu noise mapping missing keys: {sorted(missing)}")

    def num(v: float) -> str:
        # YAML 1.1 (PyYAML - and Kalibr's own ROS1-side parser) resolves a
        # dotless exponent form like `5e-05` as a STRING; render through the
        # dumper so every value round-trips as a float (`5.0e-05`).
        return yaml.safe_dump(float(v)).strip().removesuffix("\n...").strip()

    body = (
        "#Accelerometers\n"
        f"accelerometer_noise_density: {num(n['accelerometer_noise_density'])}\n"
        f"accelerometer_random_walk:   {num(n['accelerometer_random_walk'])}\n"
        "\n#Gyroscopes\n"
        f"gyroscope_noise_density:     {num(n['gyroscope_noise_density'])}\n"
        f"gyroscope_random_walk:       {num(n['gyroscope_random_walk'])}\n"
        f"\nrostopic:                    {layout.IMU_TOPIC}\n"
        f"update_rate:                 {num(update_rate)}\n"
    )
    out = Path(out)
    out.write_text(body)
    return out

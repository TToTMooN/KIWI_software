"""Scene mapping: three monocular maps, their metric frames and the shared scene frame.

Once per scene, from the designated mapping episode:

1. export every frame of the four wrist streams at the map image size, the four
   streams at once;
2. write the masks of the two front streams, then build the ``shared_rear`` map and
   the ``front_left`` and ``front_right`` maps from the 30 fps subset of their
   streams, the three builds at once;
3. localize all four full streams against the finished atlases, one after another;
4. fit each map's metric frame (scale, gravity) from those localized poses and the
   raw IMU, take the shared rear frame as the scene frame, and place each front map
   into it with the arm's fixed front-rear lever;
5. write the mapping episode's observations in scene coordinates.

``kiwi scene localize`` routes by (arm, lens): the requested arm's own
stream and calibration, against that lens's map role (front -> that arm's
front map, back -> the shared rear map). Published observations are
transformed into the scene frame first - every consumer reads poses in
the one canonical frame.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kiwi.core import layout
from kiwi.core.atomic import atomic_write_text
from kiwi.core.digest import digest_bytes
from kiwi.core.generation import file_ref, producer_info, write_record
from kiwi.episode.scene import SceneConfig, effective_min_usable_s
from kiwi.state.fusion import run_ids_from_tracked
from kiwi.track.frames import (
    MAP_FPS,
    MAP_IMAGE_SIZE,
    N_FEATURES,
    export_stream_frames,
    native_fps,
    read_timestamps,
    rescale_kb4,
    subset_times,
    write_engine_settings,
    write_timestamps,
)
from kiwi.track.masks import write_front_masks
from kiwi.track.metric_align import MetricAlignment, estimate_metric_alignment
from kiwi.track.orbslam.engine import (
    BUILD_HELP,
    ENGINE_THREADS,
    engine_binary,
    run_engine,
    vocabulary,
)
from kiwi.track.orbslam.observations import (
    Observations,
    read_observations,
    write_observations,
)
from kiwi.track.scene_frame import (
    SCENE_FRAME_NAME,
    MapToScene,
    read_scene_frame,
    solve_front_to_scene,
    write_scene_frame,
)
from kiwi.track.sequence import frame_times_ns

__all__ = [
    "MAP_ROLES",
    "BuildOutcome",
    "build_scene_maps",
    "current_maps",
    "is_localized",
    "load_imu",
    "load_unit",
    "localize_episode",
]

MAP_ROLES = ("front_left", "front_right", "shared_rear")
ARMS = ("left", "right")
# The lens names of the frame directories, the row files under scene_map/ and
# the calibration; the command line and the localization record say "back"
# for the rear lens.
LENSES = ("front", "rear")

_POINTER = "current.json"


@dataclass(frozen=True)
class RoleInputs:
    """Everything one stream's engine runs need, resolved from the episode."""

    role: str  # the map role this stream builds against
    episode_id: str
    arm: str  # whose stream and calibration this actually is
    video: Path  # the demuxed lens stream
    recording: Path  # the raw recording (IMU source)
    kb4_artifact: Path
    cam_imu_artifact: Path  # this lens's cam-IMU chain (timeshift + lever)


@dataclass(frozen=True)
class BuildOutcome:
    accepted: bool
    episode_id: str | None
    maps: dict[str, str]  # role -> maps/<id> (relative to scene_map/)
    attempts: int
    reason: str


@dataclass(frozen=True)
class Unit:
    """One camera unit's calibration, each field keyed by lens (front | rear)."""

    kb4: dict[str, np.ndarray]  # lens -> [fx fy cx cy k1 k2 k3 k4] at `resolution`
    resolution: dict[str, tuple[int, int]]  # lens -> (width, height) of the KB4 fit
    T_cam_imu: dict[str, np.ndarray]  # lens -> 4x4, p_cam = T @ p_imu
    timeshift_cam_imu: dict[str, float]  # lens -> dt, t_imu = t_cam + dt

    def T_rear_front(self) -> np.ndarray:
        """Fixed front-to-rear transform ``T_rear_front = T_rear_imu @ inv(T_front_imu)``."""
        return self.T_cam_imu["rear"] @ np.linalg.inv(self.T_cam_imu["front"])


def load_unit(arm) -> Unit:
    """Read the lens models and camera/IMU chains of one arm's camera unit."""
    from kiwi.core.camera.extrinsics import read_cam_imu
    from kiwi.core.camera.kb4 import read_kb4
    from kiwi.core.camera.registry import CameraUnit

    registry = CameraUnit.load(arm.camera_dir)
    keys = {
        "front": ("intrinsics_kb4", "extrinsic_imu"),
        "rear": ("intrinsics_kb4_back", "extrinsic_imu_back"),
    }
    paths = {
        key: registry.artifact(key)
        for key in (
            "intrinsics_kb4",
            "intrinsics_kb4_back",
            "extrinsic_imu",
            "extrinsic_imu_back",
        )
    }
    missing = [key for key, path in paths.items() if path is None]
    if missing:
        raise ValueError(f"camera {arm.camera!r} publishes no {'/'.join(missing)}")
    kb4, resolution, T_cam_imu, timeshift_cam_imu = {}, {}, {}, {}
    for lens, (kb4_key, chain_key) in keys.items():
        # The registry and the KB4 files name the rear lens "back".
        kb4[lens], doc = read_kb4(
            paths[kb4_key], expected_lens="back" if lens == "rear" else "front"
        )
        resolution[lens] = tuple(doc["resolution"])
        chain = read_cam_imu(paths[chain_key])
        T_cam_imu[lens] = chain.T_cam_imu
        timeshift_cam_imu[lens] = chain.timeshift_s or 0.0
    return Unit(kb4, resolution, T_cam_imu, timeshift_cam_imu)


def _stream_inputs(
    scene: SceneConfig, episode_id: str, arm_label: str, lens: str
) -> RoleInputs:
    """Resolve ONE (arm, lens) stream: that arm's video, recording and
    calibration - never another arm's stream relabeled. The map role is
    derived: a front lens maps against that arm's front map, any arm's
    back lens against the one shared rear map."""
    from kiwi.core import layout
    from kiwi.core.camera.registry import CameraUnit
    from kiwi.episode.config_snapshot import snapshot_config
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig

    if lens not in ("front", "back"):
        raise ValueError(f"lens must be 'front' or 'back', got {lens!r}")
    edir = scene.episode_dir(episode_id)
    snapshot_config(edir)
    cfg = EpisodeConfig.load(edir / MANIFEST_NAME)
    arm = cfg.arms.get(arm_label)
    if arm is None:
        raise ValueError(
            f"scene build needs arm {arm_label!r} in episode "
            f"{episode_id!r}, which declares {sorted(cfg.arms)}"
        )
    if lens == "back":
        role = "shared_rear"
        lens_file, kb4_key, extr_key = (
            layout.LENS1,
            "intrinsics_kb4_back",
            "extrinsic_imu_back",
        )
    else:
        role = f"front_{arm_label}"
        lens_file, kb4_key, extr_key = (
            layout.LENS0,
            "intrinsics_kb4",
            "extrinsic_imu",
        )
    unit = CameraUnit.load(arm.camera_dir)
    kb4 = unit.artifact(kb4_key)
    extr = unit.artifact(extr_key)
    if kb4 is None or extr is None:
        raise ValueError(
            f"camera {arm.camera!r} publishes no {kb4_key}/{extr_key} - "
            f"scene mapping needs the calibrated lens model and cam-IMU "
            f"chain for every stream it runs"
        )
    video = layout.stream_dir(edir, arm_label) / lens_file
    if not video.is_file():
        raise FileNotFoundError(
            f"no {video.relative_to(edir)} in episode {episode_id!r} - "
            f"run the convert stage first"
        )
    return RoleInputs(
        role=role,
        episode_id=episode_id,
        arm=arm_label,
        video=video,
        recording=arm.recording,
        kb4_artifact=kb4,
        cam_imu_artifact=extr,
    )


def current_maps(scene: SceneConfig) -> dict | None:
    """The published map set, verified complete - or None."""
    pointer = scene.root / "scene_map" / _POINTER
    if not pointer.is_file():
        return None
    doc = json.loads(pointer.read_text(encoding="utf-8"))
    if doc.get("schema") != "scene_map_pointer":
        raise ValueError(f"{pointer}: schema {doc.get('schema')!r}")
    roles = doc.get("roles") or {}
    if set(roles) != set(MAP_ROLES):
        raise ValueError(
            f"{pointer}: incomplete role set {sorted(roles)} - a partial "
            f"pointer must never be treated as a published map"
        )
    for role, rel in roles.items():
        map_dir = scene.root / "scene_map" / rel
        for artifact in ("atlas.osa",):
            if not (map_dir / artifact).is_file():
                raise ValueError(
                    f"{pointer}: {role} map at {rel} is missing {artifact}"
                )
    return doc


def load_imu(
    recording: Path, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (once per recording) and load raw IMU on the RECORDING clock.

    The extractor shifts a pre-rolling IMU record forward so its own file
    starts at t=0 and reports that shift in its meta; container frame PTS
    live on the unshifted recording clock (frame 0 at ~0), so the shift is
    undone here - camera rows then meet the IMU via the frozen equation
    ``t_imu = t_cam + timeshift`` alone. The meta rides a sidecar next to
    the cached CSV; a CSV without its sidecar is treated as half-written
    and re-extracted.
    """
    from kiwi.core.camera.imu import extract_imu_to_csv

    raw_digest = digest_bytes(recording)
    stem = raw_digest.split(":")[1][:16]
    csv = cache_dir / f"{stem}-imu.csv"
    meta_path = cache_dir / f"{stem}-imu.json"
    if not (csv.is_file() and meta_path.is_file()):
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta = extract_imu_to_csv(recording, csv)
        write_record(
            meta_path,
            {
                "clock_offset_s": float(meta.clock_offset_s),
                "producer": producer_info("kiwi.core.camera.imu.extract_imu_to_csv"),
                "inputs": {
                    "raw_recording": file_ref(
                        recording, base=cache_dir, digest=raw_digest
                    )
                },
                "outputs": {"imu": file_ref(csv, base=cache_dir)},
                "clock": "csv time = recording IMU time + clock_offset_s",
                "units": {"t": "seconds", "gyro": "rad/s", "accel": "m/s^2"},
                "body": "recording_imu_axes",
            },
        )
    offset = float(json.loads(meta_path.read_text())["clock_offset_s"])
    rows = np.loadtxt(csv, skiprows=1)
    return rows[:, 0] - offset, rows[:, 1:4], rows[:, 4:7]


# ----------------------------------------------------------------------------
# Paths under scene_map/
# ----------------------------------------------------------------------------


def role_for(arm: str, lens: str) -> str:
    return "shared_rear" if lens == "rear" else f"front_{arm}"


def frames_dir(scene_map: Path, episode_id: str, arm: str, lens: str) -> Path:
    """Exported 960 px JPEG frames of one stream, shared by mapping and localization."""
    return scene_map / "frames" / f"{episode_id}-{arm}-{lens}"


def masks_dir(scene_map: Path, episode_id: str, arm: str) -> Path:
    """Keypoint exclude masks of the front mapping frames of one arm."""
    return scene_map / "masks" / f"{episode_id}-{arm}-front"


def map_frame_rows(scene_map: Path, episode_id: str, arm: str, lens: str) -> Path:
    """Map-frame localization rows of one full stream, as the engine wrote them."""
    return scene_map / "reloc" / f"{episode_id}-{arm}-{lens}.txt"


# ----------------------------------------------------------------------------
# Map building, localization and placement
# ----------------------------------------------------------------------------


def fit_metric_frame(
    obs: Observations, timeshift: float, T_cam_imu: np.ndarray, imu
) -> MetricAlignment:
    """Metric fit on the tracked rows, with their times moved to the IMU clock."""
    imu_t, gyro, acc = imu
    ok = obs.tracked
    return estimate_metric_alignment(
        obs.t_s[ok] + timeshift,
        obs.T[ok, :3, :3],
        obs.T[ok, :3, 3],
        T_cam_imu,
        imu_t,
        gyro,
        acc,
    )


def mapping_subset(
    frames: Path, unit: Unit, lens: str
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Frame directory, 30 fps subset of frame times and map-size calibration of a stream."""
    t_ns = read_timestamps(frames / "timestamps.txt")
    subset = subset_times(t_ns, native_fps(t_ns))
    kb4 = rescale_kb4(
        unit.kb4[lens], unit.resolution[lens], (MAP_IMAGE_SIZE, MAP_IMAGE_SIZE)
    )
    return frames, subset, kb4


def write_role_masks(masker, frames: Path, unit: Unit, mask_dir: Path) -> None:
    """Write the masks of the mapping frames of one front stream."""
    frames, subset, kb4 = mapping_subset(frames, unit, "front")
    write_front_masks(masker, frames, subset, mask_dir, kb4)


def build_role(
    frames: Path, unit: Unit, lens: str, map_dir: Path, mask_dir: Path | None
) -> None:
    """Build one map from the 30 fps subset of a stream, into ``map_dir``.

    With ``mask_dir`` the engine reads the masks that ``write_role_masks`` wrote
    for the stream.
    """
    frames, subset, kb4 = mapping_subset(frames, unit, lens)
    write_timestamps(map_dir / "timestamps.txt", subset)
    settings = write_engine_settings(map_dir / "settings.yaml", kb4, MAP_FPS)
    run_engine(
        "map",
        settings,
        frames,
        map_dir / "timestamps.txt",
        map_dir / "observations.txt",
        atlas_out=map_dir / "atlas.osa",
        mask_dir=mask_dir,
        log_path=map_dir / "engine.log",
    )


def solve_placements(
    reloc: dict[tuple[str, str], Observations],
    units: dict[str, Unit],
    imu: dict[str, tuple],
    builder_arm: str,
) -> dict[str, MapToScene]:
    """Metric frame of each map and the placement of both front maps in the scene frame.

    ``reloc`` holds the map-frame rows of the four full streams by ``(arm, lens)``. The scene
    frame is the shared rear map made metric from the builder arm's rear stream.
    """
    b = builder_arm
    rear = MapToScene.from_metric_alignment(
        fit_metric_frame(
            reloc[(b, "rear")],
            units[b].timeshift_cam_imu["rear"],
            units[b].T_cam_imu["rear"],
            imu[b],
        )
    )
    placements = {"shared_rear": rear}
    for arm in ARMS:
        unit = units[arm]
        fo = reloc[(arm, "front")]
        ro = reloc[(arm, "rear")]
        front_metric = MapToScene.from_metric_alignment(
            fit_metric_frame(
                fo, unit.timeshift_cam_imu["front"], unit.T_cam_imu["front"], imu[arm]
            )
        )
        G = solve_front_to_scene(
            fo.t_s[fo.tracked] + unit.timeshift_cam_imu["front"],
            front_metric.apply(fo.T[fo.tracked]),
            ro.t_s[ro.tracked] + unit.timeshift_cam_imu["rear"],
            rear.apply(ro.T[ro.tracked]),
            unit.T_rear_front(),
            rear_run_id=run_ids_from_tracked(ro.tracked)[ro.tracked],
        )
        placements[role_for(arm, "front")] = MapToScene(
            scale=front_metric.scale,
            rotation=G[:3, :3] @ front_metric.rotation,
            translation=G[:3, 3].copy(),
        )
    return placements


def export_frames(jobs: list[tuple[Path, np.ndarray, Path]]) -> None:
    """Export every ``(video, times_ns, out_dir)`` stream, each in its own
    process, started together."""
    videos, times, out_dirs = zip(*jobs)
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        list(pool.map(export_stream_frames, videos, times, out_dirs))


def localize_stream(
    frames: Path, unit: Unit, lens: str, atlas: Path, out: Path
) -> Observations:
    """One localization session over every exported frame of a stream, against an atlas.

    The map-frame rows go to ``out``, the settings file and the engine log
    beside them; the rows are returned.
    """
    t_ns = read_timestamps(frames / "timestamps.txt")
    kb4 = rescale_kb4(
        unit.kb4[lens], unit.resolution[lens], (MAP_IMAGE_SIZE, MAP_IMAGE_SIZE)
    )
    settings = write_engine_settings(
        out.with_suffix(".settings.yaml"), kb4, native_fps(t_ns)
    )
    run_engine(
        "localize",
        settings,
        frames,
        frames / "timestamps.txt",
        out,
        atlas_in=atlas,
        log_path=out.with_suffix(".engine.log"),
    )
    return read_observations(out)


# ----------------------------------------------------------------------------
# Checks around the engine, and publication
# ----------------------------------------------------------------------------


class Masker:
    """The packaged segmentation model on the CUDA execution provider:
    ``segment(bgr)`` gives a 0/255 exclude mask."""

    def __init__(self):
        from kiwi.perception.gripper_segmentation import (
            default_model_path,
            open_runtime,
            resolve_device,
        )

        self.runtime = open_runtime(default_model_path(), resolve_device("cuda"))

    def segment(self, img_bgr):
        from kiwi.perception.gripper_segmentation import segment

        return segment(self.runtime, img_bgr)


def _require_engine() -> None:
    """Refuse to start without the built engine and its vocabulary."""
    for path in (engine_binary(), vocabulary()):
        if not path.is_file():
            raise RuntimeError(
                f"ORB-SLAM3 engine file {path} is missing - " + BUILD_HELP
            )


def _require_recipe(scene: SceneConfig) -> None:
    """Refuse a mapping profile whose engine recipe is not the one built here."""
    if (scene.profile.map_fps, scene.profile.map_image_size) != (
        MAP_FPS,
        MAP_IMAGE_SIZE,
    ):
        raise ValueError(
            f"{scene.path}: the mapping profile states "
            f"{scene.profile.map_fps:g} fps at {scene.profile.map_image_size} px; "
            f"kiwi scene build maps at {MAP_FPS:g} fps and {MAP_IMAGE_SIZE} px"
        )


def _check_session(
    rows: Path, timestamps: Path, log: Path, atlas: Path | None = None
) -> None:
    """Refuse an engine session that did not write one row per frame it was
    fed, or that was asked for an atlas and wrote none."""
    n_frames = len(read_timestamps(timestamps))
    n_rows = 0
    if rows.is_file():
        with open(rows, encoding="utf-8") as f:
            n_rows = sum(1 for _ in f)
    if n_rows != n_frames:
        raise RuntimeError(
            f"the engine wrote {n_rows} rows for {n_frames} frames - see {log}"
        )
    if atlas is not None and not atlas.is_file():
        raise RuntimeError(f"the engine wrote no atlas at {atlas} - see {log}")


def _localize_checked(
    frames: Path, unit: Unit, lens: str, atlas: Path, out: Path
) -> Observations:
    """``localize_stream`` followed by the session check, which also runs when
    the session left no row to read."""
    timestamps = frames / "timestamps.txt"
    log = out.with_suffix(".engine.log")
    # The rows of an earlier session of the stream are not this session's.
    out.unlink(missing_ok=True)
    try:
        obs = localize_stream(frames, unit, lens, atlas, out)
    except (IndexError, OSError):
        _check_session(out, timestamps, log)
        raise
    _check_session(out, timestamps, log)
    return obs


def _read_json(path: Path):
    """The parsed document, or None when the file cannot be read as JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_localized(
    scene: SceneConfig, published: dict, episode_id: str, arm: str, lens: str
) -> bool:
    """Whether a stream's observations were localized against the published map.

    ``lens`` is ``front`` or ``back``. The observation file must exist, and
    its record must name the map the role has in ``published``.
    """
    role = "shared_rear" if lens == "back" else f"front_{arm}"
    out = (
        layout.scene_dir(scene.episode_dir(episode_id), arm)
        / f"{role}_observations.txt"
    )
    record = _read_json(out.with_suffix(".inputs.json"))
    return (
        out.is_file()
        and isinstance(record, dict)
        and record.get("schema") == "scene_localization_inputs"
        and record.get("map") == published["roles"][role]
    )


def _publish_observations(
    scene: SceneConfig,
    roles: dict[str, str],
    ri: RoleInputs,
    lens: str,
    unit: Unit,
    obs: Observations,
    placement: MapToScene,
    reloc_log: Path,
    *,
    published_by: str,
    copy_log: bool,
) -> Path:
    """Write one stream's rows in the scene frame, then the record that binds
    them to their map; returns the row file.

    ``lens`` is ``front`` or ``back``. With ``copy_log`` the engine log is
    copied beside the rows; otherwise the record references it where it is.
    """
    klens = "rear" if lens == "back" else "front"
    scene_map = scene.root / "scene_map"
    atlas = scene_map / roles[ri.role] / "atlas.osa"
    frame_path = scene_map / "scene_frame.json"
    out_dir = layout.scene_dir(scene.episode_dir(ri.episode_id), ri.arm)
    out = out_dir / f"{ri.role}_observations.txt"
    record_path = out.with_suffix(".inputs.json")
    log = out_dir / f"{ri.role}.engine.log"
    for previous in (out, record_path, log):
        previous.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_observations(out, Observations(obs.t_ns, obs.state, placement.apply(obs.T)))
    engine_log = reloc_log
    if copy_log:
        shutil.copyfile(reloc_log, log)
        engine_log = log
    atlas_ref = file_ref(atlas, base=out_dir)
    # The record is written last: observations without one do not count as
    # localized.
    write_record(
        record_path,
        {
            "schema": "scene_localization_inputs",
            "map": roles[ri.role],
            "frame": SCENE_FRAME_NAME,
            "mask": None,
            "arm": ri.arm,
            "lens": lens,
            "atlas_digest": atlas_ref["digest"],
            "rows": len(obs),
            "tracked_rows": int(obs.tracked.sum()),
            "published_by": published_by,
            "producer": producer_info(
                "kiwi.scene.localize", dependencies=("numpy", "scipy")
            ),
            "inputs": {
                "video": file_ref(
                    ri.video,
                    base=out_dir,
                    record=ri.video.parent / "convert.inputs.json",
                ),
                "raw_recording": file_ref(ri.recording, base=out_dir),
                "kb4": file_ref(ri.kb4_artifact, base=out_dir),
                "atlas": atlas_ref,
                "scene_frame": file_ref(frame_path, base=out_dir),
            },
            "outputs": {
                "observations": file_ref(out, base=out_dir),
                "engine_log": file_ref(engine_log, base=out_dir),
            },
            "input_snapshots": {"scene_frame": json.loads(frame_path.read_text())},
            "input_snapshot_bases": {"scene_frame": str(frame_path.parent.resolve())},
            "clock": {
                "kind": "recording_video_pts",
                "unit": "nanoseconds",
                "offset_applied_s": 0.0,
            },
            "body": "rear_camera_optical_centre"
            if ri.role == "shared_rear"
            else "front_camera_optical_centre",
            "position_units": "metres",
            "rotation_convention": "hamilton_xyzw",
            "recipe": {
                "image_size": [MAP_IMAGE_SIZE, MAP_IMAGE_SIZE],
                "kb4_parameters": rescale_kb4(
                    unit.kb4[klens],
                    unit.resolution[klens],
                    (MAP_IMAGE_SIZE, MAP_IMAGE_SIZE),
                ).tolist(),
                "n_features": N_FEATURES,
                "engine_threads": ENGINE_THREADS,
            },
        },
    )
    return out


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------


def build_scene_maps(scene: SceneConfig) -> BuildOutcome:
    """Build the three maps of a scene from its mapping episode and place them.

    A published map set is reused. Otherwise everything under ``scene_map``
    is removed first; the build writes the maps, the map-frame rows of the
    four full streams, ``scene_frame.json``, the mapping episode's
    observations in scene coordinates and, last, ``current.json``.
    """
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig

    existing = current_maps(scene)
    if existing is not None:
        return BuildOutcome(
            accepted=True,
            episode_id=existing.get("episode"),
            maps=existing["roles"],
            attempts=0,
            reason="current map set already published and complete - reusing",
        )
    _require_recipe(scene)

    episode_id = scene.builder_episode
    # ``inputs`` and ``times`` are keyed (arm, lens) with the lens as the
    # command line names it, "front" or "back"; the frame directories, the row
    # files under scene_map/ and the units say "rear" for "back".
    streams = [
        (arm, lens, "rear" if lens == "back" else "front")
        for arm in ARMS
        for lens in ("front", "back")
    ]
    inputs = {
        (arm, lens): _stream_inputs(scene, episode_id, arm, lens)
        for arm, lens, _ in streams
    }
    cfg = EpisodeConfig.load(scene.episode_dir(episode_id) / MANIFEST_NAME)
    units = {arm: load_unit(cfg.arms[arm]) for arm in ARMS}
    times = {
        key: np.asarray(frame_times_ns(ri.video), dtype=np.int64)
        for key, ri in inputs.items()
    }

    builds = [("shared_rear", scene.map_builder_arm, "rear")]
    builds += [(role_for(arm, "front"), arm, "front") for arm in ARMS]
    floor = effective_min_usable_s(scene)
    for _, arm, klens in builds:
        t_ns = times[(arm, "back" if klens == "rear" else "front")]
        span = (t_ns[-1] - t_ns[0]) * 1e-9
        if span < floor:
            raise ValueError(
                f"episode {episode_id!r}: the {arm} {klens} stream lasts "
                f"{span:.1f} s; map building needs {floor:g} s "
                f"(min_usable_duration_s)"
            )

    # The engine and the mask model are checked before any frame is exported.
    _require_engine()
    masker = Masker()

    scene_map = scene.root / "scene_map"
    if scene_map.exists():
        shutil.rmtree(scene_map)
    maps = scene_map / "maps"
    frames = {
        (arm, klens): frames_dir(scene_map, episode_id, arm, klens)
        for arm in ARMS
        for klens in LENSES
    }
    mask_dirs = {}
    try:
        export_frames(
            [
                (inputs[(arm, lens)].video, times[(arm, lens)], frames[(arm, klens)])
                for arm, lens, klens in streams
            ]
        )

        for arm in ARMS:
            mask_dirs[arm] = masks_dir(scene_map, episode_id, arm)
            write_role_masks(masker, frames[(arm, "front")], units[arm], mask_dirs[arm])

        # One engine process per map, started together.
        with ProcessPoolExecutor(max_workers=len(builds)) as pool:
            running = [
                pool.submit(
                    build_role,
                    frames[(arm, klens)],
                    units[arm],
                    klens,
                    maps / role,
                    mask_dirs.get(arm) if klens == "front" else None,
                )
                for role, arm, klens in builds
            ]
            for build in running:
                build.result()
        for role, _, _ in builds:
            _check_session(
                maps / role / "observations.txt",
                maps / role / "timestamps.txt",
                maps / role / "engine.log",
                atlas=maps / role / "atlas.osa",
            )

        # A map directory carries the first twelve hex digits of its atlas
        # digest, so a rebuilt map set never has the names of an earlier one.
        roles = {}
        for role, _, _ in builds:
            digest = digest_bytes(maps / role / "atlas.osa").split(":")[1]
            name = f"{role}-{digest[:12]}"
            (maps / role).rename(maps / name)
            roles[role] = f"maps/{name}"

        # One engine process per stream, started together, like the map
        # builds above: each localization has its own frame directory, rows
        # file and log, and the atlas is read-only.
        reloc = {}
        with ProcessPoolExecutor(max_workers=len(ARMS) * len(LENSES)) as pool:
            pending = {
                (arm, klens): pool.submit(
                    _localize_checked,
                    frames[(arm, klens)],
                    units[arm],
                    klens,
                    scene_map / roles[role_for(arm, klens)] / "atlas.osa",
                    map_frame_rows(scene_map, episode_id, arm, klens),
                )
                for arm in ARMS
                for klens in LENSES
            }
            for key, session in pending.items():
                reloc[key] = session.result()
        for arm in ARMS:
            for klens in LENSES:
                # The placement solve needs tracked rows of all four streams.
                if not reloc[(arm, klens)].tracked.any():
                    rows = map_frame_rows(scene_map, episode_id, arm, klens)
                    raise RuntimeError(
                        f"episode {episode_id!r}: the localization of the {arm} "
                        f"{klens} stream has no tracked row - see "
                        f"{rows.with_suffix('.engine.log')}"
                    )
        imu = {
            arm: load_imu(inputs[(arm, "front")].recording, scene_map / "cache")
            for arm in ARMS
        }
        placements = solve_placements(reloc, units, imu, scene.map_builder_arm)
        write_scene_frame(scene_map / "scene_frame.json", placements, episode_id)

        for arm, lens, klens in streams:
            ri = inputs[(arm, lens)]
            rows = map_frame_rows(scene_map, episode_id, arm, klens)
            _publish_observations(
                scene,
                roles,
                ri,
                lens,
                units[arm],
                reloc[(arm, klens)],
                placements[ri.role],
                rows.with_suffix(".engine.log"),
                published_by="scene build",
                copy_log=False,
            )
    finally:
        shutil.rmtree(scene_map / "frames", ignore_errors=True)

    # The pointer is written last: without it there is no published map set.
    atomic_write_text(
        scene_map / _POINTER,
        json.dumps(
            {"schema": "scene_map_pointer", "episode": episode_id, "roles": roles},
            indent=2,
        )
        + "\n",
    )
    return BuildOutcome(
        accepted=True,
        episode_id=episode_id,
        maps=roles,
        attempts=1,
        reason=f"map set built from episode {episode_id!r}",
    )


def localize_episode(
    scene: SceneConfig, episode_id: str, arm: str, *, lens: str = "front"
) -> Path:
    """Localize one full stream against the published map; publish the rows
    in the scene frame.

    Routing is by (arm, lens): the requested arm's own stream and
    calibration, against that lens's map role. Output:
    ``<episode>/<arm>/scene/<role>_observations.txt`` (poses transformed
    by the frozen scene-frame placement), the record that binds the rows to
    the map, and the engine log. A stream already localized against the
    published map set is refused; observations of another map set are
    replaced.
    """
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig

    published = current_maps(scene)
    if published is None:
        raise ValueError(
            f"scene {scene.name!r} has no published map set - run "
            f"`kiwi scene build` first"
        )
    _require_recipe(scene)
    ri = _stream_inputs(scene, episode_id, arm, lens)
    klens = "rear" if lens == "back" else "front"
    out_dir = layout.scene_dir(scene.episode_dir(episode_id), arm)
    out = out_dir / f"{ri.role}_observations.txt"
    if is_localized(scene, published, episode_id, arm, lens):
        raise FileExistsError(f"{out} already exists - never overwritten")

    _require_engine()
    # Nothing of an earlier map set stays behind if the engine fails.
    for previous in (
        out,
        out.with_suffix(".inputs.json"),
        out_dir / f"{ri.role}.engine.log",
    ):
        previous.unlink(missing_ok=True)

    cfg = EpisodeConfig.load(scene.episode_dir(episode_id) / MANIFEST_NAME)
    unit = load_unit(cfg.arms[arm])
    scene_map = scene.root / "scene_map"
    frames = frames_dir(scene_map, episode_id, arm, klens)
    out_rows = map_frame_rows(scene_map, episode_id, arm, klens)
    atlas = scene_map / published["roles"][ri.role] / "atlas.osa"
    try:
        export_frames(
            [(ri.video, np.asarray(frame_times_ns(ri.video), dtype=np.int64), frames)]
        )
        obs = _localize_checked(frames, unit, klens, atlas, out_rows)
    finally:
        shutil.rmtree(frames, ignore_errors=True)

    placements = read_scene_frame(scene_map / "scene_frame.json")
    return _publish_observations(
        scene,
        published["roles"],
        ri,
        lens,
        unit,
        obs,
        placements[ri.role],
        out_rows.with_suffix(".engine.log"),
        published_by="scene localize",
        copy_log=True,
    )

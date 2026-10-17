"""Head-marker observations from the two fused arm states.

The head rig carries one ArUco marker. Each wrist rear camera sees it now and then, and
its pose follows from the arm's fused front-camera trajectory and the calibrated lens
geometry (``T_scene_rear = T_scene_front @ inv(T_rear_front)``), so every accepted
detection becomes a scene-frame marker pose ``T_scene_marker = T_scene_rear(t) @
T_rear_marker``. The detections of both wrists are published as one time-ordered
trajectory with a report and per-row segment ids.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kiwi.core import layout
from kiwi.core.camera.registry import CameraUnit
from kiwi.core.digest import digest_bytes
from kiwi.core.generation import file_ref, producer_info, write_record
from kiwi.core.trajectory import Trajectory, load_tum, se3_stack, write_tum
from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig

HEAD_REPORT_SCHEMA = "head_marker_state"
COMPUTED_STATUS = "computed_unvalidated"
INSUFFICIENT_STATUS = "insufficient_observations"
COMPUTED_STATUSES = frozenset({COMPUTED_STATUS})

# A gap longer than this between consecutive rows starts a new segment.
HEAD_SEGMENT_GAP_S = 1.0


@dataclass(frozen=True)
class HeadPublication:
    status: str
    reason: str
    artifacts: tuple[Path, ...]
    reused: bool
    report: dict


@dataclass(frozen=True)
class HeadArmInput:
    """Artifacts that connect one arm's rear images to its fused state."""

    label: str
    state_tum: Path
    rear_video: Path
    rear_times_ns: np.ndarray
    rear_intrinsics: Path
    rear_extrinsic: Path
    front_extrinsic: Path
    unit: object
    sync_offset_s: float


def configured_head_camera(cfg: EpisodeConfig):
    """Return the single configured head camera, or ``None``.

    :meth:`EpisodeConfig.load` refuses more than one head camera, so the
    first match is the only one.
    """
    return next((camera for camera in cfg.cameras.values() if camera.head_marker), None)


def _resolve_inputs(episode_dir: Path, cfg: EpisodeConfig) -> list[HeadArmInput]:
    from kiwi.episode.scene_build import load_unit
    from kiwi.track.sequence import frame_times_ns

    if set(cfg.arms) != {"left", "right"}:
        raise ValueError(
            "head marker estimation needs fused left and right arm states; "
            f"episode declares {sorted(cfg.arms)}"
        )
    if cfg.sync_offsets is None:
        raise ValueError(
            "head marker estimation needs the episode global clock; run the "
            "sync stage first"
        )

    inputs: list[HeadArmInput] = []
    for label in ("left", "right"):
        arm = cfg.arms[label]
        if label not in cfg.sync_offsets:
            raise ValueError(
                f"head marker estimation needs a sync offset for {label!r}; "
                "run the sync stage first"
            )
        registry = CameraUnit.load(arm.camera_dir)
        rear_intrinsics = registry.artifact("intrinsics_kb4_back")
        rear_extrinsic = registry.artifact("extrinsic_imu_back")
        front_extrinsic = registry.artifact("extrinsic_imu")
        missing = [
            name
            for name, path in (
                ("intrinsics_kb4_back", rear_intrinsics),
                ("extrinsic_imu_back", rear_extrinsic),
                ("extrinsic_imu", front_extrinsic),
            )
            if path is None
        ]
        if missing:
            raise ValueError(f"camera {arm.camera!r} publishes no {'/'.join(missing)}")
        state_tum = layout.scene_dir(episode_dir, label) / layout.STATE_TUM
        rear_video = layout.stream_dir(episode_dir, label) / layout.LENS1
        for path, what in (
            (state_tum, "fused arm state"),
            (rear_video, "converted rear video"),
            (rear_intrinsics, "rear KB4 intrinsics"),
            (rear_extrinsic, "rear camera/IMU extrinsic"),
            (front_extrinsic, "front camera/IMU extrinsic"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label}: {what} is missing: {path}")

        inputs.append(
            HeadArmInput(
                label=label,
                state_tum=state_tum,
                rear_video=rear_video,
                rear_times_ns=np.asarray(frame_times_ns(rear_video), dtype=np.int64),
                rear_intrinsics=rear_intrinsics,
                rear_extrinsic=rear_extrinsic,
                front_extrinsic=front_extrinsic,
                unit=load_unit(arm),
                sync_offset_s=float(cfg.sync_offsets[label]),
            )
        )
    return inputs


def _source_records(arms: list[HeadArmInput], output_dir: Path) -> tuple[dict, dict]:
    """Locatable production records for each arm, and the scene they share.

    The scene identity is read from each arm's ``state.inputs.json``, the
    record the state stage writes beside every state it publishes; a missing
    record, or one that declares no ``world_scene``, is refused with the stage
    to re-run. Arms declaring different identities are refused. A key that
    only some arms' records carry is published as ``None`` rather than
    borrowed from the other arm.
    """
    references, worlds = {}, []
    record_name = f"{layout.SCENE_DIR}/{layout.STATE_INPUTS}"
    for arm in arms:
        record = arm.state_tum.parent / layout.STATE_INPUTS
        if not record.is_file():
            raise ValueError(
                f"{arm.label}: no {record_name} beside the fused arm state - "
                f"re-run the state stage"
            )
        world = json.loads(record.read_text()).get("world_scene")
        if world is None:
            raise ValueError(
                f"{arm.label}: {record_name} declares no world_scene - re-run the "
                f"state stage"
            )
        worlds.append(world)
        references[arm.label] = {
            "state": file_ref(arm.state_tum, base=output_dir, record=record),
            "rear_video": file_ref(
                arm.rear_video,
                base=output_dir,
                record=arm.rear_video.parent / "convert.inputs.json",
            ),
            "rear_intrinsics": file_ref(arm.rear_intrinsics, base=output_dir),
            "rear_extrinsic": file_ref(arm.rear_extrinsic, base=output_dir),
            "front_extrinsic": file_ref(arm.front_extrinsic, base=output_dir),
        }
    shared = {}
    for key in ("frame", "scene", "map_set_digest", "scene_frame_digest"):
        values = [item.get(key) for item in worlds]
        known = {value for value in values if value is not None}
        if len(known) > 1:
            raise ValueError(f"head arm states declare different {key} identities")
        shared[key] = known.pop() if known and None not in values else None
    return references, shared


def _arm_marker_poses(
    state_tum: Path,
    rear_video: Path,
    rear_times_ns: np.ndarray,
    kb4: np.ndarray,
    resolution: tuple[int, int],
    clock_offset_s: float,
    marker: dict,
    T_rear_front: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The scene-frame marker poses of one arm's rear view, with their times."""
    from kiwi.state.head import compose_scene_marker, detect_marker_poses

    t_front, p, q, _ = load_tum(state_tum)
    T_scene_front = se3_stack(p, q)
    t_det, T_rear_marker = detect_marker_poses(
        rear_video, rear_times_ns, kb4, resolution, clock_offset_s, marker
    )
    if len(t_det) == 0:
        # No accepted detection in this rear view: the arm contributes no row.
        return t_det, np.zeros((0, 4, 4))
    return compose_scene_marker(
        t_det, T_rear_marker, t_front, T_scene_front, T_rear_front
    )


def _merge(
    per_arm: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Both arms' rows as one time-ordered track ``(t, T, arm, segment_id)``.

    ``per_arm`` is ``[(t_left, T_left), (t_right, T_right)]``; ``arm`` is 0 for
    the left arm and 1 for the right. Rows are ordered by time, the left arm
    first at equal times. A row is kept only when its time, rounded to
    nanoseconds, exceeds that of the last kept row, so the times strictly
    increase. ``segment_id`` starts at 0 and increases where the gap to the
    previous kept row exceeds ``HEAD_SEGMENT_GAP_S``.
    """
    t = np.concatenate([np.asarray(times, dtype=float) for times, _ in per_arm])
    T = np.concatenate(
        [np.asarray(poses, dtype=float).reshape(-1, 4, 4) for _, poses in per_arm]
    )
    arm = np.concatenate(
        [
            np.full(len(times), code, dtype=np.int8)
            for code, (times, _) in enumerate(per_arm)
        ]
    )
    order = np.argsort(t, kind="stable")
    kept: list[int] = []
    segment_id: list[int] = []
    for row in order:
        if not kept:
            segment_id.append(0)
        elif round(float(t[row]), 9) <= round(float(t[kept[-1]]), 9):
            continue
        else:
            gap = t[row] - t[kept[-1]]
            segment_id.append(segment_id[-1] + int(gap > HEAD_SEGMENT_GAP_S))
        kept.append(int(row))
    return t[kept], T[kept], arm[kept], np.asarray(segment_id, dtype=np.int64)


def publish_head_marker(episode_dir: Path) -> HeadPublication:
    """Compute one episode's optional head-marker publication."""
    from scipy.spatial.transform import Rotation

    from kiwi.state.geometry import quats_continuous
    from kiwi.state.head import (
        AMBIGUITY_RATIO,
        ARUCO_DICTS,
        DETECTION_MAX_EDGE_PX,
        EFFECTIVE_FPS,
        MAX_REPROJECTION_PX,
    )
    from kiwi.track.scene_frame import SCENE_FRAME_NAME

    episode_dir = Path(episode_dir).resolve()
    cfg = EpisodeConfig.load(episode_dir / MANIFEST_NAME)
    camera = configured_head_camera(cfg)
    if camera is None:
        raise ValueError("episode has no cameras.*.head_marker configuration")

    inputs = _resolve_inputs(episode_dir, cfg)
    marker = camera.head_marker
    if marker["dictionary"] not in ARUCO_DICTS:
        raise ValueError(
            f"cameras.{camera.label}.head_marker.dictionary "
            f"{marker['dictionary']!r} is not an OpenCV ArUco dictionary"
        )

    output_dir = episode_dir / layout.HEAD_DIR
    report_path = output_dir / layout.HEAD_REPORT
    trajectory_path = output_dir / layout.HEAD_TUM
    diagnostics_path = output_dir / layout.HEAD_DIAGNOSTICS
    input_files, world_scene = _source_records(inputs, output_dir)

    # The stage owns head/: every run starts from an empty directory.
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    # One process per arm, started together. The processes are spawned: OpenCV
    # runs single-threaded in a process forked from one that has already called it.
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=spawn) as pool:
        futures = [
            pool.submit(
                _arm_marker_poses,
                arm.state_tum,
                arm.rear_video,
                arm.rear_times_ns,
                arm.unit.kb4["rear"],
                arm.unit.resolution["rear"],
                arm.unit.timeshift_cam_imu["rear"] + arm.sync_offset_s,
                marker,
                arm.unit.T_rear_front(),
            )
            for arm in inputs
        ]
        per_arm = []
        for arm, future in zip(inputs, futures, strict=True):
            try:
                per_arm.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    f"head marker detection failed for the {arm.label} arm: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    t, T, arm_code, segment_id = _merge(per_arm)

    segments = []
    for segment in np.unique(segment_id):
        rows = t[segment_id == segment]
        segments.append(
            {
                "segment_id": int(segment),
                "rows": len(rows),
                "t_start": float(rows[0]),
                "t_end": float(rows[-1]),
            }
        )
    if len(t):
        status, reason = COMPUTED_STATUS, ""
    else:
        status, reason = INSUFFICIENT_STATUS, "no accepted head-marker detection"
    report = {
        "schema": HEAD_REPORT_SCHEMA,
        "status": status,
        "reason": reason,
        "frame": SCENE_FRAME_NAME,
        "body": "head_marker",
        "episode": cfg.name,
        "head_camera": camera.label,
        "marker": marker,
        "params": {
            "effective_fps": EFFECTIVE_FPS,
            "detection_max_edge_px": DETECTION_MAX_EDGE_PX,
            "ambiguity_ratio": AMBIGUITY_RATIO,
            "max_reprojection_px": MAX_REPROJECTION_PX,
            "segment_gap_s": HEAD_SEGMENT_GAP_S,
        },
        "per_arm": {
            arm.label: {
                "detections": len(times),
                "rows": int(np.count_nonzero(arm_code == code)),
            }
            for code, (arm, (times, _)) in enumerate(zip(inputs, per_arm, strict=True))
        },
        "segments": segments,
        "input_files": input_files,
        "world_scene": world_scene,
        "clock": "episode_global_seconds",
        "output": {},
        "producer": producer_info(
            "kiwi.state.head", dependencies=("opencv-python", "numpy", "scipy")
        ),
    }
    if len(t) == 0:
        write_record(report_path, report)
        return HeadPublication(status, reason, (report_path,), False, report)

    diagnostics_part = diagnostics_path.with_name(diagnostics_path.name + ".part")
    with diagnostics_part.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=np.asarray(
                json.dumps(
                    {
                        "schema": "head_marker_diagnostics",
                        "frame": SCENE_FRAME_NAME,
                        "body": "head_marker",
                        "clock": "episode_global_seconds",
                        "world_scene": world_scene,
                    },
                    sort_keys=True,
                )
            ),
            t=t,
            segment_id=segment_id,
            arm=arm_code,
        )
    os.replace(diagnostics_part, diagnostics_path)
    report["quality_outputs"] = {
        "diagnostics": file_ref(diagnostics_path, base=output_dir)
    }

    q = quats_continuous(Rotation.from_matrix(T[:, :3, :3]).as_quat())
    trajectory_part = trajectory_path.with_name(trajectory_path.name + ".part")
    write_tum(
        Trajectory(t, T[:, :3, 3], q, SCENE_FRAME_NAME, "head_marker"),
        trajectory_part,
        clock="episode global clock",
        source="kiwi.state head-marker detections",
    )
    report["output"] = {
        "path": layout.HEAD_TUM,
        "digest": digest_bytes(trajectory_part),
        "rows": len(t),
        "timestamp_span_s": [float(t[0]), float(t[-1])],
    }
    # The report is written before the trajectory it describes is renamed into place.
    write_record(report_path, report)
    os.replace(trajectory_part, trajectory_path)
    return HeadPublication(
        status, reason, (trajectory_path, diagnostics_path, report_path), False, report
    )

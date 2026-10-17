"""Build an :class:`~kiwi.core.observation_episode.ObservationEpisode` from artifacts.

:func:`build_episode` reads each arm's world-frame trajectory (by default
``<arm>/scene/state.tum``, the state stage's scene frame; ``<arm>/register.tum``
under ``world_source="marker"``), the ``eef_state_v1`` tool-state npz, and the
``<arm>/lens0.mp4`` video, places every stream on the episode global clock at a
uniform fps, and emits per-arm EE pose + tool state as ``observation.state``
plus one camera video per arm. The input is a plain
:class:`~kiwi.export.source.EpisodeSource` - never a manifest (the
``kiwi export`` verb in ``kiwi.cli`` does that flattening; see ``source.py``).

**Frame chain.** The world tum holds the CAMERA pose in the shared world frame
(its header declares frame, body and clock - refused, not guessed, when it does
not). The exported body frame is always the tool tip:

    T_world_tcp(t) = T_world_camera(t) @ T_camera_tcp
    T_camera_tcp   = inv(T_mount_cam) @ T_mount_tcp

Both legs are read from the arm's robot-description URDF by fixed-joint FK
(:func:`kiwi.core.descriptions.fixed_frame_T`), each overridable per gripper
unit (description default < unit override) for a build whose tip was re-glued
or whose camera seats off its nominal pocket. Both are expressed in the
gripper's mount frame, so the robot flange never enters the capture-side chain
(descriptions/PROVENANCE.md). Replay's solver drives the corresponding TCP site on its separately selected
robot description, so there is one body frame
(``ee_frame="tcp"``) and no knob. The state stage already normalizes poses to
the camera body, so no cam-IMU leg enters the export.

**Clock.** The world tum's timestamps are already ON the episode global
clock (declared in its header), so pose samples need no algebra at all. The
two frame-indexed streams map as ``global(frame_i) = sync_off + i / fps`` at
each stream's own native fps: the video, and the tool state (whose fps is
stored inside the ``eef_state_v1`` artifact). An undeclared clock is an
error, never a guess.

**Tool state.** ``EefState.output`` is the completed raw series, finite on every
source frame. Completion provenance follows the sampled EEF source-frame indices
into export_samples, with a source summary in tool_state. This module never
interpolates tool state: each grid instant samples the nearest source frame,
and a non-finite sample is a contract violation that raises.

**Extra camera streams.** ``cameras`` entries (e.g. a head-mounted ego view)
each contribute one more video feature - ``observation.images.<label>`` - and
nothing else: no pose, no tool state, no state columns. Same
``global(frame_i) = off + i/fps`` mapping at their own native fps. An extra
camera bounds the exported window exactly as an arm's video does.

The export window computed here (:func:`_overlap_window`: pose ∩ video ∩
tool-state, then the ``task_window``/``window`` trim) is NARROWER than the
global clock itself, whose ends are decided by video streams alone. The export
window may move with pose coverage, and is always a subinterval of the clock.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.core.config_hash import stamp_input
from kiwi.core.descriptions import fixed_frame_T, load_robot_description
from kiwi.core.generation import json_value
from kiwi.core.observation_episode import VIDEO_DTYPE, FeatureColumn, ObservationEpisode
from kiwi.core.tool_config import load_T_mount_cam, load_T_mount_tcp
from kiwi.core.trajectory import PoseTrack, load_tum, se3_stack
from kiwi.core.video import VideoInfo
from kiwi.eef import get_estimator
from kiwi.eef.state import read_eef_state

from .schema import (
    IMAGE_KEY_PREFIX,
    RELATIVE_STATE_KEY,
    STATE_KEY,
    arm_relative_names,
    arm_state_names,
    camera_pose_key,
    pose_gap_key,
    wrist_image_key,
)
from .source import WORLD_SOURCES, ArmSource, CameraSource, EpisodeSource
from .video_source import VideoSource

__all__ = [
    "WorldTumHeader",
    "build_episode",
    "read_world_tum_header",
    "world_transform",
]

# --- small artifact helpers ------------------------------------------------


def _matrix(t: np.ndarray) -> list[list[float]]:
    """4x4 SE(3) as plain nested lists (full float precision, JSON/YAML-safe)."""
    return [[float(v) for v in row] for row in np.asarray(t)]


# Anchored on the FINAL ')' so a clock that itself contains parentheses -
# "episode global clock (origin=right)" - parses whole; the optional
# "; source=..." tail is split off afterwards.
_TUM_HEADER_RE = re.compile(r"\(frame=(\S+)\s+body=(\S+);\s*t on (.+)\)\s*$")


@dataclasses.dataclass(frozen=True)
class WorldTumHeader:
    """What a world ``.tum`` header declares: frame, body, clock."""

    frame: str
    body: str
    clock: str


def read_world_tum_header(tum: Path) -> WorldTumHeader:
    """The frame/body/clock a world ``.tum`` declares, read off its own header.

    The header shape is ``kiwi.core.trajectory.write_tum``'s::

        # t tx ty tz qx qy qz qw  (frame=<f> body=<b>; t on <clock>; source=<s>)

    Three refusals:

    * no header line at all;
    * a clock that is not the episode global clock;
    * a body other than ``camera`` - the EE composition right-multiplies a
      camera-frame extrinsic.
    """
    tum = Path(tum)
    with tum.open() as fh:
        header = fh.readline()
    if not header.startswith("#"):
        raise ValueError(
            f"{tum}: no header line, so its frame/body/clock are unknowable. "
            f"Re-run the world trajectory's producer to regenerate it."
        )
    m = _TUM_HEADER_RE.search(header)
    if m is None:
        raise ValueError(
            f"{tum}: unrecognized header {header.strip()!r}; expected "
            f"'(frame=<f> body=<b>; t on <clock>; ...)' as written by the "
            f"world trajectory's producer."
        )
    clock = m.group(3).split("; source=")[0].strip()
    h = WorldTumHeader(frame=m.group(1), body=m.group(2), clock=clock)
    if not h.clock.startswith("episode global clock"):
        raise ValueError(
            f"{tum}: timestamps are on {h.clock!r}, not the episode global "
            f"clock. Re-run the world trajectory's producer after syncing so "
            f"the file is written on the global clock."
        )
    if h.body != "camera":
        raise ValueError(
            f"{tum}: body is {h.body!r}, expected 'camera'. The EE composition "
            f"applies a camera-frame extrinsic and would be wrong for any "
            f"other body."
        )
    return h


def _posetrack_from_se3(t: np.ndarray, poses: np.ndarray) -> PoseTrack:
    """Build a (Slerp-able) :class:`PoseTrack` from ``t`` + ``(N,4,4)`` poses.

    Drops non-increasing timestamps so Slerp's strictly-increasing requirement
    holds.
    """
    t = np.asarray(t, dtype=np.float64)
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, poses = t[keep], poses[keep]
    return PoseTrack(
        t=t,
        rotations=Rotation.from_matrix(poses[:, :3, :3]),
        positions=poses[:, :3, 3].copy(),
    )


def _sample_state(track: PoseTrack, grid: np.ndarray) -> np.ndarray:
    """Sample a PoseTrack at ``grid`` -> ``(N, 7)`` ``[x,y,z,qx,qy,qz,qw]``."""
    poses = np.stack([track.at(float(g)) for g in grid])
    pos = poses[:, :3, 3]
    quat = Rotation.from_matrix(poses[:, :3, :3]).as_quat()  # xyzw
    return np.hstack([pos, quat]).astype(np.float32)


def _gap_to_raw(raw_t: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Seconds from each grid instant to the nearest RAW sample time.

    The pose channel is interpolated between raw samples (Slerp + linear), so
    this is the per-frame disclosure of how far each exported pose sits from
    a measurement: ~0 on a raw sample, half the source period mid-gap on a
    healthy track, and large across a tracking dropout. Vectorized via
    searchsorted; ``PoseTrack.gap_at`` is the same quantity for one instant.
    """
    raw_t = np.asarray(raw_t, dtype=np.float64)
    i = np.searchsorted(raw_t, grid)
    left = raw_t[np.clip(i - 1, 0, len(raw_t) - 1)]
    right = raw_t[np.clip(i, 0, len(raw_t) - 1)]
    return np.minimum(np.abs(grid - left), np.abs(right - grid))


def _relative_pose(ee: np.ndarray) -> np.ndarray:
    """``(N, 7)`` absolute poses -> the same poses in the FIRST pose's frame.

    ``T_rel(t) = inv(T(t0)) @ T(t)``: row 0 is the identity, and the whole
    track is invariant to where the world anchor sat - the placement
    invariance ``--relative-state`` exists for.
    """
    t = se3_stack(ee[:, :3].astype(np.float64), ee[:, 3:7].astype(np.float64))
    rel = np.linalg.inv(t[0]) @ t
    pos = rel[:, :3, 3]
    quat = Rotation.from_matrix(rel[:, :3, :3]).as_quat()
    return np.hstack([pos, quat]).astype(np.float32)


# --- per-arm assembly ------------------------------------------------------


def world_transform(
    world_offset: tuple[float, float, float] | None,
    world_yaw: float | None,
) -> np.ndarray | None:
    """The world-frame SE(3) baked into an export, or ``None`` when both are unset.

    ``Translate(world_offset) @ Rz(world_yaw)`` - rotate about the world origin
    first, then translate in the ROTATED frame. That is the order ``kiwi replay``
    composes its own two knobs in (it yaws position and orientation, then adds
    its workspace translation in rig coordinates), so an export baked this way
    replays at yaw 0 / workspace 0 with the launch-flag numbers carried over
    unchanged.

    ``world_yaw`` is degrees about +Z. With it unset the result is a pure
    translation and orientations come out untouched.
    """
    if world_offset is None and world_yaw is None:
        return None
    t = np.eye(4)
    if world_yaw is not None:
        t[:3, :3] = Rotation.from_euler("Z", float(world_yaw), degrees=True).as_matrix()
    if world_offset is not None:
        t[:3, 3] = world_offset
    return t


def _intrinsics_block(
    camera: str | None,
    npz_path: Path,
    kb4_path: Path | None,
    repo_root: Path | None,
) -> dict:
    """The camera model as VALUES + stamps, so a dataset carries its own intrinsics.

    Reads the registry ``intrinsics.npz`` (DoubleSphere per lens, or the
    pinhole-equi variant - the two layouts ``kiwi.calib.intrinsics``
    publishes) and the optional ``kb4_intrinsics`` JSON verbatim. Only the
    FRONT lens is recorded: ``lens0.mp4`` is the demuxed front-lens stream.
    ``kb4: None`` means the unit has no KB4 fit published yet - the
    DS model is the calibration, the KB4 a derived convenience - never that
    the model is unknown (an export with no model at all is refused upstream,
    in ``build_and_write``).
    """
    with np.load(str(npz_path), allow_pickle=False) as z:
        files = set(z.files)
        block: dict = {
            "camera": camera,
            "lens": "front",
            "source": stamp_input(npz_path, root=repo_root),
        }
        if "image_size" in files:
            block["calibration_size"] = [int(v) for v in z["image_size"]]
        if "ds_front" in files:
            xi, alpha, fx, fy, cx, cy = (float(v) for v in z["ds_front"])
            block["model"] = "double_sphere"
            block["ds"] = {
                "xi": xi,
                "alpha": alpha,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }
        elif "K_front" in files:
            block["model"] = "pinhole_equi"
            block["K"] = _matrix(z["K_front"])
            block["D"] = [float(v) for v in np.asarray(z["D_front"]).ravel()]
        else:
            raise ValueError(
                f"{npz_path}: no front-lens model (neither ds_front nor "
                f"K_front) - not a kiwi.calib.intrinsics bundle. Re-publish "
                f"the unit's intrinsics npz."
            )
    if kb4_path is not None:
        block["kb4"] = json.loads(Path(kb4_path).read_text())
        block["kb4_source"] = stamp_input(kb4_path, root=repo_root)
    else:
        block["kb4"] = None
    return block


@dataclasses.dataclass
class _ArmData:
    label: str
    out_label: str
    frame: str  # the world frame the world tum declares
    # The frozen map set this arm's scene trajectory was solved against, read
    # off `<arm>/scene/state.inputs.json`. None under the marker source, whose
    # world is per-episode and has no map set.
    map_set: dict | None
    ee_track: PoseTrack
    cam_track: PoseTrack | None
    grip_v: np.ndarray  # per SOURCE frame, normalized [0,1], finite throughout
    grip_fps: float  # the eef artifact's own fps
    grip_t0: float  # global time of the first tool-state sample
    grip_t1: float  # global time of the last tool-state sample
    lens0: Path
    eef_state: Path
    sync_off: float
    fps: float  # the video's native fps
    n_video: int
    tool_meta: dict | None = None  # estimator/unit/endpoints provenance
    tool_completion: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    # The composed tip: whether every leg of it was measured, and where it
    # sits in the camera frame (the replay_state.npz `tcp` contract).
    tcp_meta: dict | None = None
    scene_frame_digest: str | None = None
    # Set by _sample_tool_state: the source frame each grid instant sampled,
    # so the completion audit can be carried along at the same indices.
    tool_frame_indices: np.ndarray | None = None
    # Every constant SE(3) used to compose this arm's exported pose, with the
    # config file (path/target/sha256) each one came from. Recorded so the
    # body-frame chain can be audited or re-derived from the dataset alone.
    extrinsics: dict | None = None
    # The camera MODEL (values + stamp, _intrinsics_block).
    intrinsics: dict | None = None


def _world_tum(arm: ArmSource, world_source: str) -> Path:
    """The arm's world trajectory for the NAMED source, or a refusal.

    An export names one world source and every arm uses it. The missing
    file is never substituted with the other source's: they are different
    world frames.
    """
    if world_source == "markerless":
        tum = arm.state_tum
        producer = (
            "The state stage writes <arm>/scene/state.tum (scene frame, "
            "camera body, episode global clock), which `--world-source "
            "markerless` selects. Run `kiwi scene build` for the episode's "
            "scene, then `kiwi run --profile global-state`."
        )
    elif world_source == "marker":
        tum = arm.register_tum
        producer = (
            "The register stage writes <arm>/register.tum (marker frame, "
            "camera body, episode global clock), which `--world-source "
            "marker` selects."
        )
    else:
        raise ValueError(
            f"world_source is {world_source!r}, expected one of {WORLD_SOURCES}"
        )
    if tum is None:
        raise FileNotFoundError(
            f"arm {arm.label!r}: this export asks for the {world_source!r} "
            f"world pose, but no trajectory path was resolved for it. "
            f"{producer}"
        )
    if not Path(tum).exists():
        raise FileNotFoundError(
            f"arm {arm.label!r}: missing {tum}, which this export needs "
            f"because it names the {world_source!r} world source. {producer} "
            f"An export never falls back to the other world source - they are "
            f"different frames."
        )
    return Path(tum)


def _map_set(arm: ArmSource, world_source: str) -> dict | None:
    """The frozen map set this arm's world trajectory was solved against.

    Under the markerless source the world is a SCENE frame, and what makes
    two episodes' poses comparable is not the frame NAME - one constant
    string for every scene there is - but the map set that froze it. The
    state stage writes that set into ``state.inputs.json`` beside every
    ``state.tum``; a trajectory without its sidecar is refused.

    ``None`` under the marker source: a marker world is per-episode and has
    no map set to compare.
    """
    if world_source != "markerless":
        return None
    sidecar = arm.state_inputs
    if sidecar is None or not Path(sidecar).is_file():
        raise FileNotFoundError(
            f"arm {arm.label!r}: missing {sidecar or '<arm>/scene/state.inputs.json'}, "
            f"the state stage's sidecar beside scene/state.tum. It names the "
            f"map set the trajectory was solved against, which is what "
            f"identifies the world these poses live in - the frame name alone "
            f"is the same constant for every scene. Re-run the state stage "
            f"(`kiwi run --profile global-state`) to write it."
        )
    doc = json.loads(Path(sidecar).read_text())
    got = doc.get("map_set")
    if not isinstance(got, dict) or not got:
        raise ValueError(
            f"{sidecar}: no `map_set` block, so the world this trajectory "
            f"lives in cannot be identified. Re-run the state stage to "
            f"rewrite the sidecar."
        )
    return got


def _map_set_digest(map_set: dict | None) -> str | None:
    """A stable short name for a map set: sha256 over its canonical JSON."""
    if map_set is None:
        return None
    canonical = json.dumps(map_set, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _mount_transforms(desc, camera_override=None, tcp_override=None):
    """Resolve calibrated legs without inventing a missing camera frame."""

    def leg(role):
        links = {}
        for required in ("default", role):
            ee = desc.end_effectors.get(required)
            if ee is None or ee.urdf_link is None:
                if required == "camera":
                    raise ValueError(
                        f"description {desc.name!r}: camera calibration is pending; "
                        "add the calibrated camera_lens0 link to the source URDF "
                        "and run `python -m kiwi.replay.generate`, or supply the "
                        "physical unit's measured build.mount_cam transform. "
                        "No camera-to-tip transform can be inferred from the mesh."
                    )
                raise ValueError(
                    f"description {desc.name!r} declares no {required!r} end-effector frame"
                )
            links[required] = ee.urdf_link
        return fixed_frame_T(desc.urdf_path, links["default"], links[role])

    camera = (
        load_T_mount_cam(camera_override)
        if camera_override is not None
        else leg("camera")
    )
    tcp = load_T_mount_tcp(tcp_override) if tcp_override is not None else leg("tcp")
    return camera, tcp


def _build_arm(
    arm: ArmSource,
    *,
    include_camera_pose: bool,
    repo_root: Path | None,
    world_source: str = "markerless",
    world_offset: tuple[float, float, float] | None = None,
    world_yaw: float | None = None,
) -> _ArmData:
    label = arm.label
    # None reaches here only for a single-stream episode; build_episode refuses
    # it above when a second stream makes the offset load-bearing.
    sync_off = 0.0 if arm.sync_off is None else float(arm.sync_off)
    world_tum = _world_tum(arm, world_source)
    map_set = _map_set(arm, world_source)
    header = read_world_tum_header(world_tum)
    t_w, p, q_xyzw, _ = load_tum(world_tum)
    t_world_cam = se3_stack(p, q_xyzw)
    t_world = world_transform(world_offset, world_yaw)
    if t_world is not None:
        # World-frame re-seating baked into the export, left-multiplied so the
        # EE track and the optional camera-pose track move together. ROTATE
        # THEN TRANSLATE (`Translate(offset) @ Rz(yaw)`) - the order `kiwi
        # replay` composes the same two knobs in, so a baked dataset replays at
        # yaw 0 / workspace 0.
        t_world_cam = t_world @ t_world_cam

    if arm.description is None:
        raise ValueError(
            f"arm {label!r}: tool {arm.tool!r} names no robot description (add "
            f"`capture.description:` to the `design:` block of "
            f"config/gripper/{arm.tool}/tool.yaml); export composes the "
            f"camera->tip "
            f"constant from the description URDF"
        )
    desc = load_robot_description(arm.description)
    if desc.urdf_path is None:
        raise ValueError(
            f"arm {label!r}: robot description {arm.description!r} registers no "
            f"URDF, so the camera->tip constant cannot be composed from it"
        )
    # Resolve only the legs not supplied by explicit per-build calibration.
    cam_overridden = arm.mount_cam_override is not None
    tcp_overridden = arm.mount_tcp_override is not None
    t_mount_cam, t_mount_tcp = _mount_transforms(
        desc, arm.mount_cam_override, arm.mount_tcp_override
    )
    # A description leg is a design fact; only an override block may declare
    # itself a placeholder, and one unmeasured leg makes the composite one.
    cam_measured = (
        bool(arm.mount_cam_override.get("measured", True)) if cam_overridden else True
    )
    tcp_measured = (
        bool(arm.mount_tcp_override.get("measured", True)) if tcp_overridden else True
    )
    measured = cam_measured and tcp_measured
    t_camera_tcp = np.linalg.inv(t_mount_cam) @ t_mount_tcp
    body = t_camera_tcp
    tcp_meta = {
        "measured": measured,
        "description": arm.description,
        "T_camera_tcp": _matrix(t_camera_tcp),
        "urdf": stamp_input(desc.urdf_path, root=repo_root),
        "t_camera_tcp_xyz": [float(v) for v in t_camera_tcp[:3, 3]],
    }
    extrinsics: dict = {
        "description": {
            "name": arm.description,
            "urdf": stamp_input(desc.urdf_path, root=repo_root),
            "meta": stamp_input(desc.meta_path, root=repo_root),
        },
        "T_mount_cam": {
            "matrix": _matrix(t_mount_cam),
            "overridden": cam_overridden,
        },
        "T_mount_tcp": {
            "matrix": _matrix(t_mount_tcp),
            "overridden": tcp_overridden,
            "measured": tcp_measured,
        },
        "T_camera_tcp": {"matrix": _matrix(t_camera_tcp)},
    }
    if cam_overridden or tcp_overridden:
        extrinsics["override"] = stamp_input(arm.override_source, root=repo_root)
    # The single constant actually applied to the trajectory:
    #     T_world_tcp(t) = T_world_camera(t) @ T_camera_ee
    extrinsics["T_camera_ee"] = {"matrix": _matrix(body)}
    t_world_ee = t_world_cam @ body

    # The tum's timestamps ARE the global clock (its header just proved it).
    ee_track = _posetrack_from_se3(t_w, t_world_ee)
    cam_track = None
    if include_camera_pose:
        cam_track = _posetrack_from_se3(t_w, t_world_cam)

    if not arm.lens0.is_file():
        raise FileNotFoundError(
            f"arm {label!r}: missing {arm.lens0}. The convert stage demuxes "
            f"the .insv into per-lens mp4s (`kiwi run --to-stage convert`)."
        )
    info = VideoInfo.read(arm.lens0)

    state = read_eef_state(arm.eef_state)
    bundle = get_estimator(state.estimator, source=str(arm.eef_state))
    # The bundle owns endpoint semantics AND their validation (missing keys,
    # degenerate closed == open) - see e.g. kiwi.eef.chopsticks.closed_open.
    closed, opened = bundle.closed_open(arm.session_block)
    # The completed raw series is finite on every source frame (the artifact
    # contract), so its support is the whole clip. Export samples it without
    # smoothing.
    grip_v = np.clip((state.output - closed) / (opened - closed), 0.0, 1.0)
    n_grip = int(grip_v.shape[0])
    tool_meta = {
        "estimator": state.estimator,
        "unit": state.unit,
        "source": arm.eef_state.name,
        "endpoints": [float(closed), float(opened)],
        "output_channel": "value_completed",
        "completion": {
            key: json_value(value)
            for key, value in state.completion.items()
            if np.ndim(value) == 0
        },
    }

    return _ArmData(
        label=label,
        out_label=arm.out_label,
        frame=header.frame,
        map_set=map_set,
        scene_frame_digest=(
            (
                json.loads(Path(arm.state_inputs).read_text()).get("world_scene") or {}
            ).get("scene_frame_digest")
            if world_source == "markerless" and arm.state_inputs
            else None
        ),
        ee_track=ee_track,
        cam_track=cam_track,
        grip_v=np.asarray(grip_v, dtype=np.float64),
        grip_fps=float(state.fps),
        grip_t0=sync_off,
        grip_t1=sync_off + float(n_grip - 1) / float(state.fps),
        lens0=arm.lens0,
        eef_state=arm.eef_state,
        sync_off=sync_off,
        fps=info.fps,
        n_video=info.n_frames,
        tool_meta=tool_meta,
        tool_completion=state.completion,
        tcp_meta=tcp_meta,
        extrinsics=extrinsics,
        intrinsics=(
            _intrinsics_block(arm.camera, arm.intrinsics, arm.intrinsics_kb4, repo_root)
            if arm.intrinsics is not None
            else None
        ),
    )


def _sample_tool_state(a: _ArmData, grid: np.ndarray) -> np.ndarray:
    """Tool state at each grid instant - nearest source frame, zero interpolation.

    The index mapping is the video sampler's (``round((t - off) * fps)``), so
    the tool value and the pixels at a grid instant come from the same source
    frame. The completed series is finite on every source frame by the
    artifact contract, so a non-finite sample here is a broken artifact, not
    a dropout to bridge: it raises, naming the arm and the source file. The
    sampled source indices are written onto the arm (not returned), so the
    return value stays the sampled array; build_episode carries the completion
    audit along them.
    """
    v = a.grip_v
    n = v.shape[0]
    pos = (grid - a.sync_off) * a.grip_fps
    idx = np.clip(np.round(pos).astype(np.int64), 0, n - 1)
    out = v[idx]
    a.tool_frame_indices = idx
    if not np.isfinite(out).all():
        bad = np.flatnonzero(~np.isfinite(out))
        raise ValueError(
            f"arm {a.label!r}: tool state is non-finite at {bad.size} grid "
            f"instant(s) inside the export window (first at {grid[bad[0]]:.3f} s "
            f"on the global clock; source {a.eef_state.name}). Completed output "
            f"is finite on every frame by contract, so this artifact is broken; "
            f"re-run the eef stage to regenerate it. The export never "
            f"interpolates tool state."
        )
    return out.astype(np.float32)


def _reject_duplicate_key(features: dict, key: str) -> None:
    """Guard against two streams resolving to the same export feature key.

    Raises, e.g. for ``--cameras left=head,right=head``, which maps both
    arms' videos to one key.
    """
    if key in features:
        raise ValueError(
            f"feature key {key!r} is used twice - two streams would collapse "
            f"into one. Check the --cameras map / `cameras.<label>.key` in "
            f"episode.yaml."
        )


@dataclasses.dataclass
class _CameraData:
    """An extra camera stream: a video and where its frame 0 sits on the clock.

    The camera-only analogue of :class:`_ArmData` - no pose track, no tool
    state.
    """

    label: str
    key: str  # export feature-key suffix
    video: Path
    sync_off: float
    fps: float
    n_video: int
    camera: str | None = None
    intrinsics: Path | None = None
    intrinsics_kb4: Path | None = None


def _build_camera(cam: CameraSource) -> _CameraData:
    if not cam.video.exists():
        raise FileNotFoundError(f"camera {cam.label!r}: missing {cam.video}")
    # Unlike an arm (where 0.0 is used for a single-arm episode whose clock
    # origin IS its frame 0), a camera offset has no default.
    if cam.sync_off is None:
        raise ValueError(
            f"camera {cam.label!r}: no sync offset in episode.yaml's `sync:` "
            f"block; run the sync stage (`kiwi run --to-stage sync`)"
        )
    info = VideoInfo.read(cam.video)
    return _CameraData(
        label=cam.label,
        key=cam.key or cam.label,
        video=cam.video,
        sync_off=float(cam.sync_off),
        fps=info.fps,
        n_video=info.n_frames,
        camera=cam.camera,
        intrinsics=cam.intrinsics,
        intrinsics_kb4=cam.intrinsics_kb4,
    )


def _overlap_window(
    arms: list[_ArmData], cams: "list[_CameraData] | tuple[()]" = ()
) -> tuple[float, float]:
    # Video end-time uses each arm's NATIVE fps (frame i -> global = sync_off
    # + i/native_fps), independent of the output grid fps. Pose bounds are
    # already global; the tool-state bounds are the eef clip's first and last
    # source frames. The completed series is finite on every frame, so a
    # filled lead-in or tail is exported, and export_samples.npz flags each
    # filled row rather than trimming it.
    lo = max(max(a.ee_track.t0, a.sync_off, a.grip_t0) for a in arms)
    hi = min(
        min(a.ee_track.t1, a.sync_off + (a.n_video - 1) / a.fps, a.grip_t1)
        for a in arms
    )
    # An extra camera bounds the window exactly as an arm's video does.
    for c in cams:
        lo = max(lo, c.sync_off)
        hi = min(hi, c.sync_off + (c.n_video - 1) / c.fps)
    return lo, hi


# --- the public builder ----------------------------------------------------


def build_episode(
    src: EpisodeSource,
    *,
    fps: float | None = None,
    window: tuple[float | None, float | None] | None = None,
    include_camera_pose: bool = False,
    include_head_marker: bool = False,
    include_ground: bool = False,
    include_extra_cameras: bool = True,
    include_relative_state: bool = False,
    include_pose_gap: bool = True,
    cameras: dict[str, str] | None = None,
    resize: tuple[int, int] | None = None,
    robot_type: str = "portable_bimanual",
    world_source: str = "markerless",
    world_offset: tuple[float, float, float] | None = None,
    world_yaw: float | None = None,
) -> ObservationEpisode:
    """Assemble one world-frame :class:`ObservationEpisode` from an episode's artifacts.

    The per-arm body frame is the tool tip, composed from the arm's robot
    description (see the module docstring). ``meta["ee_frame"]`` is the
    constant ``"tcp"``; the ``replay_state.npz`` key set is a stable format
    that tools outside this repository read, so a reader can refuse a foreign
    dump expressed at some other body.

    ``window`` is a global-clock ``(start_s, end_s)`` trim overriding the
    spec's ``task_window``. Either bound may be ``None``, meaning "do not trim
    that side" - the corresponding edge of the pose ∩ video ∩ tool-state
    overlap stands instead. Output timestamps re-base to 0 at the trimmed
    start.

    ``fps`` sets the output grid rate; each stream is still indexed at its OWN
    native rate, so passing ``29.97`` for a 119.88 fps capture decimates it
    4:1 rather than resampling. That is how mixed-rate episodes share one
    dataset.

    ``include_extra_cameras=False`` drops the spec's extra streams entirely -
    not just their features: an excluded stream must not keep bounding the
    overlap window. This is what lets an ego-equipped episode share a dataset
    with arms-only ones (a LeRobot dataset has ONE schema, so differing
    feature sets cannot mix).

    ``include_relative_state`` adds ``observation.state.relative``: each
    arm's pose re-expressed in its own first-window-frame frame
    (``inv(T_arm(t0)) @ T_arm(t)``) - the UMI-style episode-relative
    trajectory. The absolute ``observation.state`` is unchanged beside it.

    ``include_pose_gap`` (default on) adds ``observation.pose_gap.<arm>``:
    seconds from each grid instant to the arm's nearest raw pose sample (the
    raw number, not a thresholded verdict). The camera-pose feature samples
    the same world trajectory, so its gap is this gap.

    ``include_ground`` adds ``observation.ground_plane`` (the episode's floor,
    ``n . p + d = 0`` in the exported world) and ``observation.camera_height.
    <arm>`` (each arm's camera above that floor, sampled on the same grid from
    the same trajectory as the hands) from the optional ground stage's report;
    NaN, with ``meta["ground"]["status"]`` saying why, when no floor was
    computed.

    ``world_offset`` / ``world_yaw`` bake a world-frame re-seating into every
    exported pose (see :func:`world_transform`); both are recorded in ``meta``
    so the transform is auditable.
    """
    if not src.arms:
        raise ValueError(f"episode {src.name!r}: no arms in the export source")
    if window is None:
        window = src.task_window
    cameras = cameras or {}

    # An arm offset may default to 0.0 only for a single-stream episode, whose
    # clock origin IS its frame 0. With a second stream a missing entry is
    # refused, as in ``_build_camera`` below.
    n_streams = len(src.arms) + (len(src.cameras) if include_extra_cameras else 0)
    undated = [a.label for a in src.arms if a.sync_off is None]
    if undated and n_streams > 1:
        raise ValueError(
            f"episode {src.name!r}: no sync offset for arm(s) "
            f"{', '.join(repr(x) for x in undated)} in episode.yaml's `sync:` "
            f"block, and this episode has {n_streams} streams - a defaulted "
            f"offset would seat them at the global origin. Run the sync stage "
            f"(`kiwi run --to-stage sync`)."
        )

    if world_source not in WORLD_SOURCES:
        raise ValueError(
            f"world_source is {world_source!r}, expected one of {WORLD_SOURCES}"
        )
    if include_head_marker and world_source != "markerless":
        raise ValueError("head markers require the markerless scene world")
    if include_ground and world_source != "markerless":
        raise ValueError("the ground plane requires the markerless scene world")
    arms = [
        _build_arm(
            arm,
            # The camera track serves two optional consumers: its own pose
            # feature, and the camera height above the floor.
            include_camera_pose=include_camera_pose or include_ground,
            repo_root=src.repo_root,
            world_source=world_source,
            world_offset=world_offset,
            world_yaw=world_yaw,
        )
        for arm in src.arms
    ]
    frames = sorted({a.frame for a in arms})
    if len(frames) > 1:
        raise ValueError(
            f"episode {src.name!r}: arms disagree on the world frame "
            f"({frames}) - two arms in different worlds cannot share one "
            f"dataset. Re-run the stage that produces the {world_source!r} "
            f"world pose so both arms declare the same frame."
        )
    # The frame name is one constant for every scene; the arms must also
    # agree on the map-set digest.
    digests = sorted({_map_set_digest(a.map_set) or "" for a in arms})
    if len(digests) > 1:
        raise ValueError(
            f"episode {src.name!r}: arms disagree on the scene map set "
            f"(digests {[d[:12] for d in digests]}) - they were solved "
            f"against different maps, so their poses are in unrelated "
            f"coordinate systems despite sharing the frame name "
            f"{frames[0]!r}. Re-run the state stage for every arm against one "
            f"published map set."
        )
    placement_digests = {a.scene_frame_digest for a in arms if a.scene_frame_digest}
    if len(placement_digests) > 1:
        raise ValueError(
            "arms disagree on the scene_frame_digest; placements changed within the map set"
        )
    # Extra non-arm streams (e.g. a head-mounted ego view): video only.
    cams = [_build_camera(cam) for cam in src.cameras] if include_extra_cameras else []

    # The grid is defined by the ARMS; an extra camera never changes the rate.
    fps = float(fps) if fps is not None else arms[0].fps
    lo, hi = _overlap_window(arms, cams)
    narrowed_by_cams = 0.0
    if cams:
        lo_a, hi_a = _overlap_window(arms)  # cheap: pure float math
        narrowed_by_cams = max(0.0, lo - lo_a) + max(0.0, hi_a - hi)
    if window is not None:
        # Either bound may be None ("do not trim this side"), which leaves the
        # coverage bound computed above standing.
        if window[0] is not None:
            lo = max(lo, window[0])
        if window[1] is not None:
            hi = min(hi, window[1])
    if hi <= lo:
        raise ValueError(f"empty episode window [{lo:.3f}, {hi:.3f}] s")
    n = int(np.floor((hi - lo) * fps)) + 1
    grid = lo + np.arange(n) / fps
    timestamps = (grid - grid[0]).astype(np.float64)

    state_blocks: list[np.ndarray] = []
    state_names: list[str] = []
    intr_meta: dict[str, dict] = {}
    rel_blocks: list[np.ndarray] = []
    rel_names: list[str] = []
    features: dict[str, FeatureColumn] = {}
    samples = {"frame_index": np.arange(n, dtype=np.int64), "t_episode": grid.copy()}
    sample_streams = {}

    for a in arms:
        ee = _sample_state(a.ee_track, grid)  # (N, 7)
        grip = _sample_tool_state(a, grid)
        samples[f"eef_frame_index_{a.out_label}"] = a.tool_frame_indices
        for key, values in a.tool_completion.items():
            if np.ndim(values) == 1:
                samples[f"eef_completion_{key}_{a.out_label}"] = np.asarray(values)[
                    a.tool_frame_indices
                ]
        state_blocks.append(np.hstack([ee, grip[:, None]]))
        state_names += arm_state_names(a.out_label)
        if include_relative_state:
            rel_blocks.append(_relative_pose(ee))
            rel_names += arm_relative_names(a.out_label)

        # Map global grid time -> source frame index using the arm's NATIVE
        # fps (the grid fps only governs the output sample rate).
        idx = np.clip(
            np.round((grid - a.sync_off) * a.fps).astype(np.int64), 0, a.n_video - 1
        )
        samples[f"video_frame_index_{a.out_label}"] = idx.copy()
        sample_streams[a.out_label] = {
            "path": str(a.lens0),
            "fps": a.fps,
            "sync_offset_s": a.sync_off,
            "eef_path": str(a.eef_state),
            "eef_fps": a.grip_fps,
        }
        vs = VideoSource(a.lens0, idx, resize=resize)
        cam_key = (
            IMAGE_KEY_PREFIX + cameras[a.label]
            if a.label in cameras
            else wrist_image_key(a.out_label)
        )
        sample_streams[a.out_label]["feature_key"] = cam_key
        _reject_duplicate_key(features, cam_key)
        features[cam_key] = FeatureColumn(
            key=cam_key,
            dtype=VIDEO_DTYPE,
            shape=(vs.height, vs.width, 3),
            names=["height", "width", "channels"],
            data=vs,
        )
        if include_camera_pose and a.cam_track is not None:
            cam_state = _sample_state(a.cam_track, grid)
            key = camera_pose_key(a.out_label)
            features[key] = FeatureColumn(
                key=key,
                dtype="float32",
                shape=(7,),
                names=arm_state_names(a.out_label)[:7],
                data=cam_state,
            )
        if a.intrinsics is not None:
            # The model values describe calibration_size; `delivered` names
            # what the consumer actually receives, so a downloader knows to
            # rescale fx,fy,cx,cy by delivered/calibration_size - and that
            # the stream is the UNRECTIFIED front lens.
            intr_meta[cam_key] = {
                **a.intrinsics,
                "delivered": {
                    "width": vs.width,
                    "height": vs.height,
                    "rectified": False,
                },
            }
        if include_pose_gap:
            gap = _gap_to_raw(a.ee_track.t, grid).astype(np.float32)
            key = pose_gap_key(a.out_label)
            features[key] = FeatureColumn(
                key=key,
                dtype="float32",
                shape=(1,),
                names=[f"{a.out_label}_pose_gap_s"],
                data=gap[:, None],
            )

    # Extra cameras attach AFTER the arms, so the video-feature insertion
    # order (and hence meta/info.json) keeps left_wrist, right_wrist first.
    for c in cams:
        # Same model as an arm's video: a grid instant maps to a source frame
        # via the camera's NATIVE fps. The clip is a one-frame rounding guard:
        # _overlap_window already bounded the grid to this camera's coverage.
        idx = np.clip(
            np.round((grid - c.sync_off) * c.fps).astype(np.int64), 0, c.n_video - 1
        )
        samples[f"video_frame_index_{c.label}"] = idx.copy()
        sample_streams[c.label] = {
            "path": str(c.video),
            "fps": c.fps,
            "sync_offset_s": c.sync_off,
        }
        vs = VideoSource(c.video, idx, resize=resize)
        cam_key = IMAGE_KEY_PREFIX + cameras.get(c.label, c.key)
        sample_streams[c.label]["feature_key"] = cam_key
        _reject_duplicate_key(features, cam_key)
        if c.intrinsics is not None:
            intr_meta[cam_key] = {
                **_intrinsics_block(
                    c.camera, c.intrinsics, c.intrinsics_kb4, src.repo_root
                ),
                "delivered": {
                    "width": vs.width,
                    "height": vs.height,
                    "rectified": False,
                },
            }
        features[cam_key] = FeatureColumn(
            key=cam_key,
            dtype=VIDEO_DTYPE,
            shape=(vs.height, vs.width, 3),
            names=["height", "width", "channels"],
            data=vs,
        )

    state = np.hstack(state_blocks).astype(np.float32)
    features = {
        STATE_KEY: FeatureColumn(
            key=STATE_KEY,
            dtype="float32",
            shape=(state.shape[1],),
            names=state_names,
            data=state,
        ),
        **features,
    }
    if include_relative_state:
        rel = np.hstack(rel_blocks).astype(np.float32)
        features[RELATIVE_STATE_KEY] = FeatureColumn(
            key=RELATIVE_STATE_KEY,
            dtype="float32",
            shape=(rel.shape[1],),
            names=rel_names,
            data=rel,
        )

    meta: dict = {
        "robot_type": robot_type,
        "frame": frames[0],
        # Always the tool tip: what it names is what replay's solver drives.
        "ee_frame": "tcp",
        "source_episode": str(src.episode_dir),
        "arms": [a.out_label for a in arms],
        "global_t0": float(grid[0]),
        "window": [float(lo), float(hi)],
        # True when an explicit/spec trim narrowed the full overlap - surfaced
        # per-episode in export_provenance.json.
        "windowed": window is not None,
        "sync_offsets": {a.label: a.sync_off for a in arms}
        | {c.label: c.sync_off for c in cams},
        "units": "position m; quaternion xyzw; gripper [0,1]",
        # Per-arm tool-state provenance: which estimator produced it, in what
        # unit, and the closed/open endpoints the [0,1] normalization used.
        "tool_state": {a.out_label: a.tool_meta for a in arms},
    }
    if narrowed_by_cams > 1e-6:
        meta["window_narrowed_by_cameras_s"] = float(narrowed_by_cams)
    if cams:
        meta["cameras"] = {
            c.label: {
                "key": IMAGE_KEY_PREFIX + cameras.get(c.label, c.key),
                "source": str(c.video),
                "fps": c.fps,
                "n_frames": c.n_video,
                "sync_offset": c.sync_off,
            }
            for c in cams
        }
    # Per-arm TCP provenance so the export layer can surface an unmeasured
    # placeholder and cross-check the tip against a known value.
    meta["tcp"] = {a.out_label: a.tcp_meta for a in arms}
    # Full per-arm extrinsic chain (matrices + config-file stamps) - surfaced
    # into the dataset's meta/export_provenance.json by kiwi.export.write.
    meta["extrinsics"] = {a.out_label: a.extrinsics for a in arms}
    if intr_meta:
        # Keyed by the IMAGE FEATURE key, not the arm label: the model
        # describes a pixel stream (including --cameras renames and extra
        # streams).
        meta["intrinsics"] = intr_meta
    # How this episode's world pose was produced.
    meta["world_source"] = world_source
    # The episode's world IDENTITY: under the markerless source every scene
    # declares the same constant frame name. The scene name is the human half;
    # the map-set digest is the exact one (a re-built map set is a new world).
    # Both are None under the marker source, whose world is per-episode.
    meta["world_scene"] = {
        "frame": frames[0],
        "scene": src.scene if world_source == "markerless" else None,
        "map_set_digest": _map_set_digest(arms[0].map_set),
    }
    if all(a.scene_frame_digest for a in arms):
        meta["world_scene"]["scene_frame_digest"] = arms[0].scene_frame_digest
    samples["metadata"] = np.asarray(
        json.dumps(
            {
                "clock": "episode global clock",
                "streams": sample_streams,
                "video_sampling": "round((t_episode - sync_offset_s) * native_fps), clipped to video bounds",
                "eef_sampling": "nearest source frame; no interpolation",
            }
        )
    )
    meta["export_samples"] = samples
    if include_head_marker:
        from .head import sample_head

        head_features, head_meta = sample_head(
            src.head,
            grid,
            world_scene=meta["world_scene"],
            transform=world_transform(world_offset, world_yaw),
        )
        features.update(head_features)
        meta["head_marker"] = head_meta
    if include_ground:
        from .ground import sample_ground

        ground_features, ground_meta = sample_ground(
            src.ground,
            grid,
            {a.out_label: _sample_state(a.cam_track, grid)[:, :3] for a in arms},
            world_scene=meta["world_scene"],
            transform=world_transform(world_offset, world_yaw),
        )
        features.update(ground_features)
        meta["ground"] = ground_meta
    if world_offset is not None:
        # World-frame translation baked into every exported position
        # (p -> Rz(world_yaw) p + world_offset): the source world origin sits
        # at +world_offset in the exported coordinates.
        meta["world_offset"] = [float(v) for v in world_offset]
    if world_yaw is not None:
        # Applied BEFORE world_offset (see world_transform), so the offset
        # above is expressed in the rotated frame - the pair is not
        # commutative and a reader recomposing them has to know the order.
        meta["world_yaw"] = float(world_yaw)
    episode = ObservationEpisode(
        fps=fps,
        timestamps=timestamps,
        task=src.task,
        frame=frames[0],
        features=features,
        meta=meta,
    )
    episode.validate()
    return episode

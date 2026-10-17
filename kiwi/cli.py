"""The ``kiwi`` command: one verb per pipeline surface.

Verbs: ``import`` / ``init`` / ``run`` / ``viz`` /
``export`` / ``replay`` / ``calib`` / ``sync`` / ``register`` / ``eef`` /
``scene`` / ``models`` / ``gui`` / ``real2simviz``.
"""

import json
import subprocess
import sys
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import tyro

from kiwi.episode import init as init_mod
from kiwi.episode import stages


def _warn_missing_head_marker(episode_dir: Path) -> bool:
    """Warn when an episode cannot produce the optional head trajectory."""
    import yaml

    from kiwi.episode.manifest import MANIFEST_NAME

    episode_dir = Path(episode_dir)
    manifest = episode_dir / MANIFEST_NAME
    try:
        raw = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(raw, dict):
        return False
    cameras = raw.get("cameras")
    if cameras is None:
        cameras = {}
    if not isinstance(cameras, dict):
        return False
    if any(
        isinstance(camera, dict) and camera.get("head_marker") is not None
        for camera in cameras.values()
    ):
        return False
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        name = episode_dir.name
    print(
        f"warning: {name}: no cameras.*.head_marker config; head estimation "
        "is disabled and no head/head_marker.tum will be produced. Configure "
        "--head-marker-id during `kiwi init` if head output is expected.",
        file=sys.stderr,
    )
    return True


def _warn_scene_missing_head_markers(scene) -> None:
    for episode_id in sorted(scene.episodes):
        _warn_missing_head_marker(scene.episode_dir(episode_id))


@dataclass
class Run:
    """Run the offline stage pipeline over one episode directory."""

    episode: Path
    from_stage: str | None = None  # inclusive; e.g. --from-stage state
    to_stage: str | None = None  # inclusive; e.g. --to-stage sync
    resync: bool = False
    """Recompute and replace saved sync offsets; ordinary runs reuse them.
    Manual task and curation windows stay unchanged."""
    profile: str | None = None
    """Pipeline profile NAME (default: 'global-state': sync, convert, state,
    eef). 'prepare' runs preprocessing before scene maps are built.
    'global-state-head', 'global-state-ground', and 'global-state-head-ground'
    also request the corresponding optional estimators. Register remains an
    on-demand ground-truth anchor: --from-stage register --to-stage register."""

    def run(self) -> int:
        from kiwi.episode.pipeline_profile import PROFILES

        profile = None
        if self.profile is not None:
            profile = PROFILES.get(self.profile)
            if profile is None:
                print(
                    f"kiwi run: unknown profile {self.profile!r}; built-in "
                    f"profiles are {sorted(PROFILES)}",
                    file=sys.stderr,
                )
                return 2

        _warn_missing_head_marker(self.episode)

        def show(r: stages.StageResult) -> None:
            mark = "ok" if r.ok else "FAIL"
            # A refusal carries its diagnostics, which run to several lines;
            # indent them under the stage line rather than truncating.
            head, *rest = r.summary.splitlines() or [""]
            print(f"  [{mark}] {r.stage}: {head}")
            for line in rest:
                print(f"          {line}")

        try:
            results = stages.run_pipeline(
                self.episode,
                self.from_stage,
                self.to_stage,
                on_result=show,
                resync=self.resync,
                profile=profile,
            )
        except ValueError as e:  # bad --from-stage/--to-stage
            print(f"kiwi run: {e}", file=sys.stderr)
            return 2
        if results and not results[-1].ok:
            print(
                f"kiwi run: halted at '{results[-1].stage}'; the stages after it "
                f"build on what it produces.",
                file=sys.stderr,
            )
            return 1
        return 0


@dataclass
class Import:
    """Copy clips off a USB-connected camera into a directory (browser picker).

    Clips are named after the camera-registry unit the clip's telemetry serial
    resolves to, so nothing here needs to be told which arm it is looking at -
    that is a property of the sitting, and ``kiwi init`` is told it once. One
    invocation reads one body; run it again for the other, and both bodies'
    clips land side by side.

        kiwi import --dest data/scratch             # choose from the last 30
        kiwi import --dest data/scratch --no-pick   # newest clip, no questions
    """

    dest: Path
    """Directory the clips land in. Import writes no episode structure - the
    clips arrive side by side, named by unit, and sorting them into episodes
    (each one's clips flat in <episode>/raw/) is yours to do before kiwi init
    reads one."""
    pick: bool = True
    """Choose clips on a page at http://localhost:<port>, opened in the
    browser: the candidates newest first beside a video of the highlighted
    one (its LRV proxy). Arrows move, Space checks, a all/none, i imports, q
    cancels; a Left / Right / Ego switch tags the filenames _left / _right /
    _ego_video in place of the unit. The URL goes to stdout, so a run whose
    stdout is not a terminal takes the newest clip instead. Pass --no-pick to
    take the newest clip without asking."""
    port: int = 8765
    """Port the picker's page is served on (127.0.0.1 only). Over ssh, forward
    it: ssh -L 8765:localhost:8765 <host>."""
    open_browser: bool = True
    """Open the page in the local browser. --no-open-browser only prints the
    URL - the ssh case, where the browser is on the other end."""
    latest: int | None = None
    """Newest N clips to consider (default 30, or 1 with --no-pick)."""
    mount: Path | None = None
    """Camera mount point, or any folder of clips (a card's DCIM/, a copy of
    a card). When omitted, auto-detects any /media or /run/media mount
    containing DCIM/Camera01/ (the volume label is not always set); with none
    or two present the page opens on a folder browser to choose from, and a
    run without a terminal refuses."""
    label: str | None = None
    """Tag the filename with this instead of the resolved unit. The tag is
    what names an extra stream at init time - so a recording that carries no
    serial (the GO3 ego mp4) wants one, and ``kiwi init --camera
    <label>=<unit>`` refers to it."""
    unit: str | None = None
    """Registry unit to tag the filenames with instead of probing the serial,
    for a recording that carries none (the GO3 ego mp4): --unit
    insta360-GO3-001. The name is not checked against the registry, and
    --label (or the page's role switch) replaces it when given."""
    name: str | None = None
    """New basename for the copied clip; with several they are numbered
    <name>_01, <name>_02, ... in chronological order."""
    lrv: bool = False
    """Also copy each clip's low-res LRV_*.lrv proxy."""
    overwrite: bool = False
    """Re-copy over an existing destination file that differs in size."""
    dry_run: bool = False
    """Print the copy plan without writing anything."""

    def run(self) -> int:
        from kiwi.core.camera import ingest as ingest_mod
        from kiwi.core.paths import config_root

        try:
            result = ingest_mod.ingest(
                self.dest,
                config_root() / "cameras",
                mount=self.mount,
                latest=self.latest,
                pick=self.pick,
                label=self.label,
                unit=self.unit,
                name=self.name,
                lrv=self.lrv,
                overwrite=self.overwrite,
                dry_run=self.dry_run,
                port=self.port,
                open_browser=self.open_browser,
            )
        except ingest_mod.IngestError as e:
            print(f"kiwi import: {e}", file=sys.stderr)
            return 2
        print(f"next: {result.next_step}")
        return 0


@dataclass
class Init:
    """Probe an episode's raw/ recordings and write an episode.yaml skeleton.

    Recordings live flat in <episode>/raw/, as the operator sorted them there
    after `kiwi import`. Arms are bound to cameras by the clip's embedded
    serial, never by its filename. Which arm a body rides is a property of
    the sitting and is named once here. With ``--root``, one assignment
    applies to every episode under a directory.
    """

    left: str
    """Registry unit under config/cameras/ that recorded the LEFT arm."""
    right: str
    """Registry unit that recorded the RIGHT arm."""
    episode: Path | None = None
    """One episode directory."""
    root: Path | None = None
    """A directory of episode directories - one sitting."""
    tool: str | None = None
    """Tool design (config/gripper/<tool>/tool.yaml); written as TODO when omitted."""
    camera: tuple[str, ...] = ()
    """Name an extra stream's unit: --camera ego_video=insta360-GO3-001. The
    label is the filename's _<tag> suffix (`kiwi import --label`). Sync
    refuses an extra stream with no unit, since it must establish every
    camera's model and A/V delay. A .insv clip whose serial is registered gets
    its unit automatically, so this is needed only for a clip without one,
    such as a GO3 .mp4."""
    head_marker_id: int | None = None
    """ArUco ID on the optional head rig. Written under the head camera."""
    head_camera: str | None = None
    """Extra-stream label carrying the head rig; inferred when exactly one exists."""
    head_marker_dictionary: str = init_mod.DEFAULT_HEAD_MARKER_DICTIONARY
    """ArUco dictionary for --head-marker-id."""
    head_marker_size: float = init_mod.DEFAULT_HEAD_MARKER_SIZE_M
    """Black-square edge of --head-marker-id in metres."""
    force: bool = False
    """Overwrite an existing episode.yaml, discarding its hand-written context."""

    def run(self) -> int:
        if (self.episode is None) == (self.root is None):
            print(
                "kiwi init: pass exactly one of --episode <dir> or --root <dir>",
                file=sys.stderr,
            )
            return 2
        if self.episode is not None:
            targets = [self.episode]
        else:
            targets = init_mod.find_episode_dirs(self.root)
            if not targets:
                print(
                    f"kiwi init: no episode directories under {self.root}",
                    file=sys.stderr,
                )
                return 2

        extra_units: dict[str, str] = {}
        for spec in self.camera:
            label, _, unit = spec.partition("=")
            if not label or not unit:
                print(
                    f"kiwi init: --camera wants <label>=<unit>, got {spec!r}",
                    file=sys.stderr,
                )
                return 2
            extra_units[label] = unit

        failed = 0
        for ep in targets:
            out, problem = init_mod.init_episode(
                ep,
                left=self.left,
                right=self.right,
                tool=self.tool,
                extra_units=extra_units,
                head_marker_id=self.head_marker_id,
                head_camera=self.head_camera,
                head_marker_dictionary=self.head_marker_dictionary,
                head_marker_size=self.head_marker_size,
                force=self.force,
            )
            if problem:
                failed += 1
                print(f"  [FAIL] {ep.name}: {problem}", file=sys.stderr)
            else:
                print(f"  [ok] {ep.name}: wrote {out.name}")
                _warn_missing_head_marker(ep)
        if failed:
            print(
                f"kiwi init: {failed} of {len(targets)} episode(s) not written",
                file=sys.stderr,
            )
            return 1
        return 0


def _parse_camera_map(spec: str | None) -> dict[str, str]:
    """``"left=head,right=wrist"`` -> ``{"left": "head", "right": "wrist"}``."""
    if not spec:
        return {}
    out: dict[str, str] = {}
    for part in spec.split(","):
        label, _, key = part.partition("=")
        if not key:
            raise ValueError(f"--cameras entry {part!r} must be label=key")
        out[label.strip()] = key.strip()
    return out


def _scene_name(episode_dir: Path) -> str | None:
    """The name of the scene this episode was recorded in, or None.

    Read straight off ``episode.yaml``'s ``scene:`` back-reference and the
    scene manifest it points at, as plain YAML without ``load_scene``: a
    scene whose profile digest drifted or whose sibling episode moved still
    yields its name. The `episode:` id in the back-reference is
    NOT the scene identity - it names this episode WITHIN the scene - so only
    the manifest side is read; the manifest's own path stem stands in when
    the file is unreadable or names nothing.
    """
    import yaml as _yaml

    from kiwi.episode.manifest import MANIFEST_NAME

    try:
        raw = _yaml.safe_load((episode_dir / MANIFEST_NAME).read_text()) or {}
    except (OSError, _yaml.YAMLError):
        return None
    back = raw.get("scene") if isinstance(raw, dict) else None
    if not isinstance(back, dict) or not back.get("manifest"):
        return None
    target = (episode_dir / str(back["manifest"])).resolve()
    try:
        doc = _yaml.safe_load(target.read_text()) or {}
    except (OSError, _yaml.YAMLError):
        return target.stem
    name = doc.get("name") if isinstance(doc, dict) else None
    return str(name) if name else target.stem


def _export_source(episode_dir: Path, *, task: str | None = None):
    """Flatten one episode's manifest into kiwi.export's plain input spec.

    This is the composition root the DAG demands: kiwi.export may not import
    kiwi.episode (composers never import each other), so the CLI - the one
    module allowed to import both - loads the manifest ONCE and hands over
    resolved paths, floats and dicts. Precedence lives here and nowhere else.
    That includes the SCENE NAME: it lives one hop away, in the scene manifest
    the episode's `scene:` back-reference points at, and flattening it here is
    what lets the builder state an episode's world identity without ever
    reaching into kiwi.episode.
    """
    from kiwi.core import layout
    from kiwi.core.camera.registry import CameraUnit
    from kiwi.core.paths import repo_root
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig
    from kiwi.export.source import (
        ArmSource,
        CameraSource,
        EpisodeSource,
        GroundSource,
        HeadSource,
        register_tum_path,
        state_tum_path,
    )

    episode_dir = Path(episode_dir)
    cfg = EpisodeConfig.load(episode_dir / MANIFEST_NAME)
    sync = cfg.sync_offsets or {}
    scene = _scene_name(episode_dir)
    arms = []
    for label, arm in cfg.arms.items():
        # The camera->tip constant's two inputs ride the ArmSpec already: the
        # robot description the tool DESIGN declares (the default legs, by
        # fixed-joint FK) and the gripper unit's build: mount blocks (the
        # per-build overrides). A missing description stays None here - the
        # refusal, with the actionable message, belongs to the builder, which
        # is where every other geometry input is refused.
        override_source = (
            arm.gripper_unit_config
            if arm.mount_tcp is not None or arm.mount_cam is not None
            else None
        )
        intrinsics = intrinsics_kb4 = None
        if arm.camera_dir is not None:
            unit = CameraUnit.load(arm.camera_dir)
            # The camera model itself: resolved exactly as the
            # pipeline resolves it, embedded into the export's provenance.
            # A broken pointer reads as absent HERE so the refusal happens
            # at build_and_write with its actionable message, not as a
            # FileNotFoundError mid-flatten. A mode-keyed pointer (no single
            # active file) reads as absent the same way.
            try:
                intrinsics = unit.artifact("intrinsics")
            except (FileNotFoundError, ValueError):
                intrinsics = None
            try:
                intrinsics_kb4 = unit.artifact("intrinsics_kb4")
            except FileNotFoundError:
                intrinsics_kb4 = None
        sdir = layout.stream_dir(episode_dir, label)
        lens0 = arm.recording
        if lens0.suffix.lower() == ".insv":
            lens0 = sdir / layout.LENS0
        arms.append(
            ArmSource(
                label=label,
                lens0=lens0,
                state_tum=state_tum_path(episode_dir, label),
                state_inputs=layout.scene_dir(episode_dir, label) / layout.STATE_INPUTS,
                register_tum=register_tum_path(episode_dir, label),
                eef_state=sdir / layout.EEF_STATE,
                sync_off=float(sync[label]) if label in sync else None,
                session_block=dict(arm.session or {}),
                tool=arm.tool,
                description=arm.description,
                mount_tcp_override=arm.mount_tcp,
                mount_cam_override=arm.mount_cam,
                override_source=override_source,
                camera=arm.camera,
                intrinsics=intrinsics,
                intrinsics_kb4=intrinsics_kb4,
            )
        )
    cameras = []
    for label, cam in cfg.cameras.items():
        cam_intr = cam_kb4 = None
        if cam.camera_dir is not None:
            # Best-effort for extra streams (arms are the load-bearing ones,
            # enforced in build_and_write): a unit dir with no camera.yaml or
            # broken pointers records nothing rather than failing the export.
            try:
                cam_unit = CameraUnit.load(cam.camera_dir)
                cam_intr = cam_unit.artifact("intrinsics")
                cam_kb4 = cam_unit.artifact("intrinsics_kb4")
            except (FileNotFoundError, KeyError, ValueError, OSError):
                pass
        cameras.append(
            CameraSource(
                label=label,
                video=cam.recording,
                sync_off=float(sync[label]) if label in sync else None,
                key=cam.key,
                camera=cam.camera,
                intrinsics=cam_intr,
                intrinsics_kb4=cam_kb4,
            )
        )
    cameras = tuple(cameras)
    if task is None:
        desc = (cfg.meta or {}).get("description") or cfg.name
        task = str(desc).strip().splitlines()[0]
    return EpisodeSource(
        name=cfg.name,
        episode_dir=episode_dir,
        arms=tuple(arms),
        task=task,
        cameras=cameras,
        task_window=cfg.task_window,
        repo_root=repo_root(),
        scene=scene,
        head=HeadSource(
            trajectory=episode_dir / layout.HEAD_DIR / layout.HEAD_TUM,
            report=episode_dir / layout.HEAD_DIR / layout.HEAD_REPORT,
            diagnostics=episode_dir / layout.HEAD_DIR / "head_marker_diagnostics.npz",
        ),
        ground=GroundSource(
            report=episode_dir / layout.GROUND_DIR / layout.GROUND_REPORT,
            observations=episode_dir / layout.GROUND_DIR / layout.GROUND_OBSERVATIONS,
        ),
    )


@dataclass
class ExportRun:
    """Build one training dataset from processed episodes.

    Formats: `v3.0` (LeRobot v3 native, the default), `v2.1` (LeRobot v2.1;
    add --groot for Isaac-GR00T's modality.json), `umi-zarr` (the
    Diffusion-Policy replay buffer). Each episode dir needs the pipeline's
    artifacts in the arms' stream directories: `<arm>/scene/state.tum`, the
    `<arm>/eef.npz` tool state, and the demuxed `<arm>/lens0.mp4`. Only
    `--world-source marker` reads `<arm>/register.tum` instead.
    """

    output: Path
    episodes: list[Path]
    format: str = "v3.0"
    repo_id: str = "local/dataset"
    robot_type: str = "portable_bimanual"
    task: str | None = None
    window: tuple[float, float] | None = None
    curation: Path | None = None
    include_camera_pose: bool = False
    include_head_marker: bool = False
    """Include the optional head marker pose, validity and continuous segment.
    Available for LeRobot v2.1/v3 in the markerless scene world."""
    include_ground: bool = False
    """Include the optional floor plane (`observation.ground_plane`) and each
    arm's camera height above it (`observation.camera_height.<arm>`) from the
    ground stage's report; NaN where no floor was computed. Available for
    LeRobot v2.1/v3 in the markerless scene world."""
    include_extra_cameras: bool = True
    relative_state: bool = False
    pose_gap: bool = True
    """Per-arm `observation.pose_gap.<arm>`: seconds from each frame to the
    nearest raw pose sample, so an interpolated pose is distinguishable from
    a measured one. On by default; --no-pose-gap drops the column."""
    action: str = "next_state"
    cameras: str | None = None
    resize: tuple[int, int] | None = None
    fps: float | None = None
    world_source: str = "markerless"
    """Which world pose to export: markerless (the default - the state
    stage's scene frame, shared by every episode of one scene) or marker (the
    register stage's per-episode ArUco anchor, for ground-truth work).
    One dataset carries one; an export never falls back to the other."""
    world_offset: tuple[float, float, float] | None = None
    world_yaw: float | None = None
    groot: bool = False

    def run(self) -> int:
        from kiwi.export.write import build_and_write

        if not self.episodes:
            print(
                "error: pass at least one episode dir via --episodes", file=sys.stderr
            )
            return 1
        for episode in self.episodes:
            _warn_missing_head_marker(episode)
        try:
            sources = [_export_source(ep, task=self.task) for ep in self.episodes]
            result = build_and_write(
                sources,
                self.output,
                format=self.format,
                repo_id=self.repo_id,
                robot_type=self.robot_type,
                window=self.window,
                curation=self.curation,
                include_camera_pose=self.include_camera_pose,
                include_head_marker=self.include_head_marker,
                include_ground=self.include_ground,
                include_extra_cameras=self.include_extra_cameras,
                include_relative_state=self.relative_state,
                include_pose_gap=self.pose_gap,
                action=self.action,
                cameras=_parse_camera_map(self.cameras),
                resize=self.resize,
                fps=self.fps,
                world_source=self.world_source,
                world_offset=self.world_offset,
                world_yaw=self.world_yaw,
                groot=self.groot,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        for skip in result.skipped:
            print(f"skipping {skip['source']}: {skip['reason']}")
        if result.card_error is not None:
            print(
                f"warning: could not write the README dataset card "
                f"({result.card_error}); regenerate with `kiwi export card`",
                file=sys.stderr,
            )
        print(
            f"wrote {result.format} dataset ({result.episodes} episodes) to {result.out}"
        )
        return 0


@dataclass
class ExportCard:
    """(Re)render the HF dataset card from an export's provenance sidecar."""

    dataset: Path
    repo_id: str | None = None

    def run(self) -> int:
        from kiwi.export.card import write_card

        try:
            p = write_card(self.dataset, repo_id=self.repo_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"wrote {p}")
        return 0


@dataclass
class ExportState:
    """Dump one episode's EE state to replay_state.npz (the kiwi.replay seam)."""

    episode: Path
    output: Path | None = None
    fps: float | None = None
    include_head_marker: bool = False
    world_source: str = "markerless"
    """Which world pose to dump: markerless (the default - the state stage's
    scene frame) or marker (the register stage's ArUco anchor)."""
    assert_tcp_xyz: tuple[float, float, float] | None = None
    assert_tol: float = 0.01

    def run(self) -> int:
        from kiwi.core import layout
        from kiwi.export.build import build_episode
        from kiwi.export.replay_state import write_replay_state

        _warn_missing_head_marker(self.episode)
        try:
            if self.include_head_marker:
                raise ValueError(
                    "replay state does not support head markers; use LeRobot export"
                )
            src = _export_source(self.episode)
            oe = build_episode(src, fps=self.fps, world_source=self.world_source)
            result = write_replay_state(
                oe,
                self.output or (self.episode / layout.REPLAY_STATE),
                assert_tcp_xyz=self.assert_tcp_xyz,
                assert_tol=self.assert_tol,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if result.warning:
            print(f"warning: {result.warning}", file=sys.stderr)
        print(
            f"{result.frames} frames x {result.dims} dims "
            f"(ee_frame=tcp) -> {result.path}"
        )
        return 0


@dataclass
class Export:
    """Export processed episodes to training-dataset formats.

    `run` writes LeRobot v3 natively (plus the v2.1/GR00T and UMI-zarr side
    targets), `card` re-renders an export's HF README, and `state` dumps the
    EE-state npz the replay stack consumes.
    """

    command: Annotated[
        (
            Annotated[ExportRun, tyro.conf.subcommand("run", prefix_name=False)]
            | Annotated[ExportCard, tyro.conf.subcommand("card", prefix_name=False)]
            | Annotated[ExportState, tyro.conf.subcommand("state", prefix_name=False)]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


def _ensure_replay_state(
    episode: Path, *, rebuild: bool, world_source: str = "markerless"
) -> None:
    """Make sure ``<episode>/replay_state.npz`` exists and is fresh.

    Composition root: building the npz runs the export pipeline
    (``kiwi.export``), and consuming it is ``kiwi.replay`` - composers never
    import each other, so the CLI glues them. A cached npz older than any of
    its actual inputs is rebuilt: the manifest and the per-arm ``eef.npz``
    plus the world trajectory the NAMED ``world_source`` actually reads -
    ``scene/state.tum`` or ``register.tum`` - which the EE stream is derived
    from (a re-run eef, state or register stage does not touch the
    manifest). When the rebuild fails but a cached npz exists, the cache is
    used with a warning instead.

    This list is mtime-only and NOT complete, unlike the stage sidecars: it
    misses the mount legs (``T_mount_cam`` / ``T_mount_tcp``, which the eef
    stage also leaves out of its own key), the robot description, and
    ``lens0.mp4``, whose frame count bounds the export window. Recalibrating
    a mount therefore changes nothing here until the npz is rebuilt with
    ``--rebuild``.
    """
    from kiwi.core import layout
    from kiwi.episode.manifest import MANIFEST_NAME

    npz = episode / layout.REPLAY_STATE
    # `*/` reaches a stream dir only; the scene trajectory sits one level
    # deeper, so its glob has to spell the scene dir out.
    world_glob = (
        f"*/{layout.REGISTER_TUM}"
        if world_source == "marker"
        else f"*/{layout.SCENE_DIR}/{layout.STATE_TUM}"
    )
    inputs = [
        episode / MANIFEST_NAME,
        *episode.glob(f"*/{layout.EEF_STATE}"),
        *episode.glob(world_glob),
    ]
    newest = max((f.stat().st_mtime for f in inputs if f.is_file()), default=None)
    stale = npz.is_file() and newest is not None and npz.stat().st_mtime <= newest
    if npz.is_file() and not rebuild and not stale:
        return
    try:
        from kiwi.export.build import build_episode
        from kiwi.export.replay_state import write_replay_state

        src = _export_source(episode)
        oe = build_episode(src, fps=None, world_source=world_source)
        result = write_replay_state(oe, npz)
        if result.warning:
            print(f"warning: {episode.name}: {result.warning}", file=sys.stderr)
        print(f"  built {npz} ({result.frames} frames)")
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as e:
        if npz.is_file() and not rebuild:
            print(
                f"warning: {episode.name}: could not rebuild the stale "
                f"replay_state.npz ({e}); using the existing one",
                file=sys.stderr,
            )
            return
        raise


@dataclass
class _ReplayEpisodes:
    """The flag surface `replay solve` and `replay run` share: the episode
    list, the data-loading knobs, and the rig/solver parameters. Defaults
    describe one bench layout."""

    episodes: tyro.conf.Positional[list[Path]]
    """One or more episode directories (each holding, or able to build, a
    replay_state.npz)."""
    fps: float = 30.0
    """Decimate the capture clock to this rate before solving (0 = native)."""
    stride: int = 1
    """Solve every Nth frame of the (decimated) stream."""
    traj_yaw: float = 0.0
    """World-frame yaw (deg) baked into the trajectories; marker-frame
    datasets with no placement baked in typically need -90."""
    rot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Body-frame intrinsic-XYZ rotation (deg) right-multiplied onto each EE
    orientation."""
    rebuild: bool = False
    """Rebuild every replay_state.npz through the export pipeline even when
    one exists."""
    world_source: str = "markerless"
    """Which world pose a rebuilt replay_state.npz is dumped from: markerless
    (the default - the state stage's scene frame) or marker (the register
    stage's ArUco anchor). Only consulted when the npz is (re)built."""
    workspace_x: float = 0.0
    workspace_y: float = 0.0
    workspace_z: float = 0.0
    ws_yaw: float = 0.0
    ws_pitch: float = 0.0
    ws_roll: float = 0.0
    base_distance: float = 0.805
    base_z: float = 0.0
    base_roll_left: float = 0.0
    base_roll_right: float = 0.0
    base_pitch_left: float = 0.0
    base_pitch_right: float = 0.0
    base_yaw_left: float = 0.0
    base_yaw_right: float = 0.0
    start_frame: int = 0
    solve_mode: Literal["independent", "combined"] = "combined"
    """One QP per arm, or one shared QP coordinating both arms."""
    avoid_collisions: bool = False
    """Add active inter-arm + self collision avoidance to the QP."""
    collision_margin: float = 0.004
    ik_gain: float = 0.5
    ik_lm_damping: float = 1e-3
    ik_iters: int = 120
    posture_cost: float = 1e-3
    orientation_cost: float = 1.0
    offset: tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
    """Global rigid offset [dx dy dz roll pitch yaw] (m/deg) applied to ALL
    episodes on top of the workspace transform, about the shared anchor."""
    offsets_json: str | None = None
    """Per-episode offsets as JSON, e.g. '{"2": [0,0.01,0,0,0,0]}' (keys are
    episode indices into the positional list)."""
    workers: int = 1
    """Solve episodes in this many spawn-context worker processes."""

    def _ensure_episodes(self, verb: str) -> tuple[list[Path], int]:
        """Ensure every episode's replay_state.npz; return (loadable, failed)."""
        dirs: list[Path] = []
        failed = 0
        for episode in self.episodes:
            try:
                _ensure_replay_state(
                    episode, rebuild=self.rebuild, world_source=self.world_source
                )
                dirs.append(episode)
            except (ValueError, FileNotFoundError, RuntimeError, KeyError) as e:
                failed += 1
                print(f"[FAIL] {episode.name}: {e}", file=sys.stderr)
        if not dirs:
            print(f"kiwi replay {verb}: no loadable episodes", file=sys.stderr)
        return dirs, failed

    def _parsed_offsets(self, verb: str) -> dict[int, list] | None:
        """Parse --offsets-json into {episode index: offset6}; None on error.

        Keys index the POSITIONAL episode list; every key must be in range
        and every value a 6-number [dx dy dz roll pitch yaw] vector - a bad
        offset is a usage error here, not a scipy traceback mid-batch.
        """
        if not self.offsets_json:
            return {}
        try:
            raw = json.loads(self.offsets_json)
            if not isinstance(raw, dict):
                raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
            out: dict[int, list] = {}
            for k, v in raw.items():
                idx = int(k)
                if not 0 <= idx < len(self.episodes):
                    raise ValueError(
                        f"key {idx} is out of range for {len(self.episodes)} episode(s)"
                    )
                off = [float(x) for x in v]
                if len(off) != 6:
                    raise ValueError(
                        f"offset for key {idx} has {len(off)} elements, "
                        f"expected 6 ([dx dy dz roll pitch yaw])"
                    )
                out[idx] = off
            return out
        except (ValueError, KeyError, TypeError) as e:
            print(f"kiwi replay {verb}: bad --offsets-json: {e}", file=sys.stderr)
            return None

    def _remapped_offsets(
        self, per_ep: dict[int, list], dirs: list[Path]
    ) -> dict[int, list]:
        """Re-key positional-list offsets onto the loadable episode set.

        ``dirs`` is the surviving subset of ``self.episodes`` (same objects,
        same order); an offset whose episode failed to load is dropped with a
        warning instead of silently landing on the wrong episode.
        """
        out: dict[int, list] = {}
        for k, off in per_ep.items():
            src = self.episodes[k]
            if src in dirs:
                out[dirs.index(src)] = off
            else:
                print(
                    f"warning: --offsets-json key {k} refers to {src.name}, "
                    f"which failed to load; offset ignored",
                    file=sys.stderr,
                )
        return out


@dataclass
class ReplaySolve(_ReplayEpisodes):
    """Batch-solve dual-arm IK over episodes: metrics table + solved joints.

    Loads each episode's EE state (building replay_state.npz through the
    export pipeline when missing), solves differential IK for both arms on
    the dual-arm rig, prints the per-episode tracking/collision/continuity
    table, and writes <episode>/replay_solve.npz. Defaults describe one
    bench layout. Needs the replay extra (mujoco + mink):
    `pixi install -e replay`.
    """

    write: bool = True
    """Write <episode>/replay_solve.npz per solved episode (--no-write for a
    metrics-only run, e.g. a placement sweep)."""

    def run(self) -> int:
        import numpy as np

        try:
            from kiwi.replay.batch import BatchSolver, EpisodeSet
        except ImportError as e:
            print(
                f"kiwi replay solve: {e} - the solver needs the replay extra "
                f"(pip install -e .[replay] / pixi install -e replay)",
                file=sys.stderr,
            )
            return 2
        from kiwi.core import layout
        from kiwi.replay.results import write_solve_npz
        from kiwi.replay.rig import FR3_CHOPSTICK_SA

        dirs, failed = self._ensure_episodes("solve")
        if not dirs:
            return 1

        try:
            es = EpisodeSet.load(
                dirs,
                fps=self.fps,
                rot_offset_deg=self.rot_offset,
                traj_yaw_deg=self.traj_yaw,
                stride=self.stride,
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"kiwi replay solve: {e}", file=sys.stderr)
            return 1
        es.offset_global = np.asarray(self.offset, np.float64)
        per_ep = self._parsed_offsets("solve")
        if per_ep is None:
            return 1
        for k, v in self._remapped_offsets(per_ep, dirs).items():
            es.offsets[k] = np.asarray(v, np.float64)

        ws = {
            "x": self.workspace_x,
            "y": self.workspace_y,
            "z": self.workspace_z,
            "yaw": self.ws_yaw,
            "pitch": self.ws_pitch,
            "roll": self.ws_roll,
        }
        base = {
            "dist": self.base_distance,
            "z": self.base_z,
            "roll_left": self.base_roll_left,
            "roll_right": self.base_roll_right,
            "pitch_left": self.base_pitch_left,
            "pitch_right": self.base_pitch_right,
            "yaw_left": self.base_yaw_left,
            "yaw_right": self.base_yaw_right,
        }
        solver = BatchSolver(
            FR3_CHOPSTICK_SA,
            self.base_distance,
            None,  # keep the composite's configured flange attachment
            combined=self.solve_mode == "combined",
            lm_damping=self.ik_lm_damping,
            avoid_collisions=self.avoid_collisions,
            collision_margin=self.collision_margin,
            workers=self.workers,
        )
        par = f", {self.workers} workers" if self.workers > 1 and len(dirs) > 1 else ""
        print(f"Batch-solving {len(dirs)} episode(s) (headless{par})...")
        import time as _time

        t0 = _time.perf_counter()
        solver.start(
            es,
            es.ep_ids,
            ws=ws,
            base=base,
            ik_iters=self.ik_iters,
            frame_gain=self.ik_gain,
            orientation_cost=self.orientation_cost,
            posture_cost=self.posture_cost,
            start_frame=self.start_frame,
            background=False,
        )
        print(f"  done in {_time.perf_counter() - t0:.1f}s")
        print(
            f"  {'ep':>4} {'episode':>16} {'status':>9} {'cont':>5} {'frames':>6} "
            f"{'worst mm':>9} {'mean mm':>8} {'worst deg':>9} {'qp':>4} "
            f"{'self':>5} {'cross':>5} {'grip':>5}"
        )
        for ep in es.ep_ids:
            m = es.metrics(ep)
            st = es.status[ep]
            name = es.episode_dir(ep).name[:16]
            c0 = es.continuity.get(ep, {}).get("min")
            cs = "-" if c0 is None else f"{c0:.0f}"
            if m is None:
                err = es.results.get(ep, {}).get("error", "")
                print(f"  {ep:>4} {name:>16} {st:>9} {cs:>5}  {err}")
                continue
            print(
                f"  {ep:>4} {name:>16} {st:>9} {cs:>5} {m.n:>6} "
                f"{m.worst_pos * 1000:>9.1f} {m.mean_pos * 1000:>8.2f} "
                f"{np.rad2deg(m.worst_rot):>9.1f} {m.qp_failures:>4} "
                f"{m.self_f:>5} {m.cross_f:>5} {m.grip_f:>5}"
            )

        params = {
            "fps": self.fps,
            "stride": self.stride,
            "traj_yaw": self.traj_yaw,
            "rot_offset": list(self.rot_offset),
            "workspace": ws,
            "base": base,
            "start_frame": self.start_frame,
            "solve_mode": self.solve_mode,
            "avoid_collisions": self.avoid_collisions,
            "collision_margin": self.collision_margin,
            "ik_gain": self.ik_gain,
            "ik_lm_damping": self.ik_lm_damping,
            "ik_iters": self.ik_iters,
            "posture_cost": self.posture_cost,
            "orientation_cost": self.orientation_cost,
            "robot": FR3_CHOPSTICK_SA.key,
        }
        for ep in es.ep_ids:
            episode = es.episode_dir(ep)
            if es.status[ep] != "done":
                failed += 1
                if es.metrics(ep) is None:
                    continue  # error already printed in the table
                print(f"[FAIL] {episode.name}: solve {es.status[ep]}", file=sys.stderr)
                continue
            if not self.write:
                continue
            out = episode / layout.REPLAY_SOLVE
            write_solve_npz(
                out,
                es.results[ep],
                status=es.status[ep],
                continuity=es.continuity[ep],
                fps=es.meta["fps"] / max(self.stride, 1),
                decim=es.decim,
                params={**params, "offset": es.effective_offset(ep).tolist()},
            )
            print(f"[ok] {episode.name}: wrote {out}")
        return 1 if failed else 0


@dataclass
class ReplayRun(_ReplayEpisodes):
    """Interactive bimanual replay in viser: playback + live rig tuning.

    Serves the episode set in the replay viewer: a timeline over the solved
    trajectory, live base/workspace/gripper-mount sliders, direct joint
    posing, draggable live-TCP gizmos, per-component tracking-error plots,
    and the batch-solve controls that solve the whole set on a separate rig instance
    while the GUI stays interactive. Same episode inputs and rig defaults as
    `kiwi replay solve`. Needs the replay extra: `pixi install -e replay`.
    """

    port: int = 8080
    """Viser server port."""
    gui_width: Literal["small", "medium", "large"] = "large"
    """Width of viser's control panel."""
    gui_layout: Literal["floating", "collapsible", "fixed"] = "fixed"
    """How the control panel docks."""

    def run(self) -> int:
        try:
            from kiwi.replay.viewer import ViewerParams, launch
        except ImportError as e:
            print(
                f"kiwi replay run: {e} - the viewer needs the replay extra "
                f"(pip install -e .[replay] / pixi install -e replay)",
                file=sys.stderr,
            )
            return 2

        dirs, failed = self._ensure_episodes("run")
        if not dirs:
            return 1
        per_ep = self._parsed_offsets("run")
        if per_ep is None:
            return 1
        # Mechanical mapping: every ViewerParams field except the parsed
        # `offsets` is a same-named CLI flag on this dataclass; a missing
        # counterpart fails here.
        params = ViewerParams(
            **{
                f.name: getattr(self, f.name)
                for f in dataclasses.fields(ViewerParams)
                if f.name != "offsets"
            },
            offsets={
                k: tuple(v) for k, v in self._remapped_offsets(per_ep, dirs).items()
            },
        )
        try:
            launch(dirs, params)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"kiwi replay run: {e}", file=sys.stderr)
            return 1
        return 1 if failed else 0


@dataclass
class ReplayView:
    """Open a robot description in viser: FK sliders + EE frame overlay.

    Spot-check that a description's declared frames sit where the registry
    claims - every end-effector gets axes, a label, and a live [x, y, z]
    readout, including frames behind fixed joints. The `tune TCP` folder
    measures a correction and prints it in URDF <origin> units. Needs the
    optional replay feature: `pixi install -e replay` (or `pip install viser`).
    """

    description: tyro.conf.Positional[str | None] = None
    """Registry name, e.g. `fr3_chopstick_sa`. Omit to list what's available."""
    port: int = 8080
    home: bool = True
    """Open at the registry's home pose; false opens at joint-limit midpoints."""
    frames: bool = True
    """Draw the end-effector frame overlay and the TCP ruler."""
    dark: bool = False
    """Use the dark canvas instead of viser's default white one."""

    def run(self) -> int:
        from kiwi.core.descriptions import list_descriptions

        try:
            names = list_descriptions()
            if self.description is None:
                print("robot descriptions:")
                for name in names:
                    print(f"  {name}")
                return 0
            from kiwi.replay.view import launch

            launch(
                self.description,
                port=self.port,
                home=self.home,
                frames=self.frames,
                dark=self.dark,
            )
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as e:
            # KeyError.__str__ is a repr, so an unknown name would print
            # wrapped in stray quotes; take the message itself.
            msg = e.args[0] if isinstance(e, KeyError) else e
            print(f"kiwi replay view: {msg}", file=sys.stderr)
            return 2 if "viser" in str(msg) else 1
        return 0


@dataclass
class Replay:
    """Kinematic replay: solve/play episodes, or inspect a robot description."""

    command: Annotated[
        (
            Annotated[ReplayRun, tyro.conf.subcommand("run", prefix_name=False)]
            | Annotated[ReplaySolve, tyro.conf.subcommand("solve", prefix_name=False)]
            | Annotated[ReplayView, tyro.conf.subcommand("view", prefix_name=False)]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


# --------------------------------------------------------------------------- #
# `kiwi calib` - one-time per-device calibration workflows (kiwi.calib)
#
# The publishing leaves (imu noise, intrinsics publish, intrinsics kb4,
# qr av-delay) write a DATED artifact into config/cameras/<unit>/ and print
# the camera.yaml pointer line that would activate it; moving the pointer is
# the operator's reviewable publication step, never the tool's.
# --------------------------------------------------------------------------- #


def _expand_insv(insv: tuple[Path, ...]) -> list[Path]:
    """The run's part files, or one directory holding exactly one run's parts."""
    paths = list(insv)
    if len(paths) == 1 and paths[0].is_dir():
        paths = sorted(paths[0].glob("*.insv"))
    return paths


@dataclass
class CalibImuNoise:
    """Fit IMU noise densities (Allan) from a stationary on-camera log."""

    insv: tuple[Path, ...] = ()
    """The run's .insv part files, or one directory holding exactly one run's parts."""
    imu_npz: Path | None = None
    """An imu_raw_v1 .npz previously written by `kiwi calib imu extract`."""
    camera: str | None = None
    """Registry unit to write the dated artifact into (config/cameras/<unit>/)."""
    output: Path | None = None
    """Explicit output path (overrides --camera)."""
    discard_seconds: float = 0.0
    """Warmup window to discard (bias settling + thermal ramp). Defaults to 0;
    pass it explicitly (1200 is the protocol value for a ~2 h X5 log)."""
    trim_end_seconds: float = 0.0
    """Tail to discard. The operator physically stopping the recording lands in
    the longest-tau Allan clusters, where there are fewest samples. Defaults to
    0 - check the log's tail and pass it explicitly."""

    def run(self) -> int:
        from kiwi import calib
        from kiwi.core.camera.registry import CameraUnit, default_artifact_out
        from kiwi.core.paths import config_root

        if bool(self.insv) == (self.imu_npz is not None):
            print(
                "kiwi calib imu noise: pass exactly one of --insv / --imu-npz",
                file=sys.stderr,
            )
            return 2
        if self.output is None and self.camera is None:
            print(
                "kiwi calib imu noise: pass --camera <unit> or --output <path>",
                file=sys.stderr,
            )
            return 2

        if self.imu_npz is not None:
            ts, vals, meta = calib.load_imu_npz(self.imu_npz)
            sources = (
                ", ".join(s["name"] for s in meta.get("source_files", [])) or "unknown"
            )
            print(f"[load] {self.imu_npz.name}: {len(ts)} segment(s), from {sources}")
            source = str(self.imu_npz)
        else:
            paths = _expand_insv(self.insv)
            if not paths:
                print(
                    f"kiwi calib imu noise: no .insv files in {self.insv[0]}",
                    file=sys.stderr,
                )
                return 2
            ts, vals = [], []
            for p in paths:
                t, v = calib.parse_insv_imu(p)
                ts.append(t)
                vals.append(v)
                print(f"[load] {p.name}: {len(t)} samples, {t[-1] - t[0]:.1f} s")
            source = ", ".join(p.name for p in paths)
        t, gyro, accel = calib.assemble_segments(ts, vals)

        try:
            result = calib.noise_from_series(
                t, gyro, accel,
                discard_s=self.discard_seconds,
                trim_end_s=self.trim_end_seconds,
            )  # fmt: skip
        except ValueError as e:
            print(f"kiwi calib imu noise: {e}", file=sys.stderr)
            return 2

        unit_dir = (
            None if self.camera is None else config_root() / "cameras" / self.camera
        )
        out = self.output
        if out is None:
            assert unit_dir is not None
            try:
                out = default_artifact_out(unit_dir, "imu_noise")
            except FileExistsError as e:
                print(f"kiwi calib imu noise: {e}", file=sys.stderr)
                return 2
        calib.write_imu_noise(out, result, source=source, unit=self.camera)

        print(
            f"[fit] {result.n_samples} samples, {result.duration_s / 60:.1f} min "
            f"@ {result.fs_hz:.1f} Hz (warmup {self.discard_seconds:.0f} s, "
            f"tail {self.trim_end_seconds:.0f} s discarded)"
        )
        print("  axis   N (noise density)   K (random walk)")
        for a in calib.AXES:
            print(
                f"  {a:<5}  {result.per_axis_noise_density[a]:>17.3e}"
                f"  {result.per_axis_random_walk[a]:>16.3e}"
            )
        for k, v in result.noise.as_dict().items():
            print(f"  {k}: {v:.6g}")

        # Compare against the unit's currently-published artifact.
        if unit_dir is not None and unit_dir.is_dir():
            published = CameraUnit.load(unit_dir).artifact("imu_noise")
            if published is not None and published.resolve() != Path(out).resolve():
                ref, _ = calib.read_imu_noise(published)
                print(f"[vs published] {published.name}:")
                for k, v in result.noise.as_dict().items():
                    r = ref.as_dict()[k]
                    ratio = v / r if r else float("nan")
                    print(f"  {k}: x{ratio:.2f}")

        png = calib.plot_allan_curves(
            result, Path(out).parent / f"{Path(out).stem}-curves.png"
        )
        print(
            f"[write] {out}"
            + (f"\n[plot] {png}" if png else "  (no matplotlib: plot skipped)")
        )
        if self.output is None:
            print(
                f"[publish] to activate: set `imu_noise: {Path(out).name}` in "
                f"{unit_dir / 'camera.yaml'} (one reviewable commit)"
            )
        return 0


@dataclass
class CalibImuExtract:
    """Extract .insv IMU telemetry into a compact, verified imu_raw_v1 .npz."""

    insv: tuple[Path, ...]
    """The run's .insv part files, or one directory holding exactly one run's parts."""
    output: Path | None = None
    """Destination .npz (default: <dir of first part>/imu_raw.npz)."""
    delete_insv: bool = False
    """Delete the source videos after the .npz verifies (round-trip bit-exact
    AND re-stitchable). Refused without --yes: this removes the original recordings."""
    yes: bool = False
    """Confirm --delete-insv non-interactively."""

    def run(self) -> int:
        import datetime

        from kiwi import calib

        paths = _expand_insv(self.insv)
        if not paths:
            print(
                f"kiwi calib imu extract: no .insv files in {self.insv[0]}",
                file=sys.stderr,
            )
            return 2
        if self.delete_insv and not self.yes:
            print(
                "kiwi calib imu extract: --delete-insv removes the source videos "
                "permanently; pass --yes to confirm",
                file=sys.stderr,
            )
            return 2
        if self.output is None and paths[0].parent.name == "raw":
            print(
                "kiwi calib imu extract: the default output "
                f"({paths[0].parent / 'imu_raw.npz'}) would land in an "
                "episode's raw/ directory, which holds recordings only - "
                "pass --output, e.g. <episode>/<label>/imu_raw.npz",
                file=sys.stderr,
            )
            return 2
        out = self.output or paths[0].parent / "imu_raw.npz"

        meta = {
            "source_files": [
                {"name": p.name, "size_bytes": p.stat().st_size} for p in paths
            ],
            "extracted_by": "kiwi calib imu extract",
            "extracted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            **_insv_camera_meta(paths[0]),
        }
        try:
            stats = calib.extract_npz(paths, out, meta)
        except RuntimeError as e:
            print(
                f"kiwi calib imu extract: VERIFY FAILED - {e} (not deleting)",
                file=sys.stderr,
            )
            return 1
        if "stitched_samples" in stats:
            print(
                f"[verify] round-trip bit-exact; {stats['n_parts']} parts stitch "
                f"into {stats['stitched_samples']} samples"
            )
        else:
            print("[verify] round-trip bit-exact (single part, no stitching needed)")
        print(
            f"[size] {stats['insv_bytes'] / 1e9:.2f} GB of .insv -> "
            f"{stats['npz_bytes'] / 1e6:.1f} MB {out}"
        )
        if self.delete_insv:
            freed = 0
            for p in paths:
                freed += p.stat().st_size
                p.unlink()
                print(f"[delete] {p}")
            print(f"[delete] freed {freed / 1e9:.2f} GB")
        else:
            print("[keep] source .insv files untouched (--delete-insv --yes to remove)")
        return 0


def _insv_camera_meta(path: Path) -> dict:
    """Camera identity from the .insv metadata block (provenance; best effort)."""
    import telemetry_parser

    try:
        meta = telemetry_parser.Parser(str(path)).telemetry()[0]["Default"]["Metadata"]
        return {
            k: meta[k]
            for k in ("camera_type", "fw_version", "serial_number")
            if k in meta
        }
    except Exception as e:  # noqa: BLE001 - provenance only, never fatal
        print(f"[meta] warning: no camera metadata in {path.name}: {e}")
        return {}


@dataclass
class CalibImuCam:
    """Solve a lens's camera<->IMU extrinsic via Kalibr (docker).

    Needs the matching lens's intrinsics to already exist (DS for the
    dual-lens X4/X5, pinhole-equi for a single-lens GO3) - run
    `kiwi calib intrinsics` first when re-shooting a lens. Kalibr's output
    lands as <output-dir>/imucam_calib-camchain-imucam.yaml; copy it to
    config/cameras/<unit>/extrinsic-imu[-back]-<date>.yaml and vet it with
    `kiwi calib imu check` BEFORE moving the pointer (single-lens units have
    no front/back cross-check; gate on the report residuals instead).

    The intrinsics MUST come from the same video mode as this recording:
    GO3 modes are different sensor crops with different optical mappings
    (square vs 16:9 is not a rescale), which the resolution guard enforces.
    """

    insv: Path
    """The IMU-cam calibration recording (board + the motion recipe): a
    dual-lens .insv, or a single-video-stream camera original like the GO3's
    .mp4 - the stream count is probed, never assumed from the suffix."""
    output_dir: Path
    """The run dir: staged frames + CSV, helper YAMLs, kalibr outputs."""
    lens: str = "front"
    """front | back. One lens per recording - they face opposite ways. A
    single-lens recording is always front."""
    camera: str | None = None
    """Registry unit whose published intrinsics to solve against."""
    intrinsics: Path | None = None
    """Explicit intrinsics npz (overrides --camera)."""
    marker_config: Path | None = None
    """Project AprilGrid yaml (default: <config>/markers/calibio_kalibr_apriltag_board.yaml)."""
    target_name: str | None = None
    """Which apriltag_grid entry (default: the file's sole entry)."""
    frame_stride: int = 5
    """Keep every Nth frame. IN FRAMES, so scale with fps to ~6 Hz: 5 at
    30 fps, 20 at 120 fps."""
    bag_from_to: tuple[float, float] | None = None
    """Seconds [from, to] where the board is actually visible (skip the
    boardless tail rather than solving on it)."""
    timeoffset_padding: float = 0.1
    """Kalibr's bound on the estimated IMU<->cam time offset (s)."""
    shm_size: str = "8g"
    """Docker /dev/shm for Kalibr's multiprocessing corner extractor."""
    imu_noise_from: str | None = None
    """Solve with a unit's measured imu-noise.yaml instead of the standard
    priors. The published extrinsics use the standard priors."""
    skip_kalibr: bool = False
    """Stop after staging (inspect frames/, run the container by hand)."""
    force_demux: bool = False
    """Re-demux even when the per-lens mp4s already exist."""

    def run(self) -> int:
        import numpy as np

        from kiwi.calib.kalibr import docker, layout, manifest, staging, yamls
        from kiwi.core.camera.registry import CameraUnit
        from kiwi.core.paths import config_root
        from kiwi.core.video import VideoInfo

        if self.lens not in ("front", "back"):
            print("kiwi calib imu cam: --lens front|back", file=sys.stderr)
            return 2
        if (self.intrinsics is None) == (self.camera is None):
            print(
                "kiwi calib imu cam: pass exactly one of --camera / --intrinsics",
                file=sys.stderr,
            )
            return 2
        if not self.insv.is_file():
            print(f"kiwi calib imu cam: {self.insv} does not exist", file=sys.stderr)
            return 2
        # One early probe serves both mode-keyed intrinsics resolution and
        # the resolution guard below: stream 0's frame size, which both X5
        # lenses share and which IS the single-lens frame.
        info = VideoInfo.read(self.insv)
        intrinsics = self.intrinsics
        if intrinsics is None:
            unit_dir = config_root() / "cameras" / self.camera
            try:
                intrinsics = CameraUnit.load(unit_dir).artifact_for_size(
                    "intrinsics", (info.width, info.height)
                )
            except (ValueError, FileNotFoundError) as e:
                print(f"kiwi calib imu cam: {e}", file=sys.stderr)
                return 2
            if intrinsics is None:
                print(
                    f"kiwi calib imu cam: {self.camera} has no published "
                    "intrinsics - run `kiwi calib intrinsics` first",
                    file=sys.stderr,
                )
                return 2
        marker_cfg = (
            self.marker_config
            or config_root() / "markers" / "calibio_kalibr_apriltag_board.yaml"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            grid_name, grid = yamls.resolve_grid(marker_cfg, self.target_name)
        except (ValueError, FileNotFoundError) as e:
            print(f"kiwi calib imu cam: {e}", file=sys.stderr)
            return 2
        yamls.write_aprilgrid(grid, self.output_dir / yamls.APRILGRID_NAME)

        noise = None
        if self.imu_noise_from is not None:
            from kiwi.calib import read_imu_noise

            src = CameraUnit.load(config_root() / "cameras" / self.imu_noise_from)
            artifact = src.artifact("imu_noise")
            if artifact is None:
                print(
                    f"kiwi calib imu cam: {self.imu_noise_from} has no published "
                    "imu-noise.yaml (run `kiwi calib imu noise` first)",
                    file=sys.stderr,
                )
                return 2
            noise = read_imu_noise(artifact)[0].as_dict()
            print(f"[imu] measured noise from {artifact} (NOT the standard priors)")
        try:
            yamls.write_camchain(
                intrinsics,
                self.lens,
                self.output_dir / "camchain.yaml",
                unit=self.camera,
                source_video=str(self.insv),
            )
        except ValueError as e:
            print(f"kiwi calib imu cam: {e}", file=sys.stderr)
            return 2
        yamls.write_imu_yaml(self.output_dir / "imu.yaml", noise=noise)

        lens0, lens1 = _demux_both(self.insv, self.force_demux)
        if self.lens == "back" and lens1 is None:
            print(
                "kiwi calib imu cam: this recording has a single video stream "
                "(single-lens camera) - there is no back lens",
                file=sys.stderr,
            )
            return 2
        src = lens0 if self.lens == "front" else lens1
        with np.load(str(intrinsics)) as d:
            npz_size = (int(d["image_size"][0]), int(d["image_size"][1]))
        if (info.width, info.height) != npz_size:
            print(
                f"kiwi calib imu cam: {src.name} is {info.width}x{info.height} "
                f"but the intrinsics were solved at {npz_size[0]}x{npz_size[1]} - "
                "intrinsics do not transfer across video modes/crops; run "
                "`kiwi calib intrinsics` on a recording in THIS mode first",
                file=sys.stderr,
            )
            return 2

        # IMU first: its first stamp decides the clock shift both streams share.
        rows = staging.read_imu_rows_ns(self.insv)
        if not rows:
            print(
                f"kiwi calib imu cam: no IMU telemetry in {self.insv}", file=sys.stderr
            )
            return 1
        offset_ns = layout.imu_clock_offset_ns(rows[0][0])
        if offset_ns:
            print(f"[clock] shifting streams +{offset_ns / 1e6:.1f} ms (IMU pre-roll)")
        frames_root = self.output_dir / layout.FRAMES_DIR
        frames_root.mkdir(parents=True, exist_ok=True)
        _, n_imu = layout.write_imu_csv(
            frames_root / layout.IMU_CSV_NAME,
            ((ts + offset_ns, v) for ts, v in rows),
        )

        staged = staging.stage_frames(
            src,
            frames_root / layout.cam_dir(0),
            frame_stride=self.frame_stride,
            offset_ns=offset_ns,
        )
        print(
            f"[{self.lens}] staged {staged.n_written}/{staged.n_read} frames "
            f"(stride {self.frame_stride}) + {n_imu} IMU rows"
        )
        if staged.n_written == 0:
            print("kiwi calib imu cam: no frames staged; aborting", file=sys.stderr)
            return 1
        expected = staged.n_written + n_imu
        manifest.write_manifest(
            self.output_dir,
            kind="imucam",
            unit=self.camera,
            cams=[
                {
                    "id": layout.cam_dir(0),
                    "lens": self.lens,
                    "topic": layout.cam_topic(0),
                    "image_size": list(staged.image_size),
                    "n_frames": staged.n_written,
                    "frame_stride": self.frame_stride,
                    "source_video": str(self.insv),
                    "fps": staged.fps,
                }
            ],
            imu={
                "topic": layout.IMU_TOPIC,
                "csv": f"{layout.FRAMES_DIR}/{layout.IMU_CSV_NAME}",
                "n_samples": n_imu,
            },
            clock={
                "fps": staged.fps,
                "period_ns": staged.period_ns,
                "offset_ns": offset_ns,
            },
            bag={"name": docker.IMUCAM_BAG, "expected_messages": expected},
        )
        if self.skip_kalibr:
            print("[kalibr] --skip-kalibr set; staged input is in", frames_root)
            return 0
        run = docker.imucam_run(
            self.output_dir,
            expected_messages=expected,
            bag_from_to=self.bag_from_to,
            timeoffset_padding=self.timeoffset_padding,
            shm_size=self.shm_size,
        )
        try:
            rc = docker.execute(run)
        except RuntimeError as e:
            print(f"kiwi calib imu cam: {e}", file=sys.stderr)
            return 2
        if rc != 0:
            print(
                f"kiwi calib imu cam: kalibr exited {rc}; see {run.log_path}",
                file=sys.stderr,
            )
            return 1
        camchain_out = (
            self.output_dir / f"{Path(docker.IMUCAM_BAG).stem}-camchain-imucam.yaml"
        )
        suffix = "" if self.lens == "front" else "-back"
        print(
            f"[kalibr] done. Next: copy {camchain_out.name} to "
            f"config/cameras/<unit>/extrinsic-imu{suffix}-<date>.yaml, run "
            "`kiwi calib imu check` on it, and only then move the pointer."
        )
        return 0


@dataclass
class CalibImuCheck:
    """Cross-check a unit's rear cam-IMU extrinsic against its front one.

    The two lenses share one IMU, so composing their solves must recover the
    camera's physical back-to-back geometry (~180 deg / 20-40 mm). Vet a
    FRESH solve before publishing it: --back <kalibr camchain-imucam.yaml>
    with a single --units entry.
    """

    units: tuple[str, ...] = ()
    """Registry units to check (default: every unit carrying both lenses)."""
    front: Path | None = None
    """Override the front camchain yaml. Single --units entry only."""
    back: Path | None = None
    """Override the rear camchain yaml - point at a fresh Kalibr output to
    vet it BEFORE it enters the registry. Single --units entry only."""

    def run(self) -> int:
        from kiwi.calib.imucam import front_back_consistency
        from kiwi.core.camera.extrinsics import read_cam_imu
        from kiwi.core.camera.registry import CameraUnit
        from kiwi.core.paths import config_root

        cameras = config_root() / "cameras"
        units = list(self.units)
        if not units:
            for d in sorted(p for p in cameras.iterdir() if p.is_dir()):
                u = CameraUnit.load(d)
                try:
                    if u.artifact("extrinsic_imu") and u.artifact("extrinsic_imu_back"):
                        units.append(d.name)
                except FileNotFoundError as e:
                    print(
                        f"kiwi calib imu check: broken registry: {e}", file=sys.stderr
                    )
                    return 2
        if not units:
            print(
                "kiwi calib imu check: no unit has both a front and a rear "
                "extrinsic (a single-lens unit is unverifiable, not passing)",
                file=sys.stderr,
            )
            return 2
        if (self.front or self.back) and len(units) != 1:
            print(
                "kiwi calib imu check: --front/--back need exactly one --units entry",
                file=sys.stderr,
            )
            return 2

        print(
            f"{'unit':<20} {'angle':>9} {'dist':>9}  {'front dt':>10} {'back dt':>10}  verdict"
        )
        all_ok = True
        for name in units:
            unit = CameraUnit.load(cameras / name)
            try:
                front = self.front or unit.artifact("extrinsic_imu")
                back = self.back or unit.artifact("extrinsic_imu_back")
            except FileNotFoundError as e:
                print(f"kiwi calib imu check: broken registry: {e}", file=sys.stderr)
                return 2
            missing = [k for k, p in (("front", front), ("back", back)) if p is None]
            if missing:
                print(
                    f"kiwi calib imu check: {name}: no {'/'.join(missing)} extrinsic",
                    file=sys.stderr,
                )
                return 2
            f, b = read_cam_imu(front), read_cam_imu(back)
            check = front_back_consistency(f.T_cam_imu, b.T_cam_imu)
            all_ok &= check.ok
            dts = "  ".join(
                f"{1000 * t: 9.2f}ms" if t is not None else "       n/a"
                for t in (f.timeshift_s, b.timeshift_s)
            )
            verdict = "PASS" if check.ok else "FAIL: " + "; ".join(check.reasons)
            print(
                f"{name:<20} {check.angle_deg:8.2f}d {check.dist_mm:8.1f}mm  {dts}  {verdict}"
            )
        if not all_ok:
            print(
                "\nA FAIL means the rear and front solves disagree about the shared "
                "IMU.\nSuspect, in order: the rear run consumed the wrong lens or the "
                "wrong ds_*;\ntoo few views; or a badly locked board pose. Re-run the "
                "solve.",
                file=sys.stderr,
            )
            return 1
        return 0


@dataclass
class CalibImu:
    """IMU-side calibration: noise densities, telemetry extraction, cam-IMU."""

    command: Annotated[
        (
            Annotated[CalibImuNoise, tyro.conf.subcommand("noise", prefix_name=False)]
            | Annotated[
                CalibImuExtract, tyro.conf.subcommand("extract", prefix_name=False)
            ]
            | Annotated[CalibImuCam, tyro.conf.subcommand("cam", prefix_name=False)]
            | Annotated[CalibImuCheck, tyro.conf.subcommand("check", prefix_name=False)]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


def _demux_both(insv: Path, force: bool) -> tuple[Path, Path | None]:
    """(lens0_mp4, lens1_mp4) beside the .insv, demuxing when missing or stale.

    A single-video-stream recording (a GO3 camera original) has nothing to
    demux: the source itself comes back as lens0 and lens1 is ``None``, with
    no sibling artifacts written. The stream count is probed, never inferred
    from the suffix - both containers are mp4 under the hood.
    """
    from kiwi.core.camera.convert import convert, count_video_streams

    if not insv.is_file():
        # Refuse by name, as the convert stage does.
        raise FileNotFoundError(
            f"{insv} is not there - a demux beside it cannot be checked "
            f"against a source that is gone. Re-import the clip, or point "
            f"--insv at a recording that exists."
        )
    if count_video_streams(insv) == 1:
        print(f"[demux] {insv.name}: single video stream; using it directly")
        return insv, None

    stem = insv.with_suffix("").name
    from kiwi.core.layout import refuse_raw_output

    lens0 = insv.parent / f"{stem}_lens0.mp4"
    refuse_raw_output(lens0, "demux")
    lens1 = insv.parent / f"{stem}_lens1.mp4"
    # One demux freshness rule everywhere (kiwi.core.camera.convert, the convert
    # stage): existence is not freshness, a lens mp4 older than the .insv beside
    # it is the previous take. mtime is the whole key and it has a known hole -
    # `kiwi import` copies with copystat, so re-importing an *older* take keeps
    # the card's mtime and its stale demux still tests fresh.
    stale = [
        p
        for p in (lens0, lens1)
        if p.exists() and p.stat().st_mtime < insv.stat().st_mtime
    ]
    if force or stale or not (lens0.exists() and lens1.exists()):
        if stale:
            names = " + ".join(p.name for p in stale)
            print(f"[demux] {names} older than {insv.name}; re-demuxing")
        print(f"[demux] {insv.name} -> {lens0.name} + {lens1.name}")
        # Rebuild the PAIR, never half of it: convert() skips an existing output
        # unless forced.
        convert(insv, lens0, lens1, force=force or bool(stale))
    else:
        print(
            f"[demux] reusing {lens0.name} + {lens1.name} "
            f"(mtime newer than {insv.name}; --force-demux to rebuild)"
        )
    return lens0, lens1


@dataclass
class CalibIntrinsicsRun:
    """Stage a recording's frames and solve intrinsics in the Kalibr container.

    The X4/X5's two lenses face opposite directions, so one recording
    calibrates ONE lens (the other sees the back of the room); `both` exists
    for the rare capture where both lenses see the board. A single-lens
    camera original (GO3 .mp4) stages as front. Intrinsics are per video
    mode: GO3 crops (square vs 16:9) are different optical mappings, so
    record in the mode the unit will actually shoot in.
    """

    insv: Path
    """The calibration recording (board shown to the selected lens): a
    dual-lens .insv, or a single-video-stream camera original (GO3 .mp4) -
    the stream count is probed, never assumed from the suffix."""
    output_dir: Path
    """The run dir: staged frames, target yaml, manifest, kalibr outputs."""
    lenses: str = "front"
    """front | back | both. A single-lens recording is always front."""
    model: str = "ds-none"
    """Kalibr camera model. ds-none (Double Sphere) is the default, for the
    ~190 degree X4/X5 lens; pinhole-equi is the model for the GO3's
    ultrawide single lens."""
    focal_init_px: float | None = None
    """Manual focal-length init (pixels) for pinhole-* models, consumed by
    Kalibr's KALIBR_MANUAL_FOCAL_LENGTH_INIT option ONLY when its automatic
    init fails. Default: staged width / 2 - an init, not a prior. DS models
    never read it."""
    marker_config: Path | None = None
    """Project AprilGrid yaml (default: <config>/markers/calibio_kalibr_apriltag_board.yaml)."""
    target_name: str | None = None
    """Which apriltag_grid entry to use (default: the file's sole entry)."""
    min_dt_ms: float = 500.0
    """Keep at most one frame per this many ms; 500 ms = 2 fps. 0 keeps all."""
    min_corners: int = 12
    """Drop frames where the AprilGrid detector finds fewer corners. Kalibr's
    DLT pose init needs >= 6 points; 12 = 3 tags. 0 disables."""
    skip_kalibr: bool = False
    """Stop after staging (inspect frames/, run the container by hand)."""
    force_demux: bool = False
    """Re-demux even when the per-lens mp4s already exist."""

    def run(self) -> int:
        from kiwi.calib.frames import build_extractor
        from kiwi.calib.kalibr import docker, layout, manifest, staging, yamls
        from kiwi.core.paths import config_root

        if self.lenses not in ("front", "back", "both"):
            print(
                "kiwi calib intrinsics run: --lenses front|back|both", file=sys.stderr
            )
            return 2
        self.output_dir.mkdir(parents=True, exist_ok=True)
        marker_cfg = (
            self.marker_config
            or config_root() / "markers" / "calibio_kalibr_apriltag_board.yaml"
        )
        try:
            grid_name, grid = yamls.resolve_grid(marker_cfg, self.target_name)
        except (ValueError, FileNotFoundError) as e:
            print(f"kiwi calib intrinsics run: {e}", file=sys.stderr)
            return 2
        yamls.write_aprilgrid(grid, self.output_dir / yamls.APRILGRID_NAME)
        print(f"[target] {grid_name}: {grid.tag_cols}x{grid.tag_rows} AprilGrid")

        lens0, lens1 = _demux_both(self.insv, self.force_demux)
        if lens1 is None and self.lenses != "front":
            print(
                f"kiwi calib intrinsics run: --lenses {self.lenses}, but this "
                "recording has a single video stream (single-lens camera) - "
                "only front exists",
                file=sys.stderr,
            )
            return 2
        sides = ("front", "back") if self.lenses == "both" else (self.lenses,)
        mp4s = {"front": lens0, "back": lens1}
        extract = (
            # Same grid_name resolve_grid just picked: the sparse filter and
            # the board kalibr solves against must be the same target.
            build_extractor(marker_cfg, self.min_corners, name=grid_name)
            if self.min_corners > 0
            else None
        )
        frames_root = self.output_dir / layout.FRAMES_DIR
        cams, topics, total = [], [], 0
        for i, side in enumerate(sides):
            staged = staging.stage_frames(
                mp4s[side],
                frames_root / layout.cam_dir(i),
                min_dt_ms=self.min_dt_ms,
                extract=extract,
            )
            dropped = staged.n_read - staged.n_written - staged.n_sparse_dropped
            print(
                f"[{side}] kept {staged.n_written}/{staged.n_read} frames "
                f"(dropped {dropped} by --min-dt-ms, {staged.n_sparse_dropped} sparse)"
            )
            if staged.n_written == 0:
                print(
                    f"kiwi calib intrinsics run: no frames staged for {side} - "
                    "wrong lens, or the board never detected; aborting before kalibr",
                    file=sys.stderr,
                )
                return 1
            cams.append(
                {
                    "id": layout.cam_dir(i),
                    "lens": side,
                    "topic": layout.cam_topic(i),
                    "image_size": list(staged.image_size),
                    "n_frames": staged.n_written,
                    "source_video": str(mp4s[side]),
                    "fps": staged.fps,
                }
            )
            topics.append(layout.cam_topic(i))
            total += staged.n_written
        manual_init = self.model.startswith("pinhole")
        focal_init = None
        if manual_init:
            focal_init = (
                self.focal_init_px
                if self.focal_init_px is not None
                else cams[0]["image_size"][0] / 2.0
            )
            docker.write_focal_init(self.output_dir, focal_init)
            print(
                f"[kalibr] manual focal init armed at {focal_init:.1f} px "
                "(read only if Kalibr's automatic init fails)"
            )
        manifest.write_manifest(
            self.output_dir,
            kind="cameras",
            cams=cams,
            bag={"name": docker.CAMERAS_BAG, "expected_messages": total},
            extras={
                "model": self.model,
                "target": grid_name,
                **(
                    {"manual_focal_init_px": focal_init}
                    if focal_init is not None
                    else {}
                ),
            },
        )
        if self.skip_kalibr:
            print("[kalibr] --skip-kalibr set; staged frames are in", frames_root)
            return 0
        run = docker.cameras_run(
            self.output_dir,
            topics=topics,
            models=[self.model] * len(topics),
            expected_messages=total,
            manual_focal_init=manual_init,
        )
        try:
            rc = docker.execute(run)
        except RuntimeError as e:
            print(f"kiwi calib intrinsics run: {e}", file=sys.stderr)
            return 2
        if rc != 0:
            print(
                f"[kalibr] exited {rc}; if the global solve diverged, the "
                f"per-batch fits remain in {run.log_path} - "
                "`kiwi calib intrinsics publish` salvages them",
                file=sys.stderr,
            )
            return 1
        print(f"[kalibr] done; outputs in {self.output_dir}")
        return 0


@dataclass
class CalibIntrinsicsPublish:
    """Distill kalibr run dir(s) into the registry intrinsics npz."""

    front_run: Path | None = None
    """Run dir for the front lens (camchain or kalibr.log inside)."""
    back_run: Path | None = None
    """Run dir for the back lens."""
    front_cam: str = "cam0"
    back_cam: str = "cam0"
    """cam1 when the back lens came from a --lenses both run."""
    base: Path | None = None
    """Currently published npz to carry the untouched lens forward from."""
    camera: str | None = None
    """Registry unit to write the dated npz into (config/cameras/<unit>/)."""
    output: Path | None = None
    """Explicit output path (overrides --camera)."""

    def run(self) -> int:
        from kiwi.calib import intrinsics
        from kiwi.core.camera.registry import CameraUnit, default_artifact_out
        from kiwi.core.paths import config_root

        if self.front_run is None and self.back_run is None:
            print(
                "kiwi calib intrinsics publish: pass --front-run and/or --back-run",
                file=sys.stderr,
            )
            return 2
        if self.output is None and self.camera is None:
            print(
                "kiwi calib intrinsics publish: pass --camera <unit> or --output",
                file=sys.stderr,
            )
            return 2
        results = {}
        try:
            for side, run_dir, cam in (
                ("front", self.front_run, self.front_cam),
                ("back", self.back_run, self.back_cam),
            ):
                if run_dir is None:
                    continue
                r, salvage = intrinsics.lens_result(run_dir, cam)
                if salvage is not None:
                    print(f"[{side}] no camchain; salvaged {len(salvage)} batch(es):")
                    for b in salvage:
                        proj = " ".join(f"{v:.2f}" for v in b.projection)
                        print(f"    views={b.views_used:>3d}  proj=[{proj}]")
                results[side] = r
                kind = "ds" if isinstance(r, intrinsics.DoubleSphereResult) else "equi"
                print(f"[{side}] {kind} fit at {r.size[0]}x{r.size[1]}")
        except ValueError as e:
            print(f"kiwi calib intrinsics publish: {e}", file=sys.stderr)
            return 1

        unit_dir = (
            None if self.camera is None else config_root() / "cameras" / self.camera
        )
        out = self.output
        mode = None
        if out is None:
            assert unit_dir is not None
            if isinstance(CameraUnit.load(unit_dir).meta.get("intrinsics"), dict):
                # Mode-keyed unit: tag the filename with the solve's crop so
                # two modes published the same day cannot collide.
                size = next(iter(results.values())).size
                mode = f"{size[0]}x{size[1]}"
            try:
                out = default_artifact_out(unit_dir, "intrinsics", mode=mode)
            except FileExistsError as e:
                print(f"kiwi calib intrinsics publish: {e}", file=sys.stderr)
                return 2
        base = self.base
        if (
            base is None
            and len(results) == 1
            and unit_dir is not None
            and unit_dir.is_dir()
        ):
            try:
                published = CameraUnit.load(unit_dir).artifact("intrinsics")
            except ValueError:
                # Mode-keyed unit: one file per mode, nothing to carry
                # forward across lenses - the note below is X5-shaped.
                published = None
            if published is not None:
                print(
                    f"note: only one lens given and no --base - the other lens "
                    f"will be ABSENT from {Path(out).name}. To carry it forward: "
                    f"--base {published}"
                )
        try:
            keys = intrinsics.save_intrinsics_npz(out, results, base=base)
        except ValueError as e:
            print(f"kiwi calib intrinsics publish: {e}", file=sys.stderr)
            return 1
        print(f"[write] {out} with keys: {keys}")
        if self.output is None:
            pointer = (
                f"intrinsics: {{{mode}: {Path(out).name}}} (or add the "
                f"{mode!r} entry to the existing mapping)"
                if mode is not None
                else f"intrinsics: {Path(out).name}"
            )
            print(
                f"[publish] to activate: set `{pointer}` in "
                f"{unit_dir / 'camera.yaml'} (one reviewable commit)"
            )
        return 0


@dataclass
class CalibIntrinsicsKb4:
    """Fit KB4/equidistant to the published DS model for scene mapping and ground.

    Writes a lens-aware registry JSON and reports projection error vs FOV.
    """

    camera: str | None = None
    """Registry unit whose published intrinsics to use."""
    intrinsics: Path | None = None
    """Explicit intrinsics npz (overrides --camera for the DS source)."""
    lens: str = "front"
    """front | back."""
    fit_max_deg: float = 85.0
    """Half-angle the fit constrains; error beyond it is reported, not fit."""
    eval_max_deg: float = 95.0
    """Half-angle out to which error is reported."""
    output: Path | None = None
    """Explicit JSON output path (default: the unit's dated artifact)."""

    def run(self) -> int:
        import numpy as np

        from kiwi.calib import kb4
        from kiwi.core.camera.registry import CameraUnit, default_artifact_out
        from kiwi.core.paths import config_root

        if self.lens not in ("front", "back"):
            print("kiwi calib intrinsics kb4: --lens front|back", file=sys.stderr)
            return 2
        if self.camera is None:
            print(
                "kiwi calib intrinsics kb4: --camera is required - a "
                "lens-aware KB4 artifact records which physical unit it "
                "models (pass --intrinsics too to fit a non-published npz)",
                file=sys.stderr,
            )
            return 2
        unit_dir = config_root() / "cameras" / self.camera
        npz = self.intrinsics
        if npz is None:
            try:
                npz = CameraUnit.load(unit_dir).artifact("intrinsics")
            except ValueError as e:
                # Mode-keyed units are pinhole-equi single-lens cameras; KB4
                # fits the DS model, so there is no mode to pick here.
                print(f"kiwi calib intrinsics kb4: {e}", file=sys.stderr)
                return 2
            if npz is None:
                print(
                    f"kiwi calib intrinsics kb4: {self.camera} has no published "
                    "intrinsics - run `kiwi calib intrinsics` first",
                    file=sys.stderr,
                )
                return 2
        with np.load(str(npz)) as z:
            key = f"ds_{self.lens}"
            if key not in z.files:
                print(f"kiwi calib intrinsics kb4: {npz} has no {key}", file=sys.stderr)
                return 1
            ds = z[key]
            resolution = tuple(int(v) for v in z["image_size"])
        fit = kb4.fit_kb4(
            ds, resolution, fit_max_deg=self.fit_max_deg, eval_max_deg=self.eval_max_deg
        )
        fx, fy, cx, cy, k1, k2, k3, k4 = fit.params
        print(f"[kb4] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
        print(f"[kb4] k=[{k1:.6f}, {k2:.6f}, {k3:.6f}, {k4:.6f}]")
        print(
            f"  FOV half-angle   RMS px   max px   (fit <= {self.fit_max_deg:.0f} deg)"
        )
        for lo, hi, rms, mx in fit.rms_by_fov:
            print(f"   {lo:3.0f}-{hi:3.0f} deg      {rms:6.3f}   {mx:6.3f}")
        for thr, crossed in fit.crossings_deg.items():
            where = f"{crossed:.0f} deg" if crossed is not None else ">eval FOV"
            print(f"  RMS exceeds {thr:.1f}px beyond {where} half-angle")

        artifact_key = (
            "intrinsics_kb4" if self.lens == "front" else "intrinsics_kb4_back"
        )
        out = self.output
        if out is None:
            try:
                out = default_artifact_out(unit_dir, artifact_key)
            except FileExistsError as e:
                print(f"kiwi calib intrinsics kb4: {e}", file=sys.stderr)
                return 2
        kb4.write_kb4_json(out, fit, unit=self.camera, lens=self.lens)
        print(f"[write] {out}")
        if self.output is None:
            print(
                f"[publish] to activate: set `{artifact_key}: {Path(out).name}` in "
                f"{unit_dir / 'camera.yaml'} (one reviewable commit)"
            )
        return 0


@dataclass
class CalibIntrinsics:
    """Fisheye intrinsics via Kalibr: run the solve, publish the npz, fit KB4."""

    command: Annotated[
        (
            Annotated[
                CalibIntrinsicsRun, tyro.conf.subcommand("run", prefix_name=False)
            ]
            | Annotated[
                CalibIntrinsicsPublish,
                tyro.conf.subcommand("publish", prefix_name=False),
            ]
            | Annotated[
                CalibIntrinsicsKb4, tyro.conf.subcommand("kb4", prefix_name=False)
            ]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


class _NoHighgui(RuntimeError):
    """This opencv build has no GUI support - a backend fact, not a missing display."""


@dataclass
class CalibQrClock:
    """Display the fullscreen rolling QR clock for the cameras to film.

    Emits KIWI:<time.monotonic()> at microsecond precision - monotonic so an
    NTP step can never rewind or repeat a label. Needs a display; run it on
    the machine driving the filmed monitor.

    Two backends draw the SAME frames (kiwi.calib.qr.clock_frames owns the
    payload grid, so neither can drift from the other): opencv highgui, and
    Tk. The pinned `opencv-python-headless` has no highgui, so `auto` falls
    back to Tk; a missing display is refused.
    """

    qr_size: int = 700
    """Displayed QR side in pixels, before the quiet-zone border."""
    margin_frac: float = 0.12
    """White quiet-zone border as a fraction of qr_size (helps lock-on)."""
    fullscreen: bool = True
    """Fullscreen the window (drag it to the filmed monitor first if not)."""
    show_timestamp: bool = True
    """Render the payload as text under the QR - an aiming aid and the
    fallback clock reading when a frame's QR is unreadable."""
    hold_ms: float = 50.0
    """Hold each code this long before resampling the clock. A frame decodes
    only if NO code transition falls inside its exposure, so codes rolling at
    the display refresh (~10 ms) lose every frame once exposure passes ~10 ms.
    ~50 ms tolerates ordinary auto-exposure. 0 = resample every loop (locked
    fast shutters only)."""
    backend: str = "auto"
    """Which window to draw in: `auto` (highgui, else Tk), `cv2`, or `tk`."""

    def run(self) -> int:
        import time

        import cv2

        from kiwi.calib.qr import clock_frames

        if self.backend not in ("auto", "cv2", "tk"):
            print(
                f"kiwi calib qr clock: --backend {self.backend!r}, "
                "expected one of auto, cv2, tk",
                file=sys.stderr,
            )
            return 2
        encoder = cv2.QRCodeEncoder.create()
        margin = int(self.qr_size * self.margin_frac)
        hold_s = max(0.0, self.hold_ms) / 1e3
        frames = clock_frames(
            encoder,
            self.qr_size,
            margin,
            self.show_timestamp,
            hold_s,
            time.monotonic,
        )
        banner = "Point every camera at this window. Press q to stop." + (
            f"  [holding each code {self.hold_ms:.0f} ms]" if hold_s else ""
        )
        if self.backend == "tk":
            return self._tk(frames, banner)
        try:
            return self._cv2(frames, banner)
        except _NoHighgui as e:
            if self.backend == "cv2":
                print(
                    f"kiwi calib qr clock: no highgui in this opencv build ({e}); "
                    "use --backend tk, or install a GUI opencv",
                    file=sys.stderr,
                )
                return 2
            print("[clock] no highgui in this opencv build; using the Tk backend")
            return self._tk(frames, banner)

    def _cv2(self, frames, banner: str) -> int:
        import time

        import cv2

        window = "KIWI QR clock (film me)"
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            raise _NoHighgui(str(e)) from e
        if self.fullscreen:
            cv2.setWindowProperty(
                window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        print(banner)
        n, t0, last = 0, time.monotonic(), None
        try:
            for payload, frame in frames:
                if frame is not last:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    last = frame
                cv2.imshow(window, bgr)
                n += 1
                if n % 120 == 0:
                    self._rate(n, time.monotonic() - t0, payload)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
        print()
        return 0

    def _tk(self, frames, banner: str) -> int:
        """Same frames in a Tk window, for an opencv without highgui.

        Tk owns the loop, so the redraw is an `after` callback rather than a
        `while`; `clock_frames` still decides when the payload advances, so
        the two backends agree frame for frame.
        """
        import time
        import tkinter as tk

        from kiwi.calib.qr import pgm_bytes

        try:
            root = tk.Tk()
        except tk.TclError as e:
            print(
                f"kiwi calib qr clock: no display available ({e}); this verb "
                "runs on the machine driving the filmed monitor",
                file=sys.stderr,
            )
            return 2
        root.title("KIWI QR clock (film me)")
        root.configure(bg="white")
        if self.fullscreen:
            root.attributes("-fullscreen", True)
        label = tk.Label(root, bg="white", borderwidth=0, highlightthickness=0)
        label.pack(expand=True)
        for seq in ("<q>", "<Escape>"):
            root.bind(seq, lambda _e: root.destroy())

        state = {"n": 0, "t0": time.monotonic()}

        def tick():
            payload, frame = next(frames)
            image = tk.PhotoImage(data=pgm_bytes(frame), format="PPM")
            label.configure(image=image)
            # Tk keeps no reference of its own, so the image would be
            # collected between callbacks and the label would go blank.
            label.image = image
            state["n"] += 1
            if state["n"] % 120 == 0:
                self._rate(state["n"], time.monotonic() - state["t0"], payload)
            root.after(1, tick)

        print(banner)
        tick()
        root.mainloop()
        print()
        return 0

    @staticmethod
    def _rate(n: int, elapsed: float, payload: str) -> None:
        if elapsed <= 0:
            return
        print(f"\rloop {n / elapsed:6.1f} Hz   payload {payload}", end="", flush=True)


@dataclass
class CalibQrProbe:
    """Fit one clip to the filmed QR clock: monotonic = video_t + offset."""

    video: tyro.conf.Positional[Path]
    """The clip that filmed the clock window (a demuxed *_lens0.mp4)."""
    start: float = 0.0
    """Video time (s) to start scanning at (skip to the filmed-clock span)."""
    duration: float | None = None
    """Scan span in video seconds (default: until --max-decodes or EOF)."""
    stride: int = 1
    """Decode every Nth frame."""
    max_decodes: int = 300
    """Stop after this many successful decodes."""
    fit_drift: bool = False
    """Model the camera-clock drift (Theil-Sen) instead of a pure offset -
    worth it only when the filmed span is minutes long."""
    undistort: Path | None = None
    """Per-lens intrinsics npz; REQUIRED for fisheye clips (the flat decoder
    locates the code on a raw fisheye frame, then fails every decode)."""
    undistort_fov: float = 130.0
    """Virtual-pinhole FOV (deg) for --undistort."""
    undistort_lens: str = "front"
    """Which lens of a dual-lens bundle to rectify with."""
    json_output: Path | None = None
    """Optional JSON report path (schema qr_time_probe_v1)."""

    def run(self) -> int:
        from kiwi.calib import qr

        try:
            scan = qr.scan_video(
                self.video,
                start=self.start,
                duration=self.duration,
                stride=self.stride,
                max_decodes=self.max_decodes,
                undistort=self.undistort,
                undistort_fov=self.undistort_fov,
                undistort_lens=self.undistort_lens,
            )
            fit = qr.fit_clip(scan, self.fit_drift)
        except RuntimeError as e:
            print(f"kiwi calib qr probe: {e}", file=sys.stderr)
            return 1
        model = (
            "Theil-Sen drift model" if self.fit_drift else "median, slope fixed at 1"
        )
        print(
            f"decoded {fit['n_decoded']}/{fit['n_scanned']} scanned frames "
            f"({100.0 * fit['n_decoded'] / fit['n_scanned']:.1f}%) "
            f"over video t = {scan.t_video.min():.2f} .. {scan.t_video.max():.2f} s "
            f"({fit['fps']:.2f} fps)"
        )
        print(f"offset (monotonic - video) = {fit['offset_s']:+.3f} s   [{model}]")
        print(f"drift (Theil-Sen, diagnostic) = {fit['drift_ppm']:+.1f} ppm")
        print(
            f"residuals: median {fit['resid_median_ms']:.1f} ms, "
            f"p95 {fit['resid_p95_ms']:.1f} ms, max {fit['resid_max_ms']:.1f} ms; "
            f"{fit['n_outliers_100ms']} outlier(s) > 100 ms"
        )
        if fit["n_outliers_100ms"]:
            print("warning: outliers present - check for a second clock/QR in frame")
        if self.json_output is not None:
            report = {
                "schema": "qr_time_probe_v1",
                "video": str(self.video),
                "fit_drift": self.fit_drift,
                **fit,
            }
            self.json_output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"wrote {self.json_output}")
        return 0


def _qr_fit_one(video: Path, cmd, label: str, undistort: Path | None) -> dict | None:
    """Scan + fit one side of a pair, with the two-failure-mode diagnosis."""
    from kiwi.calib import qr

    scan = qr.scan_video(
        video,
        start=cmd.start,
        duration=cmd.duration,
        stride=cmd.stride,
        max_decodes=cmd.max_decodes,
        undistort=undistort,
        undistort_fov=cmd.undistort_fov,
    )
    try:
        return qr.fit_clip(scan, cmd.fit_drift)
    except RuntimeError as e:
        # Never located = framing/distance/glare; located-but-undecoded =
        # exposure spanning code transitions (hold codes longer) or, on a
        # fisheye, the module grid bent past the flat decoder (rectify).
        hint = (
            "the QR was never located - check framing, distance, focus or glare"
            if qr.located_fraction(video) < 0.2
            else "the QR was LOCATED but never decoded - shorten the exposure or "
            "hold codes longer (raise qr clock --hold-ms above its 50 ms default); "
            f"for fisheye clips pass --{label}-undistort <intrinsics.npz>"
        )
        print(f"kiwi calib qr sync: {label}: {e} - {hint}", file=sys.stderr)
        return None


@dataclass
class CalibQrSync:
    """Two clips of one filmed clock -> the cross-camera offset ground truth.

    Display latency and refresh phase are common to both cameras and cancel
    in the difference; positive delta = right started later (the audio and
    episode.yaml sign convention). --audio prints an independent cross-check
    next to it.
    """

    left: tyro.conf.Positional[Path]
    """Lens clip from the LEFT camera that filmed the monitor."""
    right: tyro.conf.Positional[Path]
    """Lens clip from the RIGHT camera that filmed the same monitor."""
    start: float = 0.0
    duration: float | None = None
    stride: int = 1
    max_decodes: int = 600
    fit_drift: bool = False
    left_undistort: Path | None = None
    """Intrinsics npz for the LEFT clip (needed for fisheye clips)."""
    right_undistort: Path | None = None
    """Intrinsics npz for the RIGHT clip."""
    undistort_fov: float = 130.0
    left_insv: Path | None = None
    """Original LEFT .insv (the demuxed mp4 drops the audio track)."""
    right_insv: Path | None = None
    """Original RIGHT .insv."""
    audio: bool = False
    """Cross-check the RAW audio offset against the QR delta (needs both .insv).
    Raw on purpose: the difference IS the per-mode A/V constant - see
    `kiwi calib qr av-delay`."""
    json_output: Path | None = None
    """Optional JSON report path (schema qr_sync_v1)."""

    def run(self) -> int:
        from kiwi.calib import qr

        fit_l = _qr_fit_one(self.left, self, "left", self.left_undistort)
        fit_r = _qr_fit_one(self.right, self, "right", self.right_undistort)
        if fit_l is None or fit_r is None:
            return 1
        delta, se = qr.cross_camera_delta(fit_l, fit_r)

        print("\n== cross-camera QR ground truth ==")
        for name, f in (("left ", fit_l), ("right", fit_r)):
            print(
                f"  {name}: offset {f['offset_s']:+.4f} s  "
                f"(n={f['n_decoded']}/{f['n_scanned']}, {f['fps']:.2f} fps, "
                f"resid med {f['resid_median_ms']:.1f} ms, drift {f['drift_ppm']:+.0f} ppm)"
            )
        print(
            f"  delta (QR)  = {delta * 1e3:+.1f} ms  +/- {se:.1f} ms   "
            "[positive = right started later]"
        )

        # The cross-check composes sibling components HERE - the CLI may
        # import both kiwi.calib and kiwi.sync; the components may not.
        rows = [("QR ground truth", delta * 1e3, f"+/- {se:.1f} ms")]
        audio_ms = self._audio_offset_ms() if self.audio else None
        if audio_ms is not None:
            rows.append(
                ("audio GCC-PHAT", audio_ms, f"{audio_ms - delta * 1e3:+.1f} ms vs QR")
            )
        if self.audio:
            print("\n== comparison (offset of right vs left) ==")
            print(f"  {'method':<18}{'offset':>12}   note")
            for name, val, note in rows:
                print(f"  {name:<18}{val:>+9.1f} ms   {note}")

        if self.json_output is not None:
            report = {
                "schema": "qr_sync_v1",
                "left": str(self.left),
                "right": str(self.right),
                "delta_qr_ms": delta * 1e3,
                "delta_qr_se_ms": se,
                "audio_offset_ms": audio_ms,
                "left_fit": fit_l,
                "right_fit": fit_r,
            }
            self.json_output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"\nwrote {self.json_output}")
        return 0

    def _audio_offset_ms(self) -> float | None:
        if self.left_insv is None or self.right_insv is None:
            print("warning: --audio needs both --left-insv and --right-insv; skipping")
            return None
        try:
            from kiwi.sync import estimate_raw_offset

            raw = estimate_raw_offset(self.left_insv, self.right_insv)
        except Exception as exc:  # noqa: BLE001 - a cross-check must never fail the run
            print(f"warning: audio cross-check failed ({exc}); skipping")
            return None
        if not raw.confident:
            print(
                f"warning: audio gates failed ({', '.join(raw.failed_gates)}); "
                "reporting anyway"
            )
        return raw.offset_s * 1e3


@dataclass
class CalibQrAvDelay:
    """Measure a camera's per-mode A/V delay against the QR ground truth.

    Uses the RAW audio offset: its difference against the QR video offset
    IS the constant being measured. The result lands in the unit's dated
    av-sync.yaml; repeated runs of one mode accumulate and average. An
    estimate that failed the audio gates or exceeds the plausible bound is
    refused unwritten (--force overrides).
    """

    ref: tyro.conf.Positional[Path]
    """Reference clip (.insv) - the camera/mode that defines the zero."""
    other: tyro.conf.Positional[Path]
    """The clip (.insv) whose A/V delay is being measured."""
    camera: str | None = None
    """Registry unit the measured delay belongs to (writes its dated table)."""
    output: Path | None = None
    """Explicit output path (overrides --camera)."""
    qr_json: Path | None = None
    """A `kiwi calib qr sync --json-output` report; its delta_qr_ms is the truth."""
    video_offset_ms: float | None = None
    """Ground-truth video offset in ms, if held outside a qr report."""
    ref_delay_s: float = 0.0
    """The reference camera's own delay (0.0 = it defines the zero)."""
    run_label: str | None = None
    """Provenance name for this run (default: the qr report's parent dir)."""
    reference: str | None = None
    """What defines the scale's zero, recorded in a fresh table (default:
    derived from the ref clip's probed mode)."""
    force: bool = False
    """Write even when the audio gates failed or the delay is implausible; a
    deliberate override only."""

    def run(self) -> int:
        from kiwi.calib import avsync
        from kiwi.core.camera.registry import CameraUnit, default_artifact_out
        from kiwi.core.paths import config_root
        from kiwi.sync import estimate_raw_offset
        from kiwi.sync.audio import EXTRACTION, av_delay_from_ground_truth

        if self.qr_json is not None:
            video_off = json.loads(self.qr_json.read_text())["delta_qr_ms"] / 1e3
        elif self.video_offset_ms is not None:
            video_off = self.video_offset_ms / 1e3
        else:
            print(
                "kiwi calib qr av-delay: pass --qr-json or --video-offset-ms "
                "(run `kiwi calib qr sync ... --json-output <json>` first)",
                file=sys.stderr,
            )
            return 2
        if self.output is None and self.camera is None:
            print(
                "kiwi calib qr av-delay: pass --camera <unit> or --output",
                file=sys.stderr,
            )
            return 2

        try:
            raw = estimate_raw_offset(self.ref, self.other)
        except RuntimeError as e:
            print(f"kiwi calib qr av-delay: {e}", file=sys.stderr)
            return 1
        delay = av_delay_from_ground_truth(
            raw_audio_offset_s=raw.offset_s,
            video_offset_s=video_off,
            ref_delay_s=self.ref_delay_s,
        )
        print(f"ref mode        : {raw.ref.mode.key}")
        print(f"other mode      : {raw.other.mode.key}")
        print(f"QR video offset : {video_off:+.4f} s")
        print(
            f"raw audio offset: {raw.offset_s:+.4f} s   "
            f"(votes {raw.votes}/{raw.runner_up}, "
            f"{'OK' if raw.confident else 'GATES: ' + ','.join(raw.failed_gates)})"
        )
        print(f"=> av_delay_s   : {delay:+.4f} s ({delay * 1e3:+.1f} ms)")

        # Refuse BEFORE any write: a written value is averaged into av_delay_s
        # and then corrects every mixed-mode episode synced afterwards. Like
        # the readers, this writer refuses non-confident offsets.
        forced: list[str] = []
        if not raw.confident:
            gates = ", ".join(raw.failed_gates)
            if not self.force:
                print(
                    f"kiwi calib qr av-delay: audio gates failed ({gates}); "
                    f"nothing written. Re-shoot the run, or pass --force to "
                    f"write this estimate anyway.",
                    file=sys.stderr,
                )
                return 1
            forced.append(f"gates {gates}")
            print(
                f"[forced] writing despite failed audio gates ({gates})",
                file=sys.stderr,
            )
        bound = avsync.AV_DELAY_PLAUSIBLE_MAX_S
        # Written as "not <= bound" so a non-finite delay refuses too: NaN is
        # the least plausible value there is, and `abs(nan) > bound` is False.
        if not (abs(delay) <= bound):
            if not self.force:
                print(
                    f"kiwi calib qr av-delay: |av_delay_s| = {abs(delay):.4f} s "
                    f"exceeds the plausible bound of {bound} s; nothing written. "
                    f"Likely ref/other swapped against the qr sync left/right, "
                    f"or a periodic false lock; pass --force to write it anyway.",
                    file=sys.stderr,
                )
                return 1
            forced.append(f"|delay| {abs(delay):.4f}s > {bound}s")
            print(
                f"[forced] writing an implausible delay ({delay:+.4f} s, "
                f"bound {bound} s)",
                file=sys.stderr,
            )

        unit_dir = (
            None if self.camera is None else config_root() / "cameras" / self.camera
        )
        out = self.output
        if out is None:
            assert unit_dir is not None
            try:
                out = default_artifact_out(unit_dir, "av_sync")
            except FileExistsError as e:
                print(f"kiwi calib qr av-delay: {e}", file=sys.stderr)
                return 2
        doc = None
        if Path(out).is_file():
            doc = avsync.load_av_sync(out)
        elif unit_dir is not None and unit_dir.is_dir():
            # A new dated document STARTS from the published table. Constants
            # are only valid under the extraction recipe they were measured
            # with, so a changed recipe starts a fresh table, with a warning.
            #
            # Every failure here degrades to a fresh table with a reason on
            # stderr, never a raise.
            try:
                published = CameraUnit.load(unit_dir).artifact("av_sync")
            except (FileNotFoundError, KeyError, ValueError, OSError) as e:
                published = None
                print(f"[seed] no table to carry forward: {e}", file=sys.stderr)
            if published is not None:
                try:
                    doc = avsync.carry_forward(published, extraction=EXTRACTION)
                    print(
                        f"[seed] {published.name}: carrying "
                        f"{len(doc.get('modes') or {})} measured mode(s) forward"
                    )
                except avsync.ExtractionMismatch as e:
                    print(
                        f"[seed] {published.name} cannot carry forward:\n"
                        f"         published: {e.published!r}\n"
                        f"         current  : {e.current!r}\n"
                        f"       starting a fresh table - every mode needs "
                        f"re-measuring before this one is published.",
                        file=sys.stderr,
                    )
                except (ValueError, OSError) as e:
                    print(
                        f"[seed] {published.name} could not be read ({e}); "
                        f"starting a fresh table",
                        file=sys.stderr,
                    )
        if doc is None:
            model = None
            if unit_dir is not None and unit_dir.is_dir():
                try:
                    model = CameraUnit.load(unit_dir).meta.get("model")
                except (FileNotFoundError, ValueError, OSError):
                    pass  # the table is written without a model name
            doc = avsync.new_av_sync_doc(
                unit=self.camera or Path(out).stem,
                model=model,
                reference=self.reference
                or f"{self.ref.name} @ {raw.ref.mode.key} (defines zero)",
                extraction=EXTRACTION,
            )
        run_label = self.run_label or (
            self.qr_json.parent.name if self.qr_json is not None else "manual"
        )
        if forced:
            # Label the run itself, so the table marks its derived_ms entry -
            # the one being averaged in - as forced.
            run_label = f"{run_label} [forced: {'; '.join(forced)}]"
        entry = avsync.add_measurement(
            doc, mode_key=raw.other.mode.key, delay_s=delay, run=run_label
        )
        avsync.write_av_sync(out, doc)
        print(
            f"[write] {out}: modes[{raw.other.mode.key}] = "
            f"{entry['av_delay_s']:+.4f} s over {len(entry['runs'])} run(s)"
        )
        if self.output is None:
            print(
                f"[publish] to activate: set `av_sync: {Path(out).name}` in "
                f"{unit_dir / 'camera.yaml'} (one reviewable commit)"
            )
        return 0  # a refused estimate returned above; --force is a conscious 0


@dataclass
class CalibQr:
    """The rolling-QR clock rig: display, per-clip probe, pair delta, A/V delay."""

    command: Annotated[
        (
            Annotated[CalibQrClock, tyro.conf.subcommand("clock", prefix_name=False)]
            | Annotated[CalibQrProbe, tyro.conf.subcommand("probe", prefix_name=False)]
            | Annotated[CalibQrSync, tyro.conf.subcommand("sync", prefix_name=False)]
            | Annotated[
                CalibQrAvDelay, tyro.conf.subcommand("av-delay", prefix_name=False)
            ]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


@dataclass
class Calib:
    """Per-device calibration workflows, published to the camera registry."""

    command: Annotated[
        (
            Annotated[
                CalibIntrinsics, tyro.conf.subcommand("intrinsics", prefix_name=False)
            ]
            | Annotated[CalibImu, tyro.conf.subcommand("imu", prefix_name=False)]
            | Annotated[CalibQr, tyro.conf.subcommand("qr", prefix_name=False)]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


# --------------------------------------------------------------------------- #
# `kiwi sync` - the diagnostic front end to kiwi.sync
#
# `kiwi run` is what puts an episode on its clock; these are for looking at one
# pair when you want to know why, or whether. Measuring a camera's A/V delay
# needs a rolling-QR ground-truth capture and lives in `kiwi calib qr av-delay`,
# which owns the av_sync_v1 artifact.
# --------------------------------------------------------------------------- #


def _sync_clip(path: Path, unit: str | None, label: str):
    from kiwi.core.paths import config_root
    from kiwi.sync import Clip

    return Clip(path, (config_root() / "cameras" / unit) if unit else None, label)


@dataclass
class SyncPair:
    """Offset of `other` relative to `ref` on the shared video timeline.

    Both camera identities are required to verify the A/V correction.
    """

    ref: tyro.conf.Positional[Path]
    """Reference clip (an .insv carries its audio on stream 0:a:0)."""
    other: tyro.conf.Positional[Path]
    """The clip being placed. A positive result means it started later."""
    ref_camera: str | None = None
    """Required config/cameras/<unit> identity for `ref`."""
    other_camera: str | None = None
    """Required config/cameras/<unit> identity for `other`."""
    json_output: Path | None = None
    """Also write the full result as JSON here."""

    def run(self) -> int:
        from kiwi.sync import estimate_offset

        res = estimate_offset(
            _sync_clip(self.ref, self.ref_camera, "ref"),
            _sync_clip(self.other, self.other_camera, "other"),
        )
        print(res.report())
        if self.json_output:
            self.json_output.write_text(
                json.dumps(
                    {
                        "ref": str(self.ref),
                        "other": str(self.other),
                        "offset_s": res.offset_s,
                        "raw_audio_offset_s": res.raw.offset_s,
                        "av_correction_s": res.correction.seconds,
                        "av_provenance": res.correction.provenance,
                        "confident": res.confident,
                        "failed_gates": res.failed_gates,
                        "votes": res.raw.votes,
                        "runner_up": res.raw.runner_up,
                        "diagnostics": res.raw.diagnostics,
                    },
                    indent=1,
                )
                + "\n"
            )
            print(f"\nwrote {self.json_output}")
        return 0 if res.confident else 1


@dataclass
class SyncSurvey:
    """Run `pair` over the two arms of every episode under a directory.

    Each episode's manifest names the clip and the camera unit of each arm, so
    the survey needs no camera flags. It is the validation tool for the sync
    recipe: after changing a `SyncParams` default, re-run it over real
    episodes and compare the confident count and the gate margins.
    """

    root: tyro.conf.Positional[Path]
    """A directory of episode dirs, each holding an `episode.yaml`."""
    tmcd: bool = False
    """Also print the audio-vs-timecode skew. A cross-check only: a
    disagreement measures the timecode's own error, it does not invalidate the
    audio offset."""

    def run(self) -> int:
        from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig
        from kiwi.sync import Clip, estimate_offset

        rows = []
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            manifest = d / MANIFEST_NAME
            if not manifest.is_file():
                continue
            try:
                cfg = EpisodeConfig.load(manifest, check_exists=False)
                arms = [Clip(a.recording, a.camera_dir, k) for k, a in cfg.arms.items()]
                if len(arms) != 2:
                    raise ValueError(
                        f"a survey pairs exactly two arms; {manifest} names {len(arms)}"
                    )
                ref, other = arms
                r = estimate_offset(ref, other)
            except Exception as exc:  # noqa: BLE001 - one bad episode must not stop a survey
                print(f"{d.name:18s} ERROR {exc}")
                continue
            skew = "      --"
            if self.tmcd:
                from kiwi.sync.tmcd import read_tmcd_seconds

                try:
                    tm = read_tmcd_seconds(other.path) - read_tmcd_seconds(ref.path)
                    skew = f"{(r.offset_s - tm) * 1e3:+8.0f}"
                except Exception:  # noqa: BLE001 - timecode is optional everywhere
                    pass
            fl = "FALSE-LOCK" if r.raw.diagnostics["argmax_is_false_lock"] else ""
            print(
                f"{d.name:18s} {other.label}-{ref.label} {r.offset_s:+9.4f}s  "
                f"votes {r.raw.votes:5d}/{r.raw.runner_up:<4d} "
                f"onsets {r.raw.onsets_matched}/{r.raw.onsets_total}  "
                f"tmcd skew {skew} ms  "
                f"{'OK ' if r.confident else 'GATE-FAIL'} {fl}"
            )
            rows.append(r)
        if rows:
            n = len(rows)
            conf = sum(1 for r in rows if r.confident)
            fls = sum(1 for r in rows if r.raw.diagnostics["argmax_is_false_lock"])
            print(
                f"\n{n} pairs | confident {conf}/{n} | raw-argmax false-locks {fls}/{n}"
            )
            print("onset counts are advisory: reported, not a gate")
        else:
            print(f"no episodes (no {MANIFEST_NAME}) under {self.root}")
        return 0


@dataclass
class SyncDrift:
    """Relative media-clock rate of `other` vs `ref`, from sliding-window refines.

    A positive slope means the offset grows over time. Constant per-device A/V
    stamping delays cancel in the slope, so no camera registry is needed.
    Uses the pair estimator's measurements and acceptance policy. Missing or
    limited drift evidence lowers the reported confidence level; measured
    excessive drift remains a refusal.
    """

    ref: tyro.conf.Positional[Path]
    """Reference clip."""
    other: tyro.conf.Positional[Path]
    """The clip whose clock rate is measured against `ref`."""
    window_s: float | None = None
    """Sliding-window length in seconds (default: SyncParams.drift_window_s)."""
    hop_s: float | None = None
    """Window hop in seconds (default: SyncParams.drift_hop_s)."""

    def run(self) -> int:
        from kiwi.sync import SyncParams
        from kiwi.sync.audio import estimate_raw_offset

        p = SyncParams()
        p = dataclasses.replace(
            p,
            drift_window_s=p.drift_window_s if self.window_s is None else self.window_s,
            drift_hop_s=p.drift_hop_s if self.hop_s is None else self.hop_s,
        )
        raw = estimate_raw_offset(self.ref, self.other, params=p)
        dur = raw.diagnostics["overlap_s"]
        d = raw.drift
        print(f"pair      : {self.ref.name} -> {self.other.name}  ({dur:.0f} s shared)")
        print(f"offset    : {raw.offset_s:+.4f} s (audio timeline)")
        if d is not None:
            for t, o in d.window_offsets:
                print(f"  t={t:7.1f} s  offset {o:+.6f} s")
            sig = "CI excludes zero" if d.significant else "not significant"
            print(
                f"drift     : {d.slope_ppm:+.1f} ppm "
                f"[{d.ci_low_ppm:+.1f}, {d.ci_high_ppm:+.1f}] "
                f"({sig}; {d.n_windows} windows, scatter {d.scatter_ms:.2f} ms, "
                f"coverage {d.coverage_fraction:.0%})"
            )
            print(f"accum     : {d.accumulated_ms:+.2f} ms over the measured span")
        print(f"state     : {raw.diagnostics['drift_status']}")
        if "matched_audio_coverage_s" in raw.diagnostics:
            detail = raw.diagnostics
            print(
                f"audio used: ref {detail['ref_audio_duration_s']:.1f} s, "
                f"other {detail['other_audio_duration_s']:.1f} s decoded; "
                f"{detail['refinement_audio_coverage_s']:.1f} s in refinement windows"
            )
            print(
                f"match     : ~{detail['matched_audio_coverage_s']:.1f} s "
                f"({detail['matched_audio_bin_s']:.0f} s bins), "
                f"span {detail['vote_support_span_s']:.1f} s; "
                f"votes {raw.votes} vs {raw.runner_up} "
                f"({detail['vote_margin']:.1f}x); "
                f"landmarks {detail['landmark_matched_fraction']:.1%}; "
                f"onsets {raw.onsets_matched}/{raw.onsets_total} (advisory, not a gate)"
            )
        confidence = raw.diagnostics.get("confidence")
        if confidence is not None:
            print(f"confidence: {confidence['level']} (evidence level)")
        error = raw.diagnostics["drift_error_bound_s"]
        if error is not None:
            print(f"budget    : {error * 1e3:.2f} / {p.max_drift_error_s * 1e3:.2f} ms")
        for name, passed in raw.gates.items():
            print(f"gate      : {name} {'PASS' if passed else 'FAIL'}")
        return 0 if all(raw.gates.values()) else 1


# --------------------------------------------------------------------------
# kiwi register - one marker-defined world frame per episode (kiwi/register)
# --------------------------------------------------------------------------


@dataclass
class RegisterScan:
    """Scan ONE clip for the anchor marker -> a `marker_detections_v1` json.

    Episode-wide runs go through the register stage (`kiwi
    run --episode <dir> --from-stage register --to-stage register`), which
    scans and caches `<arm>/detections.json` per arm; this verb is for a
    standalone clip, or for re-scanning with different knobs. The scan is the
    expensive half and records EVERY detection with its distance, so the
    solve (`kiwi register anchor`) can retune --max-marker-dist without
    another pass over the video.
    """

    video: tyro.conf.Positional[Path]
    """Demuxed FRONT-lens mp4 (the convert stage's `<arm>/lens0.mp4`).
    A `lens1.mp4` is refused: the anchor is solved in the front camera's frame."""
    marker_id: int
    """ArUco id of the anchor marker (episode.yaml `anchor.marker.id`)."""
    marker_size: float
    """Marker black-square edge in METRES - sets the PnP metric scale."""
    marker_dict: str = "DICT_4X4_50"
    """OpenCV ArUco dictionary the marker belongs to."""
    camera: str | None = None
    """Registry unit for intrinsics; default: matched by the clip's serial
    (which a demuxed mp4 does not carry - name the unit)."""
    intrinsics: Path | None = None
    """Explicit intrinsics npz (escape hatch)."""
    scan_seconds: float = 120.0
    """Scan the first N seconds - the marker only needs to be seen early."""
    fov_deg: float = 140.0
    """Virtual-pinhole FOV for the DS->pinhole rectifier. Wider keeps a
    peripheral marker in view; narrower gives the marker more pixels."""
    out_size: int = 1920
    """Rectified image size (square). Larger preserves more marker detail."""
    output: Path | None = None
    """Destination json (default: `<video stem>.detections.json` beside it)."""
    debug_image: Path | None = None
    """Write the best (lowest-reproj) detection frame with the marker axes
    drawn, to sanity-check the anchor visually."""

    def run(self) -> int:
        from kiwi.register import detections as det_mod

        out = self.output or self.video.with_name(self.video.stem + ".detections.json")
        try:
            if self.output is None:
                from kiwi.core.layout import refuse_raw_output

                refuse_raw_output(out, "detections")
            # Refuse a rear clip before the intrinsics lookup, which would
            # return the unit's front-lens fit for it.
            det_mod.refuse_rear_clip(self.video)
            intr = _clip_intrinsics(self.video, self.camera, self.intrinsics)
            meta, obs = det_mod.scan_video_for_marker(
                self.video,
                intrinsics=intr,
                marker_id=self.marker_id,
                marker_size=self.marker_size,
                marker_dict=self.marker_dict,
                fov_deg=self.fov_deg,
                out_size=self.out_size,
                scan_seconds=self.scan_seconds,
            )
            det_mod.write_detections(out, meta=meta, observations=obs)
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as e:
            print(f"kiwi register scan: {e}", file=sys.stderr)
            return 1
        print(
            f"{len(obs)} detection(s) of marker {self.marker_id} in "
            f"{meta['frames_scanned']} frame(s) -> {out}"
        )
        if not obs:
            print(
                "  none found: try a larger --scan-seconds, a wider --fov-deg, "
                "or check the marker id/dictionary. The rear lens is not an "
                "option: the anchor is solved in the front camera's frame.",
                file=sys.stderr,
            )
        if self.debug_image is not None and obs:
            p = det_mod.write_debug_image(self.video, intr, meta, obs, self.debug_image)
            if p is not None:
                print(f"debug overlay -> {p}")
        return 0


@dataclass
class RegisterAnchor:
    """Anchor ONE tracked trajectory to cached detections -> a world .tum.

    The solve half: retuning --max-marker-dist re-runs against the same
    scan. Episode-wide runs go through the register
    stage (`kiwi
    run --episode <dir> --from-stage register --to-stage register`), which
    also picks the stream offset out of the manifest for you.
    """

    track_tum: tyro.conf.Positional[Path]
    """A camera trajectory such as `<arm>/scene/state.tum`, on the episode global
    clock, declared in its header (anything else is refused, not guessed)."""
    detections: Path
    """`marker_detections_v1` json from `kiwi register scan` (or the stage)."""
    output: Path
    """Destination world-frame .tum."""
    stream_offset: float = 0.0
    """Where this stream's video frame 0 sits on the episode global clock
    (the manifest's `sync.offsets.<label>`, <= 0; 0.0 for the origin
    stream)."""
    max_marker_dist: float = 1.0
    """Drop detections whose marker is farther than this (metres). Distant/
    oblique planar-marker views risk the PnP flip ambiguity - a far, wrong
    pose with an equally low reprojection error. `inf` disables."""
    flip_check: bool = True
    """Refuse to write an anchor the flip checks call corrupted.
    --no-flip-check still prints the verdicts but writes the trajectory."""

    def run(self) -> int:
        from kiwi.core.trajectory import read_tum_header, write_tum
        from kiwi.register.anchor import solve_anchor
        from kiwi.register.detections import load_detections

        try:
            _meta, obs = load_detections(self.detections)
            result = solve_anchor(
                self.track_tum,
                obs,
                stream_offset_s=self.stream_offset,
                max_marker_dist=self.max_marker_dist,
            )
        except (ValueError, FileNotFoundError, OSError) as e:
            print(f"kiwi register anchor: {e}", file=sys.stderr)
            return 1
        print(
            f"[detect] {result.n_detected} detection(s), {result.n_too_far} "
            f"beyond --max-marker-dist, {result.n_in_range} within track coverage"
        )
        print(
            f"[anchor] marker origin in the track frame = "
            f"{result.T_track_marker[:3, 3].round(4)}"
        )
        print(f"[flip-check] {result.spread[1]}")
        print(f"[flip-check] {result.reach[1]}")
        verdict, msg = result.verdict
        if verdict == "halt" and self.flip_check:
            print(
                f"kiwi register anchor: anchor rejected - {msg}\n"
                f"  (pass --no-flip-check to write it anyway)",
                file=sys.stderr,
            )
            return 1
        if verdict != "ok":
            print(f"[flip-check] {verdict.upper()}: {msg}")
        # The output inherits the input's clock declaration - solve_anchor
        # already refused anything that is not the episode global clock.
        clock = read_tum_header(self.track_tum).clock
        write_tum(
            result.trajectory, self.output, clock=clock, source=self.detections.name
        )
        print(
            f"wrote {len(result.trajectory)} poses in frame "
            f"'{result.trajectory.frame}' -> {self.output}"
        )
        return 0


@dataclass
class Register:
    """Scan for the anchor marker, or anchor one trajectory (diagnostics).

    The register stage anchors a whole episode (both arms into the one
    marker-defined world frame) as an on-demand ground-truth reference that
    no builtin pipeline profile requests; run it with `kiwi
    run --episode <dir> --from-stage register --to-stage register`. These
    subcommands are for one clip or one trajectory, when you want to know
    why, or whether.
    """

    command: Annotated[
        (
            Annotated[RegisterScan, tyro.conf.subcommand("scan", prefix_name=False)]
            | Annotated[
                RegisterAnchor, tyro.conf.subcommand("anchor", prefix_name=False)
            ]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


@dataclass
class Sync:
    """Inspect the audio time sync of one pair of clips (diagnostics).

    `kiwi run` is what puts a whole episode on its clock; these subcommands are
    for looking at a single pair when you want to know why, or whether.
    """

    # OmitArgPrefixes keeps the flags spelled `--tmcd`, not `--command.tmcd`;
    # prefix_name=False keeps the subcommands `pair`, not `command:pair`.
    command: Annotated[
        (
            Annotated[SyncPair, tyro.conf.subcommand("pair", prefix_name=False)]
            | Annotated[SyncSurvey, tyro.conf.subcommand("survey", prefix_name=False)]
            | Annotated[SyncDrift, tyro.conf.subcommand("drift", prefix_name=False)]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


# --------------------------------------------------------------------------
# kiwi eef - per-tool end-effector state (kiwi/eef)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _EefResolved:
    """The tool/unit resolution shared by the eef verbs (flags win)."""

    unit: object | None
    unit_path: Path | None
    tool: str
    estimator_key: str
    session_block: dict
    tool_session: dict
    """The DESIGN's ``session:`` block alone (the merged ladder is
    ``session_block``): what the tuner diffs against when it writes a unit's
    override, so the unit only states what differs from its design."""
    marker_id: int | None
    marker_size_m: float | None
    marker_dictionary: str | None
    fixed_pose: dict | None
    fixed_pose_source: str


def _eef_resolve(
    gripper_unit: str | None,
    tool: str | None,
    marker_id: int | None,
    marker_size: float | None,
    *,
    unit_may_be_absent: bool = False,
) -> _EefResolved:
    """The CLI-side mirror of the manifest's tool_v1 resolution.

    Inside an episode the manifest does this (plus the episode-arm session
    layer); the standalone verbs run without one, so the same shape is
    rebuilt here from the named configs: design facts from the tool file,
    build facts (marker id/size, fixed_pose) from the unit file, the session
    recipe layered tool < unit with each layer validated at its source.
    ``unit_may_be_absent=True`` is the calibrate verb: it may target a unit
    file that does not exist yet (it creates it, bootstrapped from
    --marker-id / --marker-size).
    """
    from kiwi.core.tool_config import (
        find_gripper_unit,
        gripper_unit_config_path,
        load_gripper_unit,
        load_tool_config,
        tool_config_path,
    )
    from kiwi.eef.config import validate_session_block

    unit = None
    unit_path: Path | None = None
    if gripper_unit is not None:
        if tool is not None:
            unit_path = gripper_unit_config_path(gripper_unit, tool)
            if not unit_path.is_file():
                # Not under this design - does the id exist under another?
                # Refused either way.
                try:
                    other: Path | None = find_gripper_unit(gripper_unit)
                except (FileNotFoundError, ValueError):
                    other = None
                if other is not None:
                    raise ValueError(
                        f"--tool {tool!r} pairs with gripper unit "
                        f"{gripper_unit!r}, but that unit is a build of a "
                        f"different design ({other})"
                    )
                if not unit_may_be_absent:
                    raise FileNotFoundError(
                        f"--gripper-unit {gripper_unit!r}: no config at {unit_path}"
                    )
        else:
            try:
                unit_path = find_gripper_unit(gripper_unit)
            except FileNotFoundError:
                if not unit_may_be_absent:
                    raise
                unit_path = None  # a new unit; --tool below places it
        if unit_path is not None and unit_path.is_file():
            unit = load_gripper_unit(unit_path)
    tool_name = tool or (unit.tool if unit is not None else None)
    if tool_name is None:
        raise ValueError(
            "pass --tool, or a --gripper-unit that has a config (an existing "
            "unit names its design; a unit file that does not exist yet "
            "needs --tool to say which design's units/ dir it is created in)"
        )
    tool_yaml = tool_config_path(tool_name)
    if not tool_yaml.is_file():
        raise FileNotFoundError(f"no tool config at {tool_yaml}")
    tc = load_tool_config(tool_yaml)

    validate_session_block(tc.estimator, tc.session, str(tool_yaml))
    block = dict(tc.session)
    if unit is not None:
        validate_session_block(tc.estimator, unit.session, str(unit_path))
        block.update(unit.session)

    # Flags are bootstrap inputs over the unit's build facts (calibrating a
    # unit that has no file yet, or a bare-clip PnP run with no unit at all).
    mid = marker_id if marker_id is not None else (unit.marker_id if unit else None)
    msize = (
        marker_size if marker_size is not None else (unit.marker_size if unit else None)
    )

    fixed_pose, src = None, ""
    if unit is not None and unit.fixed_pose:
        fixed_pose, src = unit.fixed_pose, str(unit_path)

    return _EefResolved(
        unit=unit,
        unit_path=unit_path,
        tool=tool_name,
        estimator_key=tc.estimator,
        session_block=block,
        tool_session=dict(tc.session),
        marker_id=mid,
        marker_size_m=float(msize) if msize is not None else None,
        marker_dictionary=tc.marker_dictionary,
        fixed_pose=fixed_pose,
        fixed_pose_source=src,
    )


def _parse_set_overrides(pairs: list[str]) -> dict:
    """``--set KEY=VALUE`` pairs -> an overrides dict for ``resolve_params``.

    Values parse as YAML scalars (``--set clip_to_image=false``,
    ``--set max_asymmetry_deg=10``); unknown keys are rejected downstream by
    ``resolve_params``, which knows the estimator's knob set.
    """
    import yaml

    out: dict = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def _clip_intrinsics(video: Path, camera: str | None, intrinsics: Path | None) -> Path:
    """Intrinsics resolution: flag > named registry unit > the clip's serial."""
    from kiwi.core.camera.registry import CameraUnit, resolve_camera_for_video
    from kiwi.core.paths import config_root

    if intrinsics is not None:
        return intrinsics
    cameras_dir = config_root() / "cameras"
    if camera is not None:
        from kiwi.core.video import VideoInfo

        # Mode-keyed units resolve by the clip's own frame size; scalar
        # pointers ignore the size (their guard stays consumer-side).
        p = CameraUnit.load(cameras_dir / camera).artifact_for_size(
            "intrinsics", VideoInfo.read(video).size
        )
        if p is None:
            raise FileNotFoundError(
                f"camera '{camera}' has no intrinsics artifact under "
                f"{cameras_dir / camera} - publish one (`kiwi calib intrinsics`)"
            )
        return p
    m = resolve_camera_for_video(video, cameras_dir)
    if m is not None and m.intrinsics is not None:
        print(f"resolved intrinsics from serial {m.serial} ({m.unit}): {m.intrinsics}")
        return m.intrinsics
    raise FileNotFoundError(
        "intrinsics required - pass --intrinsics or --camera <unit> "
        "(auto-resolution from the clip's serial found no registry match)"
    )


@dataclass
class EefRun:
    """Estimate tool state for one clip -> `<stem>_eef.npz` beside it,
    plus `<stem>_eef.png` when Matplotlib is available. Inside an episode,
    the eef stage writes `<arm>/eef.npz`.

    Episode-wide runs go through `kiwi run` (the eef stage reads everything
    from the manifest); this verb is for standalone clips and tuning
    iterations. Resolution mirrors the manifest's layering: the tool config's
    recipe, the gripper unit's overrides + calibrated fixed_pose, flags on
    top.
    """

    video: tyro.conf.Positional[Path]
    """Source recording - .insv (auto-demuxed) or a per-lens mp4."""
    gripper_unit: str | None = None
    """This build's unit config (config/gripper/<tool>/units/<unit>.yaml):
    its facts (marker id + size, the
    calibrated fixed_pose enabling fixed-pose mode) plus its session
    overrides. Fixed-pose mode needs one - transforms are build facts."""
    tool: str | None = None
    """Tool design; default: the --gripper-unit's declared tool."""
    camera: str | None = None
    """Registry unit for intrinsics; default: matched by the clip's serial."""
    intrinsics: Path | None = None
    """Explicit intrinsics npz (escape hatch)."""
    lens0_mp4: Path | None = None
    """Pre-demuxed front-lens mp4 to read INSTEAD of demuxing the positional
    video argument, which still names the outputs."""
    marker_id: int | None = None
    """Tool marker ArUco id - a bare-clip PnP bootstrap (no unit file); with
    a --gripper-unit its build.marker.id already supplies this."""
    marker_size: float | None = None
    """Tool marker edge in METRES - same bootstrap, sets the PnP scale."""
    fov_deg: float | None = None
    """Rectifier pinhole FOV override. Default: the recipe's fov_deg."""
    set: list[str] = dataclasses.field(default_factory=list)
    """Estimator recipe overrides: pass one --set followed by KEY=VALUE entries,
    e.g. `--set max_asymmetry_deg=10 clip_to_image=false`.
    Normalization endpoints belong in the YAML session block."""
    live_pnp: bool = False
    """Force per-frame marker PnP even when a fixed_pose is configured."""
    write_overlay: bool = False
    """Also write the annotated `<stem>_eef.mp4` - a human debug artifact
    that nothing downstream reads."""

    def run(self) -> int:
        from kiwi.core.camera.convert import resolve_lens0_mp4
        from kiwi.eef.run import ClipJob, run_clip

        try:
            r = _eef_resolve(
                self.gripper_unit,
                self.tool,
                self.marker_id,
                self.marker_size,
            )
            intr = _clip_intrinsics(self.video, self.camera, self.intrinsics)
            lens0 = self.lens0_mp4 or resolve_lens0_mp4(self.video)
            overrides = _parse_set_overrides(self.set)
            if self.fov_deg is not None:
                overrides["fov_deg"] = self.fov_deg
            res = run_clip(
                ClipJob(
                    recording=self.video,
                    lens0_mp4=lens0,
                    tool=r.tool,
                    estimator_key=r.estimator_key,
                    session_block=r.session_block,
                    intrinsics=intr,
                    marker_id=r.marker_id,
                    marker_size_m=r.marker_size_m,
                    marker_dictionary=r.marker_dictionary,
                    fixed_pose=r.fixed_pose,
                    fixed_pose_source=r.fixed_pose_source,
                    live_pnp=self.live_pnp,
                    overrides=overrides,
                ),
                write_overlay=self.write_overlay,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"kiwi eef run: {e}", file=sys.stderr)
            return 1
        print(res.summary())
        if res.overlay_path is not None:
            print(f"wrote {res.overlay_path}")
        minimum = 1
        if res.n_valid < minimum:
            print(
                f"kiwi eef run: fewer than {minimum} valid frames - tool-state "
                "tracking failed for the clip. Check the trace png and the "
                "tool config (marker id, fixed_pose, ROI).",
                file=sys.stderr,
            )
            return 1
        return 0


@dataclass
class EefCalibrate:
    """Calibrate the fixed camera<->marker pose from a marker clip.

    Writes `build.marker` and `build.fixed_pose` into the unit's config IN
    PLACE, refusing to replace an existing calibrated pose without
    --force. Always prints the rigidity spread - a large one means the
    mount is not actually rigid and the pose must not be trusted.
    """

    video: tyro.conf.Positional[Path]
    """A clip of the mounted tool with its marker in view (.insv or mp4)."""
    gripper_unit: str | None = None
    """Target unit config (config/gripper/<tool>/units/<unit>.yaml);
    created when absent. An existing
    unit also supplies build.marker id/size and the tool link."""
    tool: str | None = None
    """Tool design; default: the --gripper-unit's declared tool."""
    camera: str | None = None
    intrinsics: Path | None = None
    lens0_mp4: Path | None = None
    marker_id: int | None = None
    """The tag calibrated against - required via a flag or the unit's
    build.marker.id; written into build.marker beside the pose."""
    marker_size: float | None = None
    """Marker edge in METRES; sets the metric scale of the calibrated tvec,
    so a stale value bakes in a proportional bias without any error. Written
    into build.marker beside the pose."""
    fov_deg: float | None = None
    """Rectifier pinhole FOV. Default: the recipe's fov_deg (same resolution
    the run uses), so the pose is calibrated at the FOV it will run at."""
    output: Path | None = None
    """Explicit output yaml (overrides the --gripper-unit resolution)."""
    max_frames: int = 300
    """Target sample count used to choose a frame stride across the clip."""
    min_frames: int = 20
    """Require at least this many good detections, else abort."""
    max_reproj_px: float = 2.0
    """Drop a frame whose PnP reprojection error exceeds this (px)."""
    force: bool = False
    """Replace an existing calibrated fixed_pose."""
    dry_run: bool = False
    """Compute + print the pose but do not write the config."""

    def run(self) -> int:
        from kiwi.core.camera.convert import resolve_lens0_mp4
        from kiwi.eef.homography import calibrate_fixed_pose, write_fixed_pose

        try:
            r = _eef_resolve(
                self.gripper_unit,
                self.tool,
                self.marker_id,
                self.marker_size,
                unit_may_be_absent=True,
            )
            if r.marker_id is None:
                raise ValueError(
                    "marker id required - pass --marker-id (bootstrapping a "
                    "new unit file; an existing unit's build.marker.id also "
                    "works)"
                )
            if r.marker_size_m is None:
                raise ValueError(
                    "marker size required - pass --marker-size in METRES (a "
                    "build fact recorded into build.marker beside the pose; "
                    "an existing unit's build.marker.size also works)"
                )
            intr = _clip_intrinsics(self.video, self.camera, self.intrinsics)
            lens0 = self.lens0_mp4 or resolve_lens0_mp4(self.video)
            # Use the run's FOV: CLI override > session block > the 110-degree
            # builtin.
            fov = self.fov_deg or r.session_block.get("fov_deg") or 110.0
            res = calibrate_fixed_pose(
                lens0,
                intr,
                marker_id=r.marker_id,
                marker_size_m=r.marker_size_m,
                dictionary=r.marker_dictionary or "DICT_4X4_50",
                fov_deg=float(fov),
                max_frames=self.max_frames,
                min_frames=self.min_frames,
                max_reproj_px=self.max_reproj_px,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"kiwi eef calibrate: {e}", file=sys.stderr)
            return 1
        print(res.summary())

        out = self.output if self.output is not None else r.unit_path
        if out is None:
            if not self.dry_run:
                print(
                    "\n(no --gripper-unit / --output given; not writing. "
                    "Use --dry-run to silence this.)"
                )
            return 0
        if self.dry_run:
            print(f"\n(dry-run; would write fixed_pose to {out})")
            return 0
        try:
            write_fixed_pose(
                out,
                res,
                tool=r.tool,
                marker_id=r.marker_id,
                marker_size_m=r.marker_size_m,
                force=self.force,
            )
        except RuntimeError as e:
            print(f"kiwi eef calibrate: {e}", file=sys.stderr)
            return 1
        print(f"\nwrote build.fixed_pose -> {out}")
        return 0


@dataclass
class EefTune:
    """Interactive viser tuner for a tool's detection recipe (kiwi.viz).

    Serves a browser GUI that scrubs the clip and re-runs the REAL estimator
    per slider change, then writes the tuned `session:` YAML block to paste
    into the tool config (or a unit's `session:` override). Needs the
    optional tune feature:
    `pixi install -e tune` (or `pip install viser`).
    """

    video: tyro.conf.Positional[Path]
    """Source recording - .insv (auto-demuxed) or a per-lens mp4."""
    gripper_unit: str | None = None
    tool: str | None = None
    camera: str | None = None
    intrinsics: Path | None = None
    lens0_mp4: Path | None = None
    marker_id: int | None = None
    marker_size: float | None = None
    port: int = 8080
    """viser port. The tuner is interactive-only; any free port works.
    Two simultaneous arm viewers each need a distinct port."""

    def run(self) -> int:
        from kiwi.core.camera.convert import resolve_lens0_mp4

        try:
            r = _eef_resolve(
                self.gripper_unit,
                self.tool,
                self.marker_id,
                self.marker_size,
            )
            if r.marker_size_m is None:
                raise ValueError(
                    "marker size required - pass --marker-size in METRES, or "
                    "a --gripper-unit whose build.marker.size supplies it"
                )
            intr = _clip_intrinsics(self.video, self.camera, self.intrinsics)
            lens0 = self.lens0_mp4 or resolve_lens0_mp4(self.video)
            from kiwi.viz.gripper_tune import launch

            launch(
                lens0,
                intr,
                session_block=r.session_block,
                marker_id=r.marker_id,
                marker_size_m=r.marker_size_m,
                marker_dictionary=r.marker_dictionary,
                fixed_pose=r.fixed_pose,
                fixed_pose_source=r.fixed_pose_source,
                tool_session=r.tool_session,
                unit_path=r.unit_path if r.unit is not None else None,
                port=self.port,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"kiwi eef tune: {e}", file=sys.stderr)
            return 2 if "viser" in str(e) else 1
        return 0


@dataclass
class Eef:
    """Per-tool end-effector state: estimate, calibrate the pose, tune.

    `design.estimator` in the tool config selects the kiwi.eef bundle.
    The supported tool family is Chopsticks.
    """

    command: Annotated[
        (
            Annotated[EefRun, tyro.conf.subcommand("run", prefix_name=False)]
            | Annotated[
                EefCalibrate, tyro.conf.subcommand("calibrate", prefix_name=False)
            ]
            | Annotated[EefTune, tyro.conf.subcommand("tune", prefix_name=False)]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


# --- viz ---------------------------------------------------------------


def _viz_sources(episode: Path):
    """Shared CLI/GUI manifest adapter, imported only when a view is requested."""
    from kiwi.app.sources import viz_sources

    return viz_sources(episode)


def _viz_closed_open(
    estimator: str | None, session_block: dict
) -> tuple[float | None, float | None]:
    """The tool's normalization endpoints, or ``(None, None)`` if unstated.

    ``estimator`` is the ArmSpec's design key, read from the tool config
    ``design.estimator`` field.
    """
    if not estimator or not session_block:
        return (None, None)
    from kiwi.eef import get_estimator

    try:
        return get_estimator(estimator).closed_open(session_block)
    except (ValueError, KeyError):
        # Diagnostics only: draw the plot without endpoint annotations.
        return (None, None)


@dataclass
class VizPlay:
    """Open the episode in the rerun viewer: video, trajectories, tool state.

    Every layer lands on the episode's global clock, and a layer whose
    artifact does not exist yet becomes a note in the summary panel naming
    the stage that produces it.
    """

    episode: tyro.conf.Positional[Path]
    """Episode directory (the one holding episode.yaml)."""
    save: Path | None = None
    """Write an .rrd instead of opening a viewer (`rerun <file>` reopens it)."""
    proxies: bool = True
    """Build the H.264 renditions the viewer needs (reused once built).

    The rerun viewer cannot decode the cameras' HEVC, so a first run spends
    some ffmpeg time per stream; --no-proxies skips it and drops any stream
    that would have needed one."""
    force_proxies: bool = False
    """Re-encode the renditions even when they are up to date."""

    def run(self) -> int:
        from kiwi.viz.player import log_episode, plan_layers
        from kiwi.viz.proxies import playable_video

        try:
            resolved = _viz_sources(self.episode)
        except (ValueError, FileNotFoundError) as e:
            print(f"kiwi viz play: {e}", file=sys.stderr)
            return 1

        def resolve(stream):
            return playable_video(
                stream.video,
                dst=stream.proxy,
                build=self.proxies,
                force=self.force_proxies,
            )

        plan = plan_layers(resolved.sources, resolve_video=resolve)
        print(plan.summary())
        for note in plan.notes:
            print(f"  - {note}")
        if not plan.layers:
            print(
                "kiwi viz play: nothing to show yet - run the pipeline first "
                "(`kiwi run`)",
                file=sys.stderr,
            )
            return 1
        try:
            log_episode(plan, save=self.save)
        except RuntimeError as e:
            print(f"kiwi viz play: {e}", file=sys.stderr)
            return 2 if "rerun" in str(e) else 1
        if self.save is not None:
            print(f"wrote {self.save}")
        return 0


@dataclass
class VizRender:
    """Render the streams side by side into one mp4, global time burned in."""

    episode: tyro.conf.Positional[Path]
    """Episode directory (the one holding episode.yaml)."""
    output: Path | None = None
    """Output mp4 (default: <episode>/synced_global.mp4)."""
    streams: list[str] = dataclasses.field(default_factory=list)
    """Only these stream labels, in this order (default: arms, then cameras)."""
    height: int = 600
    """Per-panel height in px."""
    out_fps: float | None = None
    """Output frame rate (default: the slowest source)."""
    t_start: float | None = None
    """Global start time (s); only narrows the streams' overlap."""
    t_end: float | None = None
    """Global end time (s); only narrows the streams' overlap."""
    strip_height: int = 200
    """Height of the tool-state strip drawn below the panels."""

    def run(self) -> int:
        from kiwi.core import layout
        from kiwi.viz.render import Panel, render_synced

        try:
            resolved = _viz_sources(self.episode)
            streams = list(resolved.sources.streams)
            if self.streams:
                by_label = {s.label: s for s in streams}
                missing = [x for x in self.streams if x not in by_label]
                if missing:
                    raise ValueError(
                        f"no such stream(s): {', '.join(missing)} "
                        f"(the episode declares {', '.join(by_label)})"
                    )
                streams = [by_label[x] for x in self.streams]

            panels, unit = [], ""
            for s in streams:
                if not s.video.is_file():
                    raise FileNotFoundError(
                        f"{s.label}: no {s.video.name} (run `kiwi run --from-stage "
                        f"convert --to-stage convert`)"
                    )
                state = s.stream_dir / layout.EEF_STATE if s.kind == "arm" else None
                panels.append(
                    Panel(
                        s.label,
                        s.video,
                        s.offset_s,
                        state if state and state.is_file() else None,
                    )
                )
            for p in panels:
                if p.state is not None:
                    from kiwi.eef.state import read_eef_state

                    unit = read_eef_state(p.state).unit
                    break
            out = self.output or (resolved.sources.root / "synced_global.mp4")
            render_synced(
                panels,
                out,
                height=self.height,
                out_fps=self.out_fps,
                t_start=self.t_start,
                t_end=self.t_end,
                strip_height=self.strip_height,
                unit=unit,
                on_progress=lambda msg: print(msg),
            )
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as e:
            print(f"kiwi viz render: {e}", file=sys.stderr)
            return 1
        return 0


@dataclass
class VizTrace:
    """Render the per-episode tool-state review png (both arms, one clock)."""

    episodes: tyro.conf.Positional[list[Path]]
    """One or more episode directories."""
    output: Path | None = None
    """Output png for a single episode (default: <episode>/tool_state.png)."""
    output_dir: Path | None = None
    """Write <out-dir>/<episode>.png instead, for a batch."""
    dpi: int = 130
    """Output resolution."""
    bounds_in_ylim: bool = False
    """Always show the closed/open endpoints, even far outside the data."""

    def run(self) -> int:
        from kiwi.core import layout
        from kiwi.viz.trace import series_from_state, trace_episode

        if self.output is not None and len(self.episodes) > 1:
            print(
                "kiwi viz trace: --output takes a single episode; use "
                "--output-dir for a batch",
                file=sys.stderr,
            )
            return 1
        failed = 0
        for episode in self.episodes:
            try:
                resolved = _viz_sources(episode)
                src = resolved.sources
                series = []
                for arm in src.arms:
                    state = arm.stream_dir / layout.EEF_STATE
                    if not state.is_file():
                        continue
                    closed, opened = _viz_closed_open(
                        *resolved.arms.get(arm.label, (None, {}))
                    )
                    series.append(
                        series_from_state(
                            arm.label, state, arm.offset_s, closed=closed, open=opened
                        )
                    )
                if not series:
                    raise FileNotFoundError(
                        "no tool-state artifacts (run `kiwi run --from-stage eef "
                        "--to-stage eef`)"
                    )
                if self.output is not None:
                    out = self.output
                elif self.output_dir is not None:
                    out = self.output_dir / f"{src.name}.png"
                else:
                    out = src.root / "tool_state.png"
                trace_episode(
                    series,
                    out,
                    title=" - ".join(x for x in (src.name, src.description) if x),
                    subtitle="  |  ".join(
                        x
                        for x in (src.date, f"tool {src.tool}" if src.tool else "")
                        if x
                    ),
                    task_window=src.task_window,
                    clock_origin=src.clock.origin if src.clock else None,
                    bounds_in_ylim=self.bounds_in_ylim,
                    dpi=self.dpi,
                )
                print(f"[ok] {src.name}: wrote {out}")
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                failed += 1
                print(f"[FAIL] {episode.name}: {e}", file=sys.stderr)
                if "matplotlib" in str(e):
                    return 2
        return 1 if failed else 0


@dataclass
class Viz:
    """Look at an episode: the rerun player, or a review artifact.

    Read-only over finished artifacts. Layers appear as their producing
    stages land, and the player says which stage each absent layer waits on.
    """

    command: Annotated[
        (
            Annotated[VizPlay, tyro.conf.subcommand("play", prefix_name=False)]
            | Annotated[VizRender, tyro.conf.subcommand("render", prefix_name=False)]
            | Annotated[VizTrace, tyro.conf.subcommand("trace", prefix_name=False)]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


@dataclass
class SceneInit:
    """Create scene.yaml from the episode directories already under --root.

    Discovers episodes (direct children of the root, and of root/episodes/,
    that contain an episode.yaml), copies the named builtin mapping profile
    into <root>/profiles/ and pins its content digest. Nothing is
    overwritten, and the manifest passes every check inspect runs before
    anything is written: init never produces a scene that inspect would
    refuse, and a refused init leaves the root untouched for the retry.
    """

    root: Path
    mapping_episode: str
    """The episode that builds the scene's maps."""
    mapping_profile: str = "default"
    """A profile name under config/scene/mapping-profiles/."""
    map_builder_arm: str = "left"
    """Which arm's maps freeze the scene frame."""
    trajectory_mode: str = "global_state"
    """alignment_only | global_state (default; recipe defaults to scene-fusion)."""
    pipeline_profile: str = "global-state"
    """Pipeline profile NAME, recorded in scene.yaml and shown by kiwi scene
    inspect (kiwi run takes --profile instead)."""
    state_profile: str | None = None
    """State profile NAME (global_state mode only; default: scene-fusion)."""
    min_map_usable_s: float = 60.0
    """Usable seconds the builder episode must provide (floor 60; a profile
    may raise it further - the effective value is the maximum)."""

    def run(self) -> int:
        from kiwi.episode import scene as scene_mod

        try:
            cfg = scene_mod.init_scene(
                root=self.root,
                mapping_episode=self.mapping_episode,
                mapping_profile=self.mapping_profile,
                map_builder_arm=self.map_builder_arm,
                trajectory_mode=self.trajectory_mode,
                pipeline_profile=self.pipeline_profile,
                state_profile=self.state_profile,
                min_usable_s=self.min_map_usable_s,
            )
        except (ValueError, OSError) as e:
            msg = str(e)
            if not msg.startswith("kiwi scene init"):
                msg = f"kiwi scene init: {msg}"
            print(msg, file=sys.stderr)
            return 2
        _warn_scene_missing_head_markers(cfg)
        print(f"scene '{cfg.name}': {len(cfg.episodes)} episode(s)")
        print(f"  builder episode: {cfg.builder_episode}")
        print(f"  map builder arm: {cfg.map_builder_arm}")
        print(
            f"  trajectory mode: {cfg.trajectory_mode} "
            f"(pipeline profile: {cfg.pipeline_profile}"
            + (f", state profile: {cfg.state_profile})" if cfg.state_profile else ")")
        )
        print(f"  mapping profile: {cfg.profile.name} ({cfg.profile_digest[:34]}...)")
        print(f"  effective min usable: {scene_mod.effective_min_usable_s(cfg):g} s")
        print(f"  wrote {cfg.path}")
        print(
            f"  linked {len(cfg.episodes)} episode manifest(s) back to the scene "
            f"(`scene:` block in episode.yaml)"
        )
        return 0


@dataclass
class SceneInspect:
    """Load a scene strictly and report it - or exactly why it is invalid.

    Everything checkable without running a component is checked: schema and
    key strictness, episode directories and manifests, the pinned profile's
    content digest, and any episode back-references (both directions).
    """

    scene: Path
    """Path to scene.yaml (or its directory)."""

    def run(self) -> int:
        from kiwi.episode import scene as scene_mod

        path = self.scene
        if path.is_dir():
            path = path / scene_mod.SCENE_MANIFEST
        try:
            cfg = scene_mod.load_scene(path)
        except (ValueError, OSError) as e:
            print(f"kiwi scene inspect: {e}", file=sys.stderr)
            return 2
        _warn_scene_missing_head_markers(cfg)
        print(f"scene '{cfg.name}' ({cfg.path})")
        print(f"  builder episode: {cfg.builder_episode}")
        print(f"  map builder arm: {cfg.map_builder_arm}")
        print(
            f"  trajectory mode: {cfg.trajectory_mode} "
            f"(pipeline profile: {cfg.pipeline_profile}"
            + (f", state profile: {cfg.state_profile})" if cfg.state_profile else ")")
        )
        print(f"  mapping profile: {cfg.profile.name}")
        print(f"    path: {cfg.profile_path}")
        print(f"    digest: {cfg.profile_digest}  [verified]")
        print(
            f"  min usable: scene {cfg.min_usable_duration_s:g} s, profile "
            f"{cfg.profile.min_usable_duration_s:g} s "
            f"-> effective {scene_mod.effective_min_usable_s(cfg):g} s"
        )
        print(f"  episodes ({len(cfg.episodes)}):")
        for episode_id, rel in sorted(cfg.episodes.items()):
            tag = "  [builder]" if episode_id == cfg.builder_episode else ""
            print(f"    {episode_id}: {rel}{tag}")
        return 0


@dataclass
class SceneBuild:
    """Build the scene's immutable map set (both front maps + the shared
    rear map) from the whole mapping episode and place the maps in one scene
    frame. Requires the built engine (`pixi run -e slam-offline
    build-orbslam3`)."""

    scene: Path
    """Path to scene.yaml (or its directory)."""

    def run(self) -> int:
        from kiwi.episode import scene as scene_mod
        from kiwi.episode import scene_build

        path = self.scene
        if path.is_dir():
            path = path / scene_mod.SCENE_MANIFEST
        try:
            cfg = scene_mod.load_scene(path)
            _warn_scene_missing_head_markers(cfg)
            outcome = scene_build.build_scene_maps(cfg)
        except (ValueError, OSError, RuntimeError) as e:
            print(f"kiwi scene build: {e}", file=sys.stderr)
            return 2
        print(f"scene '{cfg.name}': {outcome.reason} ({outcome.attempts} attempt(s))")
        for role, ref in sorted(outcome.maps.items()):
            print(f"  {role}: {ref}")
        if not outcome.accepted:
            return 1
        return 0


@dataclass
class SceneLocalize:
    """Localize one episode stream, full-trajectory, against the published
    scene map."""

    scene: Path
    episode: str
    arm: str
    """left | right."""
    lens: str = "front"
    """front | back."""

    def run(self) -> int:
        from kiwi.episode import scene as scene_mod
        from kiwi.episode import scene_build

        path = self.scene
        if path.is_dir():
            path = path / scene_mod.SCENE_MANIFEST
        try:
            cfg = scene_mod.load_scene(path)
            _warn_missing_head_marker(cfg.episode_dir(self.episode))
            out = scene_build.localize_episode(
                cfg, self.episode, self.arm, lens=self.lens
            )
        except (ValueError, OSError, RuntimeError) as e:
            print(f"kiwi scene localize: {e}", file=sys.stderr)
            return 2
        print(f"wrote {out}")
        return 0


@dataclass
class Scene:
    """Group episodes of one physical scene: manifest, map build, inspect.

    The scene is the unit map building works at (one immutable map set per
    scene, every episode localized against it).
    """

    command: Annotated[
        (
            Annotated[SceneInit, tyro.conf.subcommand("init", prefix_name=False)]
            | Annotated[SceneBuild, tyro.conf.subcommand("build", prefix_name=False)]
            | Annotated[
                SceneLocalize, tyro.conf.subcommand("localize", prefix_name=False)
            ]
            | Annotated[
                SceneInspect, tyro.conf.subcommand("inspect", prefix_name=False)
            ]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


@dataclass
class ModelsFetch:
    """Fetch every declared model into the local model cache.

    Downloads only what is missing: a cached model is re-verified against
    its sidecar hash and left alone.
    """

    def run(self) -> int:
        from kiwi.perception.gripper_segmentation import (
            PACKAGED_SIDECAR,
            ArtifactError,
            resolve_model,
        )

        # Fetch each declared model; the packaged sidecar is the sole entry.
        for sidecar in (PACKAGED_SIDECAR,):
            try:
                print(resolve_model(sidecar, fetch=True))
            except ArtifactError as e:
                print(f"kiwi models fetch: {e}", file=sys.stderr)
                return 2
        return 0


@dataclass
class ModelsPath:
    """Print where the declared model(s) sit in the cache - never fetching.

    The offline check: it refuses, with the remedies, if the weights are
    not already there.
    """

    def run(self) -> int:
        from kiwi.perception.gripper_segmentation import (
            PACKAGED_SIDECAR,
            ArtifactError,
            resolve_model,
        )

        for sidecar in (PACKAGED_SIDECAR,):
            try:
                print(resolve_model(sidecar, fetch=False))
            except ArtifactError as e:
                print(f"kiwi models path: {e}", file=sys.stderr)
                return 2
        return 0


@dataclass
class Models:
    """The model store: fetch the weights the pipeline's models need.

    Weights are not repo content - they are release assets, pinned by
    sha256 in their in-repo sidecar and cached per machine.
    """

    command: Annotated[
        (
            Annotated[ModelsFetch, tyro.conf.subcommand("fetch", prefix_name=False)]
            | Annotated[ModelsPath, tyro.conf.subcommand("path", prefix_name=False)]
        ),
        tyro.conf.OmitArgPrefixes,
    ]

    def run(self) -> int:
        return self.command.run()


@dataclass
class Gui:
    """Open the local data workspace: import, manage episodes and review in Rerun."""

    root: Path = Path("data")
    """Directory containing scenes or episodes."""
    port: int = 8765
    open_browser: bool = True
    cache: Path = Path(".pixi/kiwi-gui")
    """Application jobs, review recordings and video proxies."""

    def run(self) -> int:
        try:
            from kiwi.app.ui import launch
        except ImportError as exc:
            print(
                f"kiwi gui needs the GUI environment: pixi install -e gui ({exc})",
                file=sys.stderr,
            )
            return 2
        launch(self.root, self.cache, port=self.port, open_browser=self.open_browser)
        return 0


@dataclass
class Real2simviz:
    """View a GS asset with real arm trajectories, or render saved views."""

    asset: Annotated[Path, tyro.conf.Positional]
    episodes: tuple[Path, ...] = ()
    """Optional same-scene episodes to overlay; defaults to the saved trajectories."""
    backend_repo: Path | None = None
    python: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8895
    max_resolution: int = 1920
    pose_body: Literal["camera", "tcp"] = "camera"
    output: Path | None = None
    """Screenshot/render destination; defaults to a directory inside the asset."""
    views: Path | None = None
    """Saved view JSON; renders PNGs without starting an interactive server."""

    def run(self) -> int:
        from kiwi.app.real2sim import view

        try:
            view(**dataclasses.asdict(self))
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.CalledProcessError,
        ) as exc:
            print(f"kiwi real2simviz: {exc}", file=sys.stderr)
            return 2
        return 0


Command = (
    Run
    | Import
    | Init
    | Viz
    | Export
    | Replay
    | Calib
    | Sync
    | Register
    | Eef
    | Scene
    | Models
    | Gui
    | Real2simviz
)


def main(argv: list[str] | None = None) -> int:
    cmd = tyro.cli(
        Command,
        args=argv,
        prog="kiwi",
        description=__doc__,
    )
    return cmd.run()


if __name__ == "__main__":
    raise SystemExit(main())

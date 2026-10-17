"""The ``episode.yaml`` manifest: schema ``episode_v1``, its parser and sync writer.

An episode directory holds the raw recordings; the calibration lives under
``config/cameras/<unit>/`` and the tool configs under ``config/gripper/``. The
manifest names which camera unit and tool each arm used, where the clips are,
and the global clock every downstream window is expressed on.

:meth:`EpisodeConfig.load` is the typed parser. It does not read the optional
top-level ``scene:`` back-reference, which :mod:`kiwi.episode.scene` validates
against the scene manifest. ``kiwi init`` (:mod:`kiwi.episode.init`) writes
the initial file, and :func:`write_sync_block` is the only code that edits it
in place.

``anchor:`` is OPTIONAL and describes an ArUco table marker present in the
room. Only the on-demand ``register`` stage reads it - the ground-truth
instrument no builtin pipeline profile requests - so a normal capture has no
such block and ``kiwi init`` writes none. It is parsed and validated for the
episodes recorded against a marker.

Per-unit calibration artifacts are not resolved at load: the manifest resolves
``camera_dir`` and stops there. The active artifact is whatever
``camera.yaml``'s pointer names, resolved only through
:meth:`kiwi.core.camera.registry.CameraUnit.artifact`, so loading an episode
never depends on calibration being present.

Config layering is part of the schema's meaning, not a separate feature. The
tool_v1 blocks (:mod:`kiwi.core.tool_config`) resolve here per arm:

* ``design:`` facts (estimator, marker dictionary, the capture description)
  come from the tool file alone - nothing shadows them;
* ``build:`` facts (marker id + size, the calibrated ``fixed_pose``, the
  mount-leg overrides) come from the arm's ``gripper_unit`` file alone. A
  ``gripper_unit`` naming no file is REFUSED;
* ``session:`` recipe keys layer ``tool < unit < arm session: block``, each
  layer validated against the estimator's key set at its source
  (:func:`kiwi.eef.config.validate_session_block`).

Each ``arms.<label>`` block has a closed key set: an unknown key is refused
rather than ignored.

Run: ``python -m kiwi.episode.manifest <path>`` prints the resolved layout.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import yaml

from kiwi.core.clock import GlobalClock
from kiwi.core.paths import config_root
from kiwi.core.tool_config import (
    ToolConfig,
    find_gripper_unit,
    gripper_unit_config_path,
    load_gripper_unit,
    load_tool_config,
    merge_session,
    tool_config_path,
)

__all__ = [
    "ANCHOR_FRAMES",
    "MANIFEST_NAME",
    "SCHEMA",
    "ArmSpec",
    "CameraSpec",
    "EpisodeConfig",
    "write_scene_block",
    "write_sync_block",
]

SCHEMA = "episode_v1"
MANIFEST_NAME = "episode.yaml"

#: The only anchor world-frame convention that exists: solvePnP the anchor
#: marker, compose with the pose at that timestamp, and re-express the whole
#: trajectory so the marker's own pose IS the world origin - all six DoF from
#: the marker, z along its normal. That is what co-registers the arms.
ANCHOR_FRAMES = ("full_6dof",)

# The closed key set of an ``arms.<label>`` block. The tag id/size and the
# detection recipe are not here: they live in the gripper unit's build: block
# and the layered session: block.
_ARM_KEYS = frozenset({"camera", "recording", "tool", "gripper_unit", "session"})


# --------------------------------------------------------------------------- #
# field parsers - each rejects rather than coerces
# --------------------------------------------------------------------------- #


def _parse_task_window(doc, manifest: Path) -> tuple[float | None, float | None] | None:
    """Validate the optional ``task_window: [start_s, end_s]`` clock trim.

    Either bound may be ``null``, meaning "do not trim this side - run to the
    limit of what the episode actually covers":

    ==========================  ==========================================
    ``task_window`` absent      no trim at all; export the full overlap
    ``[7.8, null]``             drop setup, keep everything to the end
    ``[null, 30.0]``            keep the start, drop teardown
    ``[7.8, 30.0]``             drop both
    ==========================  ==========================================

    The export intersects this window with real coverage, so ``null`` tracks
    whatever that turns out to be across a re-run that inits later or a re-sync
    that moves the clock.

    ``[null, null]`` is rejected rather than read as "no trim".
    """
    raw = doc.get("task_window")
    if raw is None:
        return None
    ok = (
        isinstance(raw, (list, tuple))
        and len(raw) == 2
        and all(
            v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))
            for v in raw
        )
    )
    if not ok:
        raise ValueError(
            f"{manifest}: `task_window:` must be [start_s, end_s] - two entries, "
            f"each a number or null (null = do not trim that side), got {raw!r}"
        )
    lo = None if raw[0] is None else float(raw[0])
    hi = None if raw[1] is None else float(raw[1])
    if lo is None and hi is None:
        raise ValueError(
            f"{manifest}: `task_window: [null, null]` trims nothing - omit the "
            f"key entirely to export the full episode, or fill in the bound you "
            f"meant."
        )
    if lo is not None and hi is not None and not hi > lo:
        raise ValueError(
            f"{manifest}: `task_window:` end ({hi}) must be greater than start ({lo})"
        )
    return (lo, hi)


def _parse_eval_success(doc, manifest: Path) -> bool:
    """Validate the optional ``eval_success:`` flag (default ``False``).

    Absent reads as False. Only a real YAML bool is accepted.
    """
    raw = doc.get("eval_success")
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise ValueError(
            f"{manifest}: `eval_success:` must be a bool (true/false), got {raw!r}. "
            f"Unquoted YAML true/false only - a string would be silently truthy."
        )
    return raw


def _parse_anchor(raw, manifest: Path) -> dict:
    """Validate the optional ``anchor:`` block, returning it as a plain dict.

    The marker ID is parsed here. The register stage's optional knobs -
    ``scan_seconds``, ``max_marker_dist``, ``fov_deg`` and ``out_size`` - are
    checked here too, so a bad value refuses with its key named; an omitted
    knob takes the stage's default. The marker's size and dictionary pass
    through for the register stage's scan to validate. Only ``full_6dof`` is
    implemented; omitting ``frame:`` means the same thing, and any other value
    raises.
    """
    if raw is not None and not isinstance(raw, dict):
        raise ValueError(f"{manifest}: `anchor:` must be a mapping, got {raw!r}")
    anchor = dict(raw or {})
    marker = anchor.get("marker")
    if marker is not None:
        if not isinstance(marker, dict):
            raise ValueError(
                f"{manifest}: anchor.marker must be a mapping, got {marker!r}"
            )
        marker_id = marker.get("id")
        if marker_id is not None and (
            isinstance(marker_id, bool)
            or not isinstance(marker_id, int)
            or marker_id < 0
        ):
            raise ValueError(
                f"{manifest}: `anchor:` marker.id must be a non-negative "
                f"integer, got {marker_id!r}; remove `anchor:` for a "
                "marker-free episode or set marker.id to the table marker's "
                "ArUco id for the marker-anchored register stage"
            )
    frame = anchor.get("frame")
    if frame is not None and frame not in ANCHOR_FRAMES:
        raise ValueError(
            f"{manifest}: anchor.frame must be one of {list(ANCHOR_FRAMES)} (or "
            f"omitted), got {frame!r}. Only full-6DoF marker anchoring is "
            f"implemented - a different value would be ignored, not honoured."
        )
    for key, unit in (
        ("scan_seconds", "seconds"),
        ("max_marker_dist", "metres"),
        ("fov_deg", "degrees"),
    ):
        if key in anchor and not _positive_finite(anchor[key]):
            raise ValueError(
                f"{manifest}: anchor.{key} must be a positive finite number "
                f"({unit}), got {anchor[key]!r}"
            )
    if anchor.get("fov_deg", 0) >= 180:
        raise ValueError(
            f"{manifest}: anchor.fov_deg must be below 180 - the marker scan's "
            f"pinhole view cannot reach 180 degrees, got {anchor['fov_deg']!r}"
        )
    out_size = anchor.get("out_size")
    if "out_size" in anchor and (
        isinstance(out_size, bool) or not isinstance(out_size, int) or out_size <= 0
    ):
        raise ValueError(
            f"{manifest}: anchor.out_size must be a positive integer (the "
            f"marker scan's square view, in pixels), got {out_size!r}"
        )
    return anchor


def _positive_finite(value) -> bool:
    """A real number (never a bool) that is finite and above zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:  # an int too large for a float
        return False


def _parse_sync(
    raw, manifest: Path, arm_labels, camera_labels=frozenset()
) -> tuple[dict[str, float] | None, str | None, str | None]:
    """Validate the optional ``sync:`` block - the global clock definition.

    ``offsets`` maps a stream label (an arm, or an extra ``cameras:`` entry) to
    that stream's frame-0 time on the shared clock. ``task_window`` and every
    curation window are expressed on this clock.

    ``origin:`` is required: it names the video stream that started last,
    holds exactly 0.0, and every other offset is <= 0. That is the only clock
    :class:`kiwi.core.clock.GlobalClock` represents; a block without an origin
    raises here.
    """
    if raw is None:
        return None, None, None
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("offsets"), dict)
        or not raw["offsets"]
    ):
        raise ValueError(
            f"{manifest}: `sync:` must be a mapping like `{{method: "
            f"audio_gcc_phat, origin: right, offsets: {{left: -1.234, right: "
            f"0.0}}}}`, got {raw!r}"
        )
    known = set(arm_labels) | set(camera_labels)
    offsets: dict[str, float] = {}
    for label, v in raw["offsets"].items():
        if known and label not in known:
            extra = f"; cameras: {sorted(camera_labels)}" if camera_labels else ""
            raise ValueError(
                f"{manifest}: sync.offsets names unknown stream {label!r} "
                f"(arms: {sorted(arm_labels)}{extra})"
            )
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(
                f"{manifest}: sync.offsets.{label} must be a number, got {v!r}"
            )
        offsets[label] = float(v)

    method = raw.get("method")
    origin = raw.get("origin")
    if origin is None:
        raise ValueError(
            f"{manifest}: sync.origin is required - it names the stream whose "
            f"frame 0 is t = 0 on the global clock. Delete the block and run "
            f"`kiwi run --episode {manifest.parent} --from-stage sync "
            f"--to-stage sync` to write a complete one."
        )
    if not isinstance(origin, str):
        raise ValueError(
            f"{manifest}: sync.origin must be a stream label string, got {origin!r}"
        )
    if origin not in offsets:
        raise ValueError(
            f"{manifest}: sync.origin names {origin!r}, which is not one of the "
            f"offsets {sorted(offsets)}"
        )
    positive = sorted(k for k, v in offsets.items() if v > 0.0)
    if positive:
        raise ValueError(
            f"{manifest}: sync block (origin: {origin}) has positive offset(s) "
            f"{positive}: the origin is the LAST stream to start, so every other "
            f"frame 0 is earlier and every offset must be <= 0"
        )
    if offsets[origin] != 0.0:
        raise ValueError(
            f"{manifest}: sync.offsets.{origin} must be exactly 0.0 (it is the "
            f"clock origin), got {offsets[origin]!r}"
        )
    return offsets, (str(method) if method is not None else None), origin


# --------------------------------------------------------------------------- #
# the resolved manifest
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    """One arm's capture, plus the tool config its references resolve to.

    ``camera`` names a registry unit under ``config/cameras/``; the per-unit
    calibration artifacts are NOT resolved here (see the module docstring) -
    read them through :meth:`kiwi.core.camera.registry.CameraUnit.artifact`.

    ``recording`` is the authority on which clip belongs to this arm; its file
    and directory names are not. ``kiwi init`` verifies the binding against the
    clip's embedded camera serial.
    """

    label: str
    camera: str
    recording: Path
    camera_dir: Path
    # Tool config, resolved per the tool_v1 blocks: `marker` is the merged
    # {dictionary (design), id, size (build)} the eef stage feeds ClipJob;
    # `session` is the layered recipe (tool < unit < arm); `fixed_pose` and
    # the mount legs are the unit's build facts (None without a gripper_unit).
    tool: str
    tool_config: Path
    gripper_unit: str | None
    gripper_unit_config: Path | None
    marker: dict
    fixed_pose: dict | None
    mount_tcp: dict | None
    mount_cam: dict | None
    session: dict
    # The design's `estimator` / `capture.description` (None when the tool file
    # is absent - loading an episode never requires it; the stages that need
    # them refuse actionably), and which YAML supplied the fixed_pose (the eef
    # artifact's pose_source).
    estimator: str | None
    description: str | None
    fixed_pose_source: str


@dataclasses.dataclass(frozen=True)
class CameraSpec:
    """One extra (non-arm) camera stream - an ego view, a fixed third-person.

    Unlike :class:`ArmSpec`, it has no arm track, registration, or tool state.
    Each contributes one video feature to the export and stays out of
    ``arms``. An optional ``head_marker`` describes a passive target mounted
    beside this camera; its own imagery and calibration are not used to
    estimate the target.

    The stream must exist on disk and carry a ``sync.offsets.<label>`` entry.
    The parser accepts a stream without ``camera``, but the sync stage refuses
    one: no calibration is read from the unit, yet the A/V correction must
    establish both clips' camera models, even for a same-model rig. ``kiwi
    init`` writes ``camera`` from a registered ``.insv`` serial, or from
    ``--camera`` for a clip without one (a GO3 ``.mp4``).
    """

    label: str
    recording: Path
    camera: str | None = None
    camera_dir: Path | None = None
    key: str | None = None
    head_marker: dict | None = None
    head_stationary_window: tuple[float, float] | None = None


@dataclasses.dataclass(frozen=True)
class EpisodeConfig:
    """A loaded ``episode.yaml`` with every reference resolved."""

    name: str
    root: Path
    default_tool: str | None
    arms: dict[str, ArmSpec]
    anchor: dict
    meta: dict
    task_window: tuple[float | None, float | None] | None = None
    eval_success: bool = False
    sync_offsets: dict[str, float] | None = None
    sync_method: str | None = None
    sync_origin: str | None = None
    cameras: dict[str, CameraSpec] = dataclasses.field(default_factory=dict)
    #: Configuration tree actually selected by the loader.
    config_dir: Path | None = None

    # ---- loading ---------------------------------------------------------- #

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        check_exists: bool = True,
        config_dir: Path | None = None,
    ) -> "EpisodeConfig":
        """Parse ``path``, resolve and merge every reference.

        ``config_dir`` explicitly overrides the config tree. Otherwise prefer
        ``<episode>/config``; an episode without a snapshot (one the pipeline
        has not run yet) falls back to :func:`kiwi.core.paths.config_root`,
        which honours ``$KIWI_CONFIG``. Loading is read-only; running the
        pipeline creates the local copy.
        """
        # Function-scoped: the session-key registry is component (eef)
        # knowledge, only needed when a manifest is actually loaded.
        from kiwi.eef.config import validate_session_block

        path = Path(path).resolve()
        root = path.parent
        cfg_root = (
            Path(config_dir).resolve()
            if config_dir is not None
            else config_root(root).resolve()
        )
        # A broken episode.yaml is a refusal like any other: every stage
        # reports a ValueError as a result. The parser names no file itself.
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"{path}: not valid YAML: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: manifest must be a YAML mapping")

        schema = raw.get("schema")
        if schema != SCHEMA:
            raise ValueError(f"{path}: schema is {schema!r}, expected {SCHEMA!r}")

        default_tool = raw.get("tool")
        anchor = _parse_anchor(raw.get("anchor"), path)
        tool_cache: dict[str, ToolConfig | None] = {}

        def tool_cfg(name: str) -> ToolConfig | None:
            if name not in tool_cache:
                p = tool_config_path(name, cfg_root)
                # An ABSENT tool file is tolerated at load (the eef/export
                # stages refuse actionably). A PRESENT file must be valid
                # tool_v1, its session block included; one that is not YAML
                # at all is refused as a ValueError, like a broken manifest.
                try:
                    tc = load_tool_config(p) if p.is_file() else None
                except yaml.YAMLError as e:
                    raise ValueError(f"{p}: not valid YAML: {e}") from e
                if tc is not None:
                    validate_session_block(tc.estimator, tc.session, str(p))
                tool_cache[name] = tc
            return tool_cache[name]

        arms: dict[str, ArmSpec] = {}
        for label, a in (raw.get("arms") or {}).items():
            if not isinstance(a, dict):
                raise ValueError(f"{path}: arms.{label} must be a mapping, got {a!r}")
            for key in ("camera", "recording"):
                if key not in a:
                    raise ValueError(f"{path}: arms.{label}.{key} is required")
            unknown = sorted(set(map(str, a)) - _ARM_KEYS)
            if unknown:
                raise ValueError(
                    f"{path}: arms.{label} has unknown key(s) {unknown}; "
                    f"known: {sorted(_ARM_KEYS)}"
                )
            cam_dir = cfg_root / "cameras" / a["camera"]

            tool = a.get("tool", default_tool)
            if tool is None:
                raise ValueError(
                    f"{path}: arms.{label} has no tool and no top-level `tool:` default"
                )
            tc = tool_cfg(tool)
            marker: dict = (
                {"dictionary": tc.marker_dictionary} if tc is not None else {}
            )
            session: dict = dict(tc.session) if tc is not None else {}
            fixed_pose = mount_tcp = mount_cam = None
            fixed_pose_source = ""

            gripper_unit = a.get("gripper_unit")
            gripper_unit_config: Path | None = None
            if gripper_unit is not None:
                gripper_unit_config = gripper_unit_config_path(
                    gripper_unit, tool, cfg_root
                )
                if not gripper_unit_config.is_file():
                    # A unit lives under its design's units/ dir, so pairing
                    # an arm with another design's build cannot resolve -
                    # scan the other designs to name the actual mistake.
                    try:
                        other = find_gripper_unit(gripper_unit, cfg_root)
                    except (FileNotFoundError, ValueError):
                        other = None
                    if other is not None:
                        raise ValueError(
                            f"{path}: arms.{label} pairs tool {tool!r} with "
                            f"gripper_unit {gripper_unit!r}, but that unit is "
                            f"a build of a different design ({other}) - the "
                            f"wrong detection recipe and the wrong marker "
                            f"geometry."
                        )
                    raise FileNotFoundError(
                        f"{path}: arms.{label}.gripper_unit {gripper_unit!r} "
                        f"names no config at {gripper_unit_config} - refused."
                    )
                try:
                    gu = load_gripper_unit(gripper_unit_config)
                except yaml.YAMLError as e:
                    raise ValueError(
                        f"{gripper_unit_config}: not valid YAML: {e}"
                    ) from e
                marker["id"] = gu.marker_id
                marker["size"] = gu.marker_size
                fixed_pose = gu.fixed_pose
                if fixed_pose is not None:
                    fixed_pose_source = str(gripper_unit_config)
                mount_tcp, mount_cam = gu.mount_tcp, gu.mount_cam
                if tc is not None:
                    validate_session_block(
                        tc.estimator, gu.session, str(gripper_unit_config)
                    )
                session = merge_session(session, gu.session)

            arm_session = a.get("session") or {}
            if tc is not None:
                validate_session_block(
                    tc.estimator, arm_session, f"{path}: arms.{label}.session"
                )
            session = merge_session(session, arm_session)

            arms[label] = ArmSpec(
                label=label,
                camera=a["camera"],
                recording=(root / a["recording"]),
                camera_dir=cam_dir,
                tool=tool,
                tool_config=tool_config_path(tool, cfg_root),
                gripper_unit=gripper_unit,
                gripper_unit_config=gripper_unit_config,
                marker=marker,
                fixed_pose=fixed_pose,
                mount_tcp=mount_tcp,
                mount_cam=mount_cam,
                session=session,
                estimator=tc.estimator if tc is not None else None,
                description=tc.capture_description if tc is not None else None,
                fixed_pose_source=fixed_pose_source,
            )

        cameras: dict[str, CameraSpec] = {}
        for label, c in (raw.get("cameras") or {}).items():
            # A camera sharing an arm's label would make sync.offsets ambiguous
            # and could collide on the export feature key.
            if label in arms:
                raise ValueError(
                    f"{path}: cameras.{label!r} collides with arms.{label!r}; "
                    f"camera and arm labels must be distinct"
                )
            if not isinstance(c, dict) or "recording" not in c:
                raise ValueError(
                    f"{path}: cameras.{label} must be a mapping with a "
                    f"`recording:` path, got {c!r}"
                )
            unit = c.get("camera")
            head_marker = c.get("head_marker")
            head_stationary_window = None
            if head_marker is not None:
                if not isinstance(head_marker, dict):
                    raise ValueError(
                        f"{path}: cameras.{label}.head_marker must be a mapping, "
                        f"got {head_marker!r}"
                    )
                missing = sorted({"dictionary", "id", "size"} - set(head_marker))
                if missing:
                    raise ValueError(
                        f"{path}: cameras.{label}.head_marker missing {missing}"
                    )
                dictionary = head_marker["dictionary"]
                if not isinstance(dictionary, str) or not dictionary:
                    raise ValueError(
                        f"{path}: cameras.{label}.head_marker.dictionary must "
                        "be a non-empty string"
                    )
                marker_id = head_marker["id"]
                if (
                    isinstance(marker_id, bool)
                    or not isinstance(marker_id, int)
                    or marker_id < 0
                ):
                    raise ValueError(
                        f"{path}: cameras.{label}.head_marker.id must be a "
                        "non-negative integer"
                    )
                raw_size = head_marker["size"]
                if isinstance(raw_size, bool) or not isinstance(raw_size, (int, float)):
                    raise ValueError(
                        f"{path}: cameras.{label}.head_marker.size must be a "
                        f"number in METRES, got {raw_size!r}"
                    )
                marker_size = float(raw_size)
                if not 0.001 <= marker_size <= 1.0:
                    raise ValueError(
                        f"{path}: cameras.{label}.head_marker.size {marker_size} "
                        "is out of range; use the black-square edge in METRES "
                        "(expected 0.001-1.0)"
                    )
                stationary = head_marker.get("stationary_window")
                if stationary is not None:
                    valid = (
                        isinstance(stationary, (list, tuple))
                        and len(stationary) == 2
                        and all(
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                            for value in stationary
                        )
                    )
                    if not valid:
                        raise ValueError(
                            f"{path}: cameras.{label}.head_marker."
                            "stationary_window must be [start_s, end_s] with "
                            f"two finite numbers, got {stationary!r}"
                        )
                    low, high = (float(value) for value in stationary)
                    if high <= low:
                        raise ValueError(
                            f"{path}: cameras.{label}.head_marker."
                            f"stationary_window end ({high}) must be greater "
                            f"than start ({low})"
                        )
                    head_stationary_window = (low, high)
                occupied = {
                    int(arm.marker["id"])
                    for arm in arms.values()
                    if arm.marker.get("id") is not None
                }
                anchor_marker = anchor.get("marker") or {}
                if anchor_marker.get("id") is not None:
                    occupied.add(anchor_marker["id"])
                if marker_id in occupied:
                    raise ValueError(
                        f"{path}: cameras.{label}.head_marker.id {marker_id} "
                        "collides with an arm tool or scene-anchor marker id"
                    )
                head_marker = {
                    "dictionary": dictionary,
                    "id": marker_id,
                    "size": marker_size,
                }
            cameras[label] = CameraSpec(
                label=label,
                recording=(root / c["recording"]),
                camera=unit,
                camera_dir=(cfg_root / "cameras" / unit) if unit else None,
                key=c.get("key"),
                head_marker=head_marker,
                head_stationary_window=head_stationary_window,
            )

        configured_head_cameras = [
            label for label, camera in cameras.items() if camera.head_marker
        ]
        if len(configured_head_cameras) > 1:
            raise ValueError(
                f"{path}: multiple cameras declare head_marker: "
                f"{configured_head_cameras}; one output marker frame must be "
                "unambiguous"
            )

        sync_offsets, sync_method, sync_origin = _parse_sync(
            raw.get("sync"), path, set(arms), set(cameras)
        )

        cfg = cls(
            name=raw.get("name", root.name),
            root=root,
            default_tool=default_tool,
            arms=arms,
            anchor=anchor,
            # `notes` and `review` carry the operator's account of what
            # happened, so the parser keeps them.
            meta={
                k: raw[k]
                for k in ("description", "date", "notes", "review")
                if k in raw
            },
            task_window=_parse_task_window(raw, path),
            eval_success=_parse_eval_success(raw, path),
            sync_offsets=sync_offsets,
            sync_method=sync_method,
            sync_origin=sync_origin,
            cameras=cameras,
            config_dir=cfg_root,
        )
        if check_exists:
            cfg._check_exists(path)
        return cfg

    def _check_exists(self, manifest: Path) -> None:
        """Report every missing referenced path at once, not one per run."""
        missing: list[str] = []
        for arm in self.arms.values():
            if not arm.recording.is_file():
                missing.append(f"arms.{arm.label}.recording -> {arm.recording}")
            if not arm.camera_dir.is_dir():
                missing.append(f"arms.{arm.label}.camera -> {arm.camera_dir}")
            # gripper_unit needs no check here: a named unit whose file is
            # absent already refused at load.
        for cam in self.cameras.values():
            if not cam.recording.is_file():
                missing.append(f"cameras.{cam.label}.recording -> {cam.recording}")
            if cam.camera_dir is not None and not cam.camera_dir.is_dir():
                missing.append(f"cameras.{cam.label}.camera -> {cam.camera_dir}")
        if missing:
            listed = "\n  ".join(missing)
            raise FileNotFoundError(
                f"{manifest}: referenced paths missing:\n  {listed}"
            )

    # ---- derived ---------------------------------------------------------- #

    @property
    def stream_labels(self) -> set[str]:
        """Every stream that needs a place on the clock: arms plus extras."""
        return set(self.arms) | set(self.cameras)

    def global_clock(self) -> GlobalClock | None:
        """The ``sync:`` block as a validated :class:`GlobalClock`, or None."""
        if self.sync_offsets is None or self.sync_origin is None:
            return None
        return GlobalClock(
            origin=self.sync_origin, offsets_s=dict(self.sync_offsets)
        ).validate()

    def clock_covers_streams(self) -> bool:
        """Does the stored clock place every declared stream?

        False when a stream was added after the episode was synced.
        """
        return self.sync_offsets is not None and self.stream_labels <= set(
            self.sync_offsets
        )


# --------------------------------------------------------------------------- #
# the writer
# --------------------------------------------------------------------------- #

_SYNC_HEADER = (
    "sync:  # global clock: origin = frame 0 of the video stream that started"
    " LAST (arm OR cameras: entry); `origin:` names it and holds 0.0, every other"
    " offset is <= 0. task_window and curation windows are on this clock."
    " Machine-written by the sync stage - see kiwi.sync"
)


def write_sync_block(
    manifest: str | Path,
    offsets: dict[str, float],
    *,
    method: str,
    origin: str,
) -> None:
    """Insert or replace the top-level ``sync:`` block in ``manifest``, in place.

    Label-generic: ``offsets`` may name arms and extra ``cameras:`` streams
    alike. ``method`` records how they were obtained - required, with no default.

    ``origin`` names the stream whose frame 0 is the clock origin, so the
    offsets must already be re-origined (see
    :func:`kiwi.sync.to_last_start_origin`). A nonzero origin offset or any
    positive offset is a caller bug and raises, and a ``-0.0`` rounding residue
    is normalized so ``-0.000000`` never ships.

    The update is a textual replace-or-append rather than a YAML round-trip, so
    hand-written comments survive. The result is verified to parse, and to
    parse back to exactly what was asked for, before the file is touched.
    """
    manifest = Path(manifest)
    lines = manifest.read_text().splitlines()
    rounded = {k: float(f"{float(v):.6f}") for k, v in offsets.items()}
    if origin not in rounded:
        raise ValueError(
            f"{manifest}: sync origin {origin!r} not in offsets {sorted(rounded)}"
        )
    if abs(rounded[origin]) > 0.0:
        raise ValueError(
            f"{manifest}: origin stream {origin!r} has offset {rounded[origin]!r}, "
            f"must be 0.0 - re-origin with to_last_start_origin before writing"
        )
    rounded = {k: (0.0 if v == 0.0 else v) for k, v in rounded.items()}
    positive = sorted(k for k, v in rounded.items() if v > 0.0)
    if positive:
        raise ValueError(
            f"{manifest}: positive sync offset(s) {positive} - offsets must be "
            f"re-origined (to_last_start_origin) before writing the block"
        )

    fmt = ", ".join(f"{k}: {v:.6f}" for k, v in sorted(rounded.items()))
    block = [
        _SYNC_HEADER,
        f"  method: {method}",
        f"  origin: {origin}",
        f"  offsets: {{{fmt}}}",
    ]

    start = next((i for i, ln in enumerate(lines) if ln.startswith("sync:")), None)
    if start is None:
        while lines and not lines[-1].strip():
            lines.pop()
        out = lines + [""] + block
    else:
        end = start + 1
        while end < len(lines) and lines[end][:1] in (" ", "\t") and lines[end].strip():
            end += 1
        out = lines[:start] + block + lines[end:]

    text = "\n".join(out) + "\n"
    try:
        doc = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ValueError(
            f"{manifest}: written sync block failed to round-trip (writer bug)"
        ) from e
    parsed = doc.get("sync") or {}
    ok = (
        parsed.get("method") == method
        and parsed.get("origin") == origin
        and parsed.get("offsets") == rounded
    )
    if not ok:
        raise ValueError(
            f"{manifest}: written sync block failed to round-trip (writer bug)"
        )
    manifest.write_text(text)


_SCENE_HEADER = "scene:"


def write_scene_block(
    manifest: str | Path, *, scene_manifest: str, episode_id: str
) -> None:
    """Insert or replace the top-level ``scene:`` back-reference in ``manifest``.

    ``scene_manifest`` is the POSIX path of the scene's ``scene.yaml``
    relative to the episode directory - what the state stage resolves the
    published map set through - and ``episode_id`` the id that manifest lists
    this directory under (:mod:`kiwi.episode.scene` checks the two directions
    agree). ``kiwi scene init`` writes it for every episode it discovers.

    Same textual replace-or-append as :func:`write_sync_block`: hand-written
    comments survive. Both values are written quoted, so
    an id that reads as a date or a number still comes back as the string it
    is. The result is verified to parse back to exactly what was asked for
    before the file is touched.
    """
    manifest = Path(manifest)
    if not isinstance(scene_manifest, str) or not scene_manifest.strip():
        raise ValueError(f"{manifest}: scene back-reference needs a manifest path")
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError(f"{manifest}: scene back-reference needs an episode id")
    lines = manifest.read_text().splitlines()

    def quoted(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    block = [
        _SCENE_HEADER,
        f"  manifest: {quoted(scene_manifest)}",
        f"  episode: {quoted(episode_id)}",
    ]

    start = next((i for i, ln in enumerate(lines) if ln.startswith("scene:")), None)
    if start is None:
        while lines and not lines[-1].strip():
            lines.pop()
        out = lines + [""] + block
    else:
        end = start + 1
        while end < len(lines) and lines[end][:1] in (" ", "\t") and lines[end].strip():
            end += 1
        out = lines[:start] + block + lines[end:]

    text = "\n".join(out) + "\n"
    try:
        doc = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ValueError(
            f"{manifest}: written scene block failed to round-trip (writer bug)"
        ) from e
    if doc.get("scene") != {"manifest": scene_manifest, "episode": episode_id}:
        raise ValueError(
            f"{manifest}: written scene block failed to round-trip (writer bug)"
        )
    manifest.write_text(text)


def _main(argv=None) -> int:
    """``python -m kiwi.episode.manifest <path>`` - resolve and print a manifest."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument(
        "--no-check", action="store_true", help="skip the referenced-path check"
    )
    args = ap.parse_args(argv)

    cfg = EpisodeConfig.load(args.manifest, check_exists=not args.no_check)
    print(f"{cfg.name}  ({cfg.root})")
    print(f"  tool          {cfg.default_tool}")
    print(f"  task_window   {cfg.task_window}")
    print(f"  eval_success  {cfg.eval_success}")
    clock = cfg.global_clock()
    print(f"  clock         {clock.origin if clock else '(none)'}  {cfg.sync_method}")
    for label, arm in sorted(cfg.arms.items()):
        print(f"  arm {label:6s} {arm.camera}  {arm.recording.name}")
        print(f"      marker    {arm.marker}")
    for label, cam in sorted(cfg.cameras.items()):
        print(f"  cam {label:6s} {cam.camera or '-'}  {cam.recording.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

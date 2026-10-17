"""Tool / gripper-unit deployment configs: schema ``tool_v1`` / ``chopstick_unit_v1``.

The name-resolved deployment layer for ``config/gripper/`` - one directory
per tool design (``config/gripper/<tool>/``), holding the design's
``tool.yaml`` and the unit files of its physical builds under ``units/`` -
the tool-side analogue of :mod:`kiwi.core.camera.registry`. It lives in
``core`` so that both ``eef`` (which resolves the detection recipe) and
``episode`` (whose manifest ties an arm to its tool/unit) can resolve them
without importing each other.

The schema states its own layering: every key lives in a block named for
its variance class, and each block has a CEILING - the highest layer allowed
to state it:

* ``design:`` (tool file only) - facts of the tool DESIGN, frozen: which
  ``estimator`` bundle understands it, the robot-side ``replay`` mapping,
  the ``marker`` ``dictionary``. No other layer has keys for these, so
  nothing can shadow them.
* ``build:`` (unit file only) - facts of one PHYSICAL build: its tag
  (``marker.id`` + ``marker.size``, both required - the sticker is one
  object), the calibrated camera<->marker ``fixed_pose`` (written by
  ``kiwi eef calibrate``), and the optional ``mount_tcp:`` / ``mount_cam:``
  overrides of the robot description's two legs. Builtins are never YAML:
  the mount legs default to the description URDF's fixed-joint FK, composed
  by the export builder.
* ``session:`` (layered) - the estimator's tunable recipe, resolved as
  builtin < tool < unit < episode-arm < CLI override. The key set is owned
  by the estimator bundle (:func:`kiwi.eef.config.validate_session_block`);
  this module only carries the blocks.

Every block has a closed key set. An unknown key is REJECTED, and a key
placed in the wrong block names the block it belongs to, so a misplaced
fact fails loudly instead of half-loading.

Import-light (``dataclasses`` / ``pathlib`` / ``yaml`` / ``numpy``, no cv2).
The path builders route through :func:`kiwi.core.paths.config_root`, so every
caller honors ``$KIWI_CONFIG``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from kiwi.core.paths import config_root

__all__ = [
    "GRIPPER_UNIT_SCHEMA",
    "TOOL_SCHEMA",
    "GripperUnitConfig",
    "ToolConfig",
    "find_gripper_unit",
    "gripper_unit_config_path",
    "load_T_mount_cam",
    "load_T_mount_tcp",
    "load_gripper_unit",
    "load_tool_config",
    "merge_session",
    "tool_config_path",
    "write_unit_session",
]

# Schema tags.
TOOL_SCHEMA = "tool_v1"
GRIPPER_UNIT_SCHEMA = "chopstick_unit_v1"

# The closed key registries - one per block, plus the nested mappings. A key
# outside its registry is rejected, so the layering contract is enforced by
# the loader, not by convention.
_TOOL_TOP_KEYS = {"schema", "name", "description", "design", "session"}
_DESIGN_KEYS = {"estimator", "replay", "capture", "marker"}
_DESIGN_MARKER_KEYS = {"dictionary"}
_REPLAY_KEYS = {"gripper", "description"}
_UNIT_TOP_KEYS = {"schema", "description", "tool", "build", "session"}
_BUILD_KEYS = {"marker", "fixed_pose", "mount_tcp", "mount_cam"}
_BUILD_MARKER_KEYS = {"id", "size"}
_FIXED_POSE_KEYS = {"rvec", "tvec", "fov_deg", "out_size", "lens"}
_MOUNT_KEYS = {
    "mount_tcp": {"T_mount_tcp", "measured"},
    "mount_cam": {"T_mount_cam", "measured"},
}


def tool_config_path(tool: str, config_dir: Path | None = None) -> Path:
    """On-disk path of a tool design's config: ``config/gripper/<tool>/tool.yaml``.

    One directory per design, the shape of ``config/cameras/<unit>/``: the
    design file is the fixed name ``tool.yaml``, and the design's physical
    builds live beside it under ``units/``. This function and
    :func:`gripper_unit_config_path` are the single source of that layout -
    callers resolve names through them instead of hardcoding paths.
    ``config_dir`` overrides the config tree (defaults to
    :func:`kiwi.core.paths.config_root`).
    """
    base = Path(config_dir) if config_dir is not None else config_root()
    return base / "gripper" / tool / "tool.yaml"


def gripper_unit_config_path(
    unit: str, tool: str, config_dir: Path | None = None
) -> Path:
    """On-disk path of a unit's config: ``config/gripper/<tool>/units/<unit>.yaml``.

    Each physical build is one flat YAML under its design's ``units/`` dir,
    with the unit id as the file stem - the parent-child relation is the
    filesystem, so resolving a unit through a tool can only find that tool's
    builds. Callers holding only the unit id use :func:`find_gripper_unit`.
    """
    base = Path(config_dir) if config_dir is not None else config_root()
    return base / "gripper" / tool / "units" / f"{unit}.yaml"


def find_gripper_unit(unit: str, config_dir: Path | None = None) -> Path:
    """Locate ``<unit>.yaml`` across every design's ``units/`` dir.

    For callers that hold only the unit id (``kiwi eef run --gripper-unit``
    with no ``--tool``): scan ``config/gripper/*/units/`` and require exactly
    one match. Zero matches raise :class:`FileNotFoundError`; two or more
    raise :class:`ValueError` naming every hit - unit ids are global
    vocabulary (episode manifests reference them bare), so a collision is a
    config bug to fix, not a tie to break.
    """
    base = Path(config_dir) if config_dir is not None else config_root()
    hits = sorted((base / "gripper").glob(f"*/units/{unit}.yaml"))
    if len(hits) > 1:
        raise ValueError(
            f"gripper unit {unit!r} is ambiguous - a unit id must be unique "
            f"across designs, but it matches: {', '.join(str(p) for p in hits)}"
        )
    if not hits:
        raise FileNotFoundError(
            f"gripper unit {unit!r}: nothing matches "
            f"{base / 'gripper'}/*/units/{unit}.yaml"
        )
    return hits[0]


@dataclasses.dataclass(frozen=True)
class ToolConfig:
    """One tool design (``config/gripper/<tool>/tool.yaml``, ``tool_v1``).

    Carries the ``design:`` block flattened - ``estimator`` (which
    :mod:`kiwi.eef` bundle understands this tool), the marker ``dictionary``
    (the tag family every build of the design uses), capture geometry, and
    replay target. ``capture_description`` selects export camera-to-tip
    geometry and falls back to ``replay_description`` when the design has no
    ``capture:`` description; ``replay_gripper`` and ``replay_description``
    select the robot. Also carries the raw ``session:`` recipe defaults.
    ``name`` is the design id (the file stem).
    """

    name: str
    path: Path
    estimator: str
    marker_dictionary: str
    replay_gripper: str | None
    replay_description: str | None
    session: dict
    capture_description: str | None = None


@dataclasses.dataclass(frozen=True)
class GripperUnitConfig:
    """One physical build (``config/gripper/<tool>/units/<unit>.yaml``,
    ``chopstick_unit_v1``).

    Carries the ``build:`` block flattened. ``tool`` names the design this
    unit is a build of - the default parent config for the standalone CLIs,
    and cross-checked against the design directory the file lives under (a
    mismatch is a miscopied file and raises at load). ``marker_id`` /
    ``marker_size`` are the unit's actual tag - one physical object, both
    required; the size is stated per unit even when every build of the design
    uses the same one, because the pose calibration is bound to it.
    ``fixed_pose`` is the calibrated camera<->marker pose (``None`` until
    ``kiwi eef calibrate`` writes one); the :mod:`kiwi.eef` estimator
    consumes it, so it is left untyped here. ``mount_tcp`` / ``mount_cam``
    override the two legs the robot description otherwise supplies by
    fixed-joint FK - a build whose tip was re-glued and remeasured, or whose
    camera seats off its nominal pocket. Both are mount-frame
    (``p_mount = T @ p_leg``); a tip measured in the camera frame must be
    converted through the unit's effective ``T_mount_cam`` before it is
    written here, so the two legs stay independently overridable.
    ``session`` is this unit's recipe overrides.
    """

    name: str
    path: Path
    tool: str
    marker_id: int
    marker_size: float
    fixed_pose: dict | None
    mount_tcp: dict | None
    mount_cam: dict | None
    session: dict


def _require_mapping(raw, path: Path, what: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: {what} must be a mapping, got {raw!r}")
    return raw


def _reject_keys(
    block: Mapping, allowed, path: Path, what: str, belongs: Mapping | None = None
):
    """Reject keys outside ``allowed``, naming the right block when ``belongs``
    knows it (a fact of another layer stated here)."""
    belongs = belongs or {}
    unknown = sorted(set(map(str, block)) - set(allowed))
    if not unknown:
        return
    hints = []
    for k in unknown:
        if k in belongs:
            hints.append(f"{k!r} belongs in {belongs[k]}")
        else:
            hints.append(f"{k!r} is not a {what} key (known: {sorted(allowed)})")
    raise ValueError(f"{path}: " + "; ".join(hints))


def _check_schema(raw: Mapping, path: Path, expected: str) -> None:
    schema = raw.get("schema")
    if schema != expected:
        raise ValueError(f"{path}: schema is {schema!r}, expected {expected!r}")


def _marker_size_m(raw, path: Path, field: str) -> float:
    """Validate a marker edge length in metres.

    Rejected rather than coerced; the bounds catch millimetres typed where
    metres are expected.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{path}: `{field}` must be a number in METRES, got {raw!r}")
    size = float(raw)
    if not 0.001 <= size <= 1.0:
        raise ValueError(
            f"{path}: `{field}: {size}` is out of range - it is the tag's "
            f"black-square edge in METRES (expected 0.001-1.0; 19.5 mm is "
            f"0.0195, not 19.5)"
        )
    return size


def _check_fixed_pose(block, path: Path) -> dict | None:
    """Validate a ``build.fixed_pose:`` block (calibration record).

    ``rvec``/``tvec`` are the pose; ``fov_deg``/``out_size``/``lens`` are the
    rectifier it was solved under - provenance the estimator compares against
    the session it runs with (a mismatch warns). Written as one record by
    ``kiwi eef calibrate``; a missing half or a stray key is rejected.
    """
    if block is None:
        return None
    block = _require_mapping(block, path, "build.fixed_pose")
    _reject_keys(block, _FIXED_POSE_KEYS, path, "build.fixed_pose")
    for key in ("rvec", "tvec"):
        v = block.get(key)
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise ValueError(
                f"{path}: build.fixed_pose.{key} must be a 3-vector "
                f"(got {v!r}) - `kiwi eef calibrate` writes this block"
            )
    return dict(block)


def load_tool_config(path: Path) -> ToolConfig:
    """Load and validate a ``tool_v1`` design config."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _require_mapping(raw, path, "tool config")
    _check_schema(raw, path, TOOL_SCHEMA)
    _reject_keys(raw, _TOOL_TOP_KEYS, path, "tool_v1 top-level")

    design = _require_mapping(raw.get("design"), path, "design:")
    if not design:
        raise ValueError(
            f"{path}: tool_v1 requires a `design:` block (estimator, "
            f"marker.dictionary, optionally replay)"
        )
    _reject_keys(
        design,
        _DESIGN_KEYS,
        path,
        "design:",
        {
            "fixed_pose": "the gripper unit's build.fixed_pose (a calibration "
            "is a build fact)",
        },
    )
    estimator = design.get("estimator")
    if not estimator:
        raise ValueError(
            f"{path}: design.estimator is required - it names which kiwi.eef "
            f"estimator bundle understands this tool"
        )
    marker = _require_mapping(design.get("marker"), path, "design.marker")
    _reject_keys(
        marker,
        _DESIGN_MARKER_KEYS,
        path,
        "design.marker",
        {
            "id": "the gripper unit's build.marker.id (each build carries its own tag)",
            "size": "the gripper unit's build.marker.size (the tag is one "
            "physical object - id and size live together)",
        },
    )
    dictionary = marker.get("dictionary")
    if not dictionary:
        raise ValueError(
            f"{path}: design.marker.dictionary is required (e.g. DICT_4X4_50)"
        )
    replay = _require_mapping(design.get("replay"), path, "design.replay")
    _reject_keys(replay, _REPLAY_KEYS, path, "design.replay")
    capture = _require_mapping(design.get("capture"), path, "design.capture")
    _reject_keys(capture, {"description"}, path, "design.capture")

    return ToolConfig(
        # In the registry layout the design's identity is its directory
        # (every design file is named tool.yaml); a free-standing file keeps
        # its stem.
        name=path.parent.name if path.name == "tool.yaml" else path.stem,
        path=path,
        estimator=str(estimator),
        marker_dictionary=str(dictionary),
        replay_gripper=replay.get("gripper"),
        replay_description=replay.get("description"),
        session=dict(_require_mapping(raw.get("session"), path, "session:")),
        # No design.capture.description: capture follows the replay description.
        capture_description=capture.get("description", replay.get("description")),
    )


def load_gripper_unit(path: Path) -> GripperUnitConfig:
    """Load and validate a ``chopstick_unit_v1`` build config."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _require_mapping(raw, path, "gripper unit config")
    _check_schema(raw, path, GRIPPER_UNIT_SCHEMA)
    _reject_keys(raw, _UNIT_TOP_KEYS, path, "chopstick_unit_v1 top-level")

    tool = raw.get("tool")
    if not tool:
        raise ValueError(
            f"{path}: `tool:` is required - the unit->design link that "
            f"resolves the parent config"
        )
    if path.parent.name == "units" and path.parent.parent.name != tool:
        raise ValueError(
            f"{path}: declares tool {tool!r} but lives under design "
            f"{path.parent.parent.name!r} - a unit file's home is its own "
            f"design's units/ dir; fix `tool:` or move the file"
        )
    build = _require_mapping(raw.get("build"), path, "build:")
    if not build:
        raise ValueError(
            f"{path}: chopstick_unit_v1 requires a `build:` block (at least "
            f"marker: {{id, size}} - this build's physical tag)"
        )
    _reject_keys(build, _BUILD_KEYS, path, "build:")
    marker = _require_mapping(build.get("marker"), path, "build.marker")
    _reject_keys(marker, _BUILD_MARKER_KEYS, path, "build.marker")
    mid = marker.get("id")
    if isinstance(mid, bool) or not isinstance(mid, int):
        raise ValueError(
            f"{path}: build.marker.id must be an integer ArUco id, got {mid!r} "
            f"- the tag this build actually carries"
        )
    size = _marker_size_m(marker.get("size"), path, "build.marker.size")

    for leg, key in (("mount_tcp", "T_mount_tcp"), ("mount_cam", "T_mount_cam")):
        block = build.get(leg)
        if block is not None:
            # Validated at unit load; load_T_mount_* checks the matrix again
            # at composition time.
            _load_mount_transform(
                _require_mapping(block, path, f"build.{leg}"),
                key=key,
                leg=leg,
                path=path,
            )

    return GripperUnitConfig(
        name=path.stem,
        path=path,
        tool=str(tool),
        marker_id=int(mid),
        marker_size=size,
        fixed_pose=_check_fixed_pose(build.get("fixed_pose"), path),
        mount_tcp=build.get("mount_tcp"),
        mount_cam=build.get("mount_cam"),
        session=dict(_require_mapping(raw.get("session"), path, "session:")),
    )


def write_unit_session(path: Path, session: Mapping) -> Path:
    """Replace a unit file's ``session:`` override in place, keeping every
    other key; an empty ``session`` removes the block.

    The tuner's write path (``kiwi eef tune`` -> *write to unit*): the same
    in-place convention as the calibrated ``build.fixed_pose``. The
    new document is written beside the file and loaded through
    :func:`load_gripper_unit` BEFORE it replaces the original, so a write
    the loader would refuse never lands. Note the rewrite goes through the
    YAML dumper, so hand-written comments in the unit file do not survive -
    review the git diff before committing.
    """
    import os

    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _require_mapping(raw, path, "gripper unit config")
    if session:
        raw["session"] = dict(session)
    else:
        raw.pop("session", None)
    tmp = path.with_name(path.stem + ".tmp.yaml")
    tmp.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    try:
        load_gripper_unit(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return path


def merge_session(*layers: Mapping | None) -> dict:
    """Merge ``session:`` layers, later wins; ``None``/empty layers are skipped.

    The one merge of the session ladder (tool < unit < episode-arm). Key
    VALIDATION is per layer and per estimator - the callers run
    :func:`kiwi.eef.config.validate_session_block` on each layer at its
    source, so the error names the file that carries the typo.
    """
    out: dict = {}
    for layer in layers:
        if layer:
            out.update(layer)
    return out


def _load_mount_transform(
    block: Mapping,
    *,
    key: str,
    leg: str,
    path: Path | None = None,
) -> np.ndarray:
    """Validate one ``build.mount_*`` block and return its 4x4.

    The two legs differ only in which matrix key they spell, so they share one
    validator. The OTHER leg's matrix key is an unknown key here.
    """
    where = f"{path}: " if path is not None else ""
    allowed = _MOUNT_KEYS[leg]
    unknown = sorted(set(map(str, block)) - allowed)
    if unknown:
        raise ValueError(
            f"{where}build.{leg} has unknown key(s) "
            f"{', '.join(repr(k) for k in unknown)}; known: {sorted(allowed)}"
        )
    if key not in block:
        raise KeyError(f"{where}build.{leg} has no {key!r} 4x4 matrix")
    T = np.asarray(block[key], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{where}{key} is {T.shape}, expected (4, 4)")
    if not np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-4):
        raise ValueError(
            f"{where}{key} rotation block is not orthonormal (not a valid SE(3))"
        )
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(
            f"{where}{key} bottom row is {T[3].tolist()}, expected [0, 0, 0, 1]"
        )
    return T


def load_T_mount_tcp(block: Mapping) -> np.ndarray:
    """4x4 ``T_mount_tcp`` from a unit's ``build.mount_tcp:`` block.

    ``p_mount = T_mount_tcp @ p_tcp``: the tool tip in the gripper's mount
    frame. This is the per-unit override of the leg the robot description
    supplies by fixed-joint FK, for a build whose tip differs from the
    design's CAD (a re-glued jaw, remeasured on that unit).
    """
    return _load_mount_transform(block, key="T_mount_tcp", leg="mount_tcp")


def load_T_mount_cam(block: Mapping) -> np.ndarray:
    """4x4 ``T_mount_cam`` from a unit's ``build.mount_cam:`` block.

    ``p_mount = T_mount_cam @ p_cam``: the camera optical frame in the
    gripper's mount frame - the seating half of the same pair, overridden
    when a build's camera does not sit at its nominal pocket.
    """
    return _load_mount_transform(block, key="T_mount_cam", leg="mount_cam")

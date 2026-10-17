"""``scene.yaml``: several episodes sharing one physical scene, spelled once.

A scene is the unit map building works at: one immutable set of maps
built from one builder episode, then every episode of the scene localized
against it. This module owns the manifest that records the grouping -
which episodes belong, which one builds the map, under which pinned mapping
profile.

The manifest is strict and portable by construction:

* every path is POSIX-relative to ``scene.yaml`` and may not escape its
  directory, so a scene keeps working after the root moves;
* unknown keys are refused at every level;
* the mapping profile is pinned by content digest (``canonical_yaml_sha256``:
  parsed content, so comment edits don't invalidate it) and a mismatch is a
  load error;
* an episode may carry a ``scene:`` back-reference; when present it must
  agree with the scene manifest in both directions.

The builder episode is named by the operator: ``mapping.selection.mode`` is
always ``explicit``.

The usable-duration floor merges monotonically: the schema floor is 60
seconds, and the scene and profile values are both required to be at least
that; the effective value is the maximum of all three, so disagreement can
only ever tighten.
"""

from __future__ import annotations

import math
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

from kiwi.core.atomic import atomic_write_text
from kiwi.core.digest import digest_canonical_yaml, parse_digest, verify_digest
from kiwi.core.paths import config_root

__all__ = [
    "SCENE_MANIFEST",
    "SCENE_SCHEMA",
    "SCHEMA_MIN_USABLE_S",
    "SceneConfig",
    "SceneMappingProfile",
    "effective_min_usable_s",
    "init_scene",
    "load_mapping_profile",
    "load_scene",
]

SCENE_SCHEMA = "scene"
SCENE_MANIFEST = "scene.yaml"
PROFILE_SCHEMA = "scene_mapping_profile"

#: The schema floor on usable map-build video, in seconds. Scene and profile
#: values may raise it, never lower it.
SCHEMA_MIN_USABLE_S = 60.0


@dataclass(frozen=True)
class SceneMappingProfile:
    """The pinned mapping recipe: the selection floor and the engine recipe."""

    name: str
    min_usable_duration_s: float
    #: Frames per second the map build sees. The builder maps at 30 fps and
    #: refuses a profile that states another rate; localization is never
    #: decimated.
    map_fps: float = 30.0
    #: Square edge, in pixels, of the frames the engine reads. The builder
    #: exports at 960 px and refuses a profile that states another size.
    map_image_size: int = 960


@dataclass(frozen=True)
class SceneConfig:
    """A loaded ``scene.yaml`` with every reference verified."""

    path: Path  # the manifest itself, resolved
    name: str
    episodes: dict[str, str]  # id -> POSIX-relative episode directory
    builder_episode: str  # the episode that builds the maps
    map_builder_arm: str  # which arm's maps freeze the scene frame
    #: "alignment_only" | "global_state". Read at load only, where it decides
    #: whether ``state_profile`` is required or forbidden; the state stage
    #: does not check it - the pipeline profile a run uses decides whether
    #: state runs.
    trajectory_mode: str
    masking: str  # one of kiwi.track.masks.MASKING_MODES
    #: Pipeline profile NAME, recorded in the manifest and printed by
    #: ``kiwi scene init`` and ``kiwi scene inspect``. ``kiwi run`` does not
    #: read it: it runs ``--profile``, or the default profile.
    pipeline_profile: str
    #: Required in global_state, forbidden otherwise. The state stage runs
    #: this recipe, and ``scene-fusion`` when the scene names none.
    state_profile: str | None
    min_usable_duration_s: float  # the scene's own stated value
    profile_path: str  # POSIX-relative to the manifest
    profile_digest: str
    profile: SceneMappingProfile

    @property
    def root(self) -> Path:
        return self.path.parent

    def episode_dir(self, episode_id: str) -> Path:
        return self.root / self.episodes[episode_id]


def effective_min_usable_s(scene: SceneConfig) -> float:
    """The floor that actually applies: max(schema, scene, profile) - a
    monotone merge, so no value can weaken another."""
    return max(
        SCHEMA_MIN_USABLE_S,
        scene.min_usable_duration_s,
        scene.profile.min_usable_duration_s,
    )


def _require_keys(doc: dict, allowed: set[str], required: set[str], where: str):
    unknown = set(doc) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)} - refused")
    missing = required - set(doc)
    if missing:
        raise ValueError(f"{where}: missing required key(s) {sorted(missing)}")


def _require_relative(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: must be a non-empty relative path string")
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value:
        raise ValueError(
            f"{where}: {value!r} must be POSIX-relative and stay inside the "
            f"scene directory - absolute paths and '..' do not survive a "
            f"moved or shared scene"
        )
    return value


def _require_name(value: object, where: str) -> str:
    """A profile name is a safe basename: it becomes a path component under
    the scene root, so separators, traversal, and emptiness are refused."""
    if (
        not isinstance(value, str)
        or not value
        or "/" in value
        or "\\" in value
        or value in (".", "..")
        or value.startswith(".")
    ):
        raise ValueError(f"{where}: {value!r} is not a safe profile name")
    return value


def _require_min_usable(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: min_usable_duration_s must be a number")
    v = float(value)
    if not math.isfinite(v) or v < SCHEMA_MIN_USABLE_S:
        raise ValueError(
            f"{where}: min_usable_duration_s is {value!r}; the schema floor "
            f"is {SCHEMA_MIN_USABLE_S:g} s and a stated value may raise it, "
            f"never lower it"
        )
    return v


@contextmanager
def _refuse_invalid_yaml(path: Path) -> Iterator[None]:
    """Re-raise a YAML parser error from the block as a ValueError naming ``path``.

    Every caller of this module reports a ValueError as a result, while the
    parser's own error escapes them as a traceback that names no file.
    """
    try:
        yield
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: not valid YAML: {e}") from e


def load_mapping_profile(path: Path) -> SceneMappingProfile:
    """Parse one ``scene_mapping_profile`` YAML, strictly."""
    path = Path(path)
    with _refuse_invalid_yaml(path):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: profile must be a YAML mapping")
    if doc.get("schema") != PROFILE_SCHEMA:
        raise ValueError(
            f"{path}: schema is {doc.get('schema')!r}, expected {PROFILE_SCHEMA!r}"
        )
    _require_keys(
        doc,
        {"schema", "name", "selection", "engine"},
        {"schema", "name", "selection"},
        str(path),
    )
    name = doc["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}: profile name must be a non-empty string")
    selection = doc["selection"]
    if not isinstance(selection, dict):
        raise ValueError(f"{path}: selection must be a mapping")
    _require_keys(
        selection,
        {"min_usable_duration_s"},
        {"min_usable_duration_s"},
        f"{path}: selection",
    )

    def _pos(key: str, node: dict, where: str) -> float:
        v = node[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{where}: {key} must be a number")
        v = float(v)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{where}: {key} must be positive and finite")
        return v

    engine = doc.get("engine") or {}
    if not isinstance(engine, dict):
        raise ValueError(f"{path}: engine must be a mapping")
    _require_keys(engine, {"map_fps", "map_image_size"}, set(), f"{path}: engine")
    # An omitted field takes the dataclass default.
    recipe: dict[str, float | int] = {}
    if "map_fps" in engine:
        recipe["map_fps"] = _pos("map_fps", engine, f"{path}: engine")
    if "map_image_size" in engine:
        raw_size = engine["map_image_size"]
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size <= 0:
            raise ValueError(
                f"{path}: engine.map_image_size must be a positive integer"
            )
        recipe["map_image_size"] = raw_size

    return SceneMappingProfile(
        name=name,
        min_usable_duration_s=_require_min_usable(
            selection["min_usable_duration_s"], str(path)
        ),
        **recipe,
    )


def load_scene(path: Path) -> SceneConfig:
    """Parse ``scene.yaml`` and verify every reference, failing early.

    Everything that can be checked without running a component is checked
    here: episode directories and manifests exist, the pinned profile's
    digest matches, back-references (when an episode carries one) agree in
    both directions. A scene that loads is a scene a later stage can trust
    structurally.
    """
    path = Path(path).resolve()
    with _refuse_invalid_yaml(path):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _scene_from_doc(doc, path)


def _scene_from_doc(
    doc: object, path: Path, *, profile_file: Path | None = None
) -> SceneConfig:
    """Validate a parsed manifest as the scene at ``path`` (resolved).

    Every reference resolves against ``path.parent``, so the manifest need
    not exist yet: :func:`init_scene` validates the document it is about to
    write, through the same checks :func:`load_scene` runs. The pinned
    profile is read from ``profile_file`` when given - init passes the
    builtin template the scene-local copy is made from - and otherwise from
    the manifest's own ``mapping.profile.path``.
    """
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: scene manifest must be a YAML mapping")
    if doc.get("schema") != SCENE_SCHEMA:
        raise ValueError(
            f"{path}: schema is {doc.get('schema')!r}, expected {SCENE_SCHEMA!r}"
        )
    _require_keys(
        doc,
        {"schema", "name", "episodes", "mapping", "processing"},
        {"schema", "name", "episodes", "mapping", "processing"},
        str(path),
    )
    name = doc["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}: scene name must be a non-empty string")

    episodes_raw = doc["episodes"]
    if not isinstance(episodes_raw, dict) or not episodes_raw:
        raise ValueError(f"{path}: episodes must be a non-empty mapping of id -> dir")
    episodes: dict[str, str] = {}
    for episode_id, rel in episodes_raw.items():
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"{path}: episode ids must be non-empty strings")
        episodes[episode_id] = _require_relative(rel, f"{path}: episodes[{episode_id}]")

    mapping = doc["mapping"]
    if not isinstance(mapping, dict):
        raise ValueError(f"{path}: mapping must be a mapping")
    _require_keys(
        mapping,
        {"selection", "profile", "map_builder_arm"},
        {"selection", "profile", "map_builder_arm"},
        f"{path}: mapping",
    )
    selection = mapping["selection"]
    if not isinstance(selection, dict):
        raise ValueError(f"{path}: mapping.selection must be a mapping")
    _require_keys(
        selection,
        {"mode", "episode", "min_usable_duration_s"},
        {"mode", "episode", "min_usable_duration_s"},
        f"{path}: mapping.selection",
    )
    mode = selection["mode"]
    if mode != "explicit":
        raise ValueError(
            f"{path}: mapping.selection.mode is {mode!r}; expected 'explicit' "
            f"with mapping.selection.episode naming the builder episode"
        )
    builder = selection["episode"]
    if builder not in episodes:
        raise ValueError(
            f"{path}: mapping.selection.episode {builder!r} is not one of "
            f"the scene's episodes {sorted(episodes)}"
        )
    min_usable = _require_min_usable(
        selection["min_usable_duration_s"], f"{path}: mapping.selection"
    )
    arm = mapping.get("map_builder_arm")
    if arm not in ("left", "right"):
        raise ValueError(
            f"{path}: mapping.map_builder_arm is {arm!r}, expected 'left' or "
            f"'right' - this arm's maps freeze the scene frame"
        )

    profile_ref = mapping["profile"]
    if not isinstance(profile_ref, dict):
        raise ValueError(f"{path}: mapping.profile must be a mapping")
    _require_keys(
        profile_ref, {"path", "digest"}, {"path", "digest"}, f"{path}: mapping.profile"
    )
    profile_rel = _require_relative(
        profile_ref["path"], f"{path}: mapping.profile.path"
    )
    digest = profile_ref["digest"]
    if not isinstance(digest, str):
        raise ValueError(f"{path}: mapping.profile.digest must be a string")
    algorithm, _ = parse_digest(digest)
    if algorithm != "canonical_yaml_sha256":
        raise ValueError(
            f"{path}: mapping.profile.digest uses {algorithm!r}; a YAML "
            f"profile is pinned by canonical_yaml_sha256"
        )
    profile_path = profile_file or path.parent / profile_rel
    if not profile_path.is_file():
        raise FileNotFoundError(f"{path}: pinned profile {profile_rel} is missing")
    with _refuse_invalid_yaml(profile_path):
        verify_digest(digest, profile_path)  # the canonical digest parses it
    profile = load_mapping_profile(profile_path)

    processing = doc["processing"]
    if not isinstance(processing, dict):
        raise ValueError(f"{path}: processing must be a mapping")
    _require_keys(
        processing,
        {"trajectory_mode", "masking", "pipeline_profile", "state_profile"},
        {"trajectory_mode", "masking", "pipeline_profile"},
        f"{path}: processing",
    )
    from kiwi.track.masks import MASKING_MODES

    masking = processing["masking"]
    if masking not in MASKING_MODES:
        raise ValueError(
            f"{path}: processing.masking is {masking!r}, expected one of "
            f"{MASKING_MODES}"
        )
    trajectory_mode = processing["trajectory_mode"]
    if trajectory_mode not in ("alignment_only", "global_state"):
        raise ValueError(
            f"{path}: processing.trajectory_mode is {trajectory_mode!r}, "
            f"expected 'alignment_only' or 'global_state'"
        )
    pipeline_profile = _require_name(
        processing["pipeline_profile"], f"{path}: processing.pipeline_profile"
    )
    state_profile = processing.get("state_profile")
    if trajectory_mode == "global_state":
        state_profile = _require_name(
            state_profile, f"{path}: processing.state_profile"
        )
    elif state_profile is not None:
        raise ValueError(
            f"{path}: alignment_only names a state profile it would never "
            f"use - remove it or switch trajectory_mode"
        )

    scene = SceneConfig(
        path=path,
        name=name,
        episodes=episodes,
        builder_episode=builder,
        map_builder_arm=arm,
        trajectory_mode=trajectory_mode,
        masking=masking,
        pipeline_profile=pipeline_profile,
        state_profile=state_profile,
        min_usable_duration_s=min_usable,
        profile_path=profile_rel,
        profile_digest=digest,
        profile=profile,
    )

    for episode_id in episodes:
        _check_episode(scene, episode_id)
    return scene


def _check_episode(scene: SceneConfig, episode_id: str) -> None:
    """The episode exists, has a manifest, and any back-reference agrees."""
    from kiwi.episode.manifest import MANIFEST_NAME

    edir = scene.episode_dir(episode_id)
    manifest = edir / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{scene.path}: episode {episode_id!r} -> "
            f"{scene.episodes[episode_id]} has no {MANIFEST_NAME}"
        )
    from kiwi.episode.manifest import EpisodeConfig

    try:
        EpisodeConfig.load(manifest, check_exists=False)
    except (ValueError, FileNotFoundError) as e:
        raise ValueError(
            f"{scene.path}: episode {episode_id!r} does not load through the "
            f"episode manifest contract: {e}"
        ) from e
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    back = raw.get("scene") if isinstance(raw, dict) else None
    if back is None:
        return  # a back-reference is optional
    if not isinstance(back, dict):
        raise ValueError(f"{manifest}: scene back-reference must be a mapping")
    _require_keys(back, {"manifest", "episode"}, {"manifest", "episode"}, str(manifest))
    target = (edir / str(back["manifest"])).resolve()
    if target != scene.path:
        raise ValueError(
            f"{manifest}: scene.manifest resolves to {target}, not "
            f"{scene.path} - the reference must agree in both directions"
        )
    if back["episode"] != episode_id:
        raise ValueError(
            f"{manifest}: scene.episode is {back['episode']!r} but the scene "
            f"manifest lists this directory as {episode_id!r}"
        )


def builtin_profile_dir() -> Path:
    """Where the builtin mapping-profile templates live (under
    :func:`kiwi.core.paths.config_root`, so ``$KIWI_CONFIG`` moves it)."""
    return config_root() / "scene" / "mapping-profiles"


def init_scene(
    *,
    root: Path,
    mapping_episode: str | None,
    mapping_profile: str = "default",
    map_builder_arm: str = "left",
    trajectory_mode: str = "global_state",
    pipeline_profile: str = "global-state",
    state_profile: str | None = None,
    min_usable_s: float = SCHEMA_MIN_USABLE_S,
) -> SceneConfig:
    """Create ``scene.yaml`` under ``root`` from the episodes already there.

    Episodes are discovered (direct children of ``root`` and of
    ``root/episodes/`` that contain an ``episode.yaml``), the named builtin
    profile template is *copied* into ``root/profiles/`` and pinned by
    digest - the copy is the run-time authority, so editing the builtin
    later cannot silently change an existing scene. Nothing is overwritten:
    an existing ``scene.yaml`` refuses, and so does a profile copy whose
    content differs from the template (an identical copy left by an
    interrupted init is reused). The manifest passes every :func:`load_scene`
    check before anything is written, so a refused init leaves the root as it
    found it and the retry starts clean.

    Once the manifest is written, every discovered episode gets the matching
    ``scene:`` back-reference in its own ``episode.yaml``
    (:func:`kiwi.episode.manifest.write_scene_block`); the state stage resolves
    the published map set through that block. An episode already carrying the (validated, agreeing)
    back-reference is left as it is.
    """
    from kiwi.episode.manifest import MANIFEST_NAME, write_scene_block

    _require_min_usable(min_usable_s, "kiwi scene init")
    _require_name(mapping_profile, "kiwi scene init: --mapping-profile")
    if not mapping_episode:
        raise ValueError(
            "kiwi scene init: --mapping-episode names the episode that builds "
            "the scene's maps"
        )
    if trajectory_mode == "global_state" and state_profile is None:
        state_profile = "scene-fusion"
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"scene root {root} is not a directory")
    manifest = root / SCENE_MANIFEST
    if manifest.exists():
        raise FileExistsError(f"{manifest} already exists - never overwritten")

    found: dict[str, str] = {}
    candidates = sorted(root.glob("*")) + sorted((root / "episodes").glob("*"))
    for child in candidates:
        if not child.is_dir() or not (child / MANIFEST_NAME).is_file():
            continue
        rel = child.relative_to(root).as_posix()
        if child.name in found:
            raise ValueError(
                f"two episode directories share the id {child.name!r} "
                f"({found[child.name]} and {rel}) - ids are directory names "
                f"and must be unique"
            )
        found[child.name] = rel
    if not found:
        raise FileNotFoundError(
            f"no episode directories under {root} (looked for {MANIFEST_NAME} "
            f"in its children and in episodes/)"
        )
    if mapping_episode not in found:
        raise ValueError(
            f"--mapping-episode {mapping_episode!r} is not one of the "
            f"discovered episodes {sorted(found)}"
        )

    template = builtin_profile_dir() / f"{mapping_profile}.yaml"
    if not template.is_file():
        available = sorted(p.stem for p in builtin_profile_dir().glob("*.yaml"))
        raise FileNotFoundError(
            f"no builtin mapping profile {mapping_profile!r} "
            f"(available: {available or 'none'})"
        )
    profile_copy = root / "profiles" / f"{mapping_profile}.yaml"
    with _refuse_invalid_yaml(template):
        digest = digest_canonical_yaml(template)
    # A copy left by an interrupted init is fine to retry over - but only
    # when its content matches the template; anything else is a real file
    # this init must not clobber.
    if profile_copy.exists():
        with _refuse_invalid_yaml(profile_copy):
            copy_digest = digest_canonical_yaml(profile_copy)
        if copy_digest != digest:
            raise FileExistsError(
                f"{profile_copy} exists with different content - never overwritten"
            )

    selection: dict = {
        "mode": "explicit",
        "episode": mapping_episode,
        "min_usable_duration_s": float(min_usable_s),
    }
    doc = {
        "schema": SCENE_SCHEMA,
        "name": root.name,
        "episodes": dict(sorted(found.items())),
        "mapping": {
            "selection": selection,
            "map_builder_arm": map_builder_arm,
            "profile": {
                "path": f"profiles/{mapping_profile}.yaml",
                "digest": digest,
            },
        },
        "processing": {
            "trajectory_mode": trajectory_mode,
            "masking": "learned",
            "pipeline_profile": pipeline_profile,
            **({"state_profile": state_profile} if state_profile else {}),
        },
    }
    text = yaml.safe_dump(doc, sort_keys=False)
    # Validate the exact text about to be written, before writing anything:
    # init never produces a scene that inspect would refuse, and a refusal
    # leaves nothing behind to block the retry.
    scene = _scene_from_doc(yaml.safe_load(text), manifest, profile_file=template)
    if not profile_copy.exists():
        profile_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template, profile_copy)
    atomic_write_text(manifest, text)
    for episode_id, rel in found.items():
        edir = root / rel
        raw = yaml.safe_load((edir / MANIFEST_NAME).read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict) and raw.get("scene") is not None:
            continue  # validated above: it already points here
        write_scene_block(
            edir / MANIFEST_NAME,
            scene_manifest=Path(os.path.relpath(manifest, edir)).as_posix(),
            episode_id=episode_id,
        )
    return scene

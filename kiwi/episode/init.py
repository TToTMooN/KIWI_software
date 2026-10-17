"""``kiwi init``: probe an episode's recordings and write ``episode.yaml``.

Recordings live flat in ``<episode>/raw/``, exactly as the operator sorted
them there after ``kiwi import`` (see :mod:`kiwi.core.layout`). Every media
file in ``raw/`` is a recording, by construction - no tool writes derived
files into it.

Arms are assigned by **camera serial**, never by file or directory name.
The caller names the two units and the clips are matched to them:

    kiwi init --episode data/vid51 --left insta360-X5-001 --right insta360-X5-003

``--root`` applies one assignment to every episode under a directory.

Two cases are refused: two clips resolving to the same named unit, and a
named unit that no clip resolves to. The second is also how a misnamed unit
shows up: its clip resolves to a unit nobody named, so it becomes an extra
stream (below) and leaves the named arm without a clip.

Clips that do not resolve to either named unit become extra ``cameras:``
streams, labelled by the ``_<tag>`` suffix ``kiwi import`` stamps on the
filename (``--label ego_video`` at import time), or by the whole stem for a
clip imported untagged. That is how the ego view lands: a GO3 ``.mp4``
carries no readable telemetry serial, so it cannot match an arm.

Sync still needs every extra stream bound to a registry unit, whatever the rig:
the A/V correction must establish both clips' camera models before it can
decide that a same-model, same-mode pair cancels, or read a mixed pair's delays
from the units' tables, so it refuses a stream with no unit. An ``.insv`` clip
whose embedded serial is registered is bound automatically. A clip with no
readable or registered serial (a GO3 ``.mp4``, a demuxed per-lens ``.mp4``)
cannot be bound from disk, so its unit is told with ``--camera <label>=<unit>``,
exactly like the arms:

    kiwi init --episode data/vid53 --left insta360-X5-001 --right insta360-X5-003 \
              --camera ego_video=insta360-GO3-001 \
              --head-marker-id 50

A rig whose extra streams all resolve by serial needs no ``--camera``. The
standard head target is ``DICT_4X4_100`` with a 0.049 m black-square edge;
``--head-marker-dictionary`` and ``--head-marker-size`` override those facts
for a different rig. With multiple extra streams, ``--head-camera`` names the
one carrying the target.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from kiwi.core.camera.ingest import clip_label
from kiwi.core.camera.registry import resolve_camera_for_video
from kiwi.core.layout import RAW_DIR, raw_dir
from kiwi.core.paths import config_root
from kiwi.episode.manifest import MANIFEST_NAME, SCHEMA

__all__ = [
    "DEFAULT_HEAD_MARKER_DICTIONARY",
    "DEFAULT_HEAD_MARKER_SIZE_M",
    "MEDIA_SUFFIXES",
    "Clip",
    "discover_clips",
    "find_episode_dirs",
    "init_episode",
]

MEDIA_SUFFIXES = frozenset({".insv", ".mp4", ".mov", ".mkv"})

DEFAULT_HEAD_MARKER_DICTIONARY = "DICT_4X4_100"
DEFAULT_HEAD_MARKER_SIZE_M = 0.049

_TODO = "TODO"
_UNIT_PLACEHOLDER = (
    "    # gripper_unit: <unit>   # this arm's physical build: "
    "config/gripper/<tool>/units/<unit>.yaml"
)


def _with_unit_placeholders(text: str) -> str:
    """Mark where each arm's ``gripper_unit:`` goes.

    The skeleton cannot know the build, so every ``arms.<side>:`` block gets a
    commented line showing the key and where its file lives. YAML drops the
    comment, so the manifest loads exactly as before until the key is filled.
    The dump never folds a scalar, so ``recording:`` is always one line.
    """
    out: list[str] = []
    in_arms = False
    for line in text.splitlines():
        if line and not line[0].isspace():
            in_arms = line.startswith("arms:")
        out.append(line)
        if in_arms and line.startswith("    recording: "):
            out.append(_UNIT_PLACEHOLDER)
    return "\n".join(out) + "\n"


@dataclasses.dataclass(frozen=True)
class Clip:
    """One recording found in an episode's ``raw/`` directory."""

    path: Path
    label: str  # the filename's _<tag> suffix, else its whole stem
    unit: str | None  # registry unit resolved from the embedded serial
    serial: str | None

    @property
    def rel(self) -> str:
        return f"{RAW_DIR}/{self.path.name}"


def discover_clips(episode_dir: Path, cameras_dir: Path) -> tuple[list[Clip], str]:
    """Every media file in ``<episode>/raw/``, resolved against the registry.

    Returns ``(clips, problem)``; a non-empty ``problem`` means nothing usable
    was found and names what to fix. ``raw/`` holds recordings and nothing
    else (:mod:`kiwi.core.layout`), so a media file in it IS a recording - the
    only junk filtered is the AppleDouble ``._*`` sidecars USB media pick up
    from macOS.
    """
    raw = raw_dir(episode_dir)
    if not raw.is_dir():
        return [], (
            f"no {RAW_DIR}/ under {episode_dir}. Recordings live flat in "
            f"<episode>/{RAW_DIR}/, as `kiwi import` leaves them - make the "
            f"directory and move the clips in."
        )
    clips: list[Clip] = []
    for f in sorted(raw.iterdir()):
        if (
            not f.is_file()
            or f.suffix.lower() not in MEDIA_SUFFIXES
            or f.name.startswith("._")
        ):
            continue
        match = resolve_camera_for_video(f, cameras_dir)
        clips.append(
            Clip(
                path=f,
                label=clip_label(f),
                unit=match.unit if match else None,
                serial=match.serial if match else None,
            )
        )
    if not clips:
        return [], (
            f"no recordings in {raw}. Expected the episode's clips flat in "
            f"{RAW_DIR}/, one media file per stream."
        )
    return clips, ""


def find_episode_dirs(root: Path) -> list[Path]:
    """Immediate subdirectories of ``root`` whose ``raw/`` holds a recording.

    One sitting's worth of takes, to which one arm assignment applies.
    """
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        raw = raw_dir(d)
        if raw.is_dir() and any(
            f.is_file()
            and f.suffix.lower() in MEDIA_SUFFIXES
            and not f.name.startswith("._")
            for f in raw.iterdir()
        ):
            out.append(d)
    return out


def _assign(
    clips: list[Clip], left: str, right: str, extra_units: dict[str, str] | None = None
) -> tuple[dict, dict, str]:
    """Split discovered clips into arms and extras, or explain why not."""
    if left == right:
        return {}, {}, f"--left and --right name the same unit ({left!r})"
    extra_units = dict(extra_units or {})

    wanted = {left: "left", right: "right"}
    arms: dict[str, Clip] = {}
    extras: dict[str, Clip] = {}
    reserved: list[Clip] = []
    for clip in clips:
        side = wanted.get(clip.unit) if clip.unit else None
        if side is None:
            # Not one of the named arm bodies: an ego view, a third-person rig,
            # or a clip whose telemetry could not be read.
            if clip.label in (RAW_DIR, "left", "right"):
                # Judged only after the arms are accounted for: when an arm
                # unit is misnamed, ITS clip is what lands here carrying
                # "left", and the actionable refusal is the missing arm.
                reserved.append(clip)
                continue
            if clip.label in extras:
                return {}, {}, f"two extra streams share the label {clip.label!r}"
            named = extra_units.pop(clip.label, None)
            if named is not None:
                clip = dataclasses.replace(clip, unit=named)
            extras[clip.label] = clip
            continue
        if side in arms:
            prev = arms[side]
            return (
                {},
                {},
                f"two clips both resolve to {clip.unit!r} ({prev.rel} and "
                f"{clip.rel}), so which one is the {side} arm is ambiguous. A "
                f"camera cannot record two arms at once - check the serials.",
            )
        arms[side] = clip

    if extra_units:
        labels = ", ".join(sorted(extra_units))
        found = ", ".join(sorted(extras)) or "none"
        return (
            {},
            {},
            f"--camera named {labels}, which is not an extra stream here "
            f"(extra streams found: {found}). The label is the filename's "
            f"_<tag> suffix (`kiwi import --label`), or its whole stem when "
            f"untagged.",
        )
    missing = [f"--{s} {u}" for u, s in wanted.items() if s not in arms]
    if missing:
        seen = (
            ", ".join(f"{c.rel} -> {c.unit or 'no readable serial'}" for c in clips)
            or "nothing"
        )
        return (
            {},
            {},
            f"no clip resolves to {' or '.join(missing)}. Found: {seen}. Either "
            f"the unit names are wrong, or config/cameras/ has no camera.yaml "
            f"carrying those serials.",
        )
    if reserved:
        c = reserved[0]
        return (
            {},
            {},
            f"{c.path.name} carries the stream label {c.label!r}, which is "
            f"reserved ({RAW_DIR}/ holds the recordings; left/ and right/ are "
            f"the arms' artifact directories). Re-tag the clip - rename it, "
            f"or re-import with --label.",
        )
    return arms, extras, ""


def init_episode(
    episode_dir: Path,
    *,
    left: str,
    right: str,
    tool: str | None = None,
    extra_units: dict[str, str] | None = None,
    head_marker_id: int | None = None,
    head_camera: str | None = None,
    head_marker_dictionary: str = DEFAULT_HEAD_MARKER_DICTIONARY,
    head_marker_size: float = DEFAULT_HEAD_MARKER_SIZE_M,
    config_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path | None, str]:
    """Write ``<episode_dir>/episode.yaml``; return ``(path, problem)``.

    A non-empty ``problem`` means nothing was written and says what to fix. An
    existing manifest, with the operator's notes, the scrubbed ``task_window``
    and the machine-written ``sync:`` clock it carries, is never overwritten
    without ``force``.
    """
    episode_dir = Path(episode_dir)
    if head_marker_id is not None and (
        isinstance(head_marker_id, bool)
        or not isinstance(head_marker_id, int)
        or head_marker_id < 0
    ):
        return None, (
            f"--head-marker-id must be a non-negative integer, got {head_marker_id!r}"
        )
    if head_camera is not None and head_marker_id is None:
        return None, "--head-camera requires --head-marker-id"
    if head_marker_id is not None:
        if not isinstance(head_marker_dictionary, str) or not head_marker_dictionary:
            return None, "--head-marker-dictionary must be a non-empty string"
        if (
            isinstance(head_marker_size, bool)
            or not isinstance(head_marker_size, (int, float))
            or not 0.001 <= float(head_marker_size) <= 1.0
        ):
            return None, (
                "--head-marker-size must be the black-square edge in METRES "
                f"(0.001-1.0), got {head_marker_size!r}"
            )
    out = episode_dir / MANIFEST_NAME
    if out.exists() and not force:
        return None, (
            f"{out} already exists. It carries hand-written context (notes, the "
            f"scrubbed task_window, the sync clock), so this refuses rather than "
            f"overwriting - pass --force if you meant to discard it."
        )

    cams = (
        Path(config_dir)
        if config_dir is not None
        else config_root(episode_dir) / "cameras"
    )
    clips, problem = discover_clips(episode_dir, cams)
    if problem:
        return None, problem
    arms, extras, problem = _assign(clips, left, right, extra_units)
    if problem:
        return None, problem

    head_label = None
    if head_marker_id is not None:
        if head_camera is not None:
            if head_camera not in extras:
                found = ", ".join(sorted(extras)) or "none"
                return None, (
                    f"--head-camera {head_camera!r} is not an extra stream in "
                    f"this episode (found: {found})"
                )
            head_label = head_camera
        elif len(extras) == 1:
            head_label = next(iter(extras))
        else:
            found = ", ".join(sorted(extras)) or "none"
            return None, (
                "--head-marker-id belongs to an extra head-camera stream, but "
                f"it cannot be selected from {found}; pass --head-camera "
                "<label> when more than one extra stream exists"
            )

    doc: dict = {
        "schema": SCHEMA,
        "name": episode_dir.name,
        "description": _TODO,
        "date": _TODO,
        "tool": tool or _TODO,
        # No `anchor:` block. The table marker belongs to the on-demand
        # register stage, which no builtin pipeline profile requests. The
        # header below carries the shape for episodes captured with one.
        "arms": {
            side: {
                "camera": clip.unit,
                "recording": clip.rel,
            }
            for side, clip in sorted(arms.items())
        },
    }
    if extras:
        cameras: dict[str, dict[str, object]] = {}
        for label, clip in sorted(extras.items()):
            camera: dict[str, object] = {"recording": clip.rel}
            if clip.unit:
                camera["camera"] = clip.unit
            if label == head_label:
                camera["head_marker"] = {
                    "dictionary": head_marker_dictionary,
                    "id": head_marker_id,
                    "size": float(head_marker_size),
                }
            cameras[label] = camera
        doc["cameras"] = cameras

    # The provenance header records the binding verified against each clip's
    # embedded serial.
    lines = [
        "# Written by `kiwi init`. Each arm is bound to its camera by the clip's",
        "# EMBEDDED SERIAL, never by its filename. Verified assignment:",
    ]
    for side, clip in sorted(arms.items()):
        lines.append(f"#   {side:5s} {clip.unit}  serial {clip.serial}  <- {clip.rel}")
    for label, clip in sorted(extras.items()):
        lines.append(f"#   {label:5s} (extra stream, no arm)      <- {clip.rel}")
    lines += [
        "#",
        "# Fill in description, date, tool, and each arm's `gripper_unit:` (the",
        "# PHYSICAL build, config/gripper/<tool>/units/<unit>.yaml - the eef",
        "# stage runs on its build.marker tag and calibrated fixed_pose;",
        "# transforms are build facts, there is no version-wide pose) before",
        "# running the pipeline; the sync: block is machine-written by",
        "# `kiwi run`.",
        "#",
        "# anchor: {marker: {id: <aruco id>, size: <metres>}}  # optional: only",
        "# for the on-demand register stage / ground truth. Add it when this",
        "# episode was captured with an ArUco table marker; no builtin pipeline",
        "# profile requests it, and only an explicit register run reads it",
        "# (`kiwi run --from-stage register --to-stage register`).",
        "# Optional --head-marker-id describes the marker on the extra head camera.",
        "",
    ]
    body = _with_unit_placeholders(
        yaml.safe_dump(doc, sort_keys=False, width=float("inf"))
    )
    if yaml.safe_load(body) != doc:
        raise RuntimeError("kiwi init: the skeleton does not round-trip through yaml")
    out.write_text("\n".join(lines) + body)
    return out, ""

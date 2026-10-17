"""Build episodes from resolved specs and write one training dataset.

:func:`build_and_write` is the library entry point the ``kiwi export`` verb
calls: it fans a list of :class:`~kiwi.export.source.EpisodeSource` out
through curation, builds each job's :class:`ObservationEpisode`, streams them
into the selected format writer, and stamps ``meta/export_provenance.json``
plus the HF dataset card.

No printing here: the result object carries what the CLI narrates (skipped
sources, a best-effort card failure). The tyro CLI lives in ``kiwi.cli``.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path

from kiwi.core.config_hash import stamp_inputs
from kiwi.core.generation import file_ref, producer_info
from kiwi.core.observation_episode import ObservationEpisode
from kiwi.core.paths import repo_root
from kiwi.core.provenance import provenance_block

from .archive import SourceArchive
from .build import build_episode
from .card import write_card
from .curation import Curation, expand_episodes
from .funnel import build_funnel
from .schema import SCHEMA_VERSION, STATE_KEY
from .source import EpisodeSource
from .writers import DatasetInfo, get_writer

__all__ = ["ExportResult", "build_and_write"]


def _feature_schema(o: ObservationEpisode) -> tuple[tuple[str, tuple], ...]:
    """The episode's declared columns, in a comparable form.

    Shape is what makes a column, so this catches a resolution change between
    two episodes of one camera key as well as a differing state width.
    """
    return tuple(
        (k, (f.dtype, tuple(f.shape), tuple(f.names or ())))
        for k, f in sorted(o.features.items())
    )


def _world_identity(o: ObservationEpisode) -> tuple[str, str, str]:
    """The world an episode's poses live in, as a comparable triple.

    ``(frame, scene, map_set_digest)``. The builder stamps the last two into
    ``meta["world_scene"]``; an episode from the marker source leaves them
    empty and compares on its frame alone (a marker frame is per-episode).
    """
    ws = o.meta.get("world_scene") or {}
    return (
        str(ws.get("frame") or o.frame),
        str(ws.get("scene") or ""),
        str(ws.get("map_set_digest") or ""),
    )


def _world_label(w: tuple[str, str, str]) -> str:
    """A world triple as a phrase a person can act on."""
    frame, scene, digest = w
    where = f"scene {scene!r}" if scene else "no scene named"
    maps = f", map set {digest[:12]}" if digest else ""
    return f"frame {frame!r} ({where}{maps})"


@dataclasses.dataclass(frozen=True)
class ExportResult:
    """What a finished export produced, for the caller to narrate."""

    out: Path
    format: str
    episodes: int
    skipped: list[dict]
    card_error: str | None = None


def build_and_write(
    sources: list[EpisodeSource],
    out: Path,
    *,
    format: str = "v3.0",
    repo_id: str = "local/dataset",
    robot_type: str = "portable_bimanual",
    window: tuple[float, float] | None = None,
    curation: Path | None = None,
    include_camera_pose: bool = False,
    include_head_marker: bool = False,
    include_ground: bool = False,
    include_extra_cameras: bool = True,
    include_relative_state: bool = False,
    include_pose_gap: bool = True,
    action: str = "next_state",
    cameras: dict[str, str] | None = None,
    resize: tuple[int, int] | None = None,
    fps: float | None = None,
    world_source: str = "markerless",
    world_offset: tuple[float, float, float] | None = None,
    world_yaw: float | None = None,
    groot: bool = False,
) -> ExportResult:
    """Build each episode and write them as one dataset under ``out``.

    ``sources`` are RESOLVED specs (the CLI flattens each ``episode.yaml``
    once; see :mod:`kiwi.export.source`). Every arm exports at its tool tip,
    composed from that arm's robot description (see :mod:`kiwi.export.build`).

    ``curation`` re-exports through a ``curation_v1`` file: each source
    episode becomes one output episode per kept window; unlisted /
    fully-curated-out sources are skipped and recorded in the provenance and
    the result. Mutually exclusive with ``window``.

    ``world_source`` names which world pose every episode exports from:
    ``markerless`` (the default - the state stage's scene frame) or
    ``marker`` (the register stage's ArUco anchor, the ground-truth path).
    It is threaded to :func:`kiwi.export.build.build_episode` unchanged and
    never inferred from what is on disk.

    ``groot`` additionally writes GR00T's ``meta/modality.json`` beside the
    dataset - Isaac-GR00T consumes the v2.1 layout, so it demands
    ``format="v2.1"``.
    """
    out = Path(out)
    if include_head_marker and format not in ("v2.1", "v3.0"):
        raise ValueError("head markers are supported only in LeRobot v2.1/v3 exports")
    if include_ground and format not in ("v2.1", "v3.0"):
        raise ValueError(
            "the ground plane is supported only in LeRobot v2.1/v3 exports"
        )
    if not sources:
        raise ValueError("no episodes to write")
    # An arm with no resolvable intrinsics artifact refuses HERE, before any
    # expand/build/encode work; every export format passes through this check.
    missing = sorted(
        f"{src.episode_dir.name}:{arm.label or 'main'}"
        for src in sources
        for arm in src.arms
        if arm.intrinsics is None
    )
    if missing:
        raise ValueError(
            f"no camera-intrinsics artifact resolved for arm(s) "
            f"{', '.join(missing)}. A published dataset carries the camera "
            f"model that made it: point episode.yaml's arm at a "
            f"registry unit whose camera.yaml has an `intrinsics:` pointer "
            f"to the published npz (kiwi.calib.intrinsics writes it). A "
            f"missing KB4 fit is fine - the DS model alone exports - but no "
            f"model at all does not."
        )
    if groot and format != "v2.1":
        raise ValueError(
            f"--groot writes GR00T's modality.json beside a v2.1 dataset; "
            f"format is {format!r} (pass --format v2.1)"
        )
    # Resolve the writer first so an unknown format fails fast, before doing
    # the heavy per-episode build work.
    writer = get_writer(format)

    cur = Curation.load(curation) if curation is not None else None
    jobs, skipped = expand_episodes(sources, cur, window)
    if not jobs:
        raise ValueError("curation left no episodes to write")

    obs = [
        build_episode(
            job.episode,
            fps=fps,
            window=job.window,
            include_camera_pose=include_camera_pose,
            include_head_marker=include_head_marker,
            include_ground=include_ground,
            include_extra_cameras=include_extra_cameras,
            include_relative_state=include_relative_state,
            include_pose_gap=include_pose_gap,
            cameras=cameras,
            resize=resize,
            robot_type=robot_type,
            world_source=world_source,
            world_offset=world_offset,
            world_yaw=world_yaw,
        )
        for job in jobs
    ]

    # One LeRobot dataset has ONE schema for all its episodes (meta/info.json
    # is taken from obs[0], and total_videos = n_episodes * n_video_keys), but
    # each episode writes only its OWN video features, so all episodes must
    # share one camera feature set.
    keysets = {tuple(o.video_features()) for o in obs}
    if len(keysets) > 1:
        raise ValueError(
            "episodes have differing camera feature sets; a dataset has one "
            "schema for all episodes. Export them as separate datasets. Sets "
            "seen: " + "; ".join(str(sorted(k)) for k in sorted(keysets))
        )

    # The same one-schema rule covers the rest of what ``info`` and the
    # provenance record take from ``obs[0]``: the frame rate, the world, and
    # the column shapes.
    rates = sorted({o.fps for o in obs})
    if len(rates) > 1:
        raise ValueError(
            f"episodes disagree on the frame rate ({rates}); one dataset has "
            f"one fps, taken from the first episode, so every later episode's "
            f"timestamps and video would be described at the wrong rate. Pass "
            f"--fps to put them on one grid, or export separate datasets."
        )

    # One dataset, one scene. Every scene declares the same constant frame
    # name, so the identity compared here is the full triple: the frame, the
    # scene name, and the digest of the map set that froze it.
    worlds = sorted({_world_identity(o) for o in obs})
    if len(worlds) > 1:
        raise ValueError(
            f"episodes disagree on the world they live in - poses from two "
            f"coordinate systems cannot share one dataset. Seen: "
            f"{'; '.join(_world_label(w) for w in worlds)}. Export episodes "
            f"that share one world (a marker frame is per-episode; a scene "
            f"frame is shared only within its own scene, and only across one "
            f"published map set)."
        )

    placement_digests = {
        o.meta.get("world_scene", {}).get("scene_frame_digest")
        for o in obs
        if o.meta.get("world_scene", {}).get("scene_frame_digest")
    }
    if len(placement_digests) > 1:
        raise ValueError(
            "episodes disagree on the scene_frame_digest; export one placement version"
        )

    world_sources = {o.meta.get("world_source") for o in obs}
    if len(world_sources) > 1:
        raise ValueError(
            f"episodes disagree on the world source "
            f"({sorted(map(str, world_sources))}) "
            f"- a marker world pose and a scene world pose are different "
            f"frames and cannot share one dataset."
        )

    schemas = {_feature_schema(o) for o in obs}
    if len(schemas) > 1:
        a, b = (dict(s) for s in sorted(schemas)[:2])
        # Either a key one side lacks, or a key both have with a different
        # dtype/shape (a resolution change reads as the latter).
        differing = sorted(set(a) ^ set(b)) or sorted(
            k for k in a if a.get(k) != b.get(k)
        )
        raise ValueError(
            f"episodes disagree on the feature schema; a dataset declares one "
            f"set of columns for all its episodes. Keys that differ: "
            f"{differing}. Export them as separate datasets."
        )

    info = DatasetInfo(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=obs[0].fps,
        features=obs[0].features,
        action_policy=action,
    )
    writer.open(out, info)
    task_to_index: dict[str, int] = {}
    for i, oe in enumerate(obs):
        ti = task_to_index.setdefault(oe.task, len(task_to_index))
        writer.add_episode(oe, i, ti)
    writer.close()
    if groot:
        from .writers.groot_modality import write_modality

        state = obs[0].features[STATE_KEY]
        write_modality(out, state.names or [], list(obs[0].video_features()))
    _write_provenance(
        out,
        jobs,
        obs,
        skipped,
        cur,
        sources=sources,
        resize=resize,
        repo_id=repo_id,
        recipe={
            "format": format,
            "fps": info.fps,
            "action": action,
            "codec": info.codec,
            "include_camera_pose": include_camera_pose,
            "include_head_marker": include_head_marker,
            "include_ground": include_ground,
            "include_extra_cameras": include_extra_cameras,
            "include_relative_state": include_relative_state,
            "include_pose_gap": include_pose_gap,
            "cameras": cameras,
            "resize": resize,
            "world_source": world_source,
            "world_offset": world_offset,
            "world_yaw": world_yaw,
            "groot": groot,
        },
    )
    # The HF-rendered face of the provenance above. Best-effort: a failure
    # here does not fail the export (regenerate it with `kiwi export card`).
    card_error: str | None = None
    try:
        write_card(out, repo_id=repo_id)
    except Exception as exc:  # noqa: BLE001 - the card is best-effort
        card_error = str(exc)
    return ExportResult(
        out=out,
        format=format,
        episodes=len(obs),
        skipped=skipped,
        card_error=card_error,
    )


def _write_provenance(
    out: Path,
    jobs: list,
    obs: list,
    skipped: list[dict],
    curation: Curation | None,
    *,
    sources: list | None = None,
    resize: tuple[int, int] | None,
    repo_id: str | None = None,
    recipe: dict | None = None,
) -> None:
    """Emit ``meta/export_provenance.json``: output episode -> source + window.

    This is the back-reference downstream curation is keyed on (it maps a
    dataset episode to its source dir and global-clock window). Each episode
    also records the full per-arm extrinsic chain (the two mount-frame legs
    ``T_mount_cam`` / ``T_mount_tcp`` with the description they came from and
    whether a gripper unit overrode either, plus the composed
    ``T_camera_tcp`` and the ``T_camera_ee`` actually applied to the
    trajectory), the camera INTRINSIC model per pixel stream (DS values +
    optional KB4 body + the delivered resolution), plus the tool-state
    normalization (estimator, unit, endpoints), each with a path/target/sha256
    stamp per source config.

    ``source_stamps`` pins the rest of the prep-time chain: the source dir's
    ``episode.yaml`` (the manifest the CLI read its sync/window from) and its
    ``run_status.json``, whose ``inputs`` block carries the content hashes of
    every calibration file the pipeline resolved (intrinsics, mapping
    profiles, av-sync) that the export itself never re-reads. Both files are
    mutable (re-syncing rewrites the manifest, any re-run rewrites the
    status); the dataset records the exact versions it was built from. A
    missing ``run_status.json`` is recorded as such, not fatal.
    """
    root = repo_root()
    archive = SourceArchive(out)
    episodes = []
    for i, (job, oe) in enumerate(zip(jobs, obs)):
        entry = {
            "episode_index": i,
            "source": job.source,
            "frames": int(len(oe.timestamps)),
            "window": [float(v) for v in oe.meta["window"]],
            "windowed": bool(oe.meta.get("windowed", False)),
            "source_stamps": stamp_inputs(
                {
                    "episode_yaml": job.episode.episode_dir / "episode.yaml",
                    "run_status": job.episode.episode_dir / "run_status.json",
                },
                root=root,
            ),
        }
        entry["world_scene"] = oe.meta.get("world_scene")
        entry["inputs"] = archive.episode(job.episode, oe, i)
        samples = archive.samples(oe, i)
        if samples is not None:
            entry["export_samples"] = samples
        if oe.meta.get("head_marker"):
            entry["head_marker"] = oe.meta["head_marker"]
        if oe.meta.get("ground"):
            entry["ground"] = oe.meta["ground"]
        if job.span_index is not None:
            entry["span_index"] = job.span_index
        if oe.meta.get("extrinsics"):
            entry["extrinsics"] = oe.meta["extrinsics"]
        if oe.meta.get("intrinsics"):
            entry["intrinsics"] = oe.meta["intrinsics"]
        if oe.meta.get("tool_state"):
            entry["tool_state"] = oe.meta["tool_state"]
        episodes.append(entry)
    doc = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "producer": producer_info(
            "kiwi.export", dependencies=("numpy", "scipy", "pyarrow")
        ),
        "source_records": archive.records,
        "quality": archive.quality,
        "recipe": recipe,
        "outputs": [
            file_ref(path, base=out)
            for folder, pattern in (
                ("data", "*.parquet"),
                ("videos", "*.mp4"),
                ("meta", "info.json"),
                ("meta", "stats.json"),
            )
            for path in sorted((out / folder).rglob(pattern))
        ],
        "sampling": {
            "poses": "linear position and quaternion Slerp on episode global time",
            "head": "within continuous segment only; no extrapolation",
            "ground": "one plane per episode; camera height is the sampled camera pose's signed distance from it",
            "video": "nearest native frame; see per-episode export_samples",
            "eef": "nearest source frame of value_completed, no interpolation",
        },
        # Which revision of the public field names (kiwi/export/schema.py)
        # this dataset was written against - bumped whenever a name changes.
        "schema": SCHEMA_VERSION,
        # LeRobot's own meta/info.json does NOT carry the repo id, so this
        # sidecar is the only record of what the dataset was built to be
        # pushed as.
        "repo_id": repo_id,
        # Which code + toolchain produced this dataset.
        **provenance_block(),
        # Stable key, always "tcp": the tool tip is the only body frame this
        # exporter produces.
        "ee_frame": "tcp",
        "frame": obs[0].frame,
        # marker | markerless - which world pose the episodes were exported
        # from. One dataset carries one; they are different world frames.
        "world_source": obs[0].meta.get("world_source"),
        # The world IDENTITY every episode in this dataset shares (the gate
        # above proved they do): frame name, scene name, and the digest of the
        # map set that froze the scene frame. `frame` above repeats the first
        # of these; the other two are what makes two datasets comparable, or
        # provably not.
        "world_scene": obs[0].meta.get("world_scene"),
        # World-frame translation baked into every position (None = the
        # source world frame unchanged). The source world origin sits at
        # +world_offset in the exported coordinates.
        "world_offset": obs[0].meta.get("world_offset"),
        # Degrees about +Z applied BEFORE world_offset, so the offset above is
        # in the rotated frame (None = no rotation baked).
        "world_yaw": obs[0].meta.get("world_yaw"),
        "resize": list(resize) if resize is not None else None,
        "skipped": skipped,
        # The attrition roll-up over every CONSIDERED source (kiwi.export
        # funnel): run_status verdicts per source,
        # counts on top. `skipped` above stays as the flat export-time list;
        # the funnel is the superset a dataset card states.
        "funnel": build_funnel(sources if sources is not None else [], jobs, skipped),
        "episodes": episodes,
    }
    if curation is not None:
        doc["curation"] = curation.provenance()
    p = out / "meta" / "export_provenance.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1) + "\n")

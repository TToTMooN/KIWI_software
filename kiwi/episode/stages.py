"""The stage pipeline behind ``kiwi run``: names, registry and dispatch.

Order: sync -> convert -> state -> head -> register -> eef -> ground.
The default global-state profile requests sync, convert, state and eef.
State combines ORB-SLAM3 scene localization with raw IMU, publishing the front
camera pose and the IMU-body pose. Register is an on-demand
marker evaluation of that trajectory. Head and ground optionally consume the
fused arm states plus rear/front video respectively.

Sync precedes demux because synchronization reads raw audio; demux strips it.
Profiles state every disposition. Narrowing a run to a single stage
requests it directly. Both status files name stages from ``STAGES`` only.
"""

from collections.abc import Callable
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

STAGES: tuple[str, ...] = (
    "sync",
    "convert",
    "state",
    "head",
    "register",
    "eef",
    "ground",
)


@dataclass
class StageResult:
    """What a stage hands back to the engine.

    ``data`` is an optional machine-readable supplement to ``summary``
    (persisted into the stage's ``run_status.json`` record): downstream
    aggregators - the export funnel - read it instead of parsing the human
    text.
    """

    stage: str
    ok: bool
    summary: str = ""
    artifacts: list[Path] = field(default_factory=list)
    data: dict = field(default_factory=dict)


# An entry may additionally accept keyword-only run options (`resync` for
# sync); run_stage forwards them explicitly, so a registry stand-in must
# tolerate unknown keywords or match the real entry.
StageFn = Callable[[Path], StageResult]

# Entries are wired on the episode side, never from inside the component.
# The DAG lets a composer import a component and forbids the reverse
# (tests/test_boundaries.py), so `from kiwi.episode.stages import REGISTRY`
# inside kiwi/<component>/ is a boundary violation, not the way in. Each entry
# imports its component lazily in the function body.
REGISTRY: dict[str, StageFn] = {}


def _recorded_inputs(sidecar: Path) -> dict | None:
    """What a config-hash sidecar records, or None when it cannot be read.

    A sidecar that cannot be read is a cache MISS, never a stage failure; the
    register stage treats a corrupt scan cache the same way. A payload that is
    not an object counts as unreadable too.
    """
    try:
        doc = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _recorded_hash(sidecar: Path) -> str | None:
    """The ``sha256`` a config-hash sidecar records, or None when unreadable."""
    doc = _recorded_inputs(sidecar)
    return doc.get("sha256") if doc else None


def _write_inputs(sidecar: Path, payload: dict) -> None:
    """Write a config-hash sidecar atomically (tmp file, then replace).

    Same rule as the run record: an interrupt leaves the previous sidecar or
    none, never half of one. A failed write takes its temp file with it, as
    ``run_record._write`` does.
    """
    from kiwi.core.generation import json_value

    sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n"
        )
        os.replace(tmp, sidecar)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _sync_entry(episode_dir: Path, *, resync: bool = False) -> StageResult:
    """Put every stream of ``episode_dir`` on one global clock, from audio.

    The manifest is the authority on which clip belongs to which arm, so this
    refuses an un-initialized directory rather than labelling streams by
    filename tag.
    The clock lands in ``episode.yaml``'s ``sync:`` block, written through
    :func:`kiwi.episode.manifest.write_sync_block` so the operator's comments
    survive.

    Each stream's registry unit identifies the measured ``av_sync`` table
    needed to convert its audio placement into a video placement, so an extra
    ``cameras:`` stream the manifest names no unit for is refused before any
    audio is decoded.

    A complete saved clock is reused until the operator requests ``resync``.
    Input hashes describe an actual estimation; they never decide whether to
    repeat it. Missing or stale provenance is left untouched on reuse.
    First use computes a clock; an incomplete clock needs an explicit
    re-sync request. A requested re-sync saves validated offsets
    without migrating task or curation windows, which remain human-managed.
    """
    from kiwi.core import layout
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig, write_sync_block
    from kiwi.sync import Clip, SyncGateError, SyncParams, resolve_clock

    manifest = episode_dir / MANIFEST_NAME
    if not manifest.is_file():
        return StageResult(
            "sync",
            ok=False,
            summary=(
                f"no {MANIFEST_NAME} in {episode_dir}. The manifest is what names "
                f"which clip belongs to which arm; the file names are not. "
                f"Run `kiwi init --episode {episode_dir} --left <unit> "
                f"--right <unit>` first."
            ),
        )
    try:
        cfg = EpisodeConfig.load(manifest, check_exists=False)
        clock = cfg.global_clock()
        if clock is not None:
            clock.validate()
    except (ValueError, FileNotFoundError) as e:
        return StageResult("sync", ok=False, summary=str(e))

    if cfg.sync_offsets is not None and not resync:
        if not cfg.clock_covers_streams():
            return StageResult(
                "sync",
                ok=False,
                summary=(
                    "the saved clock does not cover every declared stream; "
                    "request --resync to estimate a replacement clock"
                ),
            )
        return StageResult(
            "sync",
            ok=True,
            summary=(
                f"reused the clock in {MANIFEST_NAME} (origin={cfg.sync_origin}; "
                "input freshness is human-managed; request --resync to recompute)"
            ),
            artifacts=[manifest],
            data={"reused": True},
        )

    unitless = sorted(k for k, c in cfg.cameras.items() if c.camera_dir is None)
    if unitless:
        arm_flags = " ".join(
            f"--{side} {cfg.arms[side].camera if side in cfg.arms else '<unit>'}"
            for side in ("left", "right")
        )
        # One --camera takes every label: a repeated flag keeps only the last.
        camera_flags = "--camera " + " ".join(f"{k}=<unit>" for k in unitless)
        return StageResult(
            "sync",
            ok=False,
            summary=(
                f"{', '.join(unitless)}: no camera unit in {MANIFEST_NAME}, so "
                f"{'its' if len(unitless) == 1 else 'their'} camera model and "
                f"audio-vs-video delay cannot be established. Set "
                f"{', '.join(f'cameras.{k}.camera' for k in unitless)} in "
                f"{MANIFEST_NAME} to a camera registry unit, or re-run "
                f"`kiwi init --episode {episode_dir} {arm_flags} {camera_flags} "
                f"--force`, which rewrites {MANIFEST_NAME} and discards its hand "
                f"edits."
            ),
        )

    # Only an actual estimation reads input identities. These are provenance,
    # not an automatic reuse/invalidation policy.
    try:
        params = SyncParams()
        input_record, input_key = _sync_inputs_key(cfg, params)
    except (OSError, ValueError) as e:
        return StageResult(
            "sync",
            ok=False,
            summary=(
                f"cannot identify the recordings and A/V delay tables a clock would be "
                f"derived through ({e}), so a derived clock could not be attributed."
            ),
        )
    payload = {"schema": "sync_inputs_v1", **input_record, "sha256": input_key}
    sidecar = episode_dir / layout.SYNC_INPUTS

    arms = {k: Clip(a.recording, a.camera_dir, k) for k, a in cfg.arms.items()}
    extras = {k: Clip(c.recording, c.camera_dir, k) for k, c in cfg.cameras.items()}
    try:
        est = resolve_clock(arms, extras, params=params)
    except (SyncGateError, RuntimeError, ValueError, OSError, yaml.YAMLError) as e:
        from kiwi.episode.generation import sync_generation

        summary = str(e)
        if isinstance(e, SyncGateError) and e.report:
            summary += "\n" + e.report
        try:
            # A failed attempt has not produced the clock still in the
            # manifest. Its input identity must never attest to that clock.
            previous = _recorded_inputs(sidecar) or {}
            if previous.get("status") == "failed":
                previous = previous.get("previous_generation", {})
            _write_inputs(
                sidecar,
                {
                    **payload,
                    **sync_generation(cfg, error=e),
                    "attempted_sha256": input_key,
                    "sha256": None,
                    "previous_generation": previous,
                },
            )
        except (OSError, ValueError) as record_error:
            summary += f"; could not save sync diagnostics: {record_error}"
        return StageResult("sync", ok=False, summary=summary)

    try:
        write_sync_block(
            manifest,
            est.clock.offsets_s,
            method=est.method,
            origin=est.clock.origin,
        )
    except (OSError, ValueError) as e:
        return StageResult("sync", ok=False, summary=str(e))
    # Written only now, beside the clock it describes. A clock nothing can
    # attribute is a refusal, not a traceback. The operator can explicitly
    # retry if the record could not be saved; reuse never invents missing
    # provenance.
    try:
        from kiwi.episode.generation import sync_generation

        _write_inputs(sidecar, {**payload, **sync_generation(cfg, estimate=est)})
    except OSError as e:
        return StageResult(
            "sync",
            ok=False,
            summary=(
                f"derived the clock but could not write {layout.SYNC_INPUTS} beside "
                f"it ({e}), so nothing records which A/V tables it came from. "
                f"Fix the episode directory's permissions and retry with --resync."
            ),
            artifacts=[manifest],
        )
    # The machine-readable twin of the summary (run_status keeps it): the A/V
    # corrections are the part a consumer cannot re-derive from the clock -
    # whether a table was read, and which rule.
    data = {
        "method": est.method,
        "origin": est.clock.origin,
        "offsets_s": {k: float(v) for k, v in est.clock.offsets_s.items()},
        "loop_residuals_s": {k: float(v) for k, v in est.loop_residuals.items()},
        "edges": [
            {
                "kind": e.kind,
                "ref": e.ref,
                "other": e.other,
                "offset_s": e.offset_s,
                "av_correction_s": e.av_correction_s,
                "av_provenance": e.av_provenance,
                "modes": list(e.modes),
                "raw": e.raw,
            }
            for e in est.edges
        ],
    }
    return StageResult(
        "sync",
        ok=True,
        summary="\n".join([est.summary(), *est.match_summaries()])
        + (
            "\nExplicit re-sync saved; manual task and curation windows are unchanged."
            if resync
            else ""
        ),
        artifacts=[manifest],
        data=data,
    )


def _sync_inputs_key(cfg, params) -> tuple[dict, str]:
    """Observational identity of the inputs used for an actual estimation.

    Unit and table bindings are per label. Absolute paths are observational
    only, so moving an otherwise identical dataset/configuration is harmless.
    Recording hashes use the same stat-aware cache as generation provenance;
    replacing bytes with preserved size and mtime still changes ctime/inode.
    """
    from dataclasses import asdict

    from kiwi.core.camera.registry import CameraUnit
    from kiwi.core.config_hash import stamp_input
    from kiwi.core.generation import file_ref
    from kiwi.sync import LOOP_TOL_S
    from kiwi.sync.audio import EXTRACTION, SYNC_RECIPE

    stamps = {}
    streams = {}
    for label, spec in sorted({**cfg.arms, **cfg.cameras}.items()):
        model = ""
        table_digest = None
        if spec.camera_dir is not None:
            try:
                unit = CameraUnit.load(spec.camera_dir)
                model = str(unit.meta.get("model") or "")
                table = unit.artifact("av_sync")
            except (OSError, ValueError) as e:
                raise ValueError(f"{spec.camera_dir}: {e}") from e
            if table is not None:
                stamps[table] = stamp_input(table, root=cfg.root)
                table_digest = stamps[table].get("sha256")
                if table_digest is None:
                    raise ValueError(
                        f"{table}: {stamps[table].get('error', 'missing')}"
                    )
        recording_digest = file_ref(spec.recording, base=cfg.root)["digest"]
        if recording_digest is None:
            raise ValueError(f"{spec.recording}: recording cannot be identified")
        streams[label] = {
            "recording": recording_digest,
            "camera": spec.camera,
            "model": model,
            "av_sync": table_digest,
        }
    identity = {
        "schema": "sync_identity_v1",
        "recipe": SYNC_RECIPE,
        "streams": streams,
        "params": asdict(params),
        "extraction": EXTRACTION,
        "loop_tolerance_s": LOOP_TOL_S,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return {
        "av_sync": [stamps[p] for p in sorted(stamps)],
        "models": {label: value["model"] for label, value in streams.items()},
        "input_identity": identity,
    }, digest


REGISTRY["sync"] = _sync_entry


def _convert_entry(episode_dir: Path) -> StageResult:
    """Demux every declared ``.insv`` stream into its per-lens mp4s.

    The outputs land in the stream's own directory as ``lens0.mp4`` /
    ``lens1.mp4`` (:mod:`kiwi.core.layout`) - never beside the recording,
    because ``raw/`` holds recordings only. Which directory that is comes
    from the manifest's stream labels, so this entry reads ``episode.yaml``
    like every other stage; a plain-container stream (the GO3 mp4) needs no
    demux and is skipped.

    Fresh outputs (both lens files at least as new as their clip) are
    reused; anything else is (re-)demuxed, so a re-copied ``.insv`` that
    lands a NEWER mtime can never leave stale video behind. The test is
    mtime-only, and ``kiwi import`` copies with ``copystat``, so a re-import
    that preserves the card's timestamps is not detected - the same limit
    :func:`kiwi.core.camera.convert.resolve_lens0_mp4` states. Failures come back as ``ok=False``
    results, never exceptions.
    """
    from kiwi.core import layout
    from kiwi.core.camera import convert as convert_mod
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig

    manifest = episode_dir / MANIFEST_NAME
    if not manifest.is_file():
        return StageResult(
            "convert",
            ok=False,
            summary=f"no {MANIFEST_NAME} in {episode_dir}; run `kiwi init` first.",
        )
    try:
        # check_exists=False: this stage reads labels and recordings only, so
        # a camera dir yet to be registered must not block a demux.
        cfg = EpisodeConfig.load(manifest, check_exists=False)
    except (ValueError, FileNotFoundError) as e:
        return StageResult("convert", ok=False, summary=str(e))

    streams = [(label, arm.recording) for label, arm in sorted(cfg.arms.items())]
    streams += [(label, cam.recording) for label, cam in sorted(cfg.cameras.items())]
    artifacts: list[Path] = []
    demuxed = reused = skipped = 0
    for label, clip in streams:
        if clip.suffix.lower() != ".insv":
            skipped += 1
            continue
        sdir = layout.stream_dir(episode_dir, label)
        lens0 = sdir / layout.LENS0
        lens1 = sdir / layout.LENS1
        outs = (lens0, lens1)
        if not clip.is_file():
            # This stage loads with check_exists=False, so the manifest may
            # name a recording that is not on disk.
            return StageResult(
                "convert",
                ok=False,
                summary=(
                    f"{label}: no recording at {clip} - {MANIFEST_NAME} names "
                    f"it but the file is not there. Re-import the clip, or fix "
                    f"its `recording:` path."
                ),
                artifacts=artifacts,
            )
        if all(o.exists() and o.stat().st_mtime >= clip.stat().st_mtime for o in outs):
            reused += 1
            artifacts += outs
            continue
        try:
            convert_mod.convert(clip, lens0, lens1, force=True)
            from kiwi.episode.generation import conversion_generation

            _write_inputs(
                sdir / "convert.inputs.json",
                conversion_generation(clip, outs, base=sdir, label=label),
            )
        except (OSError, RuntimeError) as e:
            return StageResult(
                "convert",
                ok=False,
                summary=f"{label}: {clip.name}: {e}",
                artifacts=artifacts,
            )
        demuxed += 1
        artifacts += outs
    summary = (
        f"{demuxed} stream(s) demuxed, {reused} reused "
        f"({len(artifacts)} per-lens streams)"
    )
    if skipped:
        summary += f"; {skipped} plain-container stream(s) need no demux"
    return StageResult("convert", ok=True, summary=summary, artifacts=artifacts)


REGISTRY["convert"] = _convert_entry


def _register_entry(episode_dir: Path) -> StageResult:
    """Anchor every arm's state trajectory to the episode's table marker.

    The on-demand ground-truth stage: no builtin pipeline profile requests
    it, and nothing in the core chain reads what it writes. Run it by itself
    on an episode that does carry a marker::

        kiwi run --episode <dir> --from-stage register --to-stage register

    Three artifacts per arm, all in its stream directory
    (:mod:`kiwi.core.layout`): ``detections.json``
    (``marker_detections_v1`` - every marker sighting in the clip's early
    seconds, distance included), ``register.tum`` - the same trajectory as
    ``scene/state.tum`` re-expressed in the marker's own 6-DoF frame
    (``full_6dof``: marker centre is the world origin, z along its normal),
    camera body, episode global clock, all declared in the header - and
    ``register.inputs.json``, the method-discriminated sidecar binding the
    output and its inputs by digest. All configured arms anchor to one
    physical marker, which is what co-registers them; when every arm commits, the
    episode-root ``registration_bundle.json`` aggregate is written last,
    binding every configured arm's marker document by digest so readers
    can verify which publications belong together.

    Only the scan is reused: a detections file at least as new as the lens
    mp4 AND the published intrinsics, produced for the same marker (id, size,
    dictionary) and rectifier parameters over a window at least as long as
    the one now asked for (the file records them all), skips the video pass.
    The solve always re-runs and the ``.tum`` is rewritten either way, so a
    re-synced clock or a retuned ``anchor.max_marker_dist`` reaches it
    without re-scanning.

    A ``halt`` from the flip checks (:mod:`kiwi.register.health`) fails the
    arm and writes no ``.tum``. Warnings ride along in the summary.

    Extra ``cameras:`` streams are not posed here - the head stage poses the
    hat tag, not the table marker.
    """
    import json

    from kiwi.core import layout
    from kiwi.core.atomic import atomic_write_text
    from kiwi.core.camera.registry import CameraUnit
    from kiwi.core.config_hash import hash_path
    from kiwi.core.trajectory import write_tum
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig
    from kiwi.register.detections import (
        load_detections,
        require_compatible,
        scan_video_for_marker,
        write_detections,
    )

    manifest = episode_dir / MANIFEST_NAME
    if not manifest.is_file():
        return StageResult(
            "register",
            ok=False,
            summary=f"no {MANIFEST_NAME} in {episode_dir}; run `kiwi init` first.",
        )
    try:
        cfg = EpisodeConfig.load(manifest)
        clock = cfg.global_clock()
    except (ValueError, FileNotFoundError) as e:
        return StageResult("register", ok=False, summary=str(e))
    if clock is None or not cfg.clock_covers_streams():
        return StageResult(
            "register",
            ok=False,
            summary=(
                "no global clock covering every declared stream - marker "
                "detections are placed on that clock to meet the state "
                "trajectories, so without one the anchor would sample the "
                "wrong poses. Run the sync stage."
            ),
        )

    anchor = cfg.anchor or {}
    marker = anchor.get("marker") or {}
    marker_id, marker_size = marker.get("id"), marker.get("size")
    if (
        isinstance(marker_id, bool)
        or not isinstance(marker_id, int)
        or isinstance(marker_size, bool)
        or not isinstance(marker_size, (int, float))
    ):
        return StageResult(
            "register",
            ok=False,
            summary=(
                f"no usable `anchor:` block in {MANIFEST_NAME} - the register "
                f"stage anchors every arm to ONE physical table marker, whose "
                f"own pose becomes the shared world frame. It runs on demand "
                f"only, for an episode actually captured with such a marker "
                f"as ground truth; no builtin pipeline profile requests it, "
                f"and `kiwi init` writes no anchor block. If this episode has "
                f"a marker, add:\n"
                f"  anchor:\n"
                f"    marker:\n"
                f"      id: <aruco id>\n"
                f"      size: <black-square edge, METRES>\n"
                f"(`dictionary:` defaults to DICT_4X4_50; optional knobs beside "
                f"`marker:`: `scan_seconds` (120), `max_marker_dist` (1.0 m), "
                f"`fov_deg` (140), `out_size` (1920).)"
            ),
        )
    marker_dict = str(marker.get("dictionary", "DICT_4X4_50"))
    scan_seconds = float(anchor.get("scan_seconds", 120.0))
    max_dist = float(anchor.get("max_marker_dist", 1.0))
    fov_deg = float(anchor.get("fov_deg", 140.0))
    out_size = int(anchor.get("out_size", 1920))

    from kiwi.register.anchor import solve_anchor

    lines: list[str] = []
    artifacts: list[Path] = []
    arm_markers: dict[str, Path] = {}
    ran = reused = 0
    for label, arm in sorted(cfg.arms.items()):
        sdir = layout.stream_dir(episode_dir, label)
        lens0 = sdir / layout.LENS0
        state_tum = layout.scene_dir(episode_dir, label) / layout.STATE_TUM
        det_path = sdir / layout.DETECTIONS
        tum = sdir / layout.REGISTER_TUM
        try:
            if not lens0.is_file():
                raise FileNotFoundError(
                    f"no {label}/{layout.LENS0} - run the convert stage first."
                )
            if not state_tum.is_file():
                raise FileNotFoundError(
                    f"no {label}/scene/{layout.STATE_TUM} - run the state stage first."
                )
            state_inputs = layout.scene_dir(episode_dir, label) / layout.STATE_INPUTS
            if not state_inputs.is_file():
                raise FileNotFoundError(
                    f"no {label}/scene/{layout.STATE_INPUTS} - the register sidecar "
                    f"binds the upstream state publication by digest; run the "
                    f"state stage first."
                )
            # The state sidecar records the exact published trajectory in
            # outputs.trajectory. Verify it before scanning or solving, and
            # require the global clock.
            from kiwi.core.digest import verify_digest
            from kiwi.core.trajectory import read_tum_header

            state_doc = json.loads(state_inputs.read_text())
            if not isinstance(state_doc, dict) or (
                state_doc.get("schema") != "state_trajectory_inputs"
                or state_doc.get("status") != "computed"
            ):
                raise ValueError(
                    f"{state_inputs}: not a computed state publication; "
                    "run the state stage first"
                )
            outputs = state_doc.get("outputs")
            trajectory_ref = (
                outputs.get("trajectory") if isinstance(outputs, dict) else None
            )
            if not isinstance(trajectory_ref, dict) or not isinstance(
                trajectory_ref.get("digest"), str
            ):
                raise ValueError(
                    f"{state_inputs}: missing trajectory digest; "
                    "regenerate the state publication"
                )
            if not isinstance(trajectory_ref.get("path"), str):
                raise ValueError(f"{state_inputs}: missing trajectory path")
            if (
                state_inputs.parent / trajectory_ref.get("path", "")
            ).resolve() != state_tum.resolve():
                raise ValueError(
                    f"{state_inputs}: trajectory path does not name {state_tum}"
                )
            verify_digest(trajectory_ref["digest"], state_tum)
            header = read_tum_header(state_tum)
            if (
                header is None
                or header.body != "camera"
                or not header.clock.startswith("episode global clock")
            ):
                raise ValueError(
                    f"{state_tum}: expected a camera pose on the episode "
                    "global clock; run sync and state first"
                )
            intrinsics = CameraUnit.load(arm.camera_dir).artifact("intrinsics")
            if intrinsics is None:
                raise FileNotFoundError(
                    f"camera '{arm.camera}' has no intrinsics artifact under "
                    f"{arm.camera_dir} - publish one (`kiwi calib intrinsics`) "
                    f"first."
                )

            obs = None
            if det_path.is_file() and det_path.stat().st_mtime >= max(
                lens0.stat().st_mtime, intrinsics.stat().st_mtime
            ):
                try:
                    meta, cached = load_detections(det_path)
                    require_compatible(
                        meta,
                        marker_id=marker_id,
                        fov_deg=fov_deg,
                        out_size=out_size,
                        marker_size=float(marker_size),
                        # Both are manifest knobs the mtime gate cannot see.
                        marker_dict=marker_dict,
                        scan_seconds=scan_seconds,
                        # The scan below is hardcoded to the front lens, so a
                        # detections.json produced for another lens must not be
                        # reused: T_cam_marker would be in that lens's frame and
                        # the solve composes it with the front camera's state
                        # pose. One recording `back` is already refused a line
                        # above by load_detections; what this catches on its
                        # own is a cache that does not RECORD its lens, which
                        # the loader reads as front and reuse must not.
                        lens="front",
                    )
                    if meta.get("intrinsics") == str(intrinsics):
                        obs = cached
                except ValueError:
                    # Parameters changed, or the cache itself is corrupt
                    # (json.JSONDecodeError is a ValueError): re-scan below.
                    obs = None
            if obs is not None:
                reused += 1
                lines.append(f"{label}: reused {det_path.name} ({len(obs)} detections)")
            else:
                meta, obs = scan_video_for_marker(
                    lens0,
                    intrinsics=intrinsics,
                    marker_id=marker_id,
                    marker_size=float(marker_size),
                    marker_dict=marker_dict,
                    fov_deg=fov_deg,
                    out_size=out_size,
                    scan_seconds=scan_seconds,
                )
                write_detections(det_path, meta=meta, observations=obs)
                ran += 1
                lines.append(
                    f"{label}: scanned {meta['frames_scanned']} frame(s), "
                    f"{len(obs)} detection(s) of marker {marker_id}"
                )
            artifacts.append(det_path)

            from kiwi.episode.generation import registration_generation

            generation_kwargs = dict(
                directory=sdir,
                video=lens0,
                intrinsics=intrinsics,
                state_inputs=state_inputs,
                marker=marker,
                params={
                    "max_marker_dist": max_dist,
                    "scan_seconds": scan_seconds,
                    "fov_deg": fov_deg,
                    "out_size": out_size,
                    "stream_offset_s": clock.offsets_s[label],
                },
            )
            try:
                result = solve_anchor(
                    state_tum,
                    obs,
                    stream_offset_s=clock.offsets_s[label],
                    max_marker_dist=max_dist,
                )
            except (ValueError, RuntimeError) as error:
                _write_inputs(
                    sdir / layout.REGISTER_INPUTS,
                    {
                        "schema": "register_inputs",
                        "method": "marker_anchor",
                        **registration_generation(
                            None, error=error, **generation_kwargs
                        ),
                    },
                )
                raise
            verdict, msg = result.verdict
            generation = registration_generation(result, **generation_kwargs)
            if verdict == "halt":
                # No .tum for this arm.
                lines.append(f"{label}: anchor rejected - {msg}")
                _write_inputs(
                    sdir / layout.REGISTER_INPUTS,
                    {
                        "schema": "register_inputs",
                        "method": "marker_anchor",
                        **generation,
                    },
                )
                return StageResult(
                    "register", ok=False, summary="\n".join(lines), artifacts=artifacts
                )
            write_tum(
                result.trajectory,
                tum,
                clock=f"episode global clock (origin={clock.origin})",
                source=det_path.name,
            )
            artifacts.append(tum)
            # Method-discriminated sidecar: which method produced this
            # register.tum, from which inputs, with recorded digests so
            # readers can verify the exact artifact bytes.
            marker_path = sdir / layout.REGISTER_INPUTS
            atomic_write_text(
                marker_path,
                json.dumps(
                    {
                        "schema": "register_inputs",
                        "method": "marker_anchor",
                        "register_tum_bytes_sha256": hash_path(tum),
                        "detections_bytes_sha256": hash_path(det_path),
                        "state_inputs_bytes_sha256": hash_path(state_inputs),
                        **generation,
                    },
                    indent=2,
                )
                + "\n",
            )
            arm_markers[label] = marker_path
            artifacts.append(marker_path)
            detail = f"{label}: {result.spread[1]}; {result.reach[1]}"
            if result.n_too_far:
                detail += f" ({result.n_too_far} beyond {max_dist} m dropped)"
            if verdict != "ok":
                detail += f" [{verdict.upper()}: {msg}]"
            lines.append(detail)
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as e:
            lines.append(f"{label}: {e}")
            return StageResult(
                "register", ok=False, summary="\n".join(lines), artifacts=artifacts
            )

    # All configured arms committed: write the episode-root aggregate last,
    # binding each arm's marker by digest for publication readers.
    bundle = episode_dir / layout.REGISTRATION_BUNDLE
    atomic_write_text(
        bundle,
        json.dumps(
            {
                "schema": "registration_bundle",
                "method": "marker_anchor",
                "arms": {
                    label: {"register_inputs_bytes_sha256": hash_path(path)}
                    for label, path in sorted(arm_markers.items())
                },
            },
            indent=2,
        )
        + "\n",
    )
    artifacts.append(bundle)
    head = f"{ran + reused} arm(s) anchored ({ran} scanned, {reused} reused scan)"
    if cfg.cameras:
        head += f"; {len(cfg.cameras)} extra stream(s) not posed by the table marker"
    return StageResult(
        "register", ok=True, summary="\n".join([head, *lines]), artifacts=artifacts
    )


REGISTRY["register"] = _register_entry


def _eef_inputs_hash(arm, intrinsics: Path, *, config_dir: Path) -> str:
    """Content key for one arm's eef inputs - everything but the video.

    Everything ``run_clip`` reads that the stream's mtime cannot speak for:
    the layered ``session:`` recipe, the design's estimator, the build's
    ``marker`` and calibrated ``fixed_pose`` (plus the file it came from,
    which the artifact records as its ``pose_source``), and the intrinsics BY
    CONTENT - ``hash_path`` dispatches an npz to ``hash_npz``, which is stable
    across an identical regeneration where the file bytes are not. The mount
    legs stay out: export composes with them and the estimator never reads
    them.

    The completion policy version and the content of every kiwi/eef module
    the eef stage imports also enter the key, so an edit to the estimator,
    the completion policy or the builtin defaults invalidates reuse even when
    no session YAML changed. Shared ``kiwi.core`` helpers are not part of it.

    ``fixed_pose_source`` is a path, so it is relativized against
    ``config_dir`` (the config root the manifest loader resolved) the way
    ``stamp_input`` relativizes against the repo root; that file's contents
    are already in the key as ``fixed_pose``.

    An intrinsics file that cannot be hashed raises rather than degrading to
    "no key".
    """
    from kiwi.core.config_hash import _rel, hash_path
    from kiwi.episode.generation import eef_source_fingerprint

    try:
        intrinsics_hash = hash_path(intrinsics)
    except Exception as e:  # any unreadable npz, not just the OSError family
        raise ValueError(
            f"cannot hash {intrinsics} to key the eef reuse: {type(e).__name__}: {e}"
        ) from e
    payload = json.dumps(
        {
            "tool": arm.tool,
            "estimator": arm.estimator,
            "marker": arm.marker,
            "session": arm.session,
            "fixed_pose": arm.fixed_pose,
            "fixed_pose_source": (
                _rel(Path(arm.fixed_pose_source), config_dir)
                if arm.fixed_pose_source
                else ""
            ),
            "intrinsics": intrinsics_hash,
            "source_fingerprint": eef_source_fingerprint(),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _eef_entry(episode_dir: Path) -> StageResult:
    """Estimate per-arm tool state; write ``<label>/eef.npz`` (+ trace png).

    Arms only - extra ``cameras:`` streams carry no tool state, by contract.
    Everything is pre-layered by the manifest (tool defaults < gripper unit <
    episode-arm overrides), so this entry only assembles the ClipJob: the
    arm's demuxed front-lens mp4 (the convert stage's output), the intrinsics
    from the camera registry, and the estimator bundle named by the tool
    config's ``estimator:`` key. The runner itself is tool-blind.

    Reuse: an ``eef.npz`` at least as new as the stream it was estimated from
    is kept only when it carries completed output and the generating code
    and CONFIG are unchanged too - the
    layered ``session:`` recipe, the build's tag and calibrated
    ``fixed_pose``, and the intrinsics by content, hashed into
    ``eef.inputs.json`` at produce time. An arm with no published intrinsics
    has no key at all, which counts as a miss and lands on the refusal below;
    an intrinsics file that cannot be hashed refuses outright.

    An artifact requires one raw anchor and finite completed output
    throughout. Expected exceptions are translated, never raised through.
    """
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig

    manifest = episode_dir / MANIFEST_NAME
    if not manifest.is_file():
        return StageResult(
            "eef",
            ok=False,
            summary=f"no {MANIFEST_NAME} in {episode_dir}; run `kiwi init` first.",
        )
    try:
        cfg = EpisodeConfig.load(manifest)
    except (ValueError, FileNotFoundError) as e:
        return StageResult("eef", ok=False, summary=str(e))

    from kiwi.core import layout
    from kiwi.core.camera.registry import CameraUnit
    from kiwi.eef.run import ClipJob, run_clip
    from kiwi.eef.state import read_eef_state
    from kiwi.episode.generation import eef_completion_summary, eef_generation

    lines: list[str] = []
    artifacts: list[Path] = []
    ran = reused = 0
    for label, arm in sorted(cfg.arms.items()):
        insv = arm.recording
        sdir = layout.stream_dir(episode_dir, label)
        lens0 = sdir / layout.LENS0
        out = sdir / layout.EEF_STATE
        sidecar = sdir / layout.EEF_INPUTS
        src = lens0 if lens0.is_file() else insv
        try:
            # Resolved BEFORE the freshness test, because the intrinsics and
            # the layered config are inputs to eef.npz exactly as the video
            # is; `None` (nothing published) leaves no key, so the run branch
            # below meets it and refuses with the actionable message.
            intrinsics = CameraUnit.load(arm.camera_dir).artifact("intrinsics")
            cfg_hash = (
                None
                if intrinsics is None
                else _eef_inputs_hash(arm, intrinsics, config_dir=cfg.config_dir)
            )
            st = None
            if (
                cfg_hash is not None
                and out.is_file()
                and out.stat().st_mtime >= src.stat().st_mtime
                and _recorded_hash(sidecar) == cfg_hash
            ):
                st = read_eef_state(out)
            if st is not None:
                n_valid, n = int(st.valid.sum()), len(st)
                reused += 1
                lines.append(f"{label}: reused {out.name} ({n_valid}/{n} valid)")
                artifacts.append(out)
            else:
                if arm.estimator is None:
                    raise ValueError(
                        f"no tool config at {arm.tool_config} - the eef "
                        f"stage needs its design: block (the estimator key, "
                        f"the marker dictionary). Add the tool_v1 file."
                    )
                if intrinsics is None:
                    raise FileNotFoundError(
                        f"camera '{arm.camera}' has no intrinsics artifact "
                        f"under {arm.camera_dir} - publish one "
                        f"(`kiwi calib intrinsics`) first."
                    )
                res = run_clip(
                    ClipJob(
                        recording=insv,
                        lens0_mp4=lens0,
                        out_dir=sdir,
                        tool=arm.tool,
                        estimator_key=arm.estimator,
                        session_block=arm.session,
                        intrinsics=intrinsics,
                        marker_id=arm.marker.get("id"),
                        marker_size_m=arm.marker.get("size"),
                        marker_dictionary=arm.marker.get("dictionary"),
                        fixed_pose=arm.fixed_pose,
                        fixed_pose_source=arm.fixed_pose_source,
                    )
                )
                _write_inputs(
                    sidecar,
                    {
                        "schema": "eef_inputs_v1",
                        "tool": arm.tool,
                        "estimator": arm.estimator,
                        "gripper_unit": arm.gripper_unit,
                        "intrinsics": str(intrinsics),
                        "sha256": cfg_hash,
                        **eef_generation(arm, sdir, src, intrinsics, res),
                    },
                )
                ran += 1
                st = res.state
                n_valid, n = res.n_valid, len(res.state)
                lines.append(f"{label}: {res.summary()}")
                artifacts.append(res.state_path)
                if res.trace_path is not None:
                    artifacts.append(res.trace_path)
            completion = eef_completion_summary(st)
            lines.append(
                f"{label}: completion filled {completion['filled_frames']}/{n} "
                f"frames ({completion['filled_fraction']:.1%}); "
                f"raw anchors {n_valid}/{n}"
            )
            if n_valid < 1:
                lines.append(
                    f"{label}: only {n_valid}/{n} valid frame(s) - tool-state "
                    f"tracking failed for the clip. Check the trace png and "
                    f"the tool config (marker id, fixed_pose, ROI)."
                )
                return StageResult(
                    "eef", ok=False, summary="\n".join(lines), artifacts=artifacts
                )
            if not st.valid_output.all():
                lines.append(f"{label}: completed tool-state output has missing frames")
                return StageResult(
                    "eef", ok=False, summary="\n".join(lines), artifacts=artifacts
                )
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as e:
            lines.append(f"{label}: {e}")
            return StageResult(
                "eef", ok=False, summary="\n".join(lines), artifacts=artifacts
            )

    head = f"{ran + reused} arm(s) estimated ({ran} run, {reused} reused)"
    if cfg.cameras:
        head += f"; {len(cfg.cameras)} extra stream(s) carry no tool state"
    return StageResult(
        "eef", ok=True, summary="\n".join([head, *lines]), artifacts=artifacts
    )


REGISTRY["eef"] = _eef_entry


def _state_entry(episode_dir: Path) -> StageResult:
    """Global visual-inertial batch estimate per arm, in the scene frame.

    Consumes what the scene feature already published: this episode's
    front and shared-rear localization observations (both in the one
    canonical scene frame), the arm's raw IMU, the calibrated cam-IMU
    chains and Allan noise densities. Missing localizations are produced
    on the spot via ``kiwi.episode.scene_build.localize_episode`` - the
    stage routes by (arm, lens), so each arm's own streams feed its own
    estimate.

    The graph itself lives in :mod:`kiwi.state.fusion`: every
    native front-camera frame is an IMU-body state, tracked front/rear
    localizations are robust absolute pose factors, raw-IMU
    preintegration connects consecutive states, and the published output
    is the front optical camera per native frame. Per arm:
    ``<label>/scene/state.tum`` (scene frame; the episode global clock once
    the episode is synced, the recording's own IMU clock without a ``sync:``
    block) plus ``state.inputs.json``, its generation record.
    An existing ``state.tum`` is reused while its completed generation records
    the current clock, arm offset and map set. Changed or unrecorded values are
    re-estimated, including after a sync-only rerun followed by state later.

    Needs the optional ``state`` pixi environment (GTSAM); the episode
    must carry a ``scene:`` back-reference to a scene with a published
    map set.
    """
    import json
    import hashlib

    import numpy as np
    import yaml as _yaml

    from kiwi.core import layout
    from kiwi.core.digest import digest_bytes
    from kiwi.core.generation import file_ref, producer_info, write_record
    from kiwi.core.camera.extrinsics import read_cam_imu
    from kiwi.core.camera.registry import CameraUnit
    from kiwi.core.trajectory import Trajectory, write_tum
    from kiwi.calib.imu_noise import read_imu_noise
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig
    from kiwi.episode.scene import load_scene
    from kiwi.episode.scene_build import (
        current_maps,
        is_localized,
        load_imu,
        localize_episode,
    )
    from kiwi.state import STATE_PROFILES, write_state_latent
    from kiwi.state import fusion as fusion_mod
    from kiwi.state.geometry import quats_continuous, se3_inv
    from kiwi.track.orbslam.observations import read_observations
    from kiwi.track.scene_frame import SCENE_FRAME_NAME
    from scipy.spatial.transform import Rotation

    episode_dir = Path(episode_dir).resolve()
    manifest = episode_dir / MANIFEST_NAME
    if not manifest.is_file():
        return StageResult(
            "state",
            ok=False,
            summary=f"no {MANIFEST_NAME} in {episode_dir}; run `kiwi init` first.",
        )
    try:
        cfg = EpisodeConfig.load(manifest)
    except (ValueError, FileNotFoundError) as e:
        return StageResult("state", ok=False, summary=str(e))
    raw = _yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    back = raw.get("scene")
    if not isinstance(back, dict) or "manifest" not in back or "episode" not in back:
        return StageResult(
            "state",
            ok=False,
            summary=(
                "the episode carries no `scene:` back-reference - the "
                "global estimate runs against a scene's published map set. "
                "Add to episode.yaml:\n"
                "  scene:\n"
                "    manifest: <relative path to scene.yaml>\n"
                "    episode: <this episode's id in that scene>"
            ),
        )
    try:
        scene = load_scene((episode_dir / str(back["manifest"])).resolve())
        episode_id = str(back["episode"])
        published = current_maps(scene)
    except (ValueError, FileNotFoundError) as e:
        return StageResult("state", ok=False, summary=str(e))
    if published is None:
        return StageResult(
            "state",
            ok=False,
            summary=(
                f"scene {scene.name!r} has no published map set - run "
                f"`kiwi scene build` first."
            ),
        )
    profile_name = scene.state_profile or "scene-fusion"
    config = STATE_PROFILES.get(profile_name)
    if config is None:
        return StageResult(
            "state",
            ok=False,
            summary=(
                f"unknown state profile {profile_name!r}; built-in "
                f"profiles: {sorted(STATE_PROFILES)}"
            ),
        )
    params = {
        name.lower(): getattr(fusion_mod, name)
        for name in (
            "IMU_NOISE_SCALE",
            "VISUAL_INTERP_MAX_GAP_S",
            "VISUAL_FACTOR_PERIOD_S",
            "FRONT_SIGMA_ROT_DEG",
            "FRONT_SIGMA_TRANS_M",
            "REAR_SIGMA_ROT_DEG",
            "REAR_SIGMA_TRANS_M",
            "POSE_HUBER_K",
            "VELOCITY_PRIOR_SIGMA_M_S",
            "ACCEL_BIAS_PRIOR_SIGMA_M_S2",
            "GYRO_BIAS_PRIOR_SIGMA_RAD_S",
            "INTEGRATION_COV",
            "MAX_ITERATIONS",
            "GRAVITY",
        )
    }

    def _load_lens(path: Path, timeshift: float) -> tuple:
        """Tracked rows of one lens on the IMU clock."""
        obs = read_observations(path)
        ok = obs.tracked
        run_id = fusion_mod.run_ids_from_tracked(ok)
        t_cam = obs.t_s + timeshift
        return t_cam[ok], obs.T[ok], run_id[ok]

    lines: list[str] = []
    artifacts: list[Path] = []
    failed: list[str] = []
    ran = reused = 0
    for label, arm in sorted(cfg.arms.items()):
        scene_dir = layout.scene_dir(episode_dir, label)
        out_tum = scene_dir / layout.STATE_TUM
        generation = None
        try:
            sync_offset = float((cfg.sync_offsets or {}).get(label, 0.0))
            clock_name = (
                "episode_global_seconds"
                if cfg.sync_offsets
                else "recording_imu_seconds"
            )
            if out_tum.is_file():
                previous = _recorded_inputs(scene_dir / layout.STATE_INPUTS) or {}
                offsets = previous.get("clock_offsets_s")
                if (
                    previous.get("status") == "computed"
                    and previous.get("clock") == clock_name
                    and isinstance(offsets, dict)
                    and offsets.get("sync_offset") == sync_offset
                    and previous.get("map_set") == published["roles"]
                ):
                    reused += 1
                    lines.append(f"{label}: state.tum clock unchanged - reused")
                    artifacts.append(out_tum)
                    continue
                lines.append(
                    f"{label}: state clock changed or unverified - re-estimating"
                )
            generation = {
                "schema": "state_trajectory_inputs",
                "profile": profile_name,
                "frame": SCENE_FRAME_NAME,
                "status": "running",
                "producer": producer_info(
                    "kiwi.state.fusion", dependencies=("gtsam", "numpy", "scipy")
                ),
                "params": params,
                "map_set": published["roles"],
                "world_scene": {
                    "frame": SCENE_FRAME_NAME,
                    "scene": scene.name,
                    "map_set_digest": hashlib.sha256(
                        json.dumps(
                            published["roles"], sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                },
                "inputs": {
                    "raw_recording": file_ref(arm.recording, base=scene_dir),
                    "episode": file_ref(manifest, base=scene_dir),
                    "scene": file_ref(scene.path, base=scene_dir),
                },
            }
            role = f"front_{label}"
            front_obs = scene_dir / f"{role}_observations.txt"
            rear_obs = scene_dir / "shared_rear_observations.txt"
            if not is_localized(scene, published, episode_id, label, "front"):
                localize_episode(scene, episode_id, label, lens="front")
            if not is_localized(scene, published, episode_id, label, "back"):
                localize_episode(scene, episode_id, label, lens="back")
            generation["inputs"].update(
                {
                    "front_observations": file_ref(
                        front_obs,
                        base=scene_dir,
                        record=front_obs.with_suffix(".inputs.json"),
                    ),
                    "rear_observations": file_ref(
                        rear_obs,
                        base=scene_dir,
                        record=rear_obs.with_suffix(".inputs.json"),
                    ),
                }
            )
            generation["world_scene"]["scene_frame_digest"] = digest_bytes(
                scene.root / "scene_map" / "scene_frame.json"
            )

            unit = CameraUnit.load(arm.camera_dir)
            missing = [
                key
                for key in ("extrinsic_imu", "extrinsic_imu_back", "imu_noise")
                if unit.artifact(key) is None
            ]
            if missing:
                raise ValueError(
                    f"camera {arm.camera!r} publishes no {'/'.join(missing)} - "
                    f"the estimator needs both cam-IMU chains and the Allan "
                    f"noise artifact"
                )
            generation["inputs"].update(
                {
                    key: file_ref(unit.artifact(key), base=scene_dir)
                    for key in ("extrinsic_imu", "extrinsic_imu_back", "imu_noise")
                }
            )
            front_chain = read_cam_imu(unit.artifact("extrinsic_imu"))
            rear_chain = read_cam_imu(unit.artifact("extrinsic_imu_back"))
            noise = read_imu_noise(unit.artifact("imu_noise"))[0].as_dict()

            # recording-clock IMU: the loader undoes the extractor's
            # pre-roll shift, so container-PTS rows + timeshift meet it
            # on one clock (kiwi.episode.scene_build owns the recipe)
            imu_t, imu_gyro, imu_acc = load_imu(arm.recording, scene_dir)

            raw_digest = generation["inputs"]["raw_recording"]["digest"]
            imu_stem = raw_digest.split(":")[1][:16]
            generation["inputs"]["imu"] = file_ref(
                scene_dir / f"{imu_stem}-imu.csv",
                base=scene_dir,
                record=scene_dir / f"{imu_stem}-imu.json",
            )

            # every published pose lands on the episode's GLOBAL clock: the
            # per-lens cam-IMU timeshift plus the arm's sync offset.
            # The solve runs on the arm's IMU clock (timeshift only); the sync
            # offset is added to the written timestamps.
            dt_front = front_chain.timeshift_s or 0.0
            dt_rear = rear_chain.timeshift_s or 0.0
            generation.update(
                clock=clock_name,
                clock_offsets_s={
                    "front_timeshift": dt_front,
                    "rear_timeshift": dt_rear,
                    "sync_offset": sync_offset,
                },
                front_observations=front_obs.name,
                rear_observations=rear_obs.name,
            )
            generation["inputs"]["sync"] = file_ref(
                episode_dir / layout.SYNC_INPUTS, base=scene_dir
            )
            front = _load_lens(front_obs, dt_front)
            rear = _load_lens(rear_obs, dt_rear)
            if len(front[0]) == 0 and len(rear[0]) == 0:
                raise ValueError(
                    "no tracked localization row in either lens - nothing to fuse"
                )
            # One node per row of the front observation file, which lists
            # every front frame. Node and observation times come from the same
            # expression, so a tracked front row meets its node exactly.
            t_nodes = read_observations(front_obs).t_ns.astype(float) * 1e-9
            t_nodes += dt_front
            T_scene_imu = fusion_mod.fuse(
                t_nodes,
                imu_t,
                imu_gyro,
                imu_acc,
                front,
                rear,
                front_chain.T_cam_imu,
                rear_chain.T_cam_imu,
                noise,
            )
            T_front = T_scene_imu @ se3_inv(front_chain.T_cam_imu)
            q = quats_continuous(Rotation.from_matrix(T_front[:, :3, :3]).as_quat())
            t_out = t_nodes + sync_offset
            tum_tmp = scene_dir / f"{layout.STATE_TUM}.part"
            write_tum(
                Trajectory(
                    timestamps=t_out,
                    positions=T_front[:, :3, 3],
                    quaternions=q,
                    # the same spellings every other scene artifact uses:
                    # the frame is named as scene_frame.json names it, and
                    # the body is the front-lens optical centre - which the
                    # register stage and the export contract already call
                    # "camera" for the very same physical frame
                    frame=SCENE_FRAME_NAME,
                    body="camera",
                ),
                tum_tmp,
                clock=(
                    "episode global clock"
                    if cfg.sync_offsets
                    else "camera IMU clock (native front frames)"
                ),
                source="kiwi.state scene fusion",
            )
            # The nodes at which the solver's schedule finds a tracked
            # localization of each lens.
            present = []
            for obs, T_cam_imu, exact in (
                (front, front_chain.T_cam_imu, True),
                (rear, rear_chain.T_cam_imu, False),
            ):
                scheduled = fusion_mod.schedule(
                    fusion_mod.body_track(obs, T_cam_imu), t_nodes, exact_only=exact
                )
                mask = np.zeros(len(t_nodes), bool)
                mask[[m.node for m in scheduled]] = True
                present.append(mask)
            latent = scene_dir / layout.STATE_LATENT
            write_state_latent(
                latent,
                t=t_out,
                body=T_scene_imu,
                front=T_front,
                front_candidate_present=present[0],
                rear_candidate_present=present[1],
                frame=SCENE_FRAME_NAME,
                metadata={
                    "clock": generation["clock"],
                    "clock_offsets_s": generation["clock_offsets_s"],
                    "world_scene": generation["world_scene"],
                    "generation_record": {"path": layout.STATE_INPUTS},
                },
            )
            generation.update(
                status="computed",
                diagnostics={"result_status": "computed_unvalidated"},
            )
            generation["outputs"] = {
                "trajectory": {
                    **file_ref(tum_tmp, base=scene_dir),
                    "path": out_tum.name,
                },
                "latent": file_ref(latent, base=scene_dir),
            }
            write_record(scene_dir / layout.STATE_INPUTS, generation)
            # The trajectory, whose presence marks the arm reusable, is
            # published last. A failed write cannot leave a trajectory that
            # would be reused without its state/record.
            os.replace(tum_tmp, out_tum)
            ran += 1
            lines.append(f"{label}: {len(t_nodes)} native frames")
            artifacts += [out_tum, latent, scene_dir / layout.STATE_INPUTS]
        # YAMLError: the unit's cam-IMU chains and Allan noise artifact are
        # parsed here by readers that do not translate it.
        except (
            ValueError,
            RuntimeError,
            FileNotFoundError,
            OSError,
            _yaml.YAMLError,
        ) as e:
            if generation is not None:
                generation.update(status="failed", reason=f"{type(e).__name__}: {e}")
                try:
                    write_record(scene_dir / layout.STATE_INPUTS, generation)
                    artifacts.append(scene_dir / layout.STATE_INPUTS)
                except OSError as write_error:
                    lines.append(f"{label}: record write failed: {write_error}")
                (scene_dir / f"{layout.STATE_TUM}.part").unlink(missing_ok=True)
            failed.append(label)
            lines.append(f"{label}: FAILED - {e}")
    head = f"{ran} estimated, {reused} reused"
    if failed:
        head = f"FAILED for {', '.join(failed)}"
    return StageResult(
        "state",
        ok=not failed,
        summary="\n".join([head, *lines]),
        artifacts=artifacts,
    )


REGISTRY["state"] = _state_entry


def _head_entry(episode_dir: Path) -> StageResult:
    """Place the configured head marker after both arm states are fused.

    Presence of ``cameras.<label>.head_marker`` opts an episode into the
    stage. It reads the two published ``state.tum`` trajectories plus rear
    images/calibration, holds both arms fixed, and writes a separate
    whole-episode ``head/`` publication. No configuration is a typed
    successful skip. A configured episode without an accepted detection fails
    this explicitly requested stage without altering either arm artifact.
    """
    from kiwi.episode.head import (
        COMPUTED_STATUSES,
        configured_head_camera,
        publish_head_marker,
    )
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig

    episode_dir = Path(episode_dir).resolve()
    manifest = episode_dir / MANIFEST_NAME
    if not manifest.is_file():
        return StageResult(
            "head",
            ok=False,
            summary=f"no {MANIFEST_NAME} in {episode_dir}; run `kiwi init` first.",
        )
    try:
        cfg = EpisodeConfig.load(manifest)
        if configured_head_camera(cfg) is None:
            return StageResult(
                "head",
                ok=True,
                summary=(
                    "WARNING - no cameras.*.head_marker config; skipped, "
                    "so no head/head_marker.tum will be produced"
                ),
                data={"status": "skipped_no_config"},
            )
        publication = publish_head_marker(episode_dir)
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        return StageResult("head", ok=False, summary=str(exc))

    ok = publication.status in COMPUTED_STATUSES
    action = "reused" if publication.reused else "published"
    if ok:
        rows = publication.report.get("output", {}).get("rows", 0)
        segments = len(publication.report.get("segments", []))
        summary = (
            f"{action} {publication.status}; {rows} observations in {segments} segments"
        )
    else:
        summary = publication.status + (
            f" - {publication.reason}" if publication.reason else ""
        )
    return StageResult(
        "head",
        ok=ok,
        summary=summary,
        artifacts=list(publication.artifacts),
        data={"status": publication.status, "reused": publication.reused},
    )


REGISTRY["head"] = _head_entry


def _ground_entry(episode_dir: Path) -> StageResult:
    """Floor height from the fused scene-frame poses and the front videos.

    Consumes what the state stage published per arm (``<label>/scene/state.tum``:
    optical centre, metric, gravity-aligned z-up, both arms in one frame) plus the
    demuxed front video and the unit's front KB4 fit. Two multi-view geometry
    channels per second (plane sweep, rectified motion stereo) must agree; the
    floor is the deepest horizontal surface 0.8-1.8 m below the camera. Writes
    ``<episode>/ground/ground_report.json`` (the level, sigma, verdict, inputs,
    params, producer) and ``ground_observations.npz`` (every window behind it).
    Reused while every input digest and the params still match. A RED verdict
    fails this explicitly requested stage.

    Needs the optional ``ground`` pixi environment (torch, NVDEC-capable ffmpeg).
    Assumes one planar floor over the whole trajectory - see docs/estimation.md.
    """
    from kiwi.episode.ground import COMPUTED_STATUSES, publish_ground
    from kiwi.episode.manifest import MANIFEST_NAME

    episode_dir = Path(episode_dir).resolve()
    if not (episode_dir / MANIFEST_NAME).is_file():
        return StageResult(
            "ground",
            ok=False,
            summary=f"no {MANIFEST_NAME} in {episode_dir}; run `kiwi init` first.",
        )
    try:
        publication = publish_ground(episode_dir)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, ImportError) as exc:
        return StageResult("ground", ok=False, summary=str(exc))
    report = publication.report
    ok = publication.status in COMPUTED_STATUSES
    action = "reused" if publication.reused else "published"
    if ok:
        cameras = ", ".join(
            (
                f"{name} {camera['z_m']:+.4f} m "
                f"({camera['n_used']}/{camera['n_gated']} windows)"
                if camera.get("z_m") is not None
                else f"{name} no level (0/{camera['n_gated']} windows)"
            )
            for name, camera in report["cameras"].items()
        )
        summary = (
            f"{action} {publication.status}; floor z "
            f"{report['floor_z_m']:+.4f} m (sigma {100 * report['sigma_m']:.1f} cm), "
            f"verdict {report['verdict']}; {cameras}"
        )
        if report.get("reasons"):
            summary += "; " + "; ".join(report["reasons"])
    else:
        summary = f"{action} {publication.status}" + (
            f" - {publication.reason}" if publication.reason else ""
        )
    return StageResult(
        "ground",
        ok=ok,
        summary=summary,
        artifacts=list(publication.artifacts),
        data={
            "status": publication.status,
            "verdict": report.get("verdict"),
            "floor_z_m": report.get("floor_z_m"),
            "sigma_m": report.get("sigma_m"),
            "reused": publication.reused,
        },
    )


REGISTRY["ground"] = _ground_entry


def run_stage(stage: str, episode_dir: Path, *, resync: bool = False) -> StageResult:
    if stage not in REGISTRY:
        raise KeyError(f"unknown stage {stage!r}; stages are {STAGES}")
    fn = REGISTRY[stage]
    from kiwi.episode.config_snapshot import snapshot_config
    from kiwi.episode.manifest import MANIFEST_NAME

    if (episode_dir / MANIFEST_NAME).is_file():
        snapshot_config(episode_dir)
    if stage == "sync" and resync:
        return fn(episode_dir, resync=True)
    return fn(episode_dir)


def run_pipeline(
    episode_dir: Path,
    from_stage: str | None = None,
    to_stage: str | None = None,
    on_result: Callable[[StageResult], None] | None = None,
    profile=None,
    resync: bool = False,
) -> list[StageResult]:
    """Run the stage chain over one episode directory.

    ``from_stage``/``to_stage`` bound the run inclusively. ``on_result``
    (when given) receives each StageResult as it completes, so a caller
    can stream progress even if a later stage fails.

    ``resync`` explicitly replaces a saved clock before continuing through
    the requested chain. It requires sync to be inside the stage range;
    downstream stages retain their own execution and reuse policies.

    ``profile`` (default: the built-in global-state profile) states each
    stage's disposition and the trajectory mode. A ``not_requested`` stage
    is skipped while the run continues. A range narrowed
    to exactly one stage is the exception, and dispatches it whatever
    the profile says of it. That is how the on-demand stages stay
    reachable: ``--from-stage register --to-stage register`` runs the
    ground-truth anchor that no builtin profile requests.
    Outcomes land in ``pipeline_run_status.json`` (completed | failed | not_requested |
    not_executed, plus the profile name and content digest); ``run_status.json``
    records entries for started stages only.

    The chain **stops at the first stage that reports ``ok=False``**. Stages
    build on each other: ``sync`` produces the clock every later stage reads,
    and ``convert`` the video they decode. The results list ends with the
    stage that failed.

    The run is written to ``<episode>/run_status.json`` and
    ``pipeline_run_status.json`` after every stage (see
    :mod:`kiwi.episode.run_record` and :mod:`kiwi.episode.pipeline_run_status`),
    so an interrupted run still says what completed. Recording never fails the
    run it describes.
    """
    from kiwi.episode.manifest import MANIFEST_NAME
    from kiwi.episode.pipeline_profile import DEFAULT_PROFILE
    from kiwi.episode.pipeline_run_status import PipelineRunStatus
    from kiwi.episode.run_record import RunRecord

    if profile is None:
        profile = DEFAULT_PROFILE

    def _index(name: str | None, default: int) -> int:
        if name is None:
            return default
        if name not in STAGES:
            raise ValueError(f"unknown stage {name!r}; stages are {STAGES}")
        return STAGES.index(name)

    lo = _index(from_stage, 0)
    hi = _index(to_stage, len(STAGES) - 1)
    if lo > hi:
        raise ValueError(f"from_stage {from_stage!r} is after to_stage {to_stage!r}")
    chain = STAGES[lo : hi + 1]
    if resync and "sync" not in chain:
        raise ValueError("--resync requires sync in the requested stage range")
    # A single-stage request overrides the profile so on-demand stages remain
    # reachable. Wider ranges follow the profile's dispositions.
    asked = {from_stage} if from_stage is not None and from_stage == to_stage else set()
    if resync:
        asked.add("sync")
    # The recorder and the producer must resolve the same local inputs: the
    # record stamps a stage's inputs before run_stage would copy the config,
    # so an episode's first run copies it here.
    from kiwi.episode.config_snapshot import snapshot_config

    if (episode_dir / MANIFEST_NAME).is_file():
        snapshot_config(episode_dir)
    rec = RunRecord(episode_dir, episode_dir / MANIFEST_NAME, requested=chain)
    status = PipelineRunStatus.start(episode_dir, profile, asked=frozenset(asked))
    status.begin_chain(chain)
    results: list[StageResult] = []
    try:
        for name in chain:
            if not profile.requires(name) and name not in asked:
                r = StageResult(
                    name,
                    ok=True,
                    summary=f"not requested by pipeline profile '{profile.name}'",
                    data={"disposition": "not_requested"},
                )
                status.record(name, "not_requested")
                if on_result is not None:
                    on_result(r)
                results.append(r)
                continue
            try:
                # Stamping the stage's inputs loads the manifest as well, so a
                # raise there is this stage failing, recorded like one inside it.
                rec.start(name)
                t0 = time.monotonic()
                r = run_stage(
                    name,
                    episode_dir,
                    **({"resync": True} if name == "sync" and resync else {}),
                )
            except BaseException:
                # A stage that raises or is interrupted ran and did not
                # complete; run_status.json records the same stage as "error".
                status.record(name, "failed")
                raise
            rec.finish(r, time.monotonic() - t0)
            status.record(name, "completed" if r.ok else "failed")
            if on_result is not None:
                on_result(r)
            results.append(r)
            if not r.ok:
                break
    finally:
        # Preserve completed work in the records even when a stage raises.
        done = results[-1].stage if results else None
        rec.close(reached=done, halted=bool(results and not results[-1].ok))
        status.finish()  # stages the run never reached become not_executed
    return results

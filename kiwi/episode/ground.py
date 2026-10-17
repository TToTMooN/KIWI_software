"""The ground stage's composer: resolve the state stage's products, run the estimator, publish.

Mirrors the head composer's division of labour: :mod:`kiwi.ground` takes plain
typed inputs (video path, poses, kb4 parameters) and knows nothing about the
manifest; this module resolves canonical filenames, binds every input that can
change the answer into the report, reuses an existing publication only while
those identities still match, and writes the two ``<episode>/ground/``
products: ``ground_report.json`` (the level, its sigma, the verdict, the
per-camera and cross-camera summaries, inputs, params, producer) and
``ground_observations.npz`` (every per-window record behind the level).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kiwi.core import layout
from kiwi.core.camera.kb4 import read_kb4
from kiwi.core.camera.registry import CameraUnit
from kiwi.core.digest import verify_digest
from kiwi.core.generation import file_ref, producer_info, write_record
from kiwi.core.trajectory import read_tum_header
from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig

GROUND_SCHEMA = "ground_v1"
COMPUTED_STATUS = "computed"  # a level was established (verdict GREEN or YELLOW)
NO_FLOOR_STATUS = "no_floor"  # the estimate ran and found no trustworthy level (RED)
COMPUTED_STATUSES = frozenset({COMPUTED_STATUS})

#: The observations table, one row per 1 s window of one camera. Rows of
#: different cameras share the file; ``arm`` says whose a row is.
OBSERVATION_COLUMNS = {
    "arm": "camera label the window belongs to",
    "t0": "window start on the episode global clock [s]",
    "t1": "window end [s]",
    "cam_z": "median camera z over the window's frames, scene frame [m]",
    "pitch_deg": "virtual-view pitch from nadir [deg]; NaN when the window was not processed",
    "sweep_z": "plane-sweep floor z [m]; NaN when that channel found none",
    "sweep_sigma": "plane-sweep sigma [m]",
    "stereo_z": "motion-stereo floor z [m]; NaN when that channel found none",
    "stereo_sigma": "motion-stereo sigma [m]",
    "z": "the window's floor z after the channel-agreement gate [m]; NaN when rejected",
    "sigma": "the window's sigma [m]",
    "n_support": "supporting cells/pixels behind the accepted value; 0 when none",
    "channel": "'both', 'sweep', 'stereo', or '' when no channel produced a value",
    "flags": "';'-joined rejection flags; '' for an accepted window",
}


@dataclass(frozen=True)
class GroundInput:
    label: str
    video: Path
    state_tum: Path
    kb4_path: Path
    kb4_doc: dict
    frame: str


@dataclass
class GroundPublication:
    status: str
    reused: bool
    artifacts: list[Path]
    report: dict = field(default_factory=dict)
    reason: str = ""


def resolve_inputs(episode_dir: Path, cfg: EpisodeConfig) -> list[GroundInput]:
    """Per arm: the demuxed front video, the state stage's ``scene/state.tum`` and the unit's front kb4 fit."""
    out = []
    for label, arm in sorted(cfg.arms.items()):
        video = layout.stream_dir(episode_dir, label) / layout.LENS0
        tum = layout.scene_dir(episode_dir, label) / layout.STATE_TUM
        if not tum.is_file():
            raise FileNotFoundError(
                f"{label}: no {layout.SCENE_DIR}/{layout.STATE_TUM} - run the state stage "
                f"first (the ground stage reads the scene-frame trajectory it publishes)"
            )
        if not video.is_file():
            raise FileNotFoundError(
                f"{label}: no {layout.LENS0} - run the convert stage first"
            )
        unit = CameraUnit.load(arm.camera_dir)
        kb4_path = unit.artifact("intrinsics_kb4")
        if kb4_path is None:
            raise ValueError(
                f"camera {arm.camera!r} publishes no intrinsics_kb4 - the ground stage "
                f"projects through the front lens's KB4 fit"
            )
        _, doc = read_kb4(kb4_path, expected_lens="front")
        header = read_tum_header(tum)
        frame = header.frame if header is not None else "unknown"
        out.append(GroundInput(label, video, tum, kb4_path, doc, frame))
    return out


def input_references(inputs: list[GroundInput], output_dir: Path) -> dict:
    """Report-relative references to every file the estimate depends on, with their records."""
    return {
        i.label: {
            "video": file_ref(
                i.video,
                base=output_dir,
                record=i.video.parent / "convert.inputs.json",
            ),
            "state": file_ref(
                i.state_tum,
                base=output_dir,
                record=i.state_tum.parent / layout.STATE_INPUTS,
            ),
            "intrinsics_kb4": file_ref(i.kb4_path, base=output_dir),
        }
        for i in inputs
    }


def _identity(references: dict | None) -> dict:
    """What decides reuse: each input's path and digest, not its record's."""
    return {
        label: {
            name: {"path": ref.get("path"), "digest": ref.get("digest")}
            for name, ref in (files or {}).items()
        }
        for label, files in (references or {}).items()
    }


def _world_scene(inputs: list[GroundInput]) -> dict:
    """The scene identity the arm states declare.

    Read from each arm's ``state.inputs.json``, the record the state stage
    writes beside every trajectory it publishes; a missing record, or one
    that declares no ``world_scene``, is refused with the stage to re-run.
    Arms declaring different identities are refused rather than averaged
    into one floor. A key that only some arms' records carry is published as
    ``None`` rather than borrowed from the other arm.
    """
    worlds = []
    record_name = f"{layout.SCENE_DIR}/{layout.STATE_INPUTS}"
    for i in inputs:
        record = i.state_tum.parent / layout.STATE_INPUTS
        if not record.is_file():
            raise ValueError(
                f"{i.label}: no {record_name} beside the state trajectory - "
                f"re-run the state stage"
            )
        world = json.loads(record.read_text()).get("world_scene")
        if world is None:
            raise ValueError(
                f"{i.label}: {record_name} declares no world_scene - re-run the "
                f"state stage"
            )
        worlds.append(world)
    keys = ("frame", "scene", "map_set_digest", "scene_frame_digest")
    for key in keys:
        known = {world.get(key) for world in worlds if world.get(key) is not None}
        if len(known) > 1:
            raise ValueError(f"ground arm states declare different {key} identities")
    return {
        key: worlds[0].get(key)
        if all(item.get(key) is not None for item in worlds)
        else None
        for key in keys
    }


def _reuse_existing(
    report_path: Path,
    observations_path: Path,
    references: dict,
    params: dict,
) -> GroundPublication | None:
    """The existing publication when its inputs and params still match, else ``None``."""
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != GROUND_SCHEMA:
        return None
    if _identity(report.get("inputs")) != _identity(references):
        return None
    if report.get("params") != params:
        return None
    reference = (report.get("quality_outputs") or {}).get("observations") or {}
    if reference.get("path") != observations_path.name:
        return None
    try:
        verify_digest(reference["digest"], observations_path)
    except (KeyError, TypeError, ValueError, OSError):
        return None
    status = str(report.get("status"))
    reason = "; ".join(report.get("reasons") or [])
    return GroundPublication(
        status,
        True,
        [report_path, observations_path],
        report,
        reason if status != COMPUTED_STATUS else "",
    )


def _cameras(inputs: list[GroundInput]):
    from kiwi.ground.camera import Camera, load_tum, square_size

    cams = []
    for i in inputs:
        t, pos, R = load_tum(i.state_tum)
        cams.append(
            Camera(
                i.label,
                i.video,
                t,
                pos,
                R,
                i.kb4_doc["kb4"],
                square_size(i.kb4_doc, str(i.kb4_path)),
            )
        )
    return cams


def observation_arrays(windows: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    """The per-window records of every camera as equal-length columns (see OBSERVATION_COLUMNS)."""
    rows = [(arm, w) for arm, ws in windows.items() for w in ws]

    def floats(key):
        return np.asarray(
            [float("nan") if w.get(key) is None else float(w[key]) for _, w in rows],
            dtype=np.float64,
        )

    return {
        "arm": np.asarray([arm for arm, _ in rows], dtype=str),
        "t0": floats("t0"),
        "t1": floats("t1"),
        "cam_z": floats("cam_z"),
        "pitch_deg": floats("pitch_deg"),
        "sweep_z": floats("sweep_z"),
        "sweep_sigma": floats("sweep_sigma"),
        "stereo_z": floats("stereo_z"),
        "stereo_sigma": floats("stereo_sigma"),
        "z": floats("z"),
        "sigma": floats("sigma"),
        "n_support": np.asarray(
            [int(w.get("n_support") or 0) for _, w in rows], dtype=np.int64
        ),
        "channel": np.asarray([w.get("channel") or "" for _, w in rows], dtype=str),
        "flags": np.asarray(
            [";".join(w.get("flags") or ()) for _, w in rows], dtype=str
        ),
    }


def _write_observations(path: Path, arrays: dict, metadata: dict) -> Path:
    """Atomically save the table; the JSON ``metadata`` scalar never needs pickle."""
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(
                    json.dumps(metadata, sort_keys=True, allow_nan=False)
                ),
                **arrays,
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _f(v):
    v = float(v)
    return v if np.isfinite(v) else None


def publish_ground(
    episode_dir: Path,
    *,
    config=None,
    estimator=None,
) -> GroundPublication:
    """Run (or reuse) the floor estimate for one episode and publish ``<episode>/ground/``.

    ``estimator(cameras, config, log=print) -> kiwi.ground.pool.Result`` defaults to
    :func:`kiwi.ground.pipeline.estimate`; tests inject a fake. An existing
    report is reused while every input's digest and the params match and its
    observations file verifies; otherwise the estimate runs again and both
    products are rewritten.
    """
    from kiwi.ground.config import GroundConfig

    episode_dir = Path(episode_dir).resolve()
    cfg = EpisodeConfig.load(episode_dir / MANIFEST_NAME)
    inputs = resolve_inputs(episode_dir, cfg)
    if not inputs:
        raise ValueError("the manifest names no arms")
    frames = sorted({i.frame for i in inputs})
    if len(frames) > 1:
        raise ValueError(
            f"arms declare different scene frames ({frames}); one floor needs "
            f"both trajectories in one frame - re-run the state stage against "
            f"one published map set"
        )
    config = config or GroundConfig()
    params = config.as_dict()
    output_dir = layout.ground_dir(episode_dir)
    report_path = output_dir / layout.GROUND_REPORT
    observations_path = output_dir / layout.GROUND_OBSERVATIONS
    references = input_references(inputs, output_dir)
    existing = _reuse_existing(report_path, observations_path, references, params)
    if existing is not None:
        return existing
    if estimator is None:
        from kiwi.ground.pipeline import estimate as estimator

    world_scene = _world_scene(inputs)
    cams = _cameras(inputs)
    t0 = time.time()
    res = estimator(cams, config, log=print)
    runtime_s = time.time() - t0
    status = COMPUTED_STATUS if res.verdict in ("GREEN", "YELLOW") else NO_FLOOR_STATUS
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_observations(
        observations_path,
        observation_arrays(res.windows),
        metadata={
            "schema": "ground_observations",
            "frame": frames[0],
            "clock": "episode_global_seconds",
            "status": status,
            "verdict": res.verdict,
            "floor_z_m": _f(res.floor_z_m),
            "columns": OBSERVATION_COLUMNS,
            "missing_numeric": "NaN: not measured; every window has a row, pose-gated ones included",
            "world_scene": world_scene,
        },
    )
    report = {
        "schema": GROUND_SCHEMA,
        "episode": cfg.name,
        "status": status,
        "verdict": res.verdict,
        "reasons": list(res.reasons),
        "floor_z_m": _f(res.floor_z_m),
        "sigma_m": _f(res.sigma_m),
        "frame": frames[0],
        "clock": "episode_global_seconds",
        "world_scene": world_scene,
        "cameras": {
            name: {
                "z_m": _f(c["z"]),
                "sigma_m": _f(c["sigma_m"]),
                "n_used": int(c["n_used"]),
                "n_gated": int(c["n_gated"]),
                "n_total": int(c["n_total"]),
            }
            for name, c in res.cameras.items()
        },
        "cross": (
            {
                "gap_cm": _f(res.cross["gap_cm"]),
                "pair_sd_cm": _f(res.cross["pair_sd_cm"]),
                "pair_sd_raw_cm": _f(res.cross["pair_sd_raw_cm"]),
                "n_pairs": int(res.cross["n_pairs"]),
                "channel_offset_cm": _f(res.cross["channel_offset_cm"]),
            }
            if res.cross
            else None
        ),
        "params": params,
        "inputs": references,
        "inputs_base": ".",  # every reference above is report-relative
        "producer": producer_info(
            "kiwi.ground", dependencies=("torch", "opencv-python", "numpy")
        ),
        "quality_outputs": {
            "observations": file_ref(observations_path, base=output_dir),
        },
        "columns": OBSERVATION_COLUMNS,
        "runtime_s": round(runtime_s, 1),
    }
    write_record(report_path, report)
    reason = "; ".join(res.reasons) if status != COMPUTED_STATUS else ""
    return GroundPublication(
        status, False, [report_path, observations_path], report, reason
    )

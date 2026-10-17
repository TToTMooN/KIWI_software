"""Episode-dir source: ``replay_state.npz`` -> the (N, 7) replay contract.

Reads the frozen EE-state seam (``kiwi.export.replay_state`` writes it) from
a list of episode directories and converts each arm's stream into the replay
qpos layout via ``kiwi.replay.state.convert_ee_pose``.

This module only READS the npz. Building a missing one means running the
export pipeline, and composers never import each other - the CLI (the
composition root) ensures every episode's npz exists before handing the dirs
here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.core.layout import REPLAY_STATE
from kiwi.replay.state import convert_ee_pose

__all__ = [
    "load_meta",
    "episode_to_qpos",
    "read_state",
    "read_world_scene",
    "side_qpos",
    "world_scene_label",
]

# The capture stream is ~30 fps already; a higher-rate npz is decimated to
# this target unless the caller asks otherwise (0/None keeps the native rate).
DEFAULT_FPS = 30.0


def read_world_scene(npz_path: Path) -> tuple[str, str, str]:
    """The world a dump's poses live in: ``(frame, scene, map_set_digest)``.

    ``np.load`` on an npz is lazy, so this touches the two small string
    entries and never the state array. :func:`load_meta` uses it to check
    EVERY episode's world before anything is solved.

    A dump without the ``world_scene`` key (or one from the
    marker source, where the world is per-episode) yields empty scene/digest
    fields and is compared on its frame alone.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"{npz_path} missing: build it with `kiwi export state {npz_path.parent}`"
        )
    with np.load(npz_path, allow_pickle=True) as d:
        frame = str(d["frame"]) if "frame" in d.files else ""
        raw = str(d["world_scene"]) if "world_scene" in d.files else "null"
    try:
        ws = json.loads(raw) or {}
    except json.JSONDecodeError:
        ws = {}
    if not isinstance(ws, dict):
        ws = {}
    return (
        str(ws.get("frame") or frame),
        str(ws.get("scene") or ""),
        str(ws.get("map_set_digest") or ""),
    )


def world_scene_label(world: tuple[str, str, str]) -> str:
    """A world triple as a phrase a person can act on."""
    frame, scene, digest = world
    where = f"scene {scene!r}" if scene else "no scene named"
    maps = f", map set {digest[:12]}" if digest else ""
    return f"frame {frame!r} ({where}{maps})"


def read_state(npz_path: Path) -> dict:
    """Load one ``replay_state.npz`` into
    ``{state, names, sides, fps, frame, ee_frame, task}``.

    Hard-fails unless the dump is at the tool TCP: the solver targets the tip
    site, so a missing ``ee_frame`` field or any non-TCP value is refused.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"{npz_path} missing: build it with `kiwi export state {npz_path.parent}`"
        )
    d = np.load(npz_path, allow_pickle=True)
    ee_frame = str(d["ee_frame"]) if "ee_frame" in d.files else "(absent)"
    if ee_frame != "tcp":
        raise ValueError(
            f"{npz_path} was exported with ee_frame={ee_frame!r}, but the solver "
            f"drives the tool-tip site and needs ee_frame='tcp'. Rebuild it at "
            f"the tool TCP (`kiwi export state {npz_path.parent}`)."
        )
    names = [str(x) for x in d["names"]]
    # Arm labels straight from the state's own column names (e.g. left, right).
    sides = [n[: -len("_ee_x")] for n in names if n.endswith("_ee_x")]
    return {
        "state": np.asarray(d["state"], dtype=np.float64),
        "names": names,
        "sides": sides,
        "fps": float(d["fps"]),
        "frame": str(d["frame"]),
        "ee_frame": ee_frame,
        "task": str(d["task"]) if "task" in d.files else "",
    }


def side_qpos(
    state: np.ndarray,
    names: list[str],
    side: str,
    *,
    rot_offset_deg=(0.0, 0.0, 0.0),
    traj_yaw_deg: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """One arm's (N, 7) qpos from the replay-state columns.

    Contract conventions: quaternion orientation (xyzw), gripper already in
    [0, 1]; the shared yaw/roll/gripper math lives in
    ``kiwi.replay.state.convert_ee_pose``.
    """

    def col(sub: str) -> int:
        return names.index(f"{side}_{sub}")

    pos = state[:, [col("ee_x"), col("ee_y"), col("ee_z")]]
    quat = state[:, [col("ee_qx"), col("ee_qy"), col("ee_qz"), col("ee_qw")]]  # xyzw
    grip01 = np.clip(state[:, col("gripper")], 0.0, 1.0)  # 0 closed, 1 open
    qpos = convert_ee_pose(
        pos,
        Rotation.from_quat(quat),
        grip01,
        rot_offset_deg=rot_offset_deg,
        traj_yaw_deg=traj_yaw_deg,
    )
    info = {
        "n": len(qpos),
        "pos_span_mm": np.round(np.ptp(qpos[:, :3], 0) * 1000).astype(int),
        "grip01": (float(grip01.min()), float(grip01.max())),
    }
    return qpos, info


def load_meta(
    dirs: list[Path], *, fps: float | None = DEFAULT_FPS, quiet: bool = False
) -> dict:
    """Probe a list of episode dirs into one dataset-meta dict.

    ``dirs`` are the episodes, in order (episode id = position in the list);
    every dir must already hold a ``replay_state.npz``. Only the FIRST
    episode's STATE is read here (that is where names/fps come from); the rest
    are read lazily by ``episode_to_qpos``, which also checks their columns
    against the first's. Every episode's WORLD is read here though - see
    below. ``fps`` decimates the capture clock to that target (``0``/``None``
    keeps the native rate); ``lengths`` fills in as episodes are converted.

    **One world for the set.** The solver anchors an episode set on a single
    shared origin (``shared_frame`` in :mod:`kiwi.replay.solve`), which
    requires every loaded episode to be co-registered in ONE world. The check
    compares the full world identity ``(frame, scene, map_set_digest)``, and
    it happens HERE, before a single pose is anchored.
    """
    dirs = [Path(d) for d in dirs]
    if not dirs:
        raise ValueError("no episode directories given")
    worlds: dict[tuple[str, str, str], list[str]] = {}
    for d in dirs:
        worlds.setdefault(read_world_scene(d / REPLAY_STATE), []).append(d.name)
    if len(worlds) > 1:
        detail = "; ".join(
            f"{world_scene_label(w)}: {', '.join(names)}" for w, names in worlds.items()
        )
        raise ValueError(
            f"these episodes are not one dataset - they live in "
            f"{len(worlds)} different worlds, and replay anchors the whole "
            f"set on one shared origin. Seen: {detail}. Replay episodes of "
            f"one scene (and one published map set) at a time."
        )
    probe = read_state(dirs[0] / REPLAY_STATE)
    if not {"left", "right"} <= set(probe["sides"]):
        raise ValueError(
            f"{dirs[0] / REPLAY_STATE}: expected left+right arm blocks in the "
            f"state names, got arms {probe['sides'] or '(none)'}"
        )
    native_fps = probe["fps"]
    decim = 1 if not fps else max(1, int(round(native_fps / float(fps))))
    if not quiet:
        listing = ", ".join(f"{i}={d.name}" for i, d in enumerate(dirs))
        print(f"episodes ({len(dirs)}): {listing}")
        if decim > 1:
            print(
                f"  capture clock {native_fps:g} fps -> every {decim}th frame "
                f"({native_fps / decim:g} fps target {fps:g})"
            )
    return {
        "dirs": dirs,
        "names": probe["names"],
        "sides": probe["sides"],
        "fps": native_fps / decim,
        "native_fps": native_fps,
        "decim": decim,
        "frame": probe["frame"],
        # The one world the whole set shares (the check above proved it):
        # (frame, scene, map_set_digest).
        "world_scene": next(iter(worlds)),
        "task": probe["task"],
        "total_episodes": len(dirs),
        "lengths": {0: len(probe["state"][::decim])},
    }


def episode_to_qpos(
    meta: dict,
    episode: int,
    *,
    rot_offset_deg=(0.0, 0.0, 0.0),
    traj_yaw_deg: float = 0.0,
    stride: int = 1,
) -> tuple[dict[str, np.ndarray], dict]:
    """Convert one episode into ``({"left": qpos, "right": qpos}, info)``.

    ``episode`` indexes ``meta["dirs"]``; ``rot_offset_deg`` is a body-frame
    intrinsic-XYZ rotation right-multiplied onto each EE orientation,
    ``traj_yaw_deg`` a world-frame yaw baked into positions + orientations,
    ``stride`` subsamples frames ON TOP of the meta's capture-clock decimation
    (so it means every Nth frame of the ~30 fps stream, not of the raw
    capture).
    """
    dirs = meta["dirs"]
    if not 0 <= episode < len(dirs):
        raise ValueError(f"episode {episode} out of range (0-{len(dirs) - 1})")
    ep_dir = dirs[episode]
    d = read_state(ep_dir / REPLAY_STATE)
    if d["names"] != meta["names"]:
        raise ValueError(
            f"{ep_dir / REPLAY_STATE}: state columns {d['names']} differ from "
            f"the first episode's {meta['names']} - these dirs are not one "
            f"dataset."
        )
    state = d["state"][:: meta.get("decim", 1)]
    meta["lengths"][episode] = len(state)
    state = state[:: max(int(stride), 1)]

    qpos_by_side: dict[str, np.ndarray] = {}
    info: dict = {"n": len(state), "grip01": {}, "pos_span_mm": {}, "dir": ep_dir}
    for side in ("left", "right"):
        q, side_info = side_qpos(
            state,
            d["names"],
            side,
            rot_offset_deg=rot_offset_deg,
            traj_yaw_deg=traj_yaw_deg,
        )
        qpos_by_side[side] = q
        info["grip01"][side] = side_info["grip01"]
        info["pos_span_mm"][side] = side_info["pos_span_mm"]
    return qpos_by_side, info

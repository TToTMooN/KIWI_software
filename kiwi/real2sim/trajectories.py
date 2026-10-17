"""Portable, metric episode trajectories for reconstructed-scene viewing.

State timestamps already use the episode global clock. Snapshots contain only
viewing data, not source videos or another copy of the processing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from kiwi.core.camera.extrinsics import read_cam_imu
from kiwi.core.camera.registry import CameraUnit
from kiwi.core.digest import digest_bytes, verify_digest
from kiwi.core.trajectory import PoseTrack, load_tum, read_tum_header
from kiwi.real2sim.asset import ArmSource, validate_transform

FRAME = "shared_rear_metric_gravity_aligned"


def map_set_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def same_world(actual: dict, expected: dict) -> None:
    """Compare scene identity; an unknown placement (``None``) matches any."""
    for key in ("frame", "scene", "map_set_digest"):
        if not actual.get(key) or actual.get(key) != expected.get(key):
            raise ValueError(f"Trajectory and reconstruction disagree on {key}")
    left, right = actual.get("scene_frame_digest"), expected.get("scene_frame_digest")
    if left and right and left != right:
        raise ValueError("Trajectory and reconstruction use different scene placements")


@dataclass
class ArmTrack:
    times: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    rear_from_front: np.ndarray
    tcp_from_front: np.ndarray | None = None
    max_gap_s: float = 0.05

    def __post_init__(self):
        n = len(self.times)
        if (
            n < 2
            or self.positions.shape != (n, 3)
            or self.quaternions.shape != (n, 4)
            or not all(
                np.isfinite(x).all()
                for x in (self.times, self.positions, self.quaternions)
            )
            or not np.all(np.diff(self.times) > 0)
        ):
            raise ValueError("Expected finite, strictly increasing camera trajectory")
        if not np.allclose(np.linalg.norm(self.quaternions, axis=1), 1, atol=1e-3):
            raise ValueError("Trajectory quaternions must have unit length")
        validate_transform(self.rear_from_front)
        if self.tcp_from_front is not None:
            validate_transform(self.tcp_from_front)
        self._track = PoseTrack(
            self.times, Rotation.from_quat(self.quaternions), self.positions
        )

    def at(self, timestamp: float, body: str = "camera") -> np.ndarray | None:
        """No extrapolation or interpolation across unsupported tracking gaps."""
        if timestamp < self.times[0] or timestamp > self.times[-1]:
            return None
        j = int(np.searchsorted(self.times, timestamp))
        exact = j < len(self.times) and abs(self.times[j] - timestamp) < 1e-8
        if (
            not exact
            and 0 < j < len(self.times)
            and self.times[j] - self.times[j - 1] > self.max_gap_s
        ):
            return None
        pose = self._track.at(timestamp)
        if body == "rear":
            return pose @ self.rear_from_front
        if body == "tcp":
            if self.tcp_from_front is None:
                raise ValueError("This trajectory has no fixed camera-to-TCP transform")
            return pose @ self.tcp_from_front
        if body != "camera":
            raise ValueError(f"Unknown trajectory body: {body}")
        return pose

    def segments(
        self, timestamp: float, window: float, full: bool, body: str = "camera"
    ):
        ix = np.arange(len(self.times))
        if not full:
            ix = ix[(self.times >= timestamp - window) & (self.times <= timestamp)]
        if len(ix) < 2:
            return np.empty((0, 2, 3))
        # Limit drawing cost without bridging a gap hidden by decimation.
        ix = ix[np.unique(np.linspace(0, len(ix) - 1, min(len(ix), 2000), dtype=int))]
        gaps = np.r_[0, np.cumsum(np.diff(self.times) > self.max_gap_s)]
        keep = gaps[ix[:-1]] == gaps[ix[1:]]
        positions = self.positions[ix]
        if body == "tcp":
            if self.tcp_from_front is None:
                raise ValueError("This trajectory has no fixed camera-to-TCP transform")
            positions = positions + Rotation.from_quat(self.quaternions[ix]).apply(
                self.tcp_from_front[:3, 3]
            )
        elif body != "camera":
            raise ValueError(f"Unknown path body: {body}")
        return np.stack((positions[:-1], positions[1:]), axis=1)[keep]


def save_trajectories(
    sources: list[ArmSource],
    output: Path,
    *,
    tcp_transforms: dict[tuple[str, str], np.ndarray] | None = None,
) -> dict:
    """Freeze supplied arm trajectories/calibration, with paths relative to output.

    ``tcp_transforms`` maps (resolved episode path, arm label) to T_front_tcp.
    The caller resolves tool descriptions; this module never parses EpisodeConfig.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    episodes, identities, names = {}, [], {}
    for source in sources:
        episode = source.episode.resolve()
        key = str(episode)
        if key not in names:
            name = episode.name
            if name in episodes:
                name = f"{name}-{len(names) + 1}"
            names[key] = name
            episodes[name] = {"source_episode": key, "arms": {}}
        name = names[key]
        directory = episode / source.label / "scene"
        tum = directory / "state.tum"
        header = read_tum_header(tum)
        state = json.loads((directory / "state.inputs.json").read_text())
        if (
            header is None
            or header.frame != FRAME
            or header.body != "camera"
            or not header.clock.startswith("episode global clock")
        ):
            raise ValueError(
                f"{tum}: expected front-camera state on the episode global clock"
            )
        if state.get("frame") != header.frame:
            raise ValueError(f"{tum}: state generation declares a different frame")
        produced = (state.get("outputs") or {}).get("trajectory") or {}
        if not produced.get("digest"):
            raise ValueError(f"{tum}: state generation records no trajectory digest")
        verify_digest(produced["digest"], tum)
        scene_file = source.scene_manifest
        scene_name = yaml.safe_load(scene_file.read_text())["name"]
        current = json.loads((scene_file.parent / "scene_map/current.json").read_text())
        if not state.get("map_set") or state["map_set"] != current["roles"]:
            raise ValueError(f"{tum}: state map set differs from the scene publication")
        frame_file = scene_file.parent / "scene_map/scene_frame.json"
        if json.loads(frame_file.read_text()).get("frame") != FRAME:
            raise ValueError(f"{frame_file}: incompatible scene frame")
        placement = (state.get("world_scene") or {}).get("scene_frame_digest")
        if placement:
            verify_digest(placement, frame_file)
        world = dict(
            frame=FRAME,
            scene=scene_name,
            map_set_digest=map_set_digest(state["map_set"]),
            scene_frame_digest=placement,
        )
        if identities:
            same_world(world, identities[0])
        identities.append(world)
        unit = CameraUnit.load(source.camera_dir)
        front = read_cam_imu(unit.artifact("extrinsic_imu"))
        rear = read_cam_imu(unit.artifact("extrinsic_imu_back"))
        for lens, chain in (("front", front), ("rear", rear)):
            validate_transform(chain.T_cam_imu)
            saved = state.get("clock_offsets_s", {}).get(f"{lens}_timeshift")
            if (
                saved is None
                or chain.dt_imu_from_camera_s is None
                or not np.isclose(saved, chain.dt_imu_from_camera_s, atol=1e-8, rtol=0)
            ):
                raise ValueError(
                    f"{tum}: {lens} calibration differs from the state generation"
                )
        saved_sync = state.get("clock_offsets_s", {}).get("sync_offset")
        if saved_sync is None or not np.isclose(
            saved_sync, source.sync_offset_s, atol=1e-6, rtol=0
        ):
            raise ValueError(
                f"{tum}: synchronization differs from the state generation"
            )
        times, positions, quaternions, _ = load_tum(tum)
        tcp = (tcp_transforms or {}).get((key, source.label))
        track = ArmTrack(
            np.asarray(times),
            np.asarray(positions),
            np.asarray(quaternions),
            front.T_cam_imu @ np.linalg.inv(rear.T_cam_imu),
            tcp,
        )
        relative = Path(f"episode-{list(episodes).index(name):03d}-{source.label}.npz")
        arrays = dict(
            times=track.times,
            positions=track.positions,
            quaternions=track.quaternions,
            rear_from_front=track.rear_from_front,
        )
        if tcp is not None:
            arrays["tcp_from_front"] = tcp
        np.savez_compressed(output / relative, **arrays)
        episodes[name]["arms"][source.label] = dict(
            path=str(relative),
            digest=digest_bytes(output / relative),
            source_trajectory_digest=digest_bytes(tum),
            max_gap_s=track.max_gap_s,
        )
    if not episodes:
        raise ValueError("No episode trajectories supplied")
    # One unknown placement makes the snapshot's placement unknown as a whole.
    known_placements = {
        identity["scene_frame_digest"]
        for identity in identities
        if identity.get("scene_frame_digest")
    }
    if len(known_placements) > 1:
        raise ValueError("Supplied trajectories use different scene placements")
    world = identities[0].copy()
    if any(identity.get("scene_frame_digest") is None for identity in identities):
        world["scene_frame_digest"] = None
    record = dict(
        schema="kiwi_real2sim_trajectories_v1",
        world_scene=world,
        units="metres",
        clock="episode global clock",
        episodes=episodes,
    )
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    load_trajectories(output)  # Common left/right support and saved shapes.
    return record


def load_trajectories(path: Path, *, world_scene: dict | None = None) -> dict:
    path = Path(path)
    record = json.loads((path / "manifest.json").read_text())
    if record["units"] != "metres" or record["clock"] != "episode global clock":
        raise ValueError(
            "Viewer requires metric trajectories on the episode global clock"
        )
    if world_scene is not None:
        same_world(record["world_scene"], world_scene)
    episodes = {}
    for name, item in record["episodes"].items():
        tracks = {}
        if set(item["arms"]) != {"left", "right"}:
            raise ValueError(f"{name}: require both left and right trajectories")
        for arm, info in item["arms"].items():
            file = path / info["path"]
            verify_digest(info["digest"], file)
            with np.load(file, allow_pickle=False) as arrays:
                tracks[arm] = ArmTrack(
                    arrays["times"],
                    arrays["positions"],
                    arrays["quaternions"],
                    arrays["rear_from_front"],
                    arrays.get("tcp_from_front"),
                    info["max_gap_s"],
                )
        low = max(track.times[0] for track in tracks.values())
        high = min(track.times[-1] for track in tracks.values())
        if high <= low:
            raise ValueError(f"{name}: no shared left/right trajectory interval")
        episodes[name] = dict(
            tracks=tracks,
            low=float(low),
            high=float(high),
            initial_time=float((low + high) / 2),
        )
    return episodes

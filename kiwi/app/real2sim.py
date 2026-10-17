"""Connect episode configuration to the Gaussian scene viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from kiwi.core.descriptions import fixed_frame_T, load_robot_description
from kiwi.core.tool_config import load_T_mount_cam, load_T_mount_tcp
from kiwi.episode.manifest import EpisodeConfig, MANIFEST_NAME
from kiwi.real2sim.asset import ArmSource


def _camera_tcp(arm) -> np.ndarray | None:
    if arm.description is None:
        return None
    description = load_robot_description(arm.description)
    links = {
        role: description.end_effectors[role].urdf_link
        for role in ("default", "camera", "tcp")
    }
    mount_cam = (
        load_T_mount_cam(arm.mount_cam)
        if arm.mount_cam is not None
        else fixed_frame_T(description.urdf_path, links["default"], links["camera"])
    )
    mount_tcp = (
        load_T_mount_tcp(arm.mount_tcp)
        if arm.mount_tcp is not None
        else fixed_frame_T(description.urdf_path, links["default"], links["tcp"])
    )
    return np.linalg.inv(mount_cam) @ mount_tcp


def episode_sources(episodes: tuple[Path, ...] | list[Path]):
    """Resolve only arm inputs; optional ego recordings are not required."""
    sources, tcp = [], {}
    for episode in dict.fromkeys(Path(p).resolve() for p in episodes):
        manifest = episode / MANIFEST_NAME
        cfg = EpisodeConfig.load(manifest, check_exists=False)
        raw = yaml.safe_load(manifest.read_text())
        scene = raw.get("scene", {}).get("manifest")
        if not scene:
            raise ValueError(f"{manifest}: real2sim needs a shared scene map")
        for label in ("left", "right"):
            arm = cfg.arms[label]
            if cfg.sync_offsets is None or label not in cfg.sync_offsets:
                raise ValueError(f"{manifest}: missing synchronization for {label}")
            sources.append(
                ArmSource(
                    episode,
                    label,
                    arm.camera_dir,
                    cfg.sync_offsets[label],
                    (episode / scene).resolve(),
                )
            )
            transform = _camera_tcp(arm)
            if transform is not None:
                tcp[(str(episode), label)] = transform
    return sources, tcp


def view(
    asset: Path,
    *,
    backend_repo: Path | None = None,
    python: Path | None = None,
    episodes: tuple[Path, ...] = (),
    output: Path | None = None,
    views: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8895,
    max_resolution: int = 1920,
    pose_body: str = "camera",
) -> None:
    from kiwi.real2sim.asset import read_asset
    from kiwi.real2sim.backend import _backend, run_worker
    from kiwi.real2sim.trajectories import same_world, save_trajectories

    record = read_asset(asset)
    repo, executable = _backend(backend_repo, python, "train.py")
    args = [
        "--asset",
        record["root"],
        "--backend-repo",
        repo,
        "--host",
        host,
        "--port",
        port,
        "--max-resolution",
        max_resolution,
        "--pose-body",
        pose_body,
    ]
    if episodes:
        sources, tcp = episode_sources(episodes)
        base = output.parent if output and output.suffix == ".png" else output
        trajectory_dir = (base or record["root"] / "viewer") / "trajectories"
        snapshot = save_trajectories(sources, trajectory_dir, tcp_transforms=tcp)
        same_world(snapshot["world_scene"], record["world_scene"])
        args += ["--trajectories", trajectory_dir]
    if output:
        args += ["--output", output]
    if views:
        args += ["--views", views]
    run_worker(executable, "kiwi.real2sim.viewer", *args)

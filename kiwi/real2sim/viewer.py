"""Native GS viewing and same-frame trajectory overlays in an optional CUDA env.

The browser controls an OpenCV camera and uses the scene's trained renderer.
The CUDA model stays on the host. This module is optional: neither Torch nor
Viser is imported by the ordinary KIWI CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.real2sim.asset import file_identity, validate_transform, write_json
from kiwi.real2sim.overlay import composite_overlay, trajectory_overlay
from kiwi.real2sim.trajectories import load_trajectories, map_set_digest, same_world


def _training_record(checkpoint: Path):
    record_path = next(
        (p / "run.json" for p in checkpoint.parents if (p / "run.json").is_file()),
        None,
    )
    if record_path is None:
        raise ValueError("checkpoint requires its KIWI training run.json")
    return json.loads(record_path.read_text())


def load_scene(dataset: Path, checkpoint: Path):
    """Load the training views associated with this metric checkpoint."""
    manifest = json.loads((dataset / "manifest.json").read_text())
    identity = file_identity(dataset / "manifest.json")
    record = _training_record(checkpoint)
    if (
        record.get("kind") != "train"
        or record.get("status") != "completed"
        or record.get("dataset_manifest", {}).get("digest") != identity["digest"]
    ):
        raise ValueError(
            "checkpoint must be from a completed run of this exact dataset"
        )
    if manifest.get("units") != "metres":
        raise ValueError("viewer requires metric poses in metres")
    return [f for f in manifest["frames"] if f["split"] == "train"]


def upright_pose(frame):
    pose = np.asarray(frame["T_world_camera"], dtype=float).copy()
    # Keep the azimuth but start with a slightly downward perspective view
    # into the room.
    forward = pose[:3, 2].copy()
    horizontal = np.linalg.norm(forward[:2])
    if horizontal < 1e-5:
        forward[:2] = [1, 0]
        horizontal = 1.0
    forward[:2] *= math.cos(math.radians(12)) / horizontal
    forward[2] = -math.sin(math.radians(12))
    right = np.cross(forward, [0, 0, 1])
    if np.linalg.norm(right) < 1e-5:
        right = pose[:3, 0]
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose[:3, :3] = np.column_stack((right, down, forward))
    return pose


def checkpoint_method(checkpoint):
    """Select the trained renderer; never reinterpret splats as ray-traced kernels."""
    conf = checkpoint["config"]
    method = conf.get("render", {}).get("method")
    if (
        method not in {"3dgut", "3dgrt"}
        or conf.get("model", {}).get("feature_type") != "sh"
        or checkpoint.get("feature_type", "sh") != "sh"
        or conf.get("post_processing", {}).get("method") is not None
        or "post_processing" in checkpoint
        or "feature_decoder" in checkpoint
    ):
        raise ValueError(
            "viewer supports 3DGUT/3DGRT SH checkpoints without post-processing or feature decoders"
        )
    return method


class NativeRenderer:
    def __init__(self, backend_repo: Path, checkpoint_path: Path):
        sys.path.insert(0, str(backend_repo))
        import torch
        from threedgrut.model.model import MixtureOfGaussians

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("the live viewer requires the NVIDIA CUDA environment")
        # Only locally produced, manifest-verified KIWI checkpoints are accepted.
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        conf = checkpoint["config"]
        self.method = checkpoint_method(checkpoint)
        self.model = MixtureOfGaussians(conf)
        self.model.init_from_checkpoint(checkpoint, setup_optimizer=False)
        self.model.eval()
        self.model.build_acc()
        self.num_gaussians = self.model.num_gaussians
        self.ray_key = None
        del checkpoint
        torch.cuda.empty_cache()

    def render(self, pose, width, height, fov):
        from threedgrut.datasets.protocols import Batch
        from threedgrut.utils.render import apply_background

        torch = self.torch
        focal = 0.5 * height / math.tan(fov / 2)
        key = (width, height, fov)
        with torch.inference_mode():
            if key != self.ray_key:
                v, u = torch.meshgrid(
                    torch.arange(height, dtype=torch.float32, device="cuda"),
                    torch.arange(width, dtype=torch.float32, device="cuda"),
                    indexing="ij",
                )
                self.rays = torch.nn.functional.normalize(
                    torch.stack(
                        (
                            (u + 0.5 - width / 2) / focal,
                            (v + 0.5 - height / 2) / focal,
                            torch.ones_like(u),
                        ),
                        dim=-1,
                    ),
                    dim=-1,
                )[None]
                self.origins = torch.zeros_like(self.rays)
                self.ray_key = key
            batch = Batch(
                intrinsics=[focal, focal, width / 2, height / 2],
                T_to_world=torch.as_tensor(pose, dtype=torch.float32)[None],
                rays_ori=self.origins,
                rays_dir=self.rays,
            )
            outputs = apply_background(
                self.model.background, self.model(batch, train=False), batch
            )
            return (
                (outputs["pred_features"][0].clamp(0, 1) * 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )


def look_at(position, target):
    forward = np.asarray(target) - np.asarray(position)
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(forward @ up) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    pose[:3, 3] = position
    return pose


def focus_pose(episode, timestamp, frames):
    """Frame the hands from a nearby observed position, with a simple fallback."""
    poses = [track.at(timestamp) for track in episode["tracks"].values()]
    poses = [pose for pose in poses if pose is not None]
    if not poses:
        return upright_pose(frames[len(frames) // 2])
    center = np.mean([pose[:3, 3] for pose in poses], axis=0)
    candidates = np.asarray([frame["T_world_camera"] for frame in frames])
    direction = center - candidates[:, :3, 3]
    distance = np.linalg.norm(direction, axis=1)
    direction /= np.maximum(distance[:, None], 1e-9)
    angle = np.arccos(np.clip(np.sum(direction * candidates[:, :3, 2], axis=1), -1, 1))
    score = abs(distance - 1.0) + 2 * np.maximum(angle - math.radians(35), 0)
    score += 2 * np.maximum(center[2] - candidates[:, 2, 3], 0)
    score[distance < 0.4] = np.inf
    position = (
        candidates[np.argmin(score), :3, 3]
        if np.isfinite(score).any()
        else center + [1.0, 1.0, 0.8]
    )
    return look_at(position, center)


def load_asset(asset: Path, trajectories: Path | None = None):
    """Resolve a movable asset and optional same-world trajectory snapshot."""
    asset = Path(asset).resolve()
    run = json.loads((asset / "run.json").read_text())
    if run.get("status") != "completed":
        raise ValueError("Viewer requires a completed reconstruction")
    dataset, checkpoint = (asset / run[key] for key in ("dataset", "checkpoint"))
    frames = load_scene(dataset, checkpoint)
    manifest = json.loads((dataset / "manifest.json").read_text())
    world = run["world_scene"]
    if manifest.get("world_scene"):
        same_world(manifest["world_scene"], world)
    if manifest.get("frame") != world["frame"]:
        raise ValueError("Dataset and asset declare different world frames")
    for stream in manifest.get("streams", []):
        source = stream["source"]
        if map_set_digest(source["map_set"]) != world["map_set_digest"]:
            raise ValueError("Dataset and asset declare different map sets")
        placement = world.get("scene_frame_digest")
        if placement and source.get("scene_frame", {}).get("digest") != placement:
            raise ValueError("Dataset and asset declare different scene placements")
    trajectory_dir = (
        Path(trajectories) if trajectories is not None else asset / run["trajectories"]
    )
    episodes = load_trajectories(trajectory_dir, world_scene=run["world_scene"])
    return run, frames, checkpoint, episodes


def validate_view(view: dict, episodes: dict) -> dict:
    """A saved view is the camera and trajectory snapshot of one delivered image."""
    view = dict(view)
    pose = np.asarray(view["camera_pose"], dtype=float)
    validate_transform(pose)
    if not (
        256 <= int(view["width"]) <= 1920
        and 1 <= int(view["height"]) <= 7680
        and 0 < float(view["fov"]) < math.pi
    ):
        raise ValueError("Invalid render dimensions or field of view")
    episode = episodes[view["episode"]]
    if not episode["low"] <= float(view["time"]) <= episode["high"]:
        raise ValueError("Saved view time lies outside the selected episode")
    body = view.get("pose_body", "camera")
    if body not in ("camera", "tcp"):
        raise ValueError("pose_body must be camera or tcp")
    if body == "tcp" and any(
        track.tcp_from_front is None for track in episode["tracks"].values()
    ):
        raise ValueError(
            "TCP viewing requires fixed camera-to-TCP transforms for both arms"
        )
    return view


def _groups_for_view(view, episodes):
    return trajectory_overlay(
        episodes[view["episode"]]["tracks"],
        view["time"],
        **{
            key: view[key]
            for key in (
                "pose_body",
                "show_paths",
                "show_frusta",
                "full_trajectory",
                "trail_window_s",
            )
            if key in view
        },
    )


def render_asset(
    asset: Path,
    backend_repo: Path,
    view: Path,
    output: Path,
    *,
    trajectories: Path | None = None,
) -> list[Path]:
    """Render one saved view object or a JSON list, without a Viser server."""
    from PIL import Image

    _, _, checkpoint, episodes = load_asset(asset, trajectories)
    views = json.loads(Path(view).read_text())
    multiple = isinstance(views, list)
    views = views if multiple else [views]
    views = [validate_view(item, episodes) for item in views]
    renderer = NativeRenderer(Path(backend_repo), checkpoint)
    outputs = []
    for index, item in enumerate(views):
        target = (
            Path(output) / f"{index:05d}.png"
            if multiple or Path(output).suffix.lower() != ".png"
            else Path(output)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        pose = np.asarray(item["camera_pose"])
        rgb = renderer.render(pose, item["width"], item["height"], item["fov"])
        rgb = composite_overlay(
            rgb, pose, item["fov"], _groups_for_view(item, episodes)
        )
        Image.fromarray(rgb).save(target)
        target.with_suffix(".view.json").write_text(json.dumps(item, indent=2) + "\n")
        outputs.append(target)
    return outputs


def serve_asset(
    asset: Path,
    backend_repo: Path,
    *,
    trajectories: Path | None = None,
    output: Path | None = None,
    host="127.0.0.1",
    port=8895,
    max_resolution=1920,
    max_fps=15,
    pose_body="camera",
):
    _, frames, checkpoint, episodes = load_asset(asset, trajectories)
    _serve(
        frames,
        checkpoint,
        backend_repo,
        episodes,
        output or Path(asset) / "viewer",
        host,
        port,
        max_resolution,
        max_fps,
        pose_body,
        title=Path(asset).name,
    )


def _serve(
    frames,
    checkpoint,
    backend_repo,
    episodes,
    output,
    host,
    port,
    max_resolution,
    max_fps,
    pose_body,
    *,
    title,
):
    import viser
    from PIL import Image

    if not 480 <= max_resolution <= 1920 or not 1 <= max_fps <= 30:
        raise ValueError("Render width must be 480..1920 and FPS 1..30")
    if pose_body not in ("camera", "tcp"):
        raise ValueError("pose_body must be camera or tcp")
    tcp_available = all(
        track.tcp_from_front is not None
        for ep in episodes.values()
        for track in ep["tracks"].values()
    )
    if pose_body == "tcp" and not tcp_available:
        raise ValueError("TCP viewing requires fixed camera-to-TCP transforms")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    renderer = NativeRenderer(Path(backend_repo).resolve(), Path(checkpoint).resolve())
    names = tuple(episodes)
    first = names[0]
    initial_views = {
        name: focus_pose(ep, ep["initial_time"], frames)
        for name, ep in episodes.items()
    }
    initial = initial_views[first]
    initial_width = min(1440, max_resolution)
    server = viser.ViserServer(host=host, port=port, label=f"KIWI · {title}")
    if server.get_port() != port:
        server.stop()
        raise RuntimeError(f"Viewer port {port} is occupied")
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False
    server.gui.configure_theme(dark_mode=True, show_logo=False, show_share_button=False)
    server.initial_camera.position = initial[:3, 3]
    server.initial_camera.look_at = initial[:3, 3] + initial[:3, 2] * 1.5
    server.initial_camera.up_direction = -initial[:3, 1]
    server.initial_camera.fov = math.radians(70)
    clients, lock = {}, threading.RLock()

    @server.on_client_connect
    def connect(client):
        ep = episodes[first]
        client.gui.add_markdown(
            f"**{title} · {renderer.method}**\n\nBlue: left · Green: right. Drag to orbit; right-drag to pan; scroll to zoom."
        )
        episode_ui = client.gui.add_dropdown(
            "Episode", options=names, initial_value=first
        )
        timeline = client.gui.add_slider(
            "Time (s)",
            min=ep["low"],
            max=ep["high"],
            step=0.01,
            initial_value=ep["initial_time"],
        )
        play = client.gui.add_checkbox("Play trajectory", initial_value=False)
        speed = client.gui.add_slider(
            "Playback speed", min=0.25, max=8.0, step=0.25, initial_value=1.0
        )
        body_ui = client.gui.add_dropdown(
            "Path body",
            options=("camera", "tcp") if tcp_available else ("camera",),
            initial_value=pose_body,
        )
        if not tcp_available:
            client.gui.add_markdown(
                "TCP paths unavailable: this asset has no fixed tool-tip transform for every arm."
            )
        paths_ui = client.gui.add_checkbox("Show paths", initial_value=True)
        frusta_ui = client.gui.add_checkbox(
            "Show front / rear poses", initial_value=True
        )
        full_ui = client.gui.add_checkbox("Full trajectory", initial_value=False)
        window_ui = client.gui.add_slider(
            "Trail window (s)", min=1, max=90, step=1, initial_value=20
        )
        lens_ui = client.gui.add_dropdown(
            "Jump to lens",
            options=("Left front", "Left rear", "Right front", "Right rear"),
            initial_value="Left rear",
        )
        jump = client.gui.add_button("Go to selected pose")
        overview = client.gui.add_button("Overview")
        focus = client.gui.add_button("Focus on hands")
        reset = client.gui.add_button("Reset view")
        resolution = client.gui.add_slider(
            "Render width",
            min=480,
            max=max_resolution,
            step=120,
            initial_value=initial_width,
        )
        label = client.gui.add_markdown("Rendered: waiting for first frame")
        screenshot = client.gui.add_button("Save PNG")
        save = client.gui.add_button("Save view")
        view_path = client.gui.add_text("View file", initial_value="")
        restore = client.gui.add_button("Load view")
        notice = client.gui.add_markdown("")
        client.gui.add_markdown(
            "The environment is static; selecting another episode changes only its measured trajectories. Camera paths show optical centres; TCP paths use the fixed tool transform. Frusta illustrate orientation, not the full fisheye FOV. Lines draw over the scene without depth occlusion."
        )
        client.camera.near, client.camera.far = 0.01, 100.0
        client.camera.min_orbit_distance, client.camera.max_orbit_distance = 0.05, 100.0
        state = dict(
            client=client,
            pose=initial.copy(),
            fov=math.radians(70),
            aspect=1.5,
            version=0,
            rendered=-1,
            episode=first,
            generation=0,
            timeline=timeline,
            play=play,
            speed=speed,
            resolution=resolution,
            groups=[],
            snapshot=None,
            frame=None,
            image=None,
            cache_key=None,
            cache=None,
            label=label,
        )

        def build_overlay():
            selected = episodes[state["episode"]]
            snapshot = dict(
                episode=state["episode"],
                time=float(np.clip(timeline.value, selected["low"], selected["high"])),
                pose_body=body_ui.value,
                show_paths=bool(paths_ui.value),
                show_frusta=bool(frusta_ui.value),
                full_trajectory=bool(full_ui.value),
                trail_window_s=float(window_ui.value),
            )
            state.update(
                groups=_groups_for_view(snapshot, episodes),
                snapshot=snapshot,
                version=state["version"] + 1,
            )

        def update_overlay(_=None):
            with lock:
                build_overlay()

        def apply_pose(pose):
            with client.atomic():
                client.camera.position = pose[:3, 3]
                client.camera.look_at = pose[:3, 3] + pose[:3, 2] * 1.5
                client.camera.up_direction = -pose[:3, 1]
            state.update(pose=pose.copy(), version=state["version"] + 1)

        for control in (timeline, body_ui, paths_ui, frusta_ui, full_ui, window_ui):
            control.on_update(update_overlay)

        @client.camera.on_update
        def moved(camera):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(
                np.asarray(camera.wxyz)[[1, 2, 3, 0]]
            ).as_matrix()
            pose[:3, 3] = camera.position
            with lock:
                state.update(
                    pose=pose,
                    fov=float(camera.fov),
                    aspect=float(camera.aspect),
                    version=state["version"] + 1,
                )

        @resolution.on_update
        def resized(_):
            with lock:
                state["version"] += 1

        def select_episode(name):
            selected = episodes[name]
            play.value = False
            state.update(episode=name, generation=state["generation"] + 1)
            timeline.min, timeline.max = selected["low"], selected["high"]
            timeline.value = selected["initial_time"]
            build_overlay()

        @episode_ui.on_update
        def switched(_):
            with lock, client.atomic():
                if episode_ui.value != state["episode"]:
                    select_episode(episode_ui.value)
                    apply_pose(initial_views[state["episode"]])

        @jump.on_click
        def go(_):
            with lock:
                arm, lens = lens_ui.value.lower().split()
                track = episodes[state["episode"]]["tracks"].get(arm)
                pose = (
                    track.at(timeline.value, "rear" if lens == "rear" else "camera")
                    if track
                    else None
                )
                if pose is None:
                    notice.content = "No supported camera pose at this time."
                else:
                    apply_pose(pose)

        @reset.on_click
        def reset_view(_):
            with lock:
                apply_pose(initial_views[state["episode"]])

        @focus.on_click
        def focus_hands(_):
            with lock:
                apply_pose(
                    focus_pose(episodes[state["episode"]], timeline.value, frames)
                )

        @overview.on_click
        def above(_):
            with lock:
                points = [
                    track.positions
                    for track in episodes[state["episode"]]["tracks"].values()
                ]
                center = (
                    np.median(np.concatenate(points), axis=0)
                    if points
                    else initial[:3, 3]
                )
                apply_pose(look_at(center + [1.1, 1.1, 1.5], center))

        @screenshot.on_click
        def screenshot_clicked(_):
            with lock:
                if state["image"] is None:
                    return
                path = output / f"capture-{time.time_ns()}.png"
                Image.fromarray(state["image"]).save(path)
                write_json(path.with_suffix(".view.json"), state["frame"])
                notice.content = f"Saved `{path}`"

        @save.on_click
        def save_clicked(_):
            with lock:
                if state["frame"] is None:
                    return
                path = output / f"view-{time.time_ns()}.json"
                write_json(path, state["frame"])
                view_path.value = str(path)
                notice.content = f"Saved `{path}`"

        @restore.on_click
        def load_clicked(_):
            try:
                path = Path(view_path.value).expanduser()
                if not path.is_absolute():
                    path = output / path
                view = validate_view(json.loads(path.read_text()), episodes)
                if int(view["width"]) > max_resolution:
                    raise ValueError(
                        "Saved view exceeds this viewer render-width limit"
                    )
                with lock, client.atomic():
                    select_episode(view["episode"])
                    episode_ui.value = view["episode"]
                    timeline.value = view["time"]
                    body_ui.value = view.get("pose_body", "camera")
                    paths_ui.value = view.get("show_paths", True)
                    frusta_ui.value = view.get("show_frusta", True)
                    full_ui.value = view.get("full_trajectory", False)
                    window_ui.value = view.get("trail_window_s", 20.0)
                    resolution.value = view["width"]
                    state["fov"] = view["fov"]
                    client.camera.fov = view["fov"]
                    apply_pose(np.asarray(view["camera_pose"]))
                    build_overlay()
                    notice.content = f"Loaded `{path}`; playback paused."
            except (OSError, ValueError, KeyError) as exc:
                notice.content = f"Cannot load view: {exc}"

        with lock:
            build_overlay()
            state["build_overlay"] = build_overlay
            clients[client.client_id] = state

    @server.on_client_disconnect
    def disconnect(client):
        with lock:
            clients.pop(client.client_id, None)

    print(f"READY http://{host}:{port}", flush=True)
    last = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            dt, last = min(now - last, 0.2), now
            with lock:
                current = list(clients.values())
            for state in current:
                with lock:
                    if clients.get(state["client"].client_id) is not state:
                        continue
                    ep = episodes[state["episode"]]
                    if state["play"].value:
                        state["timeline"].value = ep["low"] + (
                            state["timeline"].value
                            - ep["low"]
                            + dt * state["speed"].value
                        ) % (ep["high"] - ep["low"])
                        state["build_overlay"]()
                    if state["rendered"] == state["version"]:
                        continue
                    pose, version, fov, aspect = (
                        state["pose"].copy(),
                        state["version"],
                        state["fov"],
                        state["aspect"],
                    )
                    width, generation = (
                        int(state["resolution"].value),
                        state["generation"],
                    )
                    groups, snapshot = state["groups"], state["snapshot"]
                height = max(120, round(width / max(0.25, aspect)))
                key = (pose.tobytes(), width, height, fov)
                if state["cache_key"] != key:
                    state["cache"] = renderer.render(pose, width, height, fov)
                    state["cache_key"] = key
                rgb = composite_overlay(state["cache"], pose, fov, groups)
                with lock:
                    # Camera/time changes accept a coherent older frame; episode
                    # changes reject it so old tracks never appear in the new selection.
                    if (
                        state["generation"] != generation
                        or clients.get(state["client"].client_id) is not state
                    ):
                        continue
                    view = dict(
                        camera_pose=pose.tolist(),
                        fov=fov,
                        width=width,
                        height=height,
                        **snapshot,
                    )
                    with state["client"].atomic():
                        state["client"].scene.set_background_image(
                            rgb, format="jpeg", jpeg_quality=92
                        )
                        state[
                            "label"
                        ].content = f"Rendered: **{width} × {height}** · {snapshot['episode']} · {snapshot['time']:.2f} s"
                    state.update(rendered=version, frame=view, image=rgb)
            time.sleep(max(0.005, 1 / max_fps - (time.monotonic() - now)))
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--backend-repo", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--views", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8895)
    parser.add_argument("--max-resolution", type=int, default=1920)
    parser.add_argument("--max-fps", type=int, default=15)
    parser.add_argument("--pose-body", choices=("camera", "tcp"), default="camera")
    args = parser.parse_args()
    if args.views:
        paths = render_asset(
            args.asset,
            args.backend_repo,
            args.views,
            args.output or args.asset / "renders",
            trajectories=args.trajectories,
        )
        print("\n".join(map(str, paths)))
    else:
        serve_asset(
            args.asset,
            args.backend_repo,
            trajectories=args.trajectories,
            output=args.output,
            host=args.host,
            port=args.port,
            max_resolution=args.max_resolution,
            max_fps=args.max_fps,
            pose_body=args.pose_body,
        )


if __name__ == "__main__":
    main()

"""Generate an episode review in an isolated process, with cached proxies."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from kiwi.app.media import prefetch_videos, review_video
from kiwi.app.scene import log_track, manifest_path, scene_blueprint
from kiwi.app.sources import preview_streams, viz_sources
from kiwi.app.splats import attached_scene, attachment_path, log_scene

# The GUI runs this module as a subprocess and tails its output; progress
# travels as a line on stdout.
PROGRESS_PREFIX = "@@kiwi-progress "


def report(fraction: float, message: str) -> None:
    """Announce one step of the render for the job's progress bar."""
    print(f"{PROGRESS_PREFIX}{fraction:.4f} {message}", flush=True)


def review_key(episode: Path, cache: Path | None = None) -> str:
    """Cheap cache invalidation for the files this review reads."""
    paths = [episode / "episode.yaml"]
    if cache is not None:
        paths.append(attachment_path(episode, cache))
        scene = attached_scene(episode, cache)
        if scene:
            paths.extend(Path(scene[key]) for key in ("ply", "manifest"))
    for directory in episode.iterdir():
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        paths.extend(p for p in directory.iterdir() if p.is_file())
        scene = directory / "scene"
        if scene.is_dir():
            paths.extend(p for p in scene.iterdir() if p.is_file())
    paths.extend(
        [
            Path(__file__),
            Path(__file__).with_name("scene.py"),
            Path(__file__).with_name("sources.py"),
            Path(__file__).with_name("media.py"),
            Path(__file__).with_name("splats.py"),
            Path(__file__).parents[1] / "viz" / "player.py",
            Path(__file__).parents[1] / "viz" / "proxies.py",
        ]
    )
    stamps = [
        (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(set(paths))
        if p.is_file()
    ]
    try:
        sdk_version = version("rerun-sdk")
    except PackageNotFoundError:
        sdk_version = None
    return hashlib.sha256(json.dumps([sdk_version, stamps]).encode()).hexdigest()[:24]


@dataclasses.dataclass
class Render:
    episode: Path
    output: Path
    cache: Path

    def run(self):
        import numpy as np
        import rerun as rr

        from kiwi.core.trajectory import load_tum
        from kiwi.track.orbslam.observations import read_localization_rows
        from kiwi.viz.player import _summary_markdown, log_episode, plan_layers
        from kiwi.viz.style import stream_color

        sources = viz_sources(self.episode).sources
        if sources.clock is None:
            raise ValueError(
                "Synchronize this episode before preparing a combined review. "
                "Its episode.yaml has no synchronization clock."
            )
        sources = dataclasses.replace(
            sources, name=f"{self.episode.parent.name}/{sources.name}"
        )
        timed, provisional, missing = preview_streams(sources)
        notes = (
            [
                f"Unsynchronized streams: {', '.join(missing)}. Review their recordings separately."
            ]
            if missing
            else []
        )
        notes.extend(
            f"{label}: provisional audio alignment {offset:+.4f} s from "
            f"{label}/offset_s.txt; not recorded in episode.yaml sync."
            for label, offset in provisional.items()
        )
        # Build every preview up front, together.
        prefetch_videos(
            [(stream, 0) for stream in timed.streams]
            + [(arm, 1) for arm in timed.arms],
            self.cache,
            on_progress=lambda done, total: report(
                0.05 + 0.70 * done / total,
                f"Preparing videos ({done} of {total} ready)"
                if done < total
                else "Videos ready",
            ),
        )
        plan = plan_layers(
            timed, resolve_video=lambda stream: review_video(stream, self.cache)
        )
        plan = dataclasses.replace(
            plan,
            notes=tuple(
                notes
                + [
                    n
                    for n in plan.notes
                    if not n.startswith(
                        ("world-frame layer:", "ground plane + camera height:")
                    )
                ]
            ),
        )
        if not plan.layers:
            raise ValueError(
                "No readable videos or trajectories were found. Check the recordings "
                "named in episode.yaml, or import the missing recordings."
            )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        report(0.78, "Building visualization layers")
        # GUI tracks include short trails and selectable full trajectories.
        # Do not also log the CLI player's unconditional static paths.
        log_episode(
            dataclasses.replace(
                plan, layers=tuple(ly for ly in plan.layers if ly.kind != "trajectory")
            ),
            save=self.output,
        )
        rr.log(
            "summary",
            rr.TextDocument(_summary_markdown(plan), media_type="text/markdown"),
            static=True,
        )
        videos = [
            (
                f"{ly.label} · provisional" if ly.label in provisional else ly.label,
                f"streams/{ly.label}",
            )
            for ly in plan.of_kind("video")
        ]
        tracks = []

        def track(*args, **kwargs):
            layer = log_track(rr, *args, **kwargs)
            if layer is not None:
                tracks.append(layer)

        for layer in plan.of_kind("trajectory"):
            t, p, q, _ = load_tum(layer.path)
            track(
                f"{layer.label}_fused_front",
                f"{layer.label.capitalize()} · fused_front",
                t,
                p,
                q,
                stream_color(layer.label),
                default=True,
            )

        report(0.88, "Adding trajectories and tool state")
        for arm in timed.arms:
            rear = review_video(arm, self.cache, lens=1)
            if rear is not None:
                ent = f"streams/{arm.label}_rear"
                asset = rr.AssetVideo(path=rear)
                rr.log(ent, asset, static=True)
                ts = np.asarray(asset.read_frame_timestamps_nanos())
                rr.send_columns(
                    ent,
                    indexes=[rr.TimeColumn("global", duration=arm.offset_s + ts / 1e9)],
                    columns=rr.VideoFrameReference.columns_nanos(ts),
                )
                videos.append((f"{arm.label} rear", ent))
            for role in (f"front_{arm.label}", "shared_rear"):
                path = arm.stream_dir / "scene" / f"{role}_observations.txt"
                if not path.is_file():
                    continue
                rows = read_localization_rows(path)
                keep = rows.tracked
                track(
                    f"{arm.label}_{role}_localized",
                    f"{arm.label.capitalize()} · {'front' if role.startswith('front') else 'rear'} localized",
                    rows.t_ns[keep] / 1e9 + arm.offset_s,
                    rows.p[keep],
                    rows.q_xyzw[keep],
                    (150, 160, 170),
                )
        head = self.episode / "head" / "head_marker.tum"
        if head.is_file():
            t, p, q, _ = load_tum(head)
            track(
                "head_marker",
                "Head · marker",
                np.asarray(t),
                np.asarray(p),
                np.asarray(q),
                (225, 60, 170),
                default=True,
            )
        report(0.96, "Finishing the recording")
        spatial_layers = log_scene(rr, self.episode, self.cache)
        manifest = {
            "version": 1,
            "application_id": rr.get_application_id(),
            "recording_id": rr.get_recording_id(),
            "tracks": sorted(tracks, key=lambda t: not t["default"]),
            "spatial_layers": spatial_layers,
            "videos": videos,
            "tools": [
                (ly.label, f"tool_state/{ly.label}")
                for ly in plan.of_kind("tool_state")
            ],
        }
        # A file can finish streaming after the GUI initializes its live layout.
        # Supply the standalone default without replacing an active GUI blueprint.
        rr.send_blueprint(scene_blueprint(manifest), make_active=False)
        manifest_path(self.output).write_text(json.dumps(manifest))
        print(f"Prepared {self.output}", flush=True)


if __name__ == "__main__":
    import tyro

    tyro.cli(Render).run()

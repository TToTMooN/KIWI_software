"""Scene tracks and the blueprint shared by recording generation and layer controls."""

from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path

TRAIL_SECONDS = 3.0
MAX_TRACK_GAP = 0.5
LAYER_KINDS = ("pose", "trail", "trajectory")


def manifest_path(recording: Path) -> Path:
    return recording.with_suffix(".layers.json")


def track_samples(t, p, q):
    """Keep valid poses and split paths at clock discontinuities or tracking gaps."""
    import numpy as np

    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    keep = (
        np.isfinite(t)
        & np.all(np.isfinite(p), axis=1)
        & np.all(np.isfinite(q), axis=1)
        & (np.linalg.norm(q, axis=1) > 0)
    )
    t, p, q = t[keep], p[keep], q[keep]
    connected = (np.diff(t) > 0) & (np.diff(t) <= MAX_TRACK_GAP)
    segments = [
        part for part in np.split(p, np.flatnonzero(~connected) + 1) if len(part) > 1
    ]
    trails = np.stack([p[:-1], p[1:]], axis=1)[connected]
    return t, p, q, segments, t[1:][connected], trails


def log_track(rr, entity, label, t, p, q, color, *, default=False):
    t, p, q, segments, trail_t, trails = track_samples(t, p, q)
    if not len(t):
        return None
    paths = {"pose": f"world/camera/{entity}"}
    rr.log(paths["pose"], rr.TransformAxes3D(axis_length=0.08), static=True)
    rr.send_columns(
        paths["pose"],
        indexes=[rr.TimeColumn("global", duration=t)],
        columns=rr.Transform3D.columns(translation=p, quaternion=q),
    )
    if segments:
        paths["trajectory"] = f"world/path/{entity}"
        rr.log(
            paths["trajectory"],
            rr.LineStrips3D(segments, colors=color, radii=0.003),
            static=True,
        )
        paths["trail"] = f"world/trail/{entity}"
        rr.log(
            paths["trail"],
            rr.LineStrips3D.from_fields(colors=[color], radii=[0.002]),
            static=True,
        )
        rr.send_columns(
            paths["trail"],
            indexes=[rr.TimeColumn("global", duration=trail_t)],
            columns=rr.LineStrips3D.columns(strips=trails),
        )
    return {
        "id": entity,
        "label": label,
        "color": list(color),
        "paths": paths,
        "default": default,
    }


def default_visibility(manifest):
    return {
        **{
            path: bool(track["default"] and kind != "trajectory")
            for track in manifest["tracks"]
            for kind, path in track["paths"].items()
        },
        **{
            layer["path"]: bool(layer["default"])
            for layer in manifest.get("spatial_layers", [])
        },
    }


def scene_blueprint(manifest, visibility=None):
    import rerun.blueprint as rrb

    choices = default_visibility(manifest)
    if visibility is not None:
        choices.update(
            {path: bool(value) for path, value in visibility.items() if path in choices}
        )
    overrides = {
        layer["path"]: [rrb.EntityBehavior(visible=choices[layer["path"]])]
        for layer in manifest.get("spatial_layers", [])
    }
    for track in manifest["tracks"]:
        for kind, path in track["paths"].items():
            settings = [rrb.EntityBehavior(visible=choices[path])]
            if kind == "trail":
                settings.append(
                    rrb.VisibleTimeRanges(
                        rrb.VisibleTimeRange(
                            "global",
                            start=rrb.TimeRangeBoundary.cursor_relative(
                                seconds=-TRAIL_SECONDS
                            ),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    )
                )
            overrides[path] = settings

    def identified(part, name):
        # Keep view/container identities consistent across recording generation
        # and the initial live-view setup.
        part.id = uuid.uuid5(uuid.NAMESPACE_URL, f"{manifest['application_id']}/{name}")
        return part

    scene_options = {}
    for layer in manifest.get("spatial_layers", []):
        if layer.get("eye"):
            scene_options["eye_controls"] = rrb.EyeControls3D(**layer["eye"])
            break
    scene = identified(
        rrb.Spatial3DView(
            name="Scene", origin="world", overrides=overrides, **scene_options
        ),
        "scene",
    )
    videos = [
        identified(rrb.Spatial2DView(name=label, origin=origin), f"video/{origin}")
        for label, origin in manifest["videos"]
    ]
    tools = [
        identified(
            rrb.TimeSeriesView(name=f"{label} tool", origin=origin), f"tool/{origin}"
        )
        for label, origin in manifest["tools"]
    ]
    side = []
    if videos:
        side.append(
            identified(
                rrb.Grid(*videos, grid_columns=min(2, len(videos)), name="Videos"),
                "videos",
            )
        )
    side.extend(tools)
    side.append(
        identified(rrb.TextDocumentView(name="Episode", origin="summary"), "summary")
    )
    root = identified(rrb.Tabs(*side, active_tab=0), "preview")
    if manifest["tracks"] or manifest.get("spatial_layers"):
        root = identified(
            rrb.Horizontal(
                scene,
                identified(
                    rrb.Vertical(
                        *side,
                        row_shares=[3] + [1] * (len(side) - 1) if videos else None,
                    ),
                    "side",
                ),
                column_shares=[3, 2],
            ),
            "root",
        )
    return rrb.Blueprint(
        root,
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed", timeline="global"),
    )


@lru_cache(maxsize=4)
def _gaussian_colors(recording_path: Path, mtime_ns: int):
    """Keep only the small color arrays from the immutable recording on disk."""
    from rerun.experimental import RrdReader

    colors = []
    for chunk in RrdReader(recording_path).stream():
        batch = chunk.to_record_batch()
        if "GaussianSplats3D:colors" in batch.schema.names:
            path = batch.schema.metadata[b"rerun:entity_path"].decode()
            values = batch.column("GaussianSplats3D:colors")[0].values.to_numpy().copy()
            colors.append((path, values))
    return colors


def layer_update_bytes(manifest, visibility, recording_path: Path, *, initialize=False):
    """Update display styles in-place, retaining the viewer's camera and layout.

    Initialize the live view once, then change only splat colors and line/axis
    sizes in its data stream. The downloadable RRD keeps its native blueprint
    visibility controls and original Gaussian colors unchanged.
    """
    import rerun as rr

    recording = rr.RecordingStream(
        manifest["application_id"], recording_id=manifest["recording_id"]
    )
    sink = recording.binary_stream()
    for track in manifest["tracks"]:
        for kind, path in track["paths"].items():
            if path not in visibility:
                continue
            shown = visibility[path]
            style = (
                rr.TransformAxes3D(axis_length=0.08 if shown else 0)
                if kind == "pose"
                else rr.LineStrips3D.from_fields(
                    radii=[(0.002 if kind == "trail" else 0.003) if shown else 0]
                )
            )
            recording.log(path, style, static=True)
    for layer in manifest.get("spatial_layers", []):
        if layer["path"] not in visibility:
            continue
        for path, colors in _gaussian_colors(
            recording_path, recording_path.stat().st_mtime_ns
        ):
            if path.startswith("/" + layer["path"] + "/"):
                recording.log(
                    path,
                    rr.GaussianSplats3D.from_fields(
                        colors=colors if visibility[layer["path"]] else [0]
                    ),
                    static=True,
                )
    if initialize:
        recording.send_blueprint(
            scene_blueprint(
                manifest, {path: True for path in default_visibility(manifest)}
            ),
            make_default=False,
        )
    return sink.read()

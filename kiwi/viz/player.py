"""The rerun episode player: supported layers on one global clock.

``kiwi viz play`` logs an episode into a rerun recording and lets the
viewer own playback. The module is split in two:

* :func:`plan_layers` is pure - given resolved :class:`~kiwi.viz.sources.
  EpisodeSources` and a video resolver, it decides which layers exist,
  and notes missing artifacts and layers the player does not draw.
* :func:`log_episode` takes that plan and speaks rerun (lazy import).

One timeline, ``global``, carrying seconds on the episode global clock;
every layer is placed on it, and nothing is logged on a private clock:

* **video** - the H.264 rendition as an ``AssetVideo`` plus frame
  references at ``offset + media_time``, so the viewer decodes lazily;
* **trajectory** - each arm's front-camera path from ``scene/state.tum``
  as a static line strip, plus moving pose axes. The player does not resolve
  the intrinsics needed to draw a camera frustum;
* **tool state** - raw value and the completed output from ``eef_state_v1``
  at ``offset + i / fps``. Non-finite raw samples are omitted; completion
  metadata colors filled spans.

The summary also lists existing artifact types this player does not draw:
marker-anchor trajectories, the optional head-marker trajectory, and ground
results. A head-marker pose is not an ego-camera pose; extra camera streams
contribute video only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np

from kiwi.viz.sources import EpisodeSources, StreamSource
from kiwi.viz.style import stream_color

__all__ = [
    "TIMELINE",
    "EpisodePlan",
    "Layer",
    "log_episode",
    "plan_layers",
]

TIMELINE = "global"

# Existing artifact types this player does not load. These notes describe
# viewer coverage, independently of whether their optional stages have run.
_PENDING = (
    "world-frame layer: marker-anchor <arm>/register.tum and "
    "head/head_marker.tum are not drawn; head-marker poses are not ego-camera poses",
    "ground plane + camera height: not drawn; the ground stage writes "
    "ground/ground_report.json and ground/ground_observations.npz",
)


@dataclasses.dataclass(frozen=True)
class Layer:
    """One thing to log: a kind, the stream it belongs to, its artifact."""

    kind: str  # "video" | "trajectory" | "tool_state"
    label: str
    path: Path
    offset_s: float = 0.0


@dataclasses.dataclass(frozen=True)
class EpisodePlan:
    """What the player will log, and what it cannot."""

    sources: EpisodeSources
    layers: tuple[Layer, ...]
    notes: tuple[str, ...]

    def of_kind(self, kind: str) -> tuple[Layer, ...]:
        return tuple(ly for ly in self.layers if ly.kind == kind)

    def summary(self) -> str:
        counts = {
            k: len(self.of_kind(k)) for k in ("video", "trajectory", "tool_state")
        }
        head = ", ".join(f"{n} {k}" for k, n in counts.items())
        return f"{self.sources.name}: {head}" + (
            f"; {len(self.notes)} note(s)" if self.notes else ""
        )


def plan_layers(
    sources: EpisodeSources,
    *,
    resolve_video: Callable[[StreamSource], Path | None],
) -> EpisodePlan:
    """Decide which layers this episode can show, and note every absence.

    ``resolve_video`` maps a stream to a viewer-decodable path (for example
    :func:`kiwi.viz.proxies.playable_video`, injected here so planning stays
    free of ffmpeg). Absent artifacts are never an error: the note says which
    stage fills the hole.
    """
    layers: list[Layer] = []
    notes: list[str] = []

    if sources.clock is None:
        notes.append(
            "no sync block: every stream is placed at offset 0, so streams "
            "that started at different times will not line up (run `kiwi run "
            "--from-stage sync --to-stage sync`)"
        )

    for stream in sources.streams:
        video = resolve_video(stream)
        if video is None and not stream.video.is_file():
            notes.append(
                f"video {stream.label}: no "
                f"{stream.video.parent.name}/{stream.video.name} (run "
                f"`kiwi run --from-stage convert --to-stage convert`)"
            )
        elif video is None:
            # The stream exists but is not decodable as it stands - the
            # cameras record HEVC - and no rendition was built.
            notes.append(
                f"video {stream.label}: {stream.video.name} needs an H.264 "
                f"rendition the viewer can decode; re-run without "
                f"--no-proxies (it needs ffmpeg)"
            )
        else:
            layers.append(Layer("video", stream.label, video, stream.offset_s))

    for arm in sources.arms:
        tum = sources.tum_path(arm.label)
        if tum.is_file():
            # Already on the global clock: the state stage declares that in
            # the file header, so nothing is re-based here.
            layers.append(Layer("trajectory", arm.label, tum, 0.0))
        else:
            notes.append(
                f"trajectory {arm.label}: no {tum.parent.name}/{tum.name} "
                f"(run `kiwi run --from-stage state --to-stage state`)"
            )

        state = _eef_state_path(arm)
        if state.is_file():
            layers.append(Layer("tool_state", arm.label, state, arm.offset_s))
        else:
            notes.append(
                f"tool state {arm.label}: no {state.parent.name}/{state.name} "
                f"(run `kiwi run --from-stage eef --to-stage eef`)"
            )

    notes.extend(_PENDING)
    return EpisodePlan(sources=sources, layers=tuple(layers), notes=tuple(notes))


def _eef_state_path(stream: StreamSource) -> Path:
    from kiwi.core import layout

    return Path(stream.stream_dir) / layout.EEF_STATE


def _load_rerun():
    """Import rerun, or raise the install hint instead of a traceback."""
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise RuntimeError(
            "the episode player needs rerun-sdk, an optional dependency: "
            "`pixi install -e viz` (or `pip install 'rerun-sdk>=0.36.3,<0.37'`)"
        ) from exc
    return rr, rrb


def _summary_markdown(plan: EpisodePlan) -> str:
    src = plan.sources
    lines = [f"# {src.name}", ""]
    if src.description:
        lines += [src.description, ""]
    if src.date:
        lines.append(f"**date** {src.date}")
    if src.tool:
        lines.append(f"**tool** `{src.tool}`")
    if src.clock is not None:
        lines.append(f"**clock** origin `{src.clock.origin}` (t=0 at its frame 0)")
    if src.task_window:
        lines.append(f"**task window** {src.task_window[0]} .. {src.task_window[1]} s")
    lines += ["", "| stream | kind | offset (s) |", "| --- | --- | --- |"]
    for s in src.streams:
        lines.append(f"| {s.label} | {s.kind} | {s.offset_s:+.3f} |")
    lines += ["", "## layers", ""]
    for kind in ("video", "trajectory", "tool_state"):
        got = ", ".join(ly.label for ly in plan.of_kind(kind)) or "-"
        lines.append(f"- **{kind}**: {got}")
    if plan.notes:
        lines += ["", "## not shown", ""]
        lines += [f"- {n}" for n in plan.notes]
    return "\n".join(lines)


def _log_video(rr, layer: Layer) -> None:
    entity = f"streams/{layer.label}"
    asset = rr.AssetVideo(path=layer.path)
    rr.log(entity, asset, static=True)
    ts_ns = np.asarray(asset.read_frame_timestamps_nanos())
    if ts_ns.size == 0:
        return
    rr.send_columns(
        entity,
        indexes=[rr.TimeColumn(TIMELINE, duration=layer.offset_s + ts_ns / 1e9)],
        columns=rr.VideoFrameReference.columns_nanos(ts_ns),
    )


def _log_trajectory(rr, layer: Layer) -> None:
    from kiwi.core.trajectory import load_tum

    t, p, q_xyzw, _ = load_tum(layer.path)
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    rr.log(
        f"world/path/{layer.label}",
        rr.LineStrips3D([p], colors=stream_color(layer.label)),
        static=True,
    )
    # A moving pose triad.
    entity = f"world/camera/{layer.label}"
    rr.log(entity, rr.TransformAxes3D(axis_length=0.08), static=True)
    rr.send_columns(
        entity,
        indexes=[rr.TimeColumn(TIMELINE, duration=t)],
        columns=rr.Transform3D.columns(
            translation=p, quaternion=np.asarray(q_xyzw, dtype=np.float64)
        ),
    )


# The completed channel is finite everywhere. It is coloured per sample by
# the gap severity the eef completion pass assigned.
GAP_COLORS = {
    "short": (240, 200, 60),
    "medium": (235, 140, 45),
    "long": (215, 60, 45),
}
OBSERVED_COLOR = (70, 85, 105)
RAW_COLOR = (160, 170, 180)
OBSERVED_WIDTH = 2.0
GAP_WIDTH = 4.0


def _gap_style(state, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-sample colour and width for the completed channel, or ``None``.

    ``None`` when no per-sample gap-level metadata is available, so the line
    stays a single colour without assigning a completion category.
    """
    levels = state.completion.get("gap_level")
    if levels is None:
        return None
    levels = np.asarray(levels).astype(str)
    if levels.shape != (n,):
        return None
    colors = np.tile(np.array([OBSERVED_COLOR], dtype=np.uint8), (n, 1))
    widths = np.full(n, OBSERVED_WIDTH, dtype=np.float32)
    for level, rgb in GAP_COLORS.items():
        filled = levels == level
        colors[filled] = rgb
        widths[filled] = GAP_WIDTH
    return colors, widths


def _log_tool_state(rr, layer: Layer) -> None:
    from kiwi.eef.state import read_eef_state

    state = read_eef_state(layer.path)
    n = len(state)
    t = layer.offset_s + np.arange(n) / float(state.fps)
    root = f"tool_state/{layer.label}"
    style = _gap_style(state, n)
    for name, values, width, color, gap_styled in (
        ("value", state.value, 1.0, RAW_COLOR, False),
        (
            "value_completed",
            state.output,
            OBSERVED_WIDTH,
            OBSERVED_COLOR,
            style is not None,
        ),
    ):
        values = np.asarray(values, dtype=np.float64)
        keep = np.isfinite(values)
        # A gap-styled series gets its colour and width per sample below, and
        # ONLY per sample: in rerun a static value for the same component
        # shadows every temporal one.
        line = {"names": f"{layer.label} {name} [{state.unit}]"}
        if not gap_styled:
            line.update(widths=width, colors=color)
        rr.log(f"{root}/{name}", rr.SeriesLines(**line), static=True)
        if not keep.any():
            continue
        columns = list(rr.Scalars.columns(scalars=values[keep]))
        if gap_styled:
            colors, widths = style
            columns.extend(
                rr.SeriesLines.columns(colors=colors[keep], widths=widths[keep])
            )
        # Only finite samples are logged; missing measurements never become
        # synthetic zero values.
        rr.send_columns(
            f"{root}/{name}",
            indexes=[rr.TimeColumn(TIMELINE, duration=t[keep])],
            columns=columns,
        )


def _blueprint(rrb, plan: EpisodePlan):
    video_views = [
        rrb.Spatial2DView(name=ly.label, origin=f"streams/{ly.label}")
        for ly in plan.of_kind("video")
    ]
    series_views = [
        rrb.TimeSeriesView(
            name=f"{ly.label} tool state", origin=f"tool_state/{ly.label}"
        )
        for ly in plan.of_kind("tool_state")
    ]
    left: list = [rrb.Spatial3DView(name="world", origin="world")]
    if video_views:
        left.append(rrb.Grid(*video_views))
    right: list = list(series_views)
    right.append(rrb.TextDocumentView(name="episode", origin="summary"))
    return rrb.Blueprint(
        rrb.Horizontal(rrb.Vertical(*left), rrb.Vertical(*right), column_shares=[2, 1]),
        collapse_panels=True,
    )


def log_episode(
    plan: EpisodePlan,
    *,
    save: Path | None = None,
    spawn: bool = True,
) -> None:
    """Log ``plan`` into a rerun recording (spawning the viewer by default).

    ``save`` writes an ``.rrd`` instead of opening a viewer - the shareable
    form: ``rerun episode.rrd`` reopens it with no artifacts in reach.
    """
    rr, rrb = _load_rerun()
    blueprint = _blueprint(rrb, plan)
    rr.init(
        f"kiwi - {plan.sources.name}",
        spawn=spawn and save is None,
        default_blueprint=blueprint,
    )
    if save is not None:
        rr.save(str(save), default_blueprint=blueprint)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "summary",
        rr.TextDocument(_summary_markdown(plan), media_type="text/markdown"),
        static=True,
    )

    for layer in plan.layers:
        if layer.kind == "video":
            _log_video(rr, layer)
        elif layer.kind == "trajectory":
            _log_trajectory(rr, layer)
        elif layer.kind == "tool_state":
            _log_tool_state(rr, layer)

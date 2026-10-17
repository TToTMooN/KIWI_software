"""Render an episode's streams side by side with the GLOBAL time burned in.

The shareable review artifact: one mp4 an operator can scrub, attach to a
review, or hand to someone with no repo.

Global-time model, the same one every KIWI artifact speaks: frame ``i`` of
a stream sits at ``offset + i / fps``, where ``offset`` is the global time
of that stream's frame 0. Offsets come from the episode manifest's sync
block and nowhere else.

The window is the overlap where EVERY panel has frames, with no implicit
floor at 0: on an episode where the origin is the ego stream, both arm
offsets are negative and that pre-zero overlap is real footage.

With ``eef_state_v1`` artifacts present, a strip below the panels plots
each stream's tool state against the same global clock, with a playhead
tracking the video. Each stream draws TWO curves in one hue: a thin dim
raw ``value`` under a thick bright consumer ``output`` (the completed raw
values). Keeping both visible shows missing raw observations alongside what
export consumes.

Renders whatever streams the episode declares.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

from kiwi.viz.style import bgr, stream_color

__all__ = ["Panel", "Series", "global_window", "render_synced", "tool_state_series"]


@dataclasses.dataclass(frozen=True)
class Panel:
    """One stream in the render: its video, its offset, its tool state."""

    label: str
    video: Path
    offset_s: float = 0.0
    state: Path | None = None  # an eef_state_v1 npz, when the eef stage ran


@dataclasses.dataclass(frozen=True)
class Series:
    """One curve on the strip: times on the global clock + its values."""

    label: str
    times: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    color: tuple[int, int, int]  # BGR
    thickness: int


def global_window(
    spans: Sequence[tuple[float, float]],
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple[float, float]:
    """Global ``[lo, hi]`` where EVERY span has frames; the bounds only narrow.

    ``spans`` are ``(first, last)`` global times per stream. No implicit
    floor at 0: on an ego-origin episode every arm offset is negative and
    the pre-zero overlap is real footage.
    """
    if not spans:
        raise ValueError("no streams to render")
    lo = max(s[0] for s in spans)
    hi = min(s[1] for s in spans)
    if t_start is not None:
        lo = max(lo, t_start)
    if t_end is not None:
        hi = min(hi, t_end)
    return lo, hi


def tool_state_series(panel: Panel) -> list[Series]:
    """The raw + consumer output curves for one panel, on the global clock.

    Empty when the panel has no artifact. The raw curve keeps its own mask,
    so missing observations stay visible beside the completed output, which
    is finite throughout without changing finite raw anchors.
    """
    if panel.state is None or not Path(panel.state).is_file():
        return []
    from kiwi.eef.state import read_eef_state

    state = read_eef_state(panel.state)
    t = panel.offset_s + np.arange(len(state)) / float(state.fps)
    color = bgr(stream_color(panel.label))
    dim = tuple(int(c * 0.45) for c in color)
    raw = np.asarray(state.value, dtype=np.float64)
    output = np.asarray(state.output, dtype=np.float64)
    return [
        Series(f"{panel.label} raw", t, raw, np.isfinite(raw), dim, 1),
        Series(panel.label, t, output, np.isfinite(output), color, 2),
    ]


class _Reader:
    """A video opened for forward (monotonic) frame access by index.

    Sequential reads with a backward-seek fallback.
    """

    def __init__(self, path: Path) -> None:
        from kiwi.core.video import VideoInfo, open_capture

        self.cap = open_capture(path)
        info = VideoInfo.from_capture(self.cap)
        self.fps, self.n = info.fps, info.n_frames
        self.w, self.h = info.width, info.height
        self._pos = 0
        self._last: np.ndarray | None = None
        self._last_idx = -1

    def at(self, idx: int) -> np.ndarray | None:
        idx = max(0, min(idx, self.n - 1))
        if idx < self._last_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._pos = idx
            self._last = None
        while self._pos <= idx:
            ok, frame = self.cap.read()
            self._pos += 1
            if ok:
                self._last = frame
            else:
                break
        self._last_idx = idx
        return self._last

    def release(self) -> None:
        self.cap.release()


def _text(img, s, org, scale, color=(255, 255, 255), thick=2) -> None:
    cv2.putText(
        img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA
    )
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _centered(img, s, y, scale, color=(255, 255, 255), thick=2) -> None:
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    _text(img, s, ((img.shape[1] - tw) // 2, y), scale, color, thick)


def _ticks(lo: float, extent: float, n: int) -> list[tuple[float, str]]:
    """``n`` tick values with just enough decimals to stay distinct."""
    prec = 0 if extent >= 10 else (1 if extent >= 2 else 2)
    return [
        (lo + k / (n - 1) * extent, f"{lo + k / (n - 1) * extent:.{prec}f}")
        for k in range(n)
    ]


def _render_strip(width: int, height: int, lo: float, hi: float, series: list[Series]):
    """Pre-render the static tool-state strip (cv2 only - no matplotlib).

    Returns ``(image, time_to_x, value_to_y, y_top, y_bottom)`` so the caller
    can draw the moving playhead onto a copy each frame.
    """
    img = np.full((height, width, 3), 28, np.uint8)
    ml, mr, mt, mb = 56, 14, 16, 26
    x0, x1, y0, y1 = ml, width - mr, mt, height - mb

    vals = [s.values[s.valid] for s in series if s.valid.any()]
    vmin, vmax = (
        (float(np.concatenate(vals).min()), float(np.concatenate(vals).max()))
        if vals
        else (0.0, 1.0)
    )
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    pad = 0.06 * (vmax - vmin)
    vmin, vmax = vmin - pad, vmax + pad
    span = max(1e-9, hi - lo)

    def tx(t: float) -> int:
        return int(round(x0 + (t - lo) / span * (x1 - x0)))

    def vy(v: float) -> int:
        return int(round(y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)))

    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    for v, lab in _ticks(vmin, vmax - vmin, 4):
        yy = vy(v)
        cv2.line(img, (x0 + 1, yy), (x1, yy), (48, 48, 48), 1)
        _text(img, lab, (6, yy + 4), 0.4, (170, 170, 170), 1)
    for t, lab in _ticks(lo, span, 6):
        xx = tx(t)
        cv2.line(img, (xx, y0), (xx, y1 - 1), (48, 48, 48), 1)
        _text(img, f"{lab}s", (xx - 12, y1 + 18), 0.4, (170, 170, 170), 1)

    lx = x1 - 4
    for s in series:
        seg: list[tuple[int, int]] = []
        for i in range(len(s.times)):
            if s.valid[i] and lo <= s.times[i] <= hi:
                seg.append((tx(s.times[i]), vy(s.values[i])))
            else:
                # A dropout breaks the line.
                if len(seg) >= 2:
                    cv2.polylines(
                        img,
                        [np.array(seg, np.int32)],
                        False,
                        s.color,
                        s.thickness,
                        cv2.LINE_AA,
                    )
                seg = []
        if len(seg) >= 2:
            cv2.polylines(
                img, [np.array(seg, np.int32)], False, s.color, s.thickness, cv2.LINE_AA
            )
        (tw, _), _ = cv2.getTextSize(s.label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        lx -= tw
        _text(img, s.label, (lx, y0 + 16), 0.5, s.color, 2)
        lx -= 24

    return img, tx, vy, y0, y1


def _value_at(s: Series, g: float) -> float | None:
    """The drawn value of ``s`` at global time ``g``, or None outside it."""
    t, v = s.times[s.valid], s.values[s.valid]
    if t.size == 0 or g < t[0] or g > t[-1]:
        return None
    return float(np.interp(g, t, v))


def render_synced(
    panels: Sequence[Panel],
    out: Path,
    *,
    height: int = 600,
    out_fps: float | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    strip_height: int = 200,
    unit: str = "",
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """Render ``panels`` side by side onto one global clock. Returns ``out``."""
    step = on_progress or (lambda _msg: None)
    if not panels:
        raise ValueError("no streams to render")

    readers = [_Reader(p.video) for p in panels]
    try:
        spans = []
        for panel, r in zip(panels, readers, strict=True):
            if not r.fps > 0 or r.n <= 0:
                raise RuntimeError(
                    f"{panel.video.name}: no frames or no frame rate; the clip "
                    f"may be truncated"
                )
            spans.append((panel.offset_s, panel.offset_s + r.n / r.fps))
            step(
                f"{panel.label}: {r.w}x{r.h} fps={r.fps:.3f} n={r.n} "
                f"(frame 0 @ global {panel.offset_s:+.4f}s)"
            )

        lo, hi = global_window(spans, t_start, t_end)
        if hi <= lo:
            raise RuntimeError(
                f"empty window: the streams do not overlap on the global clock "
                f"([{lo:.3f}, {hi:.3f}]s)"
            )
        fps_out = out_fps if out_fps is not None else min(r.fps for r in readers)
        n_out = int(np.floor((hi - lo) * fps_out)) + 1
        step(
            f"window: global [{lo:.3f}, {hi:.3f}]s -> {n_out} frames @ {fps_out:.3f} fps"
        )

        series = [s for p in panels for s in tool_state_series(p)]
        # A panel's live readout tracks its BRIGHT curve (what export
        # consumes), which is the last series that panel contributed.
        readout = {}
        for p in panels:
            own = [s for s in series if s.label in (p.label, f"{p.label} raw")]
            if own:
                readout[p.label] = own[-1]

        header = max(70, height // 8)
        panel_w = height
        canvas_w = panel_w * len(panels)
        canvas_h = header + height
        strip = None
        strip_h = 0
        if series:
            strip_h = strip_height
            strip_bg, tx, vy, y0, y1 = _render_strip(canvas_w, strip_h, lo, hi, series)
            strip = (strip_bg, tx, vy, y0, y1)
            step(f"tool-state strip: {len(series)} series in a {strip_h}px band")

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_out,
            (canvas_w, canvas_h + strip_h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open a VideoWriter for {out}")

        blank = np.zeros((height, panel_w, 3), np.uint8)
        sc = height / 720
        try:
            for j in range(n_out):
                g = lo + j / fps_out
                canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
                for k, (panel, r) in enumerate(zip(panels, readers, strict=True)):
                    idx = int(round((g - panel.offset_s) * r.fps))
                    frame = r.at(idx)
                    tile = (
                        cv2.resize(frame, (panel_w, height))
                        if frame is not None
                        else blank.copy()
                    )
                    x = k * panel_w
                    canvas[header:, x : x + panel_w] = tile
                    _text(canvas, panel.label, (x + 10, header + 28), sc * 0.85)
                    s = readout.get(panel.label)
                    cur = _value_at(s, g) if s is not None else None
                    if cur is not None:
                        _text(
                            canvas,
                            f"{cur:.1f}{(' ' + unit) if unit else ''}",
                            (x + 10, header + 28 + int(sc * 30)),
                            sc * 0.8,
                            s.color,
                        )
                    _text(
                        canvas,
                        f"frame {idx}",
                        (x + 10, canvas_h - 14),
                        sc * 0.72,
                        (200, 230, 255),
                    )
                _centered(
                    canvas,
                    f"GLOBAL t = {g:7.3f} s",
                    int(header * 0.7),
                    sc * 1.2,
                    (0, 255, 255),
                )
                progress_x = int(canvas_w * j / max(1, n_out - 1))
                cv2.line(
                    canvas,
                    (0, canvas_h - 3),
                    (progress_x, canvas_h - 3),
                    (0, 200, 255),
                    3,
                )

                if strip is None:
                    writer.write(canvas)
                else:
                    strip_bg, tx, vy, y0, y1 = strip
                    band = strip_bg.copy()
                    xp = tx(g)
                    cv2.line(band, (xp, y0), (xp, y1), (0, 0, 255), 2)
                    for s in series:
                        cur = _value_at(s, g)
                        if cur is None:
                            continue
                        rad = 4 if s.thickness > 1 else 3
                        cv2.circle(band, (xp, vy(cur)), rad, s.color, -1)
                        if rad > 3:  # a white ring marks the exported series
                            cv2.circle(band, (xp, vy(cur)), rad, (255, 255, 255), 1)
                    writer.write(np.vstack([canvas, band]))

                if j % 100 == 0 or j == n_out - 1:
                    step(f"  {j + 1}/{n_out} (global {g:.2f}s)")
        finally:
            writer.release()
    finally:
        for r in readers:
            r.release()

    step(
        f"wrote {out} ({canvas_w}x{canvas_h + strip_h}, {n_out} frames @ {fps_out:.2f} fps)"
    )
    return out

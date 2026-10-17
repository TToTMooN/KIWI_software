"""The interactive gripper tuner: a viser app over the REAL estimator.

The tuner runs the SAME per-frame path the batch estimator runs: every
slider feeds :func:`kiwi.eef.chopsticks.make_frame_estimator` as an override and the
frame is processed by the estimator itself (its ``debug`` hook exposes the
warped plane view and the threshold mask).

Persistence: the **export configs** button renders the tuned values as the
exact YAML block to paste into ``config/gripper/<tool>/tool.yaml`` (or a
unit's ``session:`` override), prints it, and writes it next to the clip.
The normalization endpoints the source block carried ride along, so a
wholesale paste keeps them.

viser is an optional dependency (``pixi install -e tune`` or
``pip install -e '.[tune]'`` from the checkout); this module imports without it, and
:func:`launch` fails with the install hint rather than a traceback. This is
a viz application over the eef component (a composer may import a
component); only the ``kiwi eef tune`` CLI leaf sits above both.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import yaml

from kiwi.core.tool_config import load_gripper_unit, write_unit_session
from kiwi.core.viz.preview import hold_until_interrupt
from kiwi.eef import EstimatorContext, get_estimator
from kiwi.eef.chopsticks import WARP_PX
from kiwi.eef.config import (
    DEFAULT_THRESHOLD_BY_POLARITY,
    config_from_block,
    resolve_params,
    validate_session_block,
)

__all__ = ["endpoints_from_sweep", "launch", "session_diff", "tuned_block"]

# The knobs the tuner exposes, in GUI order: (field, kind, lo, hi, step).
_SLIDERS = (
    ("threshold", "int", 0, 255, 1),
    ("min_area", "int", 1, 3000, 1),
    ("max_area", "int", 10, 8000, 10),
    ("min_circularity", "float", 0.0, 1.0, 0.01),
    ("extent_x", "float", 1.0, 10.0, 0.1),
    ("extent_y", "float", 1.0, 10.0, 0.1),
    ("roi_radius_mm", "float", 5.0, 200.0, 0.5),
    ("roi_inner_radius_mm", "float", 0.0, 150.0, 0.5),
    ("roi_center_x_mm", "float", -100.0, 100.0, 0.5),
    ("roi_center_y_mm", "float", -100.0, 100.0, 0.5),
    ("max_asymmetry_deg", "float", 0.5, 60.0, 0.5),
    ("outlier_trim_mm", "float", 0.5, 10.0, 0.1),
    ("outlier_trim_mad_k", "float", 0.5, 10.0, 0.1),
)

# GUI labels that differ from the session: key they drive. The key itself
# is the config schema (tool.yaml, DotParams) and is what the emitted block
# carries; the label is only what the slider reads as.
_SLIDER_LABELS = {"roi_radius_mm": "roi_outer_radius_mm"}

# What the emitted session: block carries - the detection/fit recipe the
# batch estimator reads. Retry/rescue/completion knobs stay out; the
# normalization endpoints pass through from the source block (export-side
# values the tuner does not touch but must not delete).
_EMIT_KEYS = (
    "polarity",
    "threshold",
    "min_area",
    "max_area",
    "min_circularity",
    "fov_deg",
    "extent_x",
    "extent_y",
    "roi_radius_mm",
    "roi_inner_radius_mm",
    "roi_center_x_mm",
    "roi_center_y_mm",
    "roi_semicircle_keep",
    "clip_to_image",
    "max_asymmetry_deg",
    "enable_outlier_trim",
    "outlier_trim_mm",
    "outlier_trim_mad_k",
)


def tuned_block(values: dict, passthrough: dict | None = None) -> str:
    """The tuned values as the exact ``session:`` YAML block to paste.

    Only the recipe keys travel (see ``_EMIT_KEYS``); ``None`` values (an
    unset annulus) are dropped. ``passthrough`` carries the source block's
    export-side keys (the normalization endpoints) unchanged. The output
    round-trips through :func:`kiwi.eef.config.config_from_block` by
    construction - that is what the CI test pins.
    """
    block = {k: values[k] for k in _EMIT_KEYS if values.get(k) is not None}
    block.update(passthrough or {})
    return yaml.safe_dump({"session": block}, sort_keys=False)


# The normalization endpoints: export-side session keys the tuner reads
# off the sweep (the full-travel stops) and writes beside the recipe.
_ENDPOINT_KEYS = ("angle_closed_deg", "angle_open_deg")


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
    return a == b


def session_diff(
    values: Mapping, tool_session: Mapping, endpoints: Mapping | None = None
) -> dict:
    """A unit's ``session:`` override: only what differs from its design.

    ``values`` is the tuned recipe (the tuner's live dict; an inner radius
    of 0.0 means no annulus), ``tool_session`` the DESIGN's ``session:``
    block. Each emitted key is compared against what the tool layer
    resolves to (builtin < tool), so a knob the tool leaves at the builtin
    still compares correctly. Equal keys stay OUT of the override, so a
    later change to the design recipe still reaches this build. ``endpoints``
    (``angle_closed_deg`` / ``angle_open_deg``) join when they differ from
    the tool's.
    """
    base = dataclasses.asdict(resolve_params(config_from_block(dict(tool_session))))
    emit = {k: values.get(k) for k in _EMIT_KEYS}
    if emit.get("roi_inner_radius_mm") == 0.0:
        emit["roi_inner_radius_mm"] = None
    diff = {k: v for k, v in emit.items() if not _same(v, base.get(k))}
    for k in _ENDPOINT_KEYS:
        v = (endpoints or {}).get(k)
        if v is not None and not _same(v, tool_session.get(k)):
            diff[k] = float(v)
    return diff


def endpoints_from_sweep(value: np.ndarray) -> tuple[float, float] | None:
    """``(closed, open)`` read off a sweep's value series, or ``None``.

    The full-travel stops, not the typical cycle: the 10th percentile of
    the trough values and the 90th of the peak values.
    """
    peaks, troughs = _extrema(value)
    if peaks.size == 0 or troughs.size == 0:
        return None
    v = np.asarray(value, dtype=np.float64)
    closed = float(np.percentile(v[troughs], 10))
    opened = float(np.percentile(v[peaks], 90))
    return round(closed, 1), round(opened, 1)


def _load_viser():
    try:
        import viser
    except ImportError as e:
        raise RuntimeError(
            "the tuner needs viser, an optional dependency: "
            "`pixi install -e tune` (or `pip install viser`)"
        ) from e
    return viser


@dataclasses.dataclass
class _Clip:
    """Frame access for scrubbing: seek + a small rectified-frame cache."""

    cap: cv2.VideoCapture
    rect: object
    n_frames: int
    _cache: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)

    def rectified(self, idx: int) -> np.ndarray:
        if idx not in self._cache:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = self.cap.read()
            if not ok:
                raise RuntimeError(f"could not read frame {idx}")
            if len(self._cache) > 32:
                self._cache.pop(next(iter(self._cache)))
            self._cache[idx] = self.rect.apply(bgr)
        return self._cache[idx]


_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_GRAY = (110, 110, 110)
_WHITE = (255, 255, 255)


def _annotate_plane(
    view: np.ndarray,
    state,
    *,
    min_area: int,
    max_area: int,
    ref_xy: tuple[int, int],
) -> None:
    """Annotate the plane-warp view in place (BGR).

    Every dot candidate rings up where the detector found it - kept green,
    dropped gray - with its area in px^2 beside it, so a borderline dot can
    be read straight against the sliders. Then the reference
    circles: a green circle of ``min_area`` and a red one of ``max_area``
    about ``ref_xy`` (warp px), for parking over a real dot or a stray speck.
    """
    d = state.detail
    centers = d.get("warp_centers")
    areas = d.get("warp_areas")
    kept = {int(i) for i in (d.get("kept_indices") or [])}
    if centers is not None:
        for i, (cx, cy) in enumerate(np.asarray(centers)):
            c = (int(round(cx)), int(round(cy)))
            ok = i in kept
            color = _GREEN if ok else _GRAY
            cv2.circle(view, c, 8 if ok else 5, color, 2 if ok else 1, cv2.LINE_AA)
            if areas is not None:
                cv2.putText(
                    view,
                    f"{int(round(float(areas[i])))}",
                    (c[0] + 10, c[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    min_r = max(1, int(round(math.sqrt(min_area / math.pi))))
    max_r = max(min_r + 1, int(round(math.sqrt(max_area / math.pi))))
    cx, cy = ref_xy
    cv2.circle(view, (cx, cy), min_r, _GREEN, 2, cv2.LINE_AA)
    cv2.circle(view, (cx, cy), max_r, _RED, 2, cv2.LINE_AA)
    cv2.drawMarker(
        view, (cx, cy), _WHITE, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1
    )
    for row, (text, color) in enumerate(
        (
            (f"min={min_area}px^2 r~={min_r}px", _GREEN),
            (f"max={max_area}px^2 r~={max_r}px", _RED),
        )
    ):
        cv2.putText(
            view,
            text,
            (12, 28 + 28 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


@dataclasses.dataclass
class _SweepResult:
    """Per-sampled-frame estimator outputs from one sweep."""

    frames: np.ndarray  # (N,) int frame indices
    n_kept: np.ndarray  # (N,) int, kept dots (both jaws)
    n_left: np.ndarray  # (N,) int, kept dots on the left jaw (plane x < 0)
    n_right: np.ndarray  # (N,) int, kept dots on the right jaw (plane x >= 0)
    n_detected: np.ndarray  # (N,) int, candidates before the ROI filter
    value: np.ndarray  # (N,) float, NaN where the fit gave nothing
    has_pose: np.ndarray  # (N,) bool

    @property
    def n_weak(self) -> np.ndarray:
        """Kept dots on the weaker jaw - the count that starves the fit."""
        return np.minimum(self.n_left, self.n_right)

    def lowest(self, k: int = 5) -> list[tuple[int, int, int]]:
        """The ``k`` posed frames whose weaker jaw kept the fewest dots, as
        (frame, left, right).

        Frames strictly below the k-th lowest count all make the list; the
        count that ties at the cut is sampled evenly across the clip rather
        than taken in frame order, so a flat sweep (everything tied) points
        at five spread-out frames instead of the first five.
        """
        weak = self.n_weak
        idx = np.flatnonzero(self.has_pose)
        if idx.size == 0:
            idx = np.arange(len(self.frames))
        order = idx[np.argsort(weak[idx], kind="stable")]
        if len(order) <= k:
            picked = order
        else:
            cut = weak[order[k - 1]]
            below = order[weak[order] < cut]
            tied = order[weak[order] == cut]
            need = k - len(below)
            if len(tied) > need:
                tied = np.sort(tied)[
                    np.round(np.linspace(0, len(tied) - 1, need)).astype(int)
                ]
            picked = np.sort(np.concatenate([below, tied]))
        return [
            (int(self.frames[i]), int(self.n_left[i]), int(self.n_right[i]))
            for i in picked
        ]

    def lowest_labels(self, k: int = 5) -> list[str]:
        return [f"{fr}: L{nl} R{nr}" for fr, nl, nr in self.lowest(k)]


def _extrema(
    value: np.ndarray, *, prominence_frac: float = 0.15, min_separation: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the plateau peaks (maxima) and troughs (minima) of a
    NaN-gapped value series.

    An extremum must stand out by ``prominence_frac`` of the series' range
    and sit ``min_separation`` samples from its neighbours, so the jitter
    on a plateau labels once, not per wobble. NaN gaps are bridged linearly
    before the search, so a dropout neither splits a plateau nor holds an
    extremum of its own, and a plateau at either end of the clip counts.
    """
    from scipy.signal import find_peaks

    v = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(v)
    if finite.sum() < 3:
        return np.zeros(0, int), np.zeros(0, int)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    span = hi - lo
    if span <= 0:
        return np.zeros(0, int), np.zeros(0, int)
    prom = prominence_frac * span
    # Bridge NaN gaps linearly so a dropout inside a plateau does not split
    # it into two extrema, then pad both ends with the far extreme so a
    # plateau that runs into the start or the end of the clip still counts
    # (find_peaks never places a peak on a border).
    idx = np.arange(v.size)
    vi = np.interp(idx, idx[finite], v[finite])
    up = np.r_[lo, vi, lo]
    down = np.r_[-hi, -vi, -hi]
    peaks, _ = find_peaks(up, prominence=prom, distance=min_separation)
    troughs, _ = find_peaks(down, prominence=prom, distance=min_separation)
    return peaks.astype(int) - 1, troughs.astype(int) - 1


def _sweep_clip(
    lens0_mp4: Path,
    rect,
    est,
    detector,
    *,
    stride: int,
    n_frames: int,
    progress=None,
) -> _SweepResult:
    """Run ``est`` (+ ``detector``) on every ``stride``-th frame of the clip.

    Reads sequentially on a private capture - ``grab`` over the skipped
    frames. ``progress(done, total)`` is called every few frames.
    """
    from kiwi.core.pipeline import Detections

    frames = list(range(0, n_frames, stride))
    n_kept = np.zeros(len(frames), np.int32)
    n_left = np.zeros(len(frames), np.int32)
    n_right = np.zeros(len(frames), np.int32)
    n_det = np.zeros(len(frames), np.int32)
    value = np.full(len(frames), np.nan)
    has_pose = np.zeros(len(frames), bool)
    empty = Detections(
        corners=np.zeros((0, 4, 2), np.float32), ids=np.zeros(0, np.int32)
    )
    cap = cv2.VideoCapture(str(lens0_mp4))
    try:
        pos = 0
        for k, target in enumerate(frames):
            while pos < target:
                if not cap.grab():
                    raise RuntimeError(f"clip ended early at frame {pos}")
                pos += 1
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"could not read frame {target}")
            pos += 1
            rectified = rect.apply(bgr)
            dets = detector(rectified) if detector is not None else empty
            state = est(rectified, dets, rect.K_pinhole)
            n_kept[k] = state.n_kept
            n_det[k] = state.n_detected
            has_pose[k] = bool(state.has_pose)
            # Same jaw split the hinge fit uses: plane x < 0 is the left jaw.
            kept_xy = state.detail.get("centers_plane")
            if kept_xy is not None and len(kept_xy):
                left = np.asarray(kept_xy)[:, 0] < 0
                n_left[k] = int(left.sum())
                n_right[k] = int((~left).sum())
            if state.value is not None:
                value[k] = float(state.value)
            if progress is not None and (k % 10 == 0 or k == len(frames) - 1):
                progress(k + 1, len(frames))
    finally:
        cap.release()
    return _SweepResult(
        np.array(frames), n_kept, n_left, n_right, n_det, value, has_pose
    )


def _render_sweep(
    res: _SweepResult, *, title: str, unit: str, out_png: Path
) -> np.ndarray:
    """Plot the sweep (kept dots on top, fitted value below) to ``out_png``
    and return it as an RGB array for the main view."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = res.frames
    fig, (ax_k, ax_v) = plt.subplots(
        2, 1, sharex=True, figsize=(10.0, 4.6), dpi=110, height_ratios=(1, 1.2)
    )
    weak = res.n_weak
    ax_k.plot(fr, res.n_left, color="C0", lw=1.0, label="left jaw")
    ax_k.plot(fr, res.n_right, color="C1", lw=1.0, label="right jaw")
    ax_k.axhline(2, color="C3", ls="--", lw=0.8, label="fit min (2)")
    starved = res.has_pose & (weak < 2)
    if starved.any():
        ax_k.scatter(fr[starved], weak[starved], s=12, color="C3", zorder=3)
    no_pose = ~res.has_pose
    if no_pose.any():
        ax_k.scatter(
            fr[no_pose],
            np.zeros(int(no_pose.sum())),
            s=14,
            marker="x",
            color="0.5",
            zorder=4,
            label="no pose",
        )
    lowest = res.lowest()
    if lowest:
        ax_k.scatter(
            [f for f, _, _ in lowest],
            [min(nl, nr) for _, nl, nr in lowest],
            s=36,
            facecolors="none",
            edgecolors="C2",
            linewidths=1.2,
            zorder=5,
            label="5 lowest",
        )
        for f, nl, nr in lowest:
            ax_k.annotate(
                str(f),
                (f, min(nl, nr)),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color="C2",
            )
    ax_k.set_ylabel("kept dots / jaw")
    # 5 dots per jaw by design; stretch only if a frame kept more (a stray
    # passed the filters), so an overflow is visible rather than clipped.
    top = max(5, int(res.n_left.max(initial=0)), int(res.n_right.max(initial=0)))
    ax_k.set_ylim(-0.3, top + 0.7)
    ax_k.set_yticks(range(0, top + 1))
    ax_k.grid(True, alpha=0.3)
    ax_k.legend(loc="upper right", fontsize=7, ncol=5)
    ax_k.set_title(f"sweep - {title}", fontsize=9)

    ax_v.plot(fr, res.value, color="C2", lw=1.0, label="value")
    rejected = res.has_pose & (weak >= 2) & np.isnan(res.value)
    if rejected.any():
        ax_v.scatter(
            fr[rejected],
            np.zeros(int(rejected.sum())),
            s=14,
            marker="x",
            color="C3",
            zorder=4,
            label="posed, no value",
        )
    # Label every plateau: the open and closed values are what the
    # normalization endpoints are set from, so read them off the sweep.
    peaks, troughs = _extrema(res.value)
    n_labels = len(peaks) + len(troughs)
    label_fs = 7 if n_labels <= 24 else 6
    # Neighbouring labels closer than this share of the x range would
    # overprint; every other one of a close run is pushed further out.
    close_x = 0.035 * max(1.0, float(fr[-1] - fr[0]))
    for idx, marker, dy, va in ((peaks, "^", 5, "bottom"), (troughs, "v", -5, "top")):
        if idx.size == 0:
            continue
        ax_v.scatter(fr[idx], res.value[idx], s=16, marker=marker, color="C4", zorder=4)
        staggered = False
        for n, i in enumerate(idx):
            near = n > 0 and (fr[i] - fr[idx[n - 1]]) < close_x
            staggered = near and not staggered
            ax_v.annotate(
                f"{res.value[i]:.1f}",
                (fr[i], res.value[i]),
                textcoords="offset points",
                xytext=(0, dy * (2.6 if staggered else 1.0)),
                ha="center",
                va=va,
                fontsize=label_fs,
                color="C4",
            )
    ax_v.set_xlabel("frame")
    ax_v.set_ylabel(f"value ({unit})")
    ax_v.set_ylim(bottom=0)
    if peaks.size:
        # Headroom for the labels over the highest peak, staggered ones
        # included, so none of them prints across the axes frame.
        ax_v.set_ylim(top=float(np.nanmax(res.value)) * 1.22 + 1.0)
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


class _MainView:
    """The views as textured planes in the viewport (not the sidebar):
    rectified on the left, plane warp on the right, both one unit tall, a
    label over each, and the sweep plot under both once a sweep has run.
    Each connecting client gets a camera looking straight at them; the
    mouse pans and zooms from there."""

    _GAP = 0.08
    _FLIP = (0.0, 1.0, 0.0, 0.0)

    def __init__(
        self, server, *, rect_wh: tuple[int, int], plane_wh: tuple[int, int]
    ) -> None:
        rw = rect_wh[0] / rect_wh[1]
        pw = plane_wh[0] / plane_wh[1]
        # viser lays an image in the x-y plane with its top row toward -y;
        # a half-turn about x puts the top at +y, and a camera on +z with
        # +y up then sees it upright and unmirrored (the other side shows the
        # mirror image).
        flip = self._FLIP
        self._server = server
        self._rect = server.scene.add_image(
            "/rectified",
            np.zeros((rect_wh[1], rect_wh[0], 3), dtype=np.uint8),
            render_width=rw,
            render_height=1.0,
            wxyz=flip,
            position=(-(rw + self._GAP) / 2, 0.0, 0.0),
            format="jpeg",
            jpeg_quality=90,
        )
        self._plane = server.scene.add_image(
            "/plane_warp",
            np.zeros((plane_wh[1], plane_wh[0], 3), dtype=np.uint8),
            render_width=pw,
            render_height=1.0,
            wxyz=flip,
            position=((pw + self._GAP) / 2, 0.0, 0.0),
            format="jpeg",
            jpeg_quality=90,
        )
        # Children of the flipped nodes: local -y is world +y (above).
        server.scene.add_label(
            "/rectified/label", "rectified", position=(0.0, -0.55, 0.0)
        )
        server.scene.add_label(
            "/plane_warp/label", "plane warp", position=(0.0, -0.55, 0.0)
        )
        self._span = rw + pw + self._GAP
        self._plot = None
        span = self._span

        # Aim a little right of centre.
        @server.on_client_connect
        def _(client) -> None:
            def frame(cam) -> None:
                cam.up_direction = (0.0, 1.0, 0.0)
                cam.position = (0.2 * span, 0.0, 0.55 * span)
                cam.look_at = (0.2 * span, 0.0, 0.0)

            frame(client.camera)
            framed = False

            # 2-D mode. viser has no camera lock, so every camera update is
            # snapped back to a straight-on view over the orbit target: a
            # pan moves the target (kept), a zoom changes the orbit distance
            # (kept, floored just in front of the planes), a rotate changes
            # neither and is undone. The setters skip unchanged values, so a
            # camera already straight-on sends nothing back. Until the client
            # has reported a straight-on view once, its updates are stale
            # pre-framing state (the default orbit), so re-frame instead.
            @client.camera.on_update
            def _(cam) -> None:
                nonlocal framed
                pos, tgt = np.asarray(cam.position), np.asarray(cam.look_at)
                dist = max(0.05, float(np.linalg.norm(pos - tgt)))
                # The tolerance sits above the float32 round-trip noise of
                # the client's orbit controls.
                tol = 2e-3 * dist
                straight = (
                    np.allclose(pos[:2], tgt[:2], atol=tol)
                    and pos[2] > tgt[2]
                    and abs(float(tgt[2])) < tol
                    and np.allclose(cam.up_direction, (0.0, 1.0, 0.0), atol=2e-3)
                )
                if straight:
                    framed = True
                    return
                if not framed:
                    frame(cam)
                    return
                tx, ty = float(tgt[0]), float(tgt[1])
                cam.position = (tx, ty, dist)
                cam.look_at = (tx, ty, 0.0)
                cam.up_direction = (0.0, 1.0, 0.0)

    @property
    def rectified(self) -> np.ndarray:
        return self._rect.image

    @rectified.setter
    def rectified(self, bgr: np.ndarray) -> None:
        self._rect.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @property
    def plane(self) -> np.ndarray:
        return self._plane.image

    @plane.setter
    def plane(self, bgr: np.ndarray) -> None:
        self._plane.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @property
    def plot(self) -> np.ndarray | None:
        return None if self._plot is None else self._plot.image

    @plot.setter
    def plot(self, rgb: np.ndarray) -> None:
        """The sweep plot (RGB), as a plane spanning both images below them;
        created on first use so an unswept session shows no empty slot."""
        if self._plot is None:
            h, w = rgb.shape[:2]
            height = self._span * h / w
            self._plot = self._server.scene.add_image(
                "/sweep",
                rgb,
                render_width=self._span,
                render_height=height,
                wxyz=self._FLIP,
                position=(0.0, -(0.5 + self._GAP + height / 2), 0.0),
                format="png",
            )
        else:
            self._plot.image = rgb


def launch(
    lens0_mp4: Path,
    intrinsics: Path,
    *,
    session_block: dict,
    marker_id: int | None,
    marker_size_m: float,
    marker_dictionary: str | None = None,
    fixed_pose: dict | None = None,
    fixed_pose_source: str = "",
    tool_session: Mapping | None = None,
    unit_path: Path | None = None,
    out_size: tuple[int, int] = (1920, 1920),
    port: int = 8080,
    block: bool = True,
) -> None:
    """Serve the tuner at ``http://localhost:<port>`` until Ctrl-C.

    ``session_block`` seeds the sliders (the layered tool < unit recipe);
    every change rebuilds the estimator through the same factory the batch
    runner uses and reprocesses the scrubbed frame. With ``tool_session``
    (the design's block alone) and ``unit_path``, the *write to unit*
    button writes the tuned recipe's DIFF against the design, plus the
    endpoints, into that unit file's ``session:`` override in place.
    """
    viser = _load_viser()
    from kiwi.core.rectify import load_rectifier_auto
    from kiwi.core.video import VideoInfo

    bundle = get_estimator("chopsticks")
    info = VideoInfo.read(lens0_mp4)
    # Validate + seed through the SAME config layer the batch path uses; an
    # unknown session key fails here too.
    seeded = resolve_params(config_from_block(session_block))
    values: dict = dataclasses.asdict(seeded)

    rect = load_rectifier_auto(
        intrinsics,
        (info.width, info.height),
        lens="front",
        out_size=out_size,
        fov_deg=values["fov_deg"],
    )
    clip = _Clip(
        cap=cv2.VideoCapture(str(lens0_mp4)), rect=rect, n_frames=info.n_frames
    )

    def make_pair(debug: bool = True):
        """Estimator + matching detector from ONE context, every refresh -
        so toggling fixed-pose off really switches to detected PnP."""
        ctx = EstimatorContext(
            session_block={},
            K=rect.K_pinhole,
            image_wh=tuple(rect.out_size),
            marker_id=marker_id,
            marker_size_m=marker_size_m,
            marker_dictionary=marker_dictionary,
            fixed_pose=fixed_pose if values.get("use_fixed_pose", True) else None,
            fixed_pose_source=fixed_pose_source,
            live_pnp=not values.get("use_fixed_pose", True),
            overrides={k: values[k] for k in dataclasses.asdict(seeded)},
        )
        est = bundle.make_frame_estimator(ctx)
        est.debug = debug
        return est, bundle.make_detector(ctx)

    server = viser.ViserServer(port=port)
    server.scene.world_axes.visible = False
    lock = threading.Lock()

    frame_slider = server.gui.add_slider(
        "frame", min=0, max=max(0, info.n_frames - 1), step=1, initial_value=0
    )
    polarity = server.gui.add_dropdown(
        "polarity",
        options=tuple(DEFAULT_THRESHOLD_BY_POLARITY),
        initial_value=values["polarity"],
    )
    keep = server.gui.add_dropdown(
        "roi_semicircle_keep",
        options=("+y", "-y", "+x", "-x"),
        initial_value=values["roi_semicircle_keep"],
    )
    clip_edges = server.gui.add_checkbox("clip_to_image", values["clip_to_image"])
    outlier_trim = server.gui.add_checkbox(
        "enable_outlier_trim", values["enable_outlier_trim"]
    )
    use_fixed = None
    if fixed_pose is not None:
        values["use_fixed_pose"] = True
        use_fixed = server.gui.add_checkbox("use fixed pose", True)
    handles = {}
    for name, kind, lo, hi, step in _SLIDERS:
        init = values.get(name)
        if init is None:
            init = lo if name != "roi_inner_radius_mm" else 0.0
        handles[name] = server.gui.add_slider(
            _SLIDER_LABELS.get(name, name),
            min=lo,
            max=hi,
            step=step,
            initial_value=int(init) if kind == "int" else float(init),
        )
    # Reference circles: green = min_area, red = max_area, parked
    # anywhere on the plane warp (the grid the areas are measured on) so a
    # real dot or a stray speck can be compared against them by eye.
    with server.gui.add_folder("ref circle"):
        ref_x = server.gui.add_slider(
            "ref circle X", min=0, max=WARP_PX - 1, step=1, initial_value=WARP_PX // 2
        )
        ref_y = server.gui.add_slider(
            "ref circle Y", min=0, max=WARP_PX - 1, step=1, initial_value=WARP_PX // 2
        )
    # Dots-per-frame sweep, over the REAL estimator: run
    # the current recipe across the clip at a stride and plot kept dots + the
    # fitted value per frame.
    with server.gui.add_folder("sweep clip"):
        sweep_stride = server.gui.add_number(
            "stride (frames)",
            initial_value=max(1, round(info.n_frames / 500)),
            min=1,
            max=max(1, info.n_frames - 1),
            step=1,
        )
        sweep_btn = server.gui.add_button("sweep + plot")
        sweep_status = server.gui.add_text(
            "sweep", initial_value="(not run)", disabled=True
        )
        sweep_jump = server.gui.add_dropdown(
            "jump to lowest", options=("-",), initial_value="-", disabled=True
        )
    # The normalization endpoints ride beside the recipe: seeded from the
    # source block, set from the sweep's full-travel stops, written with
    # the block.
    with server.gui.add_folder("endpoints"):
        angle_closed = server.gui.add_number(
            "angle_closed_deg",
            initial_value=float(session_block.get("angle_closed_deg") or 0.0),
            step=0.5,
        )
        angle_open = server.gui.add_number(
            "angle_open_deg",
            initial_value=float(session_block.get("angle_open_deg") or 0.0),
            step=0.5,
        )
        endpoints_btn = server.gui.add_button("set from sweep", disabled=True)
    status = server.gui.add_text("status", initial_value="-", disabled=True)
    write_btn = server.gui.add_button("export configs")
    unit_btn = None
    if unit_path is not None and tool_session is not None:
        unit_btn = server.gui.add_button(f"write to unit: {Path(unit_path).stem}")
    last_sweep: dict = {"res": None}

    # The two views live in the main viewport as side-by-side planes (pan /
    # zoom with the mouse), not in the sidebar: rectified on the left, plane
    # warp on the right, both one unit tall.
    views = _MainView(server, rect_wh=tuple(rect.out_size), plane_wh=(WARP_PX, WARP_PX))

    def refresh(_=None) -> None:
        with lock:
            values["polarity"] = polarity.value
            values["roi_semicircle_keep"] = keep.value
            values["clip_to_image"] = bool(clip_edges.value)
            values["enable_outlier_trim"] = bool(outlier_trim.value)
            if use_fixed is not None:
                values["use_fixed_pose"] = bool(use_fixed.value)
            for name, kind, *_ in _SLIDERS:
                v = handles[name].value
                values[name] = int(v) if kind == "int" else float(v)
            if values["roi_inner_radius_mm"] == 0.0:
                values["roi_inner_radius_mm"] = None  # 0 = no annulus cut
            try:
                est, detector = make_pair()
                rectified = clip.rectified(int(frame_slider.value))
                from kiwi.core.pipeline import Detections

                empty = Detections(
                    corners=np.zeros((0, 4, 2), np.float32),
                    ids=np.zeros(0, np.int32),
                )
                dets = detector(rectified) if detector is not None else empty
                state = est(rectified, dets, rect.K_pinhole)
            except (ValueError, RuntimeError) as e:
                status.value = f"ERROR: {e}"
                return
            canvas = rectified.copy()
            if bundle.draw_overlay is not None:
                bundle.draw_overlay(canvas, state)
            views.rectified = canvas
            warped = state.detail.get("warped")
            if warped is not None:
                plane_view = warped.copy()
                _annotate_plane(
                    plane_view,
                    state,
                    min_area=int(values["min_area"]),
                    max_area=int(values["max_area"]),
                    ref_xy=(int(ref_x.value), int(ref_y.value)),
                )
                views.plane = plane_view
            if state.value is not None:
                status.value = (
                    f"value {state.value:.2f} {bundle.unit}  "
                    f"kept {state.n_kept}/{state.n_detected}  "
                    f"mode {state.detail.get('mode')}"
                )
            else:
                status.value = (
                    f"{state.error or 'no pose'}  "
                    f"kept {state.n_kept}/{state.n_detected}"
                )
            if values["roi_inner_radius_mm"] is None:
                values["roi_inner_radius_mm"] = 0.0  # back to the slider's 0

    def endpoints() -> dict:
        """The endpoint fields, or {} when they do not form a valid pair
        (a source block without them leaves both at 0)."""
        closed, opened = float(angle_closed.value), float(angle_open.value)
        if not opened > closed:
            return {}
        return {"angle_closed_deg": closed, "angle_open_deg": opened}

    def write_block(_=None) -> None:
        emit = dict(values)
        if emit.get("roi_inner_radius_mm") == 0.0:
            emit["roi_inner_radius_mm"] = None
        text = tuned_block(emit, passthrough=endpoints())
        out = lens0_mp4.with_name(lens0_mp4.stem + "_tuned-session.yaml")
        out.write_text(text)
        print(
            f"\n# paste into config/gripper/<tool>/tool.yaml (or a unit's session: "
            f"override)\n{text}\nwrote {out}"
        )
        status.value = f"wrote {out.name}"

    def write_to_unit(_=None) -> None:
        """Write the diff against the design into the unit's session:
        override, keeping the unit's keys the tuner does not manage."""
        assert unit_path is not None and tool_session is not None
        with lock:
            diff = session_diff(values, tool_session, endpoints=endpoints())
        try:
            validate_session_block(bundle.name, diff, str(unit_path))
            existing = load_gripper_unit(unit_path).session
            managed = set(_EMIT_KEYS) | set(_ENDPOINT_KEYS)
            new = {k: v for k, v in existing.items() if k not in managed}
            new.update(diff)
            write_unit_session(unit_path, new)
        except (ValueError, RuntimeError, OSError) as e:
            status.value = f"ERROR: {e}"
            return
        shown = (
            yaml.safe_dump({"session": diff}, sort_keys=False)
            if diff
            else "(no override - the recipe equals the design's)\n"
        )
        print(
            f"\n# {unit_path}: session: override (diff against the design)\n"
            f"{shown}wrote {unit_path} - review the git diff (comments do not "
            f"survive the rewrite)"
        )
        status.value = f"wrote {Path(unit_path).name}: {len(diff)} override key(s)"

    def set_endpoints(_=None) -> None:
        res = last_sweep["res"]
        got = endpoints_from_sweep(res.value) if res is not None else None
        if got is None:
            status.value = "ERROR: no plateaus in the sweep to read endpoints from"
            return
        angle_closed.value, angle_open.value = got
        status.value = f"endpoints from sweep: closed {got[0]}  open {got[1]}"

    def run_sweep(_=None) -> None:
        """Snapshot the recipe, then sweep on a worker thread so the GUI
        keeps scrubbing; the plot lands in the main view when done."""
        with lock:
            est, detector = make_pair(debug=False)
            stride = max(1, int(sweep_stride.value))
            label = f"stride={stride} polarity={values['polarity']}"
        sweep_btn.disabled = True
        sweep_jump.disabled = True

        def progress(done: int, total: int) -> None:
            sweep_status.value = f"sweeping {done}/{total}"

        def work() -> None:
            try:
                res = _sweep_clip(
                    lens0_mp4,
                    rect,
                    est,
                    detector,
                    stride=stride,
                    n_frames=info.n_frames,
                    progress=progress,
                )
                out = lens0_mp4.with_name(lens0_mp4.stem + "_tune-sweep.png")
                views.plot = _render_sweep(
                    res, title=label, unit=bundle.unit, out_png=out
                )
            except Exception as e:  # a worker thread: surface, never swallow
                sweep_status.value = f"ERROR: {e}"
                raise
            finally:
                sweep_btn.disabled = False
            valid = ~np.isnan(res.value)
            sweep_status.value = (
                f"L med {int(np.median(res.n_left))} min {int(res.n_left.min())}  "
                f"R med {int(np.median(res.n_right))} min {int(res.n_right.min())}  "
                f"valid {int(valid.sum())}/{len(valid)}  -> {out.name}"
            )
            sweep_jump.options = ("-", *res.lowest_labels())
            sweep_jump.value = "-"
            sweep_jump.disabled = False
            last_sweep["res"] = res
            endpoints_btn.disabled = False
            print(f"sweep: {sweep_status.value}")

        threading.Thread(target=work, name="tune-sweep", daemon=True).start()

    def jump(_=None) -> None:
        if sweep_jump.value != "-":
            frame_slider.value = int(sweep_jump.value.split(":")[0])

    for h in (
        frame_slider,
        polarity,
        keep,
        clip_edges,
        outlier_trim,
        ref_x,
        ref_y,
        *handles.values(),
    ):
        h.on_update(refresh)
    if use_fixed is not None:
        use_fixed.on_update(refresh)
    write_btn.on_click(write_block)
    if unit_btn is not None:
        unit_btn.on_click(write_to_unit)
    endpoints_btn.on_click(set_endpoints)
    sweep_btn.on_click(run_sweep)
    sweep_jump.on_update(jump)

    refresh()
    print(f"tuner: http://localhost:{port}  (Ctrl-C to exit)")
    if block:  # pragma: no cover - interactive loop
        try:
            hold_until_interrupt(msg=None)  # the URL line above already said how
        finally:
            clip.cap.release()

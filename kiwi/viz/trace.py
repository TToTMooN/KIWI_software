"""One tool-state review PNG per episode: every arm on the global clock.

Arms are stacked on a shared x axis, that axis being the episode's global
synced clock - the same clock ``task_window`` and the curation windows
live on.

Frame ``i`` sits at ``offset + i / fps``, the mapping export uses to emit
the tool-state feature.

Both raw ``value`` and the consumer ``output`` (``value_completed``) are
drawn. Completion fills every gap without changing finite raw anchors.
Dropout bands show where the raw series is missing.

Tool-agnostic: the unit and the closed/open endpoints come from the artifact
and the tool config.
"""

from __future__ import annotations

import collections
import dataclasses
from pathlib import Path

import numpy as np

from kiwi.viz.style import PALETTE, stream_color

__all__ = ["ArmSeries", "dropout_spans", "stat_line", "trace_episode"]

# Reddish-purple, distinct from both arm hues.
_DROPOUT_COLOR = "#CC79A7"
_INK = "#1a1a1a"
_MUTED = "#6b6b6b"


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02X%02X%02X" % rgb


@dataclasses.dataclass(frozen=True)
class ArmSeries:
    """One arm's tool state, already placed on the global clock."""

    label: str
    t: np.ndarray
    value: np.ndarray
    output: np.ndarray
    fps: float
    offset_s: float
    unit: str
    errors: collections.Counter
    closed: float | None = None
    open: float | None = None

    @property
    def coverage(self) -> float:
        n = self.value.size
        return float(np.count_nonzero(np.isfinite(self.value)) / n) if n else 0.0


def series_from_state(
    label: str,
    state_path: Path,
    offset_s: float,
    *,
    closed: float | None = None,
    open: float | None = None,
) -> ArmSeries:
    """Load one arm's ``eef_state_v1`` npz onto the global clock."""
    from kiwi.eef.state import read_eef_state

    st = read_eef_state(state_path)
    errors = collections.Counter(str(e) for e in np.asarray(st.error) if str(e))
    return ArmSeries(
        label=label,
        t=offset_s + np.arange(len(st)) / float(st.fps),
        value=np.asarray(st.value, dtype=np.float64),
        output=np.asarray(st.output, dtype=np.float64),
        fps=float(st.fps),
        offset_s=float(offset_s),
        unit=str(st.unit),
        errors=errors,
        closed=closed,
        open=open,
    )


def dropout_spans(t: np.ndarray, valid: np.ndarray) -> list[tuple[float, float]]:
    """``[(start, end)]`` global-time spans where no value exists."""
    if valid.all():
        return []
    edges = np.flatnonzero(np.diff(np.r_[False, ~valid, False].astype(np.int8)))
    return [
        (float(t[a]), float(t[min(b, t.size - 1)])) for a, b in edges.reshape(-1, 2)
    ]


def stat_line(s: ArmSeries) -> str:
    """The one-line quantitative summary drawn above each arm's axes."""
    have = np.isfinite(s.value)
    n_bad = int(np.count_nonzero(~have))
    u = f" {s.unit}" if s.unit else ""
    bits = [f"{s.value.size} frames @ {s.fps:.2f} fps", f"offset {s.offset_s:+.3f}s"]
    if have.any():
        bits.append(
            f"value {np.nanmin(s.value):.1f}-{np.nanmax(s.value):.1f}{u} "
            f"(median {np.nanmedian(s.value):.1f}{u})"
        )
    if s.closed is not None and s.open is not None:
        bits.append(f"closed/open {s.closed:g}-{s.open:g}{u}")
    bits.append(f"coverage {100 * s.coverage:.2f}%")
    if n_bad:
        why = ", ".join(f"{k} {v}" for k, v in s.errors.most_common(3))
        bits.append(f"{n_bad} dropped ({why})" if why else f"{n_bad} dropped")
    return "  |  ".join(bits)


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise RuntimeError(
            "the tool-state trace needs matplotlib, an optional dependency: "
            "`pixi install -e viz` (or `pip install matplotlib`)"
        ) from exc
    return plt


def _fmt(v: float | None, fallback: float) -> str:
    return f"{fallback:.1f}*" if v is None else f"{v:.1f}"


def trace_episode(
    series: list[ArmSeries],
    out: Path,
    *,
    title: str = "",
    subtitle: str = "",
    task_window: tuple[float | None, float | None] | None = None,
    clock_origin: str | None = None,
    bounds_in_ylim: bool = False,
    dpi: int = 130,
) -> Path:
    """Render the stacked review plot for ``series``. Returns ``out``."""
    if not series:
        raise ValueError("no arms to plot")
    plt = _load_pyplot()

    n = len(series)
    header_in = 0.95  # inches reserved for the figure title block
    fig_h = 3.1 * n + header_in
    fig, axes = plt.subplots(n, 1, figsize=(13, fig_h), sharex=True, squeeze=False)
    axes = axes[:, 0]

    t_lo = min(float(s.t[0]) for s in series)
    t_hi = max(float(s.t[-1]) for s in series)
    lo_w, hi_w = task_window or (None, None)

    # One y range for every arm.
    finite = np.concatenate(
        [s.value[np.isfinite(s.value)] for s in series] or [np.zeros(0)]
    )
    y_lo, y_hi = (
        (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    )
    if bounds_in_ylim:
        b = [x for s in series for x in (s.closed, s.open) if x is not None]
        if b:
            y_lo, y_hi = min(y_lo, min(b)), max(y_hi, max(b))
    pad = 0.06 * max(y_hi - y_lo, 1.0)
    y_lo, y_hi = y_lo - pad, y_hi + pad

    for ax, s in zip(axes, series, strict=True):
        color = _hex(stream_color(s.label))
        unit = f" ({s.unit})" if s.unit else ""

        # Outside the exported task window: greyed out.
        if lo_w is not None and lo_w > t_lo:
            ax.axvspan(t_lo, lo_w, color="#000000", alpha=0.055, lw=0, zorder=0)
        if hi_w is not None and hi_w < t_hi:
            ax.axvspan(hi_w, t_hi, color="#000000", alpha=0.055, lw=0, zorder=0)
        for edge in (lo_w, hi_w):
            if edge is not None:
                ax.axvline(edge, color=_MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1)

        # The export-normalisation endpoints, drawn only when they fall in
        # view; the stat line states them either way.
        for name, val in (("closed", s.closed), ("open", s.open)):
            if val is None or not (y_lo < val < y_hi):
                continue
            ax.axhline(val, color=_MUTED, ls=(0, (1, 3)), lw=1.0, zorder=1)
            ax.annotate(
                f"{name} {val:g}",
                xy=(1.0, val),
                xycoords=("axes fraction", "data"),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color=_MUTED,
            )

        # Every dropout span is widened to a visible minimum width; the count
        # in the stat line stays the quantitative answer.
        min_w = 0.004 * max(t_hi - t_lo, 1e-9)
        for i, (lo_d, hi_d) in enumerate(dropout_spans(s.t, np.isfinite(s.value))):
            hi_d += 1.0 / s.fps
            grow = max(0.0, min_w - (hi_d - lo_d)) / 2
            ax.axvspan(
                lo_d - grow,
                hi_d + grow,
                color=_DROPOUT_COLOR,
                alpha=0.35,
                lw=0,
                zorder=1.5,
                label="no value" if i == 0 else None,
            )

        ax.plot(
            s.t,
            s.value,
            color=color,
            lw=2.4,
            alpha=0.28,
            solid_capstyle="round",
            label="raw",
        )
        ax.plot(s.t, s.output, color=color, lw=1.4, label="output", zorder=3)

        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel(f"tool state{unit}", fontsize=9, color=_INK)
        ax.set_title(f"{s.label}", loc="left", fontsize=11, color=_INK, pad=15)
        ax.annotate(
            stat_line(s),
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8,
            color=_MUTED,
        )
        ax.grid(axis="y", color="#000000", alpha=0.08, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#cccccc")
        ax.tick_params(colors=_MUTED, labelsize=8.5)
        # Above the axes, on the title row.
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=8,
            frameon=False,
            ncols=3,
            handlelength=1.6,
            columnspacing=1.2,
            labelcolor=_MUTED,
        )

    axes[-1].set_xlim(t_lo, t_hi)
    origin = f", origin = {clock_origin} frame 0" if clock_origin else ""
    axes[-1].set_xlabel(f"global synced clock (s){origin}", fontsize=9, color=_INK)

    open_end = task_window is not None and (lo_w is None or hi_w is None)
    window = (
        "no task_window (full overlap exported)"
        if task_window is None
        else f"task_window [{_fmt(lo_w, t_lo)}, {_fmt(hi_w, t_hi)}] s (shaded = outside"
        + ("; * = open end, run to end of coverage)" if open_end else ")")
    )
    fig.suptitle(
        title or "episode",
        x=0.008,
        y=1 - 0.34 / fig_h,
        va="top",
        ha="left",
        fontsize=13.5,
        color=_INK,
    )
    fig.text(
        0.008,
        1 - 0.62 / fig_h,
        "  |  ".join(x for x in (subtitle, window) if x),
        va="top",
        ha="left",
        fontsize=8.5,
        color=_MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.80 / fig_h))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# The arm hues as hex strings, importable without the plotting stack.
ARM_COLORS = {k: _hex(v) for k, v in PALETTE.items()}

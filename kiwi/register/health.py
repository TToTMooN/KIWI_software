"""Detect a corrupted marker anchor (the planar-PnP flip ambiguity).

A planar fiducial has a well-known two-fold pose ambiguity: a wrong solution
mirrored about the marker plane can carry a reprojection error as low as the
true one.

The anchor solve drops detections beyond ``max_marker_dist``. This module
checks the anchor with two independent signals:

**Displacement** (:func:`anchor_displacement`). The marker is physically
static, so every per-detection anchor candidate must agree up to VIO/PnP
noise. The anchor is a weighted mean of the candidates; the check compares
that mean to a weighted median of the same candidates. Blind, though, to a
*uniform* flip: if every detection flips the same way (a whole capture shot
from one bad viewpoint) they agree with each other, and mean and median move
together.

**Reach** (:func:`trajectory_reach`) - the backstop for exactly that case. The
anchored trajectory is a handheld wrist path around a table marker, so it
stays close to the marker origin.

Pure numpy (no cv2).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DISPLACEMENT_HALT_M",
    "DISPLACEMENT_WARN_M",
    "REACH_HALT_M",
    "REACH_WARN_M",
    "anchor_displacement",
    "marker_origin_spread",
    "reach_verdict",
    "spread_verdict",
    "trajectory_reach",
    "worst",
]

# --- thresholds (metres) --------------------------------------------------- #
# Reach: median distance of the anchored trajectory from the marker origin.
REACH_WARN_M = 0.80
REACH_HALT_M = 1.25

# Displacement: how far the written anchor (a weighted MEAN) sits from a robust
# weighted MEDIAN of the same candidates.
DISPLACEMENT_WARN_M = 0.05
DISPLACEMENT_HALT_M = 0.15


def _weighted_mean(x: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return x.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != x.shape[0] or not np.isfinite(w).all() or w.sum() <= 0:
        return x.mean(axis=0)
    return (x * w[:, None]).sum(axis=0) / w.sum()


def _weighted_median(x: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Coordinate-wise weighted median (not rotation-equivariant)."""
    if weights is None:
        return np.median(x, axis=0)
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != x.shape[0] or not np.isfinite(w).all() or w.sum() <= 0:
        return np.median(x, axis=0)
    out = np.empty(3)
    for j in range(3):
        order = np.argsort(x[:, j])
        cw = np.cumsum(w[order])
        out[j] = x[order][np.searchsorted(cw, 0.5 * cw[-1]), j]
    return out


def marker_origin_spread(
    origins: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    measurements: dict | None = None,
) -> tuple[float, float]:
    """(median, max) distance of per-detection marker origins from their mean.

    ``origins`` is (N,3): the translation of each candidate anchor. Uses the
    same reprojection weighting the anchor average uses, so the centre it
    measures against is the anchor that will actually be written.
    """
    o = np.asarray(origins, dtype=float).reshape(-1, 3)
    if len(o) < 2:
        return (0.0, 0.0)  # a single detection has no dispersion to measure
    d = np.linalg.norm(o - _weighted_mean(o, weights), axis=1)
    if measurements is not None:
        measurements["candidate_scatter_m"] = d.tolist()
    return (float(np.median(d)), float(d.max()))


def trajectory_reach(
    p_marker: np.ndarray, *, measurements: dict | None = None
) -> tuple[float, float]:
    """(median, p95) distance of an anchored trajectory from the marker origin.

    ``p_marker`` is (N,3), the xyz columns of a ``<arm>/register.tum``.
    """
    p = np.asarray(p_marker, dtype=float).reshape(-1, 3)
    if len(p) == 0:
        return (float("inf"), float("inf"))
    n = np.linalg.norm(p, axis=1)
    if measurements is not None:
        measurements["trajectory_reach_m"] = n.tolist()
    return (float(np.median(n)), float(np.percentile(n, 95)))


def anchor_displacement(
    origins: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """How far the written anchor sits from a robust estimate of the same anchor.

    The anchor is a weighted MEAN of per-detection marker origins; the robust
    estimate is their weighted MEDIAN.
    """
    o = np.asarray(origins, dtype=float).reshape(-1, 3)
    if len(o) < 3:
        return 0.0  # too few to tell an outlier from the cluster
    return float(
        np.linalg.norm(_weighted_mean(o, weights) - _weighted_median(o, weights))
    )


def spread_verdict(
    origins: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    measurements: dict | None = None,
) -> tuple[str, str]:
    """('ok'|'warn'|'halt', message) for a flip-corrupted anchor."""
    med, mx = marker_origin_spread(origins, weights, measurements=measurements)
    disp = anchor_displacement(origins, weights)
    if measurements is not None:
        measurements.update(
            anchor_displacement_m=disp, scatter_median_m=med, scatter_max_m=mx
        )
    n = len(np.asarray(origins).reshape(-1, 3))
    detail = (
        f"{n} detections, anchor offset {disp * 100:.1f} cm "
        f"(scatter median {med * 100:.1f} cm / max {mx * 100:.1f} cm)"
    )
    if disp > DISPLACEMENT_HALT_M:
        return (
            "halt",
            f"anchor pulled off its own candidates: {detail} - the weighted mean sits "
            f"> {DISPLACEMENT_HALT_M * 100:.0f} cm from the robust median, so flipped "
            f"detections have moved it. This is the planar-PnP flip ambiguity; lower "
            f"anchor.max_marker_dist to drop the far/oblique views.",
        )
    if disp > DISPLACEMENT_WARN_M:
        return ("warn", f"anchor is being pulled by outlier detections: {detail}")
    return ("ok", detail)


def reach_verdict(
    p_marker: np.ndarray, *, measurements: dict | None = None
) -> tuple[str, str]:
    """('ok'|'warn'|'halt', message) for an anchored trajectory's reach."""
    med, p95 = trajectory_reach(p_marker, measurements=measurements)
    if measurements is not None:
        measurements.update(reach_median_m=med, reach_p95_m=p95)
    detail = f"median {med:.2f} m / p95 {p95:.2f} m from the marker"
    if med > REACH_HALT_M:
        return (
            "halt",
            f"anchored trajectory sits {detail} (> {REACH_HALT_M} m). A handheld path "
            f"around a table marker does not; this is the anchor, not the motion. "
            f"Check anchor.max_marker_dist, then re-run the register stage.",
        )
    if med > REACH_WARN_M:
        return ("warn", f"anchored trajectory is far from the marker: {detail}")
    return ("ok", detail)


def worst(*verdicts: tuple[str, str]) -> tuple[str, str]:
    """Combine verdicts, keeping the most severe (halt > warn > ok)."""
    rank = {"ok": 0, "warn": 1, "halt": 2}
    return max(verdicts, key=lambda v: rank[v[0]])

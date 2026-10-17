"""Complete a dense, uniformly sampled EEF series with explicit provenance.

Gap labels describe duration, not uncertainty. An ``observed`` value is a
provided finite anchor, which may itself come from upstream inference; it
does not imply independent image evidence. Completion never changes anchors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["COMPLETION_POLICY_VERSION", "complete_series"]

COMPLETION_POLICY_VERSION = 1


def _real_scalar(value: float, name: str) -> float:
    scalar = np.asarray(value)
    if scalar.ndim != 0 or scalar.dtype.kind not in "iuf" or not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real numeric scalar excluding bool")
    return scalar[()]


def complete_series(
    values: ArrayLike,
    fps: float,
    *,
    short_max_s: float = 0.167,
    medium_max_s: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Fill every NaN run, preserving all original finite samples exactly.

    Input must be one-dimensional real numeric samples at uniform ``fps``.
    Interior gaps use linear interpolation by frame index; edge gaps hold
    the nearest original finite anchor. The input is never modified.
    Empty/all-NaN input, infinities, and invalid configuration raise ValueError.

    Gap duration is missing frame count / fps, whereas anchor span includes
    the interval to the next anchor: (right - left) / fps. Duration labels
    use inclusive maxima in seconds, without rounding or tolerance, and do
    not affect filling. Missing anchors have index -1 and no anchor span.
    Observed frames anchor to themselves, with zero distances and span.
    Only linear fills have a finite interpolation weight (toward the right).

    Metadata values are per-frame or 0-d NumPy arrays, safe to save without
    pickle. Aggregate ``policy`` identifies the algorithm and ``version``
    identifies this metadata/policy contract. Notes summarize frame counts.
    """
    fps = _real_scalar(fps, "fps")
    short_max_s = _real_scalar(short_max_s, "short_max_s")
    medium_max_s = _real_scalar(medium_max_s, "medium_max_s")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not 0 <= short_max_s <= medium_max_s:
        raise ValueError("thresholds must satisfy 0 <= short_max_s <= medium_max_s")

    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(
            "values must be a one-dimensional real numeric array excluding bool"
        )
    if np.isinf(raw).any():
        raise ValueError(
            "values must not contain infinity; use NaN for missing samples"
        )
    finite = np.isfinite(raw)
    if not finite.any():
        raise ValueError("no usable EEF samples: values are empty or all missing (NaN)")

    # Integer arrays cannot contain NaNs and keep their dtype; floats are
    # promoted without narrowing extended precision.
    out = (
        raw.copy()
        if raw.dtype.kind in "iu"
        else raw.astype(np.result_type(raw.dtype, np.float64), copy=True)
    )
    n = raw.size
    frames = np.arange(n)
    filled = ~finite
    method = np.full(n, "observed", dtype="U13")
    gap_level = np.full(n, "none", dtype="U6")
    gap_id = np.full(n, -1, dtype=np.int64)
    gap_frames = np.zeros(n, dtype=np.int64)
    gap_duration_s = np.zeros(n)
    left_anchor_frame = frames.copy()
    right_anchor_frame = frames.copy()
    nearest_anchor_distance_s = np.zeros(n)
    anchor_span_s = np.zeros(n)
    interpolation_weight = np.full(n, np.nan)

    # Transitions delimit maximal runs in the original missing mask, so
    # synthesized values can never become anchors for subsequent fills.
    edges = np.flatnonzero(np.diff(np.r_[False, filled, False]))
    max_gap_s = 0.0
    for identifier, (start, stop) in enumerate(edges.reshape(-1, 2)):
        gap = slice(start, stop)
        count = stop - start
        duration = count / fps
        left = start - 1
        right = stop if stop < n else -1
        gap_id[gap] = identifier
        gap_frames[gap] = count
        gap_duration_s[gap] = duration
        gap_level[gap] = (
            "short"
            if duration <= short_max_s
            else "medium"
            if duration <= medium_max_s
            else "long"
        )
        left_anchor_frame[gap] = left
        right_anchor_frame[gap] = right
        max_gap_s = max(max_gap_s, duration)

        if left < 0:
            out[gap] = raw[right]
            method[gap] = "hold_next"
            nearest_anchor_distance_s[gap] = (right - frames[gap]) / fps
            anchor_span_s[gap] = np.nan
        elif right < 0:
            out[gap] = raw[left]
            method[gap] = "hold_previous"
            nearest_anchor_distance_s[gap] = (frames[gap] - left) / fps
            anchor_span_s[gap] = np.nan
        else:
            weight = (frames[gap] - left) / (right - left)
            # Convex combination of the two flanking anchors.
            out[gap] = (1 - weight) * out[left] + weight * out[right]
            method[gap] = "linear"
            interpolation_weight[gap] = weight
            nearest_anchor_distance_s[gap] = (
                np.minimum(frames[gap] - left, right - frames[gap]) / fps
            )
            anchor_span_s[gap] = (right - left) / fps

    filled_frames = int(filled.sum())
    gap_count = edges.size // 2
    metadata = {
        "filled": filled,
        "method": method,
        "gap_level": gap_level,
        "gap_id": gap_id,
        "gap_frames": gap_frames,
        "gap_duration_s": gap_duration_s,
        "left_anchor_frame": left_anchor_frame,
        "right_anchor_frame": right_anchor_frame,
        "nearest_anchor_distance_s": nearest_anchor_distance_s,
        "anchor_span_s": anchor_span_s,
        "interpolation_weight": interpolation_weight,
        "filled_frames": np.asarray(filled_frames),
        "filled_fraction": np.asarray(filled_frames / n),
        "gap_count": np.asarray(gap_count),
        "max_gap_s": np.asarray(max_gap_s),
        "total_filled_s": np.asarray(filled_frames / fps),
        "short_max_s": np.asarray(short_max_s),
        "medium_max_s": np.asarray(medium_max_s),
        "policy": np.asarray("linear_interior_nearest_edge"),
        "version": np.asarray(COMPLETION_POLICY_VERSION),
    }
    counts = ", ".join(
        f"{name}={np.count_nonzero(method == name)}"
        for name in ("observed", "linear", "hold_next", "hold_previous")
    )
    notes = [
        f"EEF completion filled {filled_frames}/{n} frames ({filled_frames / n:.1%}) "
        f"across {gap_count} gaps; longest gap {max_gap_s:.6g} s. "
        f"Method frame counts: {counts}.",
        "Labels describe provenance and gap duration, not uncertainty estimates; "
        "observed means a provided finite anchor, not independent image evidence.",
    ]
    return out, metadata, notes

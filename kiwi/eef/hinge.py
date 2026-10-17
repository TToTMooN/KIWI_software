"""Pure hinge-angle math: per-jaw fits, ROI, plane-warp geometry, rescue.

``hinge_angle_in_plane`` is the geometric heart of the Chopsticks estimator:
given N>=2 dot positions in marker-plane metres, fit a line through each side
(PCA with an outlier trim) and return the angle between the two chopstick
lines plus per-side diagnostics, falling back to one jaw when the other is
occluded. The supporting functions cover the drop-dot retry, the long-gap
rescue, the spatial ROI filter and the shared plane-warp geometry.

Everything here is numpy-only (the warp and dot detection that feed it live
in :mod:`kiwi.eef.chopsticks`, which owns the cv2 half). Dots are always
detected in the warped plane (the ROI filter works in plane coordinates).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hinge_angle_in_plane",
    "hinge_angle_with_drop_dot_retry",
    "plane_warp_geometry",
    "roi_keep_indices",
    "long_gap_rescue",
    "wrap180",
]

# Per-side outlier-trim defaults (overridable via config / CLI). A dot whose
# perpendicular residual to the side line exceeds ``max(TRIM_M, TRIM_MAD_K *
# 1.4826 * MAD)`` is treated as an outlier and dropped. TRIM_M is the absolute
# floor in metres (it dominates the clean case where MAD is ~0); the MAD term
# adapts to noisier fits.
_DEFAULT_OUTLIER_TRIM_M = 0.003
_DEFAULT_OUTLIER_TRIM_MAD_K = 3.0


def wrap180(a: float) -> float:
    """Wrap an angle difference into [-180, 180) degrees."""
    return (a + 180.0) % 360.0 - 180.0


def _pca_line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid + outward-oriented principal axis (PCA) of ``pts`` (N>=2)."""
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    # 2x2 covariance; eigh returns eigenvalues in ascending order, so the
    # principal axis is the LAST column of the eigenvector matrix.
    cov = centred.T @ centred
    _evals, evecs = np.linalg.eigh(cov)
    direction = evecs[:, -1]
    # Orient outward (away from origin). PCA returns direction up to a sign;
    # pick the sign so the vector and centroid lie in the same half-plane.
    if float(np.dot(direction, centroid)) < 0:
        direction = -direction
    return centroid, direction


def _perp_residuals(
    pts: np.ndarray, centroid: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Perpendicular distance of each point to the line (centroid, direction)."""
    normal = np.array([-direction[1], direction[0]])
    return np.abs((pts - centroid) @ normal)


def _fit_side_line(
    pts: np.ndarray,
    *,
    outlier_trim: bool = True,
    trim_m: float = _DEFAULT_OUTLIER_TRIM_M,
    trim_mad_k: float = _DEFAULT_OUTLIER_TRIM_MAD_K,
) -> dict:
    """PCA line fit through ``pts`` (N>=2, plane coords) with an outlier trim.

    With ``outlier_trim`` and N>=3, each leave-one-out subset is fit and the
    removal that leaves the smallest squared residual in the remaining points
    is selected; that point is dropped when its deleted residual exceeds
    ``max(trim_m, trim_mad_k * 1.4826 * MAD)`` (MAD over the cleaned set),
    refitting until clean or only two points remain (a line needs two). Clean
    input is left untouched (fit identical to plain PCA) and skips the
    leave-one-out work.

    Returns a dict with:
      - ``centroid``: mean of the inlier pts ((2,) float64)
      - ``direction``: unit vector along the principal axis, oriented so
        ``dot(direction, centroid) > 0`` - i.e., pointing outward from the
        marker origin.
      - ``angle_to_normal_deg``: signed angle of ``direction`` from the
        marker's +Y axis, in degrees. Negative = tilts toward -X (left),
        positive = tilts toward +X (right). For a symmetric gripper the left
        side's value is the negative of the right's.
      - ``endpoint_a`` / ``endpoint_b``: line endpoints obtained by projecting
        the inliers onto ``direction``. ``a`` is the "base" end (smaller
        projection), ``b`` is the "tip" end (larger projection).
      - ``n_dots``: number of points fed in (>=2).
      - ``n_inliers`` / ``n_dropped``: post-trim inlier / dropped counts.
      - ``inlier_indices``: indices into the input points retained by the fit.
      - ``rms_residual_m``: RMS perpendicular residual of the inlier fit - a
        per-side quality metric (meaningful only once a side has >2 dots; a
        line through 2 points always fits exactly).
    """
    pts = np.asarray(pts, dtype=np.float64)
    keep = np.ones(len(pts), dtype=bool)
    if outlier_trim:
        # Drop the worst leave-one-out outlier and refit, until clean or only
        # a line's worth of points (2) remains. Needs >=3 so the others-fit
        # has >=2.
        while keep.sum() >= 3:
            cur_idx = np.flatnonzero(keep)
            cur = pts[cur_idx]
            # Quick exit: if the joint fit's worst residual is under the
            # absolute floor, the side is clean - skip the leave-one-out work.
            # (Floor only, not the MAD threshold.)
            c0, d0 = _pca_line(cur)
            if float(_perp_residuals(cur, c0, d0).max()) <= float(trim_m):
                break
            n = len(cur)
            deleted = np.empty(n)
            remaining_cost = np.empty(n)
            for j in range(n):
                others = cur[np.arange(n) != j]
                cj, dj = _pca_line(others)
                deleted[j] = _perp_residuals(cur[j][None, :], cj, dj)[0]
                remaining_cost[j] = float(np.sum(_perp_residuals(others, cj, dj) ** 2))
            # For three points every remaining pair fits exactly; use the
            # deleted residual to break numerical ties, independent of units.
            tolerance = np.finfo(float).eps * max(
                float(np.sum((cur - cur.mean(0)) ** 2)), 1e-30
            )
            best = np.flatnonzero(remaining_cost <= remaining_cost.min() + tolerance)
            j = int(best[np.argmax(deleted[best])])
            others = cur[np.arange(n) != j]
            cj, dj = _pca_line(others)
            res_o = _perp_residuals(others, cj, dj)
            mad_o = float(np.median(np.abs(res_o - np.median(res_o))))
            thr = max(float(trim_m), float(trim_mad_k) * 1.4826 * mad_o)
            if float(deleted[j]) <= thr:
                break
            keep[cur_idx[j]] = False
    inliers = pts[keep]
    centroid, direction = _pca_line(inliers)
    # Signed angle from +Y axis. atan2(x, y) is intentional: rotating
    # +Y toward +X gives a positive angle (right-leaning chopstick).
    angle_rad = np.arctan2(float(direction[0]), float(direction[1]))
    proj = (inliers - centroid) @ direction
    endpoint_a = centroid + float(proj.min()) * direction
    endpoint_b = centroid + float(proj.max()) * direction
    res = _perp_residuals(inliers, centroid, direction)
    return {
        "centroid": centroid,
        "direction": direction,
        "angle_to_normal_deg": float(np.degrees(angle_rad)),
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "n_dots": int(len(pts)),
        "n_inliers": int(keep.sum()),
        "n_dropped": int(len(pts) - keep.sum()),
        "inlier_indices": np.flatnonzero(keep),
        "rms_residual_m": float(np.sqrt(np.mean(res**2))),
    }


def hinge_angle_in_plane(
    centers_plane: np.ndarray,
    *,
    outlier_trim: bool = True,
    trim_m: float = _DEFAULT_OUTLIER_TRIM_M,
    trim_mad_k: float = _DEFAULT_OUTLIER_TRIM_MAD_K,
    axis_offset_deg: float = 0.0,
    allow_one_jaw: bool = True,
) -> tuple[float, dict[str, object]]:
    """Hinge open angle from per-side PCA line fits.

    ``centers_plane`` is an ``(N, 2)`` array of dot positions in the marker
    plane (metres, +X right, +Y up). Dots are split by ``sign(plane_x)`` into
    left (x<0) and right (x>=0) sides - this is marker-anchored, so it works
    with any per-side count. Each side fits a PCA line with the outlier trim
    (:func:`_fit_side_line`, ``outlier_trim`` / ``trim_m`` / ``trim_mad_k``
    are forwarded), which drops stray dots one at a time until the side is
    clean or only a line's worth (2) remains - so a 4-dot side can lose two,
    not just one.

    Two modes:

    - **both** (both sides have >=2 dots, the primary path): the open angle
      is the angle *between* the two side vectors - independent of any marker
      mounting rotation, and it carries the ``asymmetry_deg`` cross-check.
    - **one_jaw** (``allow_one_jaw`` and exactly one side has >=2 dots, the
      one-jaw fallback): the gripper is symmetric, so the open angle is
      ``2 * |vis - axis_offset_deg|`` where ``vis`` is the visible side's
      signed angle from +Y and ``axis_offset_deg`` (delta) is the marker ->
      symmetry-axis misalignment. This recovers frames where one jaw is
      occluded (the both-side path would raise). ``asymmetry_deg`` is NaN (no
      cross-side check available).

    Raises ``ValueError`` only when neither side has >=2 dots.

    Returns ``(total_open_deg, info)``. ``info`` always has ``mode``
    (``"both"`` / ``"one_jaw"``) and ``asymmetry_deg``; the both path
    adds ``info["left"]`` / ``info["right"]`` (see :func:`_fit_side_line`);
    the one-jaw path adds the visible side under its name plus
    ``which_side`` and ``vis_angle_to_normal_deg``.
    """
    left_pts = centers_plane[centers_plane[:, 0] < 0]
    right_pts = centers_plane[centers_plane[:, 0] >= 0]
    n_left, n_right = len(left_pts), len(right_pts)

    if n_left >= 2 and n_right >= 2:
        left = _fit_side_line(
            left_pts, outlier_trim=outlier_trim, trim_m=trim_m, trim_mad_k=trim_mad_k
        )
        right = _fit_side_line(
            right_pts, outlier_trim=outlier_trim, trim_m=trim_m, trim_mad_k=trim_mad_k
        )
        # Angle between the two chopstick lines. Both direction vectors are
        # oriented outward from the marker (via the centroid trick above), so
        # the signed dot is well-defined here - no need for ``abs``.
        cos_theta = float(
            np.clip(float(np.dot(left["direction"], right["direction"])), -1.0, 1.0)
        )
        total_open_deg = float(np.degrees(np.arccos(cos_theta)))
        asymmetry_deg = abs(left["angle_to_normal_deg"] + right["angle_to_normal_deg"])
        return total_open_deg, {
            "left": left,
            "right": right,
            "asymmetry_deg": asymmetry_deg,
            "mode": "both",
        }

    if allow_one_jaw and (n_left >= 2 or n_right >= 2):
        which = "left" if n_left >= 2 else "right"
        side = _fit_side_line(
            left_pts if which == "left" else right_pts,
            outlier_trim=outlier_trim,
            trim_m=trim_m,
            trim_mad_k=trim_mad_k,
        )
        vis = float(side["angle_to_normal_deg"])
        # Symmetric gripper: each jaw tilts |vis - delta| from the symmetry
        # axis, so the full open angle is twice that. Wrap the difference into
        # [-180, 180] first so the formula holds for both fan directions
        # (dots fanning +y give |vis| < 90; dots fanning -y with
        # outward-oriented jaws give |vis| in (90, 180]).
        d = wrap180(vis - axis_offset_deg)
        total_open_deg = 2.0 * abs(d)
        return total_open_deg, {
            which: side,
            "asymmetry_deg": float("nan"),
            "mode": "one_jaw",
            "which_side": which,
            "vis_angle_to_normal_deg": vis,
        }

    raise ValueError(
        f"need >=2 dots on at least one side, got left={n_left} right={n_right}"
    )


def hinge_angle_with_drop_dot_retry(
    centers_plane: np.ndarray,
    *,
    max_asymmetry_deg: float,
    min_side_dots: int = 3,
    outlier_trim: bool = True,
    trim_m: float = _DEFAULT_OUTLIER_TRIM_M,
    trim_mad_k: float = _DEFAULT_OUTLIER_TRIM_MAD_K,
    axis_offset_deg: float = 0.0,
    allow_one_jaw: bool = True,
) -> tuple[float, dict[str, object]]:
    """:func:`hinge_angle_in_plane` with the drop-dot retry: refit without the
    single worst dot when the both-side asymmetry gate fails.

    Opt-in via ``enable_drop_dot_retry``; the ``chopstick`` tool config
    enables it.

    The retry drops the single dot whose removal **minimises asymmetry** and
    keeps the result only if it now passes the gate. Two guards apply:

    - a dot is only droppable if its own side still has ``min_side_dots``
      (default 3), so at least two remain;
    - the retried fit must pass the *same* gate, not a relaxed one.

    On failure the ORIGINAL fit and asymmetry are returned unchanged, so the
    caller still rejects the frame with its true diagnostics. On success
    ``info`` carries ``drop_dot_retry = {dropped_index, asymmetry_before,
    asymmetry_after, angle_before}``.
    """
    if min_side_dots < 3:
        raise ValueError(
            f"min_side_dots must be >= 3 so a drop leaves a 2-point line that "
            f"is not one real dot plus one stray, got {min_side_dots}"
        )
    pts = np.asarray(centers_plane, dtype=np.float64).reshape(-1, 2)
    kw = dict(
        outlier_trim=outlier_trim,
        trim_m=trim_m,
        trim_mad_k=trim_mad_k,
        axis_offset_deg=axis_offset_deg,
        allow_one_jaw=allow_one_jaw,
    )
    angle, info = hinge_angle_in_plane(pts, **kw)
    if info.get("mode") != "both":
        return angle, info
    asym0 = float(info["asymmetry_deg"])
    if asym0 <= max_asymmetry_deg:
        return angle, info

    is_left = pts[:, 0] < 0
    n_left, n_right = int(is_left.sum()), int((~is_left).sum())
    best: tuple[float, int, float, dict] | None = None
    for j in range(len(pts)):
        side_n = n_left if is_left[j] else n_right
        if side_n < min_side_dots:
            continue  # dropping here would leave fewer than two on that side
        try:
            a_j, i_j = hinge_angle_in_plane(np.delete(pts, j, axis=0), **kw)
        except ValueError:
            continue
        if i_j.get("mode") != "both":
            continue  # a one-jaw result has no asymmetry to check
        s_j = float(i_j["asymmetry_deg"])
        if not np.isfinite(s_j):
            continue
        if best is None or s_j < best[0]:
            best = (s_j, j, a_j, i_j)

    if best is None or best[0] > max_asymmetry_deg:
        return angle, info
    asym1, j, angle1, info1 = best
    info1 = dict(info1)
    info1["drop_dot_retry"] = {
        "dropped_index": int(j),
        "asymmetry_before": asym0,
        "asymmetry_after": asym1,
        "angle_before": float(angle),
    }
    return angle1, info1


def long_gap_rescue(
    valid: np.ndarray,
    tilt_left: np.ndarray,
    tilt_right: np.ndarray,
    *,
    long_gap_min_frames: int,
    axis_offset_deg: float,
    max_jaw_drift_deg: float = 3.0,
    max_open_deg: float = 180.0,
) -> list[tuple[int, float, dict]]:
    """The long-gap rescue: recover rejected frames in long runs from the jaw
    that stayed consistent with the nearest accepted frame.

    A whole-series post-pass, not a per-frame fit. Frames rejected by the
    asymmetry gate (after the drop-dot retry) are considered only in runs
    longer than ``long_gap_min_frames``. For each such frame, pick the jaw
    whose tilt drifted least from the same jaw on the nearest
    ORIGINALLY-valid frame, and if that drift is within
    ``max_jaw_drift_deg``, estimate the opening from that single jaw:
    ``2 * |wrap180(tilt - axis_offset_deg)|`` - the same formula as the
    one-jaw fallback. These rescued raw estimates become anchors for the
    runner's completion step. Any remaining missing intervals are completed
    independently by interior interpolation or edge holds;
    ``long_gap_min_frames`` limits this rescue pass, not completion.

    Limits:

    - **Runs <= long_gap_min_frames are never touched.** Their missing raw
      values remain available for the runner's completion step.
    - **The drift test is against originally-valid frames only** - rescued
      frames never become references, so estimates cannot chain and drift.
      Inside a long run every frame compares against the same flanking valid
      frames. A frame remains rejected when neither jaw has a finite
      reference tilt within ``max_jaw_drift_deg``.
    - ``max_open_deg`` re-applies the one-jaw fallback's plausibility cap.

    Inputs are per-frame arrays aligned to the series: ``valid`` marks frames
    that carry a value; ``tilt_left``/``tilt_right`` are the per-jaw
    ``angle_to_normal_deg`` (NaN where absent - both are present on
    gate-failed both-mode frames). Returns ``(frame_index, angle_deg, audit)``
    tuples, where ``audit`` holds ``side``, ``drift_deg``, ``ref_frame`` and
    ``delta_deg``; the caller writes the value and records ``audit`` so
    rescued frames stay identifiable in the artifact.
    """
    valid = np.asarray(valid, dtype=bool)
    tl = np.asarray(tilt_left, dtype=np.float64)
    tr = np.asarray(tilt_right, dtype=np.float64)
    n = valid.size
    valid_idx = np.flatnonzero(valid)
    out: list[tuple[int, float, dict]] = []
    if valid_idx.size == 0:
        return out
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if j - i > long_gap_min_frames:
            for f in range(i, j):
                best: tuple[float, str, float, int] | None = None
                for tilts, side in ((tl, "left"), (tr, "right")):
                    tf = float(tilts[f])
                    if not np.isfinite(tf):
                        continue
                    k = int(np.searchsorted(valid_idx, f))
                    cands = [
                        int(g)
                        for g in (
                            valid_idx[k - 1 : k].tolist()
                            + valid_idx[k : k + 1].tolist()
                        )
                        if np.isfinite(tilts[int(g)])
                    ]
                    if not cands:
                        continue
                    g = min(cands, key=lambda g: abs(g - f))
                    drift = abs(wrap180(tf - float(tilts[g])))
                    if best is None or drift < best[0]:
                        best = (drift, side, tf, g)
                if best is None or best[0] > max_jaw_drift_deg:
                    continue
                drift, side, tf, g = best
                est = 2.0 * abs(wrap180(tf - axis_offset_deg))
                if est > max_open_deg:
                    continue
                out.append(
                    (
                        f,
                        est,
                        {
                            "side": side,
                            "drift_deg": round(drift, 3),
                            "ref_frame": g,
                            "delta_deg": round(float(axis_offset_deg), 3),
                        },
                    )
                )
        i = j
    return out


def roi_keep_indices(
    plane_mm: np.ndarray,
    img_xy: np.ndarray,
    *,
    half_marker_mm: float,
    image_wh: tuple[int, int],
    roi_radius_mm: float | None = None,
    roi_inner_radius_mm: float | None = None,
    roi_center_mm: tuple[float, float] = (0.0, 0.0),
    roi_semicircle_keep: str = "-y",
    clip_to_image: bool = True,
) -> list[int]:
    """Indices of dots that pass the dot-search ROI filters.

    ``plane_mm`` is ``(N, 2)`` dot positions in marker-plane millimetres and
    ``img_xy`` the matching ``(N, 2)`` rectified-image pixel coordinates.
    Three filters, all marker-anchored:

    - **in-marker square**: drop dots inside ``+-half_marker_mm`` on both
      axes (the marker's own border dots).
    - **ROI shape**: a **semicircle** of ``roi_radius_mm`` about
      ``roi_center_mm`` when ``roi_radius_mm`` is set - the half-disk on the
      ``roi_semicircle_keep`` side of the center (``"+y"``/``"-y"``/``"+x"``/
      ``"-x"``). Pick the half the dots fan into from the hinge (for the
      chopstick tool the marker sits above the dots, so ``"-y"``). No shape
      filter when unset.
    - **image edges**: when ``clip_to_image``, drop dots whose image pixel
      falls outside ``[0, W) x [0, H)`` (e.g. the semicircle extending past
      the image when the marker is near an edge).

    ``roi_inner_radius_mm`` turns the half-disk into a **half-annulus**,
    dropping everything inside that radius as well. ``None`` keeps the plain
    half-disk.
    """
    plane = np.asarray(plane_mm, dtype=np.float64).reshape(-1, 2)
    n = len(plane)
    if n == 0:
        return []
    px, py = plane[:, 0], plane[:, 1]
    keep = ~((np.abs(px) <= half_marker_mm) & (np.abs(py) <= half_marker_mm))
    if roi_radius_mm is not None:
        cx, cy = roi_center_mm
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        keep &= d2 <= roi_radius_mm**2
        if roi_inner_radius_mm is not None:
            if roi_inner_radius_mm >= roi_radius_mm:
                raise ValueError(
                    f"roi_inner_radius_mm ({roi_inner_radius_mm}) must be "
                    f"smaller than roi_radius_mm ({roi_radius_mm}); the "
                    f"annulus would be empty"
                )
            keep &= d2 >= roi_inner_radius_mm**2
        half = {
            "+y": py >= cy,
            "-y": py <= cy,
            "+x": px >= cx,
            "-x": px <= cx,
        }.get(roi_semicircle_keep)
        if half is None:
            raise ValueError(
                f"roi_semicircle_keep must be +y/-y/+x/-x, got {roi_semicircle_keep!r}"
            )
        keep &= half
    if clip_to_image:
        ixy = np.asarray(img_xy, dtype=np.float64).reshape(-1, 2)
        w, h = image_wh
        keep &= (ixy[:, 0] >= 0) & (ixy[:, 0] < w) & (ixy[:, 1] >= 0) & (ixy[:, 1] < h)
    return np.flatnonzero(keep).tolist()


def plane_warp_geometry(
    px_w: int,
    px_h: int,
    extent_x_m: float,
    extent_y_m: float,
    origin_x_frac: float = 0.5,
    origin_y_frac: float = 0.5,
) -> tuple[np.ndarray, float, float, float, float]:
    """Geometry of a fronto-parallel plane-warp window.

    Maps marker-plane metres (+X right, +Y up) onto a ``px_w`` x ``px_h``
    pixel grid whose full width/height spans ``2*extent_x_m`` /
    ``2*extent_y_m``, with plane (0, 0) landing at pixel
    ``(origin_x_frac * px_w, origin_y_frac * px_h)``. +X right gives a
    positive x scale; +Y up gives a negative y scale (image rows count
    down). Single source for the warp used by the estimator and the tuner.

    Returns ``(H_plane_to_warped, s_x, s_y, ox_px, oy_px)`` with ``s_x`` /
    ``s_y`` in px per metre and ``ox_px`` / ``oy_px`` the origin pixel.
    """
    s_x = px_w / (2.0 * extent_x_m)
    s_y = px_h / (2.0 * extent_y_m)
    ox_px = origin_x_frac * px_w
    oy_px = origin_y_frac * px_h
    H_plane_to_warped = np.array(
        [[s_x, 0.0, ox_px], [0.0, -s_y, oy_px], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return H_plane_to_warped, s_x, s_y, ox_px, oy_px

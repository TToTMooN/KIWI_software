"""Unit tests for kiwi.eef.hinge - per-jaw fits, ROI, warp geometry, rescue.

Run: pixi run test
Pure numpy geometry - no OpenCV / video needed (kiwi.eef.hinge is cv2-free
by design; the warp + detection that feed it live in kiwi.eef.chopsticks).

Synthetic gripper: each side's dots lie on a ray ``t * [+-sin a, cos a]`` from
the marker origin (left x<0, right x>=0). For a symmetric pair at +-a from +Y the
angle between the two side vectors is ``2a`` and the asymmetry is 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from kiwi.eef.hinge import _fit_side_line, hinge_angle_in_plane, roi_keep_indices

_A = 20.0  # per-side tilt from +Y (deg) -> total open = 2*_A
_DL = np.array([-np.sin(np.radians(_A)), np.cos(np.radians(_A))])  # left ray dir
_DR = np.array([np.sin(np.radians(_A)), np.cos(np.radians(_A))])  # right ray dir


def _ray(direction, ts):
    return np.asarray(ts)[:, None] * direction


def test_clean_symmetric_angle():
    """Clean 4+4 dots -> total = 2a, zero asymmetry, nothing dropped."""
    ts = [0.01, 0.02, 0.03, 0.04]
    pts = np.vstack([_ray(_DL, ts), _ray(_DR, ts)])
    total, info = hinge_angle_in_plane(pts)
    assert total == pytest.approx(2 * _A, abs=1e-4)
    assert info["asymmetry_deg"] == pytest.approx(0.0, abs=1e-4)
    assert info["left"]["n_dropped"] == 0
    assert info["right"]["n_dropped"] == 0
    assert info["left"]["rms_residual_m"] == pytest.approx(0.0, abs=1e-9)


def _data_with_right_outlier():
    """4 clean left, 3 clean right + 1 right dot pushed 2 cm off its ray."""
    left = _ray(_DL, [0.01, 0.02, 0.03, 0.04])
    right_clean = _ray(_DR, [0.01, 0.02, 0.03])
    # Perpendicular toward +x so the displaced dot stays on the right side
    # (a -x push of this size would cross the axis and be split as a left dot).
    perp_out = np.array([_DR[1], -_DR[0]])
    outlier = 0.025 * _DR + 0.02 * perp_out  # on the ray, then 2 cm perpendicular
    right = np.vstack([right_clean, outlier])
    return np.vstack([left, right])


def test_outlier_trim_drops_single_outlier():
    """The outlier trim drops the one bad dot and recovers the clean angle."""
    pts = _data_with_right_outlier()
    total, info = hinge_angle_in_plane(pts)  # outlier trim on by default
    assert info["right"]["n_dropped"] == 1
    assert info["right"]["n_inliers"] == 3
    assert total == pytest.approx(2 * _A, abs=1e-3)


def test_without_outlier_trim_the_fit_is_corrupted():
    """Plain least-squares PCA keeps the outlier and gets a wrong angle."""
    pts = _data_with_right_outlier()
    total, info = hinge_angle_in_plane(pts, outlier_trim=False)
    assert info["right"]["n_dropped"] == 0
    assert info["right"]["n_inliers"] == 4
    assert abs(total - 2 * _A) > 2.0  # the outlier visibly skews the fit


def test_two_dot_side_not_trimmed():
    """A side with exactly 2 dots is never trimmed (a line needs two)."""
    fit = _fit_side_line(_ray(_DR, [0.01, 0.02]))
    assert fit["n_inliers"] == 2
    assert fit["n_dropped"] == 0


def test_clean_fit_matches_plain_pca():
    """On clean input the trimmed fit equals plain PCA (no points dropped)."""
    pts = _ray(_DR, [0.01, 0.02, 0.03, 0.04])
    trimmed = _fit_side_line(pts, outlier_trim=True)
    plain = _fit_side_line(pts, outlier_trim=False)
    assert trimmed["n_dropped"] == 0
    np.testing.assert_allclose(trimmed["direction"], plain["direction"], atol=1e-12)
    assert trimmed["angle_to_normal_deg"] == pytest.approx(
        plain["angle_to_normal_deg"], abs=1e-9
    )


# --- one-jaw fallback (occlusion) ----------------------------------------


def test_both_mode_tagged():
    ts = [0.01, 0.02, 0.03, 0.04]
    pts = np.vstack([_ray(_DL, ts), _ray(_DR, ts)])
    _, info = hinge_angle_in_plane(pts)
    assert info["mode"] == "both"


def test_one_jaw_fallback_recovers_angle():
    """One jaw occluded (all dots on one side) -> 2x the visible tilt."""
    pts = _ray(_DR, [0.01, 0.02, 0.03, 0.04])  # right side only (left x<0 empty)
    total, info = hinge_angle_in_plane(pts)
    assert info["mode"] == "one_jaw"
    assert info["which_side"] == "right"
    assert np.isnan(info["asymmetry_deg"])
    assert total == pytest.approx(2 * _A, abs=1e-4)  # equals the both-side answer


def test_one_jaw_applies_axis_offset():
    """The symmetry-axis offset delta shifts the one-jaw estimate."""
    pts = _ray(_DR, [0.01, 0.02, 0.03])  # vis tilt = +_A
    total, _ = hinge_angle_in_plane(pts, axis_offset_deg=5.0)
    assert total == pytest.approx(2 * abs(_A - 5.0), abs=1e-4)


def test_one_jaw_minus_y_fan_wraps():
    """Dots fanning -y (marker above the jaws), so the visible
    jaw's angle-to-normal is >90deg and the symmetry axis sits near +-180deg.
    The one-jaw open angle must wrap the (vis - delta) difference into
    [-180, 180] before doubling; without it an 80deg-open gripper reads as
    ~360-80 and gets rejected. Here 2a=40deg must be recovered, not ~320."""
    ts = [0.01, 0.02, 0.03, 0.04]
    dl2 = np.array([-np.sin(np.radians(_A)), -np.cos(np.radians(_A))])  # left, fans -y
    dr2 = np.array([np.sin(np.radians(_A)), -np.cos(np.radians(_A))])  # right, fans -y

    # A both-side frame still recovers 2a (arccos of the ray dot is
    # orientation-free), and its symmetry axis is the circular mean of the two
    # jaw angles - ~180deg, which a plain arithmetic mean would wrongly give 0.
    both_total, both_info = hinge_angle_in_plane(
        np.vstack([_ray(dl2, ts), _ray(dr2, ts)])
    )
    assert both_total == pytest.approx(2 * _A, abs=1e-4)
    la = np.radians(both_info["left"]["angle_to_normal_deg"])
    ra = np.radians(both_info["right"]["angle_to_normal_deg"])
    delta = float(
        np.degrees(np.arctan2(np.sin(la) + np.sin(ra), np.cos(la) + np.cos(ra)))
    )
    assert abs(abs(delta) - 180.0) < 5.0  # symmetry axis is the -y direction

    # One jaw occluded (left only): with that delta the wrapped formula recovers
    # 2a; without the wrap, 2*|vis - delta| would give ~680deg here.
    total, info = hinge_angle_in_plane(_ray(dl2, ts), axis_offset_deg=delta)
    assert info["mode"] == "one_jaw"
    assert total == pytest.approx(2 * _A, abs=1e-4)


def test_one_jaw_disabled_raises():
    pts = _ray(_DR, [0.01, 0.02, 0.03])
    with pytest.raises(ValueError):
        hinge_angle_in_plane(pts, allow_one_jaw=False)


def test_neither_side_enough_raises():
    """One dot per side -> no side has >=2 -> ValueError even with fallback on."""
    pts = np.vstack([_ray(_DL, [0.01]), _ray(_DR, [0.01])])
    with pytest.raises(ValueError):
        hinge_angle_in_plane(pts)


# --- circle ROI + image-edge clip -------------------------------------

_IMG = (1920, 1920)
_INSIDE = np.full((4, 2), 500.0)  # all rectified-image pixels well in-frame


def test_roi_semicircle_keeps_inside_drops_outside():
    plane = np.array([[50.0, 0.0], [0.0, 60.0], [100.0, 0.0], [0.0, 0.0]])
    keep = roi_keep_indices(
        plane,
        _INSIDE,
        half_marker_mm=7.0,
        image_wh=_IMG,
        roi_radius_mm=80.0,
        roi_semicircle_keep="+y",
    )
    # (50,0) in; (0,60) in (+y half); (100,0) outside r=80; (0,0) inside marker square
    assert keep == [0, 1]


def test_roi_image_edge_clip():
    plane = np.array([[50.0, 0.0], [50.0, 0.0]])  # both inside the circle
    img = np.array([[500.0, 500.0], [-5.0, 500.0]])  # second pixel is off-frame
    keep = roi_keep_indices(
        plane, img, half_marker_mm=7.0, image_wh=_IMG, roi_radius_mm=80.0
    )
    assert keep == [0]
    keep_noclip = roi_keep_indices(
        plane,
        img,
        half_marker_mm=7.0,
        image_wh=_IMG,
        roi_radius_mm=80.0,
        clip_to_image=False,
    )
    assert keep_noclip == [0, 1]


def test_roi_marker_square_excluded():
    plane = np.array([[3.0, 3.0], [5.0, -4.0]])  # both within +-7mm marker square
    img = np.full((2, 2), 500.0)
    keep = roi_keep_indices(
        plane, img, half_marker_mm=7.0, image_wh=_IMG, roi_radius_mm=80.0
    )
    assert keep == []


def test_roi_semicircle_keeps_chosen_half():
    plane = np.array(
        [[50.0, 20.0], [50.0, -20.0]]
    )  # upper dot, lower dot (both in r=80)
    img = np.full((2, 2), 500.0)
    kw = dict(half_marker_mm=7.0, image_wh=_IMG, roi_radius_mm=80.0)
    keep_up = roi_keep_indices(plane, img, roi_semicircle_keep="+y", **kw)
    assert keep_up == [0]  # +y keeps the upper dot
    keep_lo = roi_keep_indices(plane, img, roi_semicircle_keep="-y", **kw)
    assert keep_lo == [1]  # -y keeps the lower dot (the chopsticks default)


# --- drop-dot retry (opt-in via enable_drop_dot_retry) ----------------------


# A recorded frame's geometry (plane mm) with one stray dot.
# Left side: 3 real jaw dots plus a stray at r=73 mm sitting between them.
_STRAY_DOT_GEOMETRY = (
    np.array(
        [
            [
                -72.3,
                46.9,
            ],  # 0  real, r=89
            [-67.0, 24.6],  # 1  STRAY, r=73 <- what should be dropped
            [-51.1, 21.5],  # 2  real, r=58
            [-45.1, 14.2],  # 3  real, r=49
            [43.2, 11.8],  # 4  real right
            [48.6, 18.2],  # 5  real right
            [66.6, 39.5],  # 6  real right
        ]
    )
    / 1000.0
)


def test_independent_trim_recovers_recorded_outlier_without_drop_dot_retry():
    from kiwi.eef.hinge import hinge_angle_with_drop_dot_retry

    plain, info_plain = hinge_angle_in_plane(_STRAY_DOT_GEOMETRY)
    # The independent fit drops the stray and preserves all real dots.
    assert info_plain["left"]["n_dropped"] == 1
    assert list(info_plain["left"]["inlier_indices"]) == [0, 2, 3]
    assert info_plain["left"]["angle_to_normal_deg"] == pytest.approx(-39.8, abs=0.5)
    assert float(info_plain["asymmetry_deg"]) <= 2.5
    assert plain == pytest.approx(80.70, abs=1.0)

    # The angular estimator accepts the independent fit without retry.
    total, info = hinge_angle_with_drop_dot_retry(
        _STRAY_DOT_GEOMETRY, max_asymmetry_deg=2.5
    )
    assert "drop_dot_retry" not in info
    assert float(info["asymmetry_deg"]) <= 2.5
    assert info["left"]["angle_to_normal_deg"] == pytest.approx(-39.8, abs=0.5)
    assert total == pytest.approx(80.70, abs=1.0)


def test_drop_dot_retry_gives_up_rather_than_reporting_a_partial_clean():
    """When the best single drop still fails the gate, the caller must see the
    ORIGINAL asymmetry."""
    from kiwi.eef.hinge import hinge_angle_with_drop_dot_retry

    a0, i0 = hinge_angle_in_plane(_STRAY_DOT_GEOMETRY)
    a, i = hinge_angle_with_drop_dot_retry(_STRAY_DOT_GEOMETRY, max_asymmetry_deg=0.2)
    assert "drop_dot_retry" not in i
    assert float(i["asymmetry_deg"]) == pytest.approx(
        float(i0["asymmetry_deg"]), abs=1e-12
    )
    assert a == pytest.approx(a0, abs=1e-12)


def test_drop_dot_retry_is_a_no_op_on_a_clean_frame():
    from kiwi.eef.hinge import hinge_angle_with_drop_dot_retry

    ts = [0.01, 0.02, 0.03]
    pts = np.vstack([_ray(_DL, ts), _ray(_DR, ts)])
    a0, i0 = hinge_angle_in_plane(pts)
    a1, i1 = hinge_angle_with_drop_dot_retry(pts, max_asymmetry_deg=2.5)
    assert "drop_dot_retry" not in i1
    assert a1 == pytest.approx(a0, abs=1e-12)


def test_drop_dot_retry_will_not_drop_from_a_two_dot_side():
    """min_side_dots guards the manufacturing risk: a 2-dot side must never become a
    1-dot side, and a 3-dot side must not be reduced to one real dot plus a stray."""
    from kiwi.eef.hinge import hinge_angle_with_drop_dot_retry

    pts = np.vstack([_ray(_DL, [0.01, 0.02]), _ray(_DR, [0.01, 0.02, 0.03, 0.09])])
    _, i = hinge_angle_with_drop_dot_retry(pts, max_asymmetry_deg=0.0, min_side_dots=5)
    assert "drop_dot_retry" not in i  # no side has 5, so nothing is droppable


def test_drop_dot_retry_rejects_too_few_side_dots():
    from kiwi.eef.hinge import hinge_angle_with_drop_dot_retry

    pts = _STRAY_DOT_GEOMETRY
    with pytest.raises(ValueError, match="min_side_dots must be >= 3"):
        hinge_angle_with_drop_dot_retry(pts, max_asymmetry_deg=2.5, min_side_dots=2)


def test_roi_inner_radius_makes_a_half_annulus():
    """The hand-blob cluster sits well inside the innermost real jaw dot, so an
    inner radius drops it without touching the dots. Radii here: blob ~34 mm,
    real dots >= 41 mm."""
    plane = np.array([[34.0, 1.0], [0.0, 34.0], [43.0, 10.0], [0.0, 60.0]])
    img = np.full((4, 2), 500.0)
    kw = dict(
        half_marker_mm=7.0, image_wh=_IMG, roi_radius_mm=95.0, roi_semicircle_keep="+y"
    )
    assert roi_keep_indices(plane, img, **kw) == [
        0,
        1,
        2,
        3,
    ]  # plain half-disk keeps all
    # r measured about the ROI centre (0, 0) here: 34.0, 34.0, 44.1, 60.0
    assert roi_keep_indices(plane, img, roi_inner_radius_mm=38.0, **kw) == [2, 3]


def test_roi_inner_radius_is_measured_from_the_roi_centre():
    """Not from the plane origin - the deployed centre is offset (0, -5)."""
    plane = np.array([[0.0, 30.0]])  # r=30 from the origin, r=35 from (0,-5)
    img = np.full((1, 2), 500.0)
    kw = dict(
        half_marker_mm=7.0,
        image_wh=_IMG,
        roi_radius_mm=95.0,
        roi_semicircle_keep="+y",
        roi_center_mm=(0.0, -5.0),
    )
    assert roi_keep_indices(plane, img, roi_inner_radius_mm=33.0, **kw) == [
        0
    ]  # 35 >= 33
    assert roi_keep_indices(plane, img, roi_inner_radius_mm=36.0, **kw) == []  # 35 < 36


def test_roi_inner_radius_rejects_an_empty_annulus():
    plane = np.array([[50.0, 10.0]])
    img = np.full((1, 2), 500.0)
    with pytest.raises(ValueError, match="must be smaller than"):
        roi_keep_indices(
            plane,
            img,
            half_marker_mm=7.0,
            image_wh=_IMG,
            roi_radius_mm=95.0,
            roi_inner_radius_mm=95.0,
        )


def test_roi_inner_radius_ignored_without_an_outer_radius():
    """Without roi_radius_mm there is no radial component, so an inner radius
    must not silently apply (resolve_params rejects the combination upstream;
    the geometry itself just has no annulus to cut)."""
    plane = np.array([[10.0, 10.0]])  # r=14.1, well inside any inner radius
    img = np.full((1, 2), 500.0)
    keep = roi_keep_indices(
        plane,
        img,
        half_marker_mm=7.0,
        image_wh=_IMG,
        roi_inner_radius_mm=38.0,
    )
    assert keep == [0]


def test_plane_warp_geometry_round_trip():
    from kiwi.eef.hinge import plane_warp_geometry

    H, s_x, s_y, ox_px, oy_px = plane_warp_geometry(800, 400, 0.077, 0.0385)
    # Plane origin lands at the configured origin pixel.
    origin = H @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(origin[:2] / origin[2], [ox_px, oy_px])
    # +X right, +Y up: a point up-right of the origin maps right and UP (row decreases).
    p = H @ np.array([0.01, 0.01, 1.0])
    u, v = p[:2] / p[2]
    assert u > ox_px and v < oy_px
    # The full window spans exactly 2*extent on each axis.
    assert np.isclose(s_x * 2 * 0.077, 800)
    assert np.isclose(s_y * 2 * 0.0385, 400)
    # Inverting the pixel mapping recovers the plane coordinate.
    x_back = (u - ox_px) / s_x
    y_back = (oy_px - v) / s_y
    assert np.allclose([x_back, y_back], [0.01, 0.01])


def test_plane_warp_geometry_off_center_origin():
    from kiwi.eef.hinge import plane_warp_geometry

    H, _, _, ox_px, oy_px = plane_warp_geometry(
        800, 800, 0.05, 0.05, origin_x_frac=0.25, origin_y_frac=0.75
    )
    assert (ox_px, oy_px) == (200.0, 600.0)
    origin = H @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(origin[:2] / origin[2], [200.0, 600.0])


# --- long-gap rescue --------------------------------------------------------


def _rescue(**kw):
    from kiwi.eef.hinge import long_gap_rescue

    return long_gap_rescue(**kw)


def _series(n, run_lo, run_hi, tl_val=-20.0, tr_val=20.0, tl_run=None, tr_run=None):
    """Valid frames with mirrored tilts (+-20 -> angle 40), one invalid run.

    ``tl_run``/``tr_run`` set the run's per-jaw tilts (rejected rows keep them).
    """
    valid = np.ones(n, bool)
    valid[run_lo:run_hi] = False
    tl = np.full(n, tl_val)
    tr = np.full(n, tr_val)
    if tl_run is not None:
        tl[run_lo:run_hi] = tl_run
    if tr_run is not None:
        tr[run_lo:run_hi] = tr_run
    return valid, tl, tr


def test_long_gap_rescue_rescues_a_static_hold_from_the_clean_jaw():
    """Left jaw contaminated (tilt swung 30 deg), right stable: every frame of
    the long run recovers as 2*|right tilt| with the right-side audit."""
    valid, tl, tr = _series(100, 30, 70, tl_run=-50.0, tr_run=20.0)
    out = _rescue(
        valid=valid,
        tilt_left=tl,
        tilt_right=tr,
        long_gap_min_frames=20,
        axis_offset_deg=0.0,
    )
    assert [f for f, _, _ in out] == list(range(30, 70))
    for _, est, audit in out:
        assert est == pytest.approx(40.0, abs=1e-9)
        assert audit["side"] == "right"
        assert audit["drift_deg"] == pytest.approx(0.0, abs=1e-9)
        assert audit["ref_frame"] in (29, 70)


def test_long_gap_rescue_never_touches_short_runs():
    """A run short enough for completion to interpolate must stay untouched."""
    valid, tl, tr = _series(100, 40, 55, tr_run=20.0)  # 15 <= long_gap_min_frames 20
    assert (
        _rescue(
            valid=valid,
            tilt_left=tl,
            tilt_right=tr,
            long_gap_min_frames=20,
            axis_offset_deg=0.0,
        )
        == []
    )


def test_long_gap_rescue_moving_gripper_stays_an_honest_gap():
    """Both jaws drifted past max_jaw_drift_deg since the flanking accepted
    frames: no jaw is trustworthy, so nothing ships."""
    valid, tl, tr = _series(100, 30, 70, tl_run=-28.0, tr_run=28.0)  # 8 deg drift
    assert (
        _rescue(
            valid=valid,
            tilt_left=tl,
            tilt_right=tr,
            long_gap_min_frames=20,
            axis_offset_deg=0.0,
            max_jaw_drift_deg=3.0,
        )
        == []
    )


def test_long_gap_rescue_picks_the_jaw_that_drifted_least():
    valid, tl, tr = _series(100, 30, 70, tl_run=-21.0, tr_run=35.0)  # drift 1 vs 15
    out = _rescue(
        valid=valid,
        tilt_left=tl,
        tilt_right=tr,
        long_gap_min_frames=20,
        axis_offset_deg=0.0,
    )
    assert out and all(a["side"] == "left" for _, _, a in out)
    assert out[0][1] == pytest.approx(42.0, abs=1e-9)  # 2*|-21|


def test_long_gap_rescue_applies_the_axis_offset_and_wrap():
    """delta shifts the estimate; a -y fan (tilt ~ -160, delta ~ 180)
    must wrap to the small true angle, not ~680."""
    valid, tl, tr = _series(60, 20, 50, tl_run=-160.0, tr_run=160.0)
    tl[valid] = -160.0
    tr[valid] = 160.0
    out = _rescue(
        valid=valid,
        tilt_left=tl,
        tilt_right=tr,
        long_gap_min_frames=20,
        axis_offset_deg=180.0,
    )
    assert out and out[0][1] == pytest.approx(40.0, abs=1e-6)


def test_long_gap_rescue_respects_the_open_cap():
    valid, tl, tr = _series(100, 30, 70, tr_run=95.0)
    tr[valid] = 95.0  # no drift, but 2*95 = 190 > 180
    assert (
        _rescue(
            valid=valid,
            tilt_left=np.full(100, np.nan),
            tilt_right=tr,
            long_gap_min_frames=20,
            axis_offset_deg=0.0,
            max_open_deg=180.0,
        )
        == []
    )


def test_long_gap_rescue_no_chaining():
    """References are ORIGINALLY-valid frames only: with no valid frame at all,
    nothing rescues (nothing to measure drift against)."""
    valid = np.zeros(50, bool)
    assert (
        _rescue(
            valid=valid,
            tilt_left=np.full(50, -20.0),
            tilt_right=np.full(50, 20.0),
            long_gap_min_frames=10,
            axis_offset_deg=0.0,
        )
        == []
    )

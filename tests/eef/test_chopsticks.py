"""Tests for kiwi.eef.chopsticks + the kiwi.eef registry.

The E2E half renders synthetic red dots of KNOWN plane geometry into images
(drawn with cv2, not real footage) by projecting plane
points through the same camera model the estimator is given, then asserts
the whole warp -> detect -> ROI -> fit chain recovers the known hinge angle.
The postprocess half builds frame states directly and checks the auto-delta
correction, the rescue wiring, and the long-gap threshold's frame count.

Run: pixi run test
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from kiwi.core.pipeline import Detections, GripperEstimator, GripperState  # noqa: E402
from kiwi.eef import ESTIMATOR_KEYS, EstimatorContext, get_estimator  # noqa: E402
from kiwi.eef.chopsticks import long_gap_min_frames, postprocess  # noqa: E402
from kiwi.eef.config import resolve_params  # noqa: E402
from kiwi.core.geometry import apply_homography_batch  # noqa: E402

# --- the synthetic scene ---------------------------------------------------

W = H = 640
K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 320.0], [0.0, 0.0, 1.0]])
MARKER_SIZE = 0.014
# Marker at 0.26 m with a realistic mount tilt. Deliberately NOT
# fronto-parallel: a flat square is the classic IPPE two-fold ambiguity and
# solvePnP may return the mirrored pose, which no real mount ever presents.
FIXED_POSE = {
    "rvec": [2.7, 0.12, 0.05],
    "tvec": [0.0, 0.02, 0.26],
    "fov_deg": 110.0,
    "out_size": [W, H],
    "lens": "front",
}
_A = 20.0  # per-jaw tilt from +Y -> true open angle 2*_A

# extent 5.5 x 14 mm marker = a 77 mm half-extent plane warp; every real dot
# below sits inside it, the ROI annulus is 38..65 mm. This is the merged
# `session:` recipe an ArmSpec hands the runner (tool < unit < episode-arm).
SESSION_BLOCK = {
    "polarity": "red",
    "min_area": 100,
    "max_area": 900,
    "threshold": 150,
    "extent_x": 5.5,
    "roi_radius_mm": 65.0,
    "roi_inner_radius_mm": 38.0,
    "roi_semicircle_keep": "+y",
}


def _jaw_points(side: int, tilt_deg: float, radii_mm=(42, 52, 62)) -> np.ndarray:
    """Dots on a ray tilted ``tilt_deg`` from +Y, on the given side (in m)."""
    d = np.array([np.sin(np.radians(tilt_deg)), np.cos(np.radians(tilt_deg))])
    pts = np.array([r / 1000.0 * d for r in radii_mm])
    assert np.all(np.sign(pts[:, 0]) == side) or tilt_deg == 0.0
    return pts


def _ctx(**over) -> EstimatorContext:
    base = dict(
        session_block=SESSION_BLOCK,
        K=K,
        image_wh=(W, H),
        marker_id=4,
        marker_size_m=MARKER_SIZE,
        fixed_pose=FIXED_POSE,
        fixed_pose_source="config/gripper/chopsticks-v3-1/units/test-unit.yaml",
    )
    base.update(over)
    return EstimatorContext(**base)


def _render(est, plane_pts_m: np.ndarray, dot_radius_px: int = 6) -> np.ndarray:
    """Red dots at the given plane positions, via the estimator's own H.

    Drawn at sub-pixel centres (``shift=4``): integer rounding would move
    each dot up to 0.5 px (~0.16 mm in plane), which over a short 3-dot
    baseline bends the fitted jaw line by ~0.5 deg - a rendering artifact,
    not an estimator error.
    """
    frame = np.full((H, W, 3), 128, dtype=np.uint8)
    H_plane_to_img = est.fixed_H[0]
    img = apply_homography_batch(H_plane_to_img, plane_pts_m)
    for u, v in img:
        cv2.circle(
            frame,
            (int(round(u * 16)), int(round(v * 16))),
            dot_radius_px * 16,
            (0, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
            shift=4,
        )
    return frame


def _no_dets() -> Detections:
    return Detections(
        corners=np.zeros((0, 4, 2), np.float32), ids=np.zeros(0, np.int32)
    )


def _make(ctx=None):
    bundle = get_estimator("chopsticks")
    est = bundle.make_frame_estimator(ctx or _ctx())
    return bundle, est


# --- registry ---------------------------------------------------------------


def test_registry_returns_the_chopsticks_bundle():
    bundle, est = _make()
    assert bundle.name == "chopsticks" and bundle.unit == "deg"
    assert isinstance(est, GripperEstimator)  # runtime Protocol check
    assert est.pose_mode == "fixed_pose"
    assert est.pose_source == "config/gripper/chopsticks-v3-1/units/test-unit.yaml"
    assert "chopsticks" in ESTIMATOR_KEYS


def test_registry_unknown_key_names_the_source_and_the_known_keys():
    """An estimator key with no registered bundle gets the file, the known keys,
    and the field the name was read from - the frozen `design.estimator`."""
    with pytest.raises(
        ValueError,
        match=r"parallel.*parallel-v1/tool\.yaml.*chopsticks.*design\.estimator",
    ):
        get_estimator("parallel", source="config/gripper/parallel-v1/tool.yaml")


def test_the_session_block_is_strict_at_estimator_construction():
    """A typo'd knob refuses here rather than silently running the builtin:
    the frame estimator resolves the merged session: block through the strict
    config_from_block, so a bad key never reaches a written artifact."""
    with pytest.raises(ValueError, match=r"unknown session key.*min_aera"):
        _make(_ctx(session_block={**SESSION_BLOCK, "min_aera": 100}))


# --- the per-frame path, end to end ----------------------------------------


def test_fixed_pose_recovers_the_known_angle():
    _, est = _make()
    pts = np.vstack([_jaw_points(-1, -_A), _jaw_points(+1, +_A)])
    # Three decoys the ROI must drop: inside the marker square, past the
    # outer radius (still inside the warp), inside the inner annulus cut
    # (the operator's-hand cluster).
    decoys = np.array([[0.003, 0.003], [0.0, 0.072], [0.020, 0.025]])
    state = est(_render(est, np.vstack([pts, decoys])), _no_dets(), K)

    assert state.has_pose  # unconditional in fixed-pose mode
    assert state.n_detected == 9 and state.n_kept == 6
    assert state.error is None
    assert state.value == pytest.approx(2 * _A, abs=0.8)
    assert state.detail["mode"] == "both"
    assert abs(state.detail["asymmetry_deg"]) < 1.0
    assert state.detail["n_left"] == 3 and state.detail["n_right"] == 3


def test_one_jaw_fallback_on_occlusion():
    _, est = _make()
    state = est(_render(est, _jaw_points(+1, +_A)), _no_dets(), K)
    assert state.detail["mode"] == "one_jaw"
    assert state.detail["which_side"] == "right"
    assert state.value == pytest.approx(2 * _A, abs=0.8)


def test_asymmetry_gate_rejects_but_keeps_the_diagnostics():
    _, est = _make()
    pts = np.vstack([_jaw_points(-1, -_A), _jaw_points(+1, +70.0)])  # asym ~50
    state = est(_render(est, pts), _no_dets(), K)
    assert state.value is None
    assert state.error is not None and state.error.startswith("asymmetry")
    assert state.rejected_value is not None
    # The long-gap rescue needs the tilts of gate-failed both-mode frames.
    assert state.detail["mode"] == "both"
    assert np.isfinite(state.detail["left_angle_to_normal_deg"])


def test_every_dropout_carries_a_reason():
    """Too-few-dots frames get the fit's reason string instead of a silent no-value row."""
    _, est = _make()
    state = est(_render(est, np.array([[0.05, 0.05]])), _no_dets(), K)
    assert state.value is None
    assert state.error is not None and state.error.startswith("need >=2 dots")


def test_marker_pnp_mode_uses_detections():
    ctx = _ctx(fixed_pose=None)
    bundle, est = _make(ctx)
    assert est.pose_mode == "marker_pnp"
    assert any("no fixed_pose" in w for w in est.warnings)
    assert bundle.make_detector(ctx) is not None  # PnP mode needs a detector

    # No marker in the frame: no pose, no value, and no error string -
    # has_pose is the signal here.
    fixed_est = _make()[1]  # to render with the true pose
    frame = _render(fixed_est, _jaw_points(+1, +_A))
    state = est(frame, _no_dets(), K)
    assert not state.has_pose and state.value is None and state.error is None

    # Feed the true marker corners as Detections: PnP recovers the pose and
    # the chain lands on the same angle.
    obj = np.array(
        [
            [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
            [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
            [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
            [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
        ],
        dtype=np.float32,
    )
    proj, _ = cv2.projectPoints(
        obj,
        np.asarray(FIXED_POSE["rvec"]),
        np.asarray(FIXED_POSE["tvec"]),
        K,
        np.zeros(5),
    )
    dets = Detections(
        corners=proj.reshape(1, 4, 2).astype(np.float32),
        ids=np.array([4], dtype=np.int32),
    )
    both = _render(fixed_est, np.vstack([_jaw_points(-1, -_A), _jaw_points(+1, +_A)]))
    state = est(both, dets, K)
    assert state.has_pose
    assert state.value == pytest.approx(2 * _A, abs=0.8)


def test_pnp_mode_without_a_marker_id_is_refused():
    with pytest.raises(ValueError, match="marker id"):
        _make(_ctx(fixed_pose=None, marker_id=None))


def test_fixed_pose_mismatch_warnings():
    fp = dict(FIXED_POSE, fov_deg=160.0, out_size=[1920, 1920])
    _, est = _make(_ctx(fixed_pose=fp))
    assert est.pose_mode == "fixed_pose"
    assert any("fov_deg" in w for w in est.warnings)
    assert any("out_size" in w for w in est.warnings)


# --- whole-series post-passes ----------------------------------------------


def _both_state(tilt_l: float, tilt_r: float) -> GripperState:
    return GripperState(
        value=abs(tilt_r - tilt_l),
        detail={
            "mode": "both",
            "left_angle_to_normal_deg": tilt_l,
            "right_angle_to_normal_deg": tilt_r,
            "asymmetry_deg": abs(tilt_l + tilt_r),
            "n_left": 4,
            "n_right": 4,
        },
    )


def _one_jaw_state(vis: float, *, value=None) -> GripperState:
    return GripperState(
        value=2 * abs(vis) if value is None else value,
        detail={
            "mode": "one_jaw",
            "which_side": "right",
            "vis_angle_to_normal_deg": vis,
            "n_vis": 4,
        },
    )


def test_postprocess_auto_delta_corrects_one_jaw_frames():
    """True axis offset delta=5: jaws at delta-+a. One-jaw frames ran at
    delta=0 (value 2|vis|); the post-pass recomputes them to 2|vis - delta|."""
    delta, a = 5.0, 20.0
    states = [_both_state(delta - a, delta + a) for _ in range(25)]
    states.append(_one_jaw_state(delta + a))  # loop value = 2*(delta+a) = 50
    params = resolve_params(None, {"polarity": "red"})
    detail, notes = postprocess(states, params, fps=30.0)

    assert float(detail["delta_deg"]) == pytest.approx(delta, abs=1e-6)
    assert states[-1].value == pytest.approx(2 * a, abs=1e-6)
    assert any("auto axis offset" in n for n in notes)


def test_postprocess_auto_delta_can_revoke_past_the_cap():
    delta, a = 5.0, 20.0
    states = [_both_state(delta - a, delta + a) for _ in range(25)]
    states.append(_one_jaw_state(delta + 30.0))  # corrected: 60 > cap 50
    params = resolve_params(None, {"one_jaw_max_open_deg": 50.0})
    postprocess(states, params, fps=30.0)
    s = states[-1]
    assert s.value is None
    assert s.error is not None and s.error.startswith("one-jaw")


def test_postprocess_too_few_both_frames_leaves_delta_zero():
    states = [_both_state(-20.0, 20.0) for _ in range(5)]
    states.append(_one_jaw_state(25.0))
    detail, notes = postprocess(states, resolve_params(None), fps=30.0)
    assert float(detail["delta_deg"]) == 0.0
    assert states[-1].value == pytest.approx(50.0)  # untouched
    assert any("too few both-side frames" in n for n in notes)


def test_postprocess_configured_delta_is_never_recalibrated():
    states = [_both_state(-15.0, 25.0) for _ in range(30)]  # circular mean 5
    params = resolve_params(None, {"symmetry_axis_offset_deg": 2.0})
    detail, _ = postprocess(states, params, fps=30.0)
    assert float(detail["delta_deg"]) == 2.0


def test_postprocess_long_gap_rescue_wiring():
    """A long gate-rejected run with one steady jaw: values land, errors
    clear, and the rescue audit arrays mark exactly those frames."""
    tilt = 20.0
    states = [_both_state(-tilt, tilt) for _ in range(100)]
    for i in range(30, 70):
        s = states[i]
        s.value = None
        s.error = "asymmetry 40.0deg > 30deg"
        s.rejected_value = 60.0
        s.detail["left_angle_to_normal_deg"] = -50.0  # contaminated jaw
        s.detail["right_angle_to_normal_deg"] = tilt  # steady jaw
    params = resolve_params(
        None,
        {
            "enable_long_gap_rescue": True,
            "symmetry_axis_offset_deg": 0.0,
            "long_gap_min_s": 0.167,  # 5 frames at 30 fps << the 40-run
        },
    )
    detail, notes = postprocess(states, params, fps=30.0)
    assert all(states[i].value == pytest.approx(2 * tilt) for i in range(30, 70))
    assert all(states[i].error is None for i in range(30, 70))
    assert all(states[i].rejected_value == 60.0 for i in range(30, 70))
    rescued = detail["long_gap_rescued"]
    assert rescued[30:70].all() and not rescued[:30].any()
    assert set(detail["long_gap_rescue_side"][30:70]) == {"right"}
    assert (detail["long_gap_rescue_ref_frame"][30:70] >= 0).all()
    assert np.allclose(detail["long_gap_rescue_drift_deg"][30:70], 0.0)
    assert any("long-gap rescue: 40 frames" in n for n in notes)


def test_long_gap_min_frames_follows_the_clip_frame_rate():
    """The duration is converted per clip: the default 0.167 s is 5 frames at
    29.97 fps and ~4x that at 119.88 fps, and never rounds below one."""
    assert long_gap_min_frames(0.167, 29.97) == 5
    assert long_gap_min_frames(0.167, 119.88) == 20
    assert long_gap_min_frames(0.200, 30.0) == 6
    assert long_gap_min_frames(0.001, 29.97) == 1


@pytest.mark.parametrize("fps", [0.0, -1.0])
def test_long_gap_min_frames_rejects_nonpositive_fps(fps):
    """Never guess 30: a non-positive fps raises."""
    with pytest.raises(ValueError, match="fps must be positive"):
        long_gap_min_frames(0.167, fps)


def test_postprocess_empty_clip_yields_empty_audit_columns():
    detail, _ = postprocess([], resolve_params(None), fps=30.0)
    assert detail["long_gap_rescued"].shape == (0,) and detail["mode"].shape == (0,)


def test_closed_open_endpoints():
    """The endpoints come off the MERGED session mapping (tool defaults <
    this unit's measured hinge < episode-arm), so closed_open takes the same
    block the estimator ran on."""
    bundle, _ = _make()
    merged = {**SESSION_BLOCK, "angle_closed_deg": 70.0, "angle_open_deg": 165.0}
    assert bundle.closed_open(merged) == (70.0, 165.0)
    with pytest.raises(
        ValueError, match="session config carries no angle_closed_deg.*session: block"
    ):
        bundle.closed_open({"polarity": "red"})
    with pytest.raises(ValueError, match="divide by zero"):
        bundle.closed_open({"angle_closed_deg": 5.0, "angle_open_deg": 5.0})


def test_roi_outline_is_drawn_on_both_panels():
    """The tuner/overlay must draw the ROI outline.
    The estimator stashes the ROI geometry per frame, draw_overlay outlines
    it on the rectified canvas, and the debug warp view comes back annotated."""
    from kiwi.eef.chopsticks import _roi_paths, draw_overlay

    bundle, est = _make()
    est.debug = True
    frame = _render(est, _jaw_points(+1, +_A))
    state = est(frame, _no_dets(), K)

    roi = state.detail["roi"]
    assert roi["radius_mm"] == 65.0 and roi["inner_mm"] == 38.0
    paths = _roi_paths(roi)
    assert len(paths) == 4  # marker square, outer arc, chord, inner arc
    # Every path is inside the kept (+y) half, in metres.
    assert all(p.ndim == 2 and p.shape[1] == 2 for p in paths)
    assert paths[1][:, 1].min() >= -1e-9  # outer arc stays in +y

    canvas = frame.copy()
    draw_overlay(canvas, state)
    assert not np.array_equal(canvas, frame)  # something was drawn

    # The debug warp view is annotated (differs from a raw re-warp).
    warped_raw = cv2.warpPerspective(
        frame, est._H_plane_to_warped @ est.fixed_H[1], est._warp_wh
    )
    assert not np.array_equal(state.detail["warped"], warped_raw)
    assert state.detail["warped"].shape == warped_raw.shape

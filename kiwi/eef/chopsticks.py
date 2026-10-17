"""The Chopsticks bundle: plane pose -> warp -> red dots -> per-jaw fit.

The batch runner and interactive tuner share this per-frame implementation
through the :class:`kiwi.core.pipeline.GripperEstimator` contract.

Per frame (:class:`ChopsticksFrameEstimator`):

1. **Plane pose** - by ONE of two mutually exclusive modes, chosen once at
   construction and used for every frame (they never mix, and neither falls
   back to the other mid-run):

   - *fixed-pose* (default when the unit's ``build.fixed_pose`` resolves):
     reuse the calibrated plane <-> image homographies without detecting the
     marker. Marker visibility does not affect plane-pose availability;
     usable jaw dots are still needed for an angle. ``has_pose`` is then
     unconditionally True, not a detection-quality signal.
   - *marker-based* (``live_pnp``, or no ``fixed_pose`` anywhere): filter
     the shared ``Detections`` for the tool marker, solvePnP (IPPE_SQUARE),
     then ``homography_from_aruco_pose``.

   Either way the homographies are anchored on a plane offset
   ``z_offset_m`` along marker -Z.
2. **Dot detection** in the fronto-parallel plane warp, converted to
   marker-plane mm.
3. **ROI filter**: in-marker square + the semicircle / annulus + the
   image-edge clip (:func:`kiwi.eef.hinge.roi_keep_indices`).
4. **Per-jaw fit + gates**: :func:`kiwi.eef.hinge.hinge_angle_in_plane`
   (a PCA line per jaw with the outlier trim, the drop-dot retry when
   enabled, and the one-jaw fallback when a jaw is occluded), then the
   per-mode sanity gate - both-side asymmetry, one-jaw plausibility cap. A
   gated frame keeps its fit in ``rejected_value`` + the reason in ``error``.

Dot-fit failures and gate rejections record their reason in ``error``. A missing
marker pose returns ``has_pose=False`` with no angle and may leave ``error=None``.

Whole-series (:func:`postprocess`, run after the frame loop): the auto
symmetry-axis calibration (delta from the circular mean of both-side jaw
tilts) and the long-gap rescue of rejected runs longer than ``long_gap_min_s``.
The applied delta and every rescue stay auditable in the artifact's
``chopsticks/`` namespace. The batch runner then completes the corrected raw
values.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING

import cv2
import numpy as np

from kiwi.core.geometry import apply_homography_batch, homography_from_aruco_pose
from kiwi.core.markers import marker_object_points, resolve_aruco_dict
from kiwi.core.pipeline import Detections, GripperState
from kiwi.eef.config import DotParams, config_from_block, resolve_params
from kiwi.eef.hinge import (
    hinge_angle_in_plane,
    hinge_angle_with_drop_dot_retry,
    plane_warp_geometry,
    roi_keep_indices,
    long_gap_rescue,
    wrap180,
)

if TYPE_CHECKING:
    from kiwi.eef import EstimatorContext

__all__ = [
    "DELTA_MIN_BOTH_FRAMES",
    "DOT_PLANE_Z_OFFSET_M",
    "WARP_PX",
    "ChopsticksFrameEstimator",
    "closed_open",
    "long_gap_min_frames",
    "make_frame_estimator",
    "postprocess",
]

# The dot plane sits this far along marker -Z from the marker face (the dots
# are printed on a recessed surface).
DOT_PLANE_Z_OFFSET_M = -0.005

# Width/height of the plane-detection warp in pixels. min_area / max_area in
# the config are tuned at THIS resolution.
WARP_PX = 800

# Both-side frames needed before the auto delta calibration trusts itself.
DELTA_MIN_BOTH_FRAMES = 20

_DEFAULT_DICTIONARY = "DICT_4X4_50"


def _empty_detections() -> Detections:
    return Detections(
        corners=np.zeros((0, 4, 2), dtype=np.float32),
        ids=np.zeros(0, dtype=np.int32),
    )


@dataclasses.dataclass
class ChopsticksFrameEstimator:
    """Per-frame Chopsticks state, satisfying the core GripperEstimator.

    Build through :func:`make_frame_estimator`, which resolves the config,
    picks the plane-pose mode, and precomputes the warp geometry. Frames go
    through ``__call__(rectified, detections, K)``; in fixed-pose mode
    ``detections`` is ignored.
    """

    params: DotParams
    marker_size_m: float
    image_wh: tuple[int, int]
    pose_mode: str
    pose_source: str
    fixed_H: tuple[np.ndarray, np.ndarray] | None
    marker_id: int | None
    axis_offset_deg: float
    z_offset_m: float = DOT_PLANE_Z_OFFSET_M
    warp_px: int = WARP_PX
    warp_height_px: int | None = None
    origin_x_frac: float = 0.5
    origin_y_frac: float = 0.5
    warnings: list[str] = dataclasses.field(default_factory=list)
    # Tuner hook: stash the warped plane view, the threshold mask and every
    # dot candidate in warp px ("warped" / "mask" / "warp_centers" /
    # "warp_areas") so the tuner can show the exact grid min_area /
    # max_area are measured on.
    debug: bool = False

    def __post_init__(self) -> None:
        w = int(self.warp_px)
        h = int(self.warp_height_px) if self.warp_height_px else w
        ex = self.marker_size_m * self.params.extent_x
        ey = self.marker_size_m * self.params.extent_y
        H, s_x, s_y, ox, oy = plane_warp_geometry(
            w, h, ex, ey, self.origin_x_frac, self.origin_y_frac
        )
        self._warp_wh = (w, h)
        self._H_plane_to_warped = H
        self._scale = (s_x, s_y)
        self._origin_px = (ox, oy)
        self._half_marker_mm = self.marker_size_m * 1000.0 / 2.0
        self._obj_pts = marker_object_points(self.marker_size_m)

    # ---- plane pose ------------------------------------------------------ #

    def _pose(
        self, detections: Detections, K: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.fixed_H is not None:
            return self.fixed_H
        hits = np.flatnonzero(np.asarray(detections.ids) == self.marker_id)
        if hits.size == 0:
            return None
        corners = np.asarray(detections.corners[hits[0]], dtype=np.float32).reshape(
            4, 2
        )
        ok, rvec, tvec = cv2.solvePnP(
            self._obj_pts,
            corners,
            K,
            np.zeros(5, dtype=np.float64),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None
        return homography_from_aruco_pose(K, rvec, tvec, z_offset_m=self.z_offset_m)

    # ---- the frame ------------------------------------------------------- #

    def __call__(
        self,
        rectified: np.ndarray,
        detections: Detections,
        K: np.ndarray,
    ) -> GripperState:
        p = self.params
        state = GripperState(
            has_pose=False,
            H_plane_to_img=None,
            H_img_to_plane=None,
            detail={
                "mode": "",
                "drop_dot_retry": None,
                # Static per run; carried per frame so the drawing side (the
                # overlay video, the tuner) can outline the ROI.
                "roi": {
                    "half_marker_mm": self._half_marker_mm,
                    "radius_mm": p.roi_radius_mm,
                    "inner_mm": p.roi_inner_radius_mm,
                    "center_mm": (p.roi_center_x_mm, p.roi_center_y_mm),
                    "keep": p.roi_semicircle_keep,
                },
            },
        )
        pose = self._pose(detections, K)
        if pose is None:
            return state
        H_plane_to_img, H_img_to_plane = pose
        state.has_pose = True
        state.H_plane_to_img = H_plane_to_img
        state.H_img_to_plane = H_img_to_plane

        # Detect in the fronto-parallel plane warp; convert to plane mm.
        w, h = self._warp_wh
        s_x, s_y = self._scale
        ox, oy = self._origin_px
        warped = cv2.warpPerspective(
            rectified,
            self._H_plane_to_warped @ H_img_to_plane,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        from kiwi.eef.dots import detect_dots

        dets, mask = detect_dots(
            warped,
            p.polarity,
            threshold=p.threshold,
            min_area=p.min_area,
            max_area=p.max_area,
            min_circularity=p.min_circularity,
        )
        if self.debug:
            view = warped.copy()
            _draw_roi(
                view,
                state.detail["roi"],
                lambda pts_m: np.column_stack(
                    [ox + pts_m[:, 0] * s_x, oy - pts_m[:, 1] * s_y]
                ),
            )
            state.detail["warped"] = view
            state.detail["mask"] = mask
            state.detail["warp_centers"] = np.array(
                [d["center"] for d in dets], dtype=np.float64
            ).reshape(-1, 2)
            state.detail["warp_areas"] = np.array(
                [d["area"] for d in dets], dtype=np.float64
            )
        state.n_detected = len(dets)
        kept_plane_m = np.empty((0, 2))
        if dets:
            centers_w = np.array([d["center"] for d in dets], dtype=np.float64)
            plane_x_mm = (centers_w[:, 0] - ox) / s_x * 1000.0
            plane_y_mm = (oy - centers_w[:, 1]) / s_y * 1000.0
            plane_pts_m = np.column_stack([plane_x_mm / 1000.0, plane_y_mm / 1000.0])
            img_centers = apply_homography_batch(H_plane_to_img, plane_pts_m)
            kept = roi_keep_indices(
                np.column_stack([plane_x_mm, plane_y_mm]),
                img_centers,
                half_marker_mm=self._half_marker_mm,
                image_wh=self.image_wh,
                roi_radius_mm=p.roi_radius_mm,
                roi_inner_radius_mm=p.roi_inner_radius_mm,
                roi_center_mm=(p.roi_center_x_mm, p.roi_center_y_mm),
                roi_semicircle_keep=p.roi_semicircle_keep,
                clip_to_image=p.clip_to_image,
            )
            state.n_kept = len(kept)
            kept_plane_m = plane_pts_m[kept]
            state.detail["centers_plane"] = kept_plane_m
            state.detail["img_centers"] = img_centers
            state.detail["kept_indices"] = kept

        # Fit + gates. The fit itself names the reason when it cannot run
        # (too few dots on every side), so every dropout carries one.
        try:
            if p.enable_drop_dot_retry:
                value, info = hinge_angle_with_drop_dot_retry(
                    kept_plane_m,
                    max_asymmetry_deg=p.max_asymmetry_deg,
                    min_side_dots=p.drop_dot_min_side_dots,
                    outlier_trim=p.enable_outlier_trim,
                    trim_m=p.outlier_trim_mm / 1000.0,
                    trim_mad_k=p.outlier_trim_mad_k,
                    axis_offset_deg=self.axis_offset_deg,
                    allow_one_jaw=p.enable_one_jaw_fallback,
                )
            else:
                value, info = hinge_angle_in_plane(
                    kept_plane_m,
                    outlier_trim=p.enable_outlier_trim,
                    trim_m=p.outlier_trim_mm / 1000.0,
                    trim_mad_k=p.outlier_trim_mad_k,
                    axis_offset_deg=self.axis_offset_deg,
                    allow_one_jaw=p.enable_one_jaw_fallback,
                )
        except ValueError as e:
            state.error = str(e)
            return state

        # Per-mode sanity gate. Both-side: the cross-side asymmetry. One-jaw:
        # no cross-check, only a cap on the angle.
        mode = str(info.get("mode"))
        asym = float(info["asymmetry_deg"])
        reject: str | None = None
        if mode == "one_jaw":
            if value > p.one_jaw_max_open_deg:
                reject = f"one-jaw {value:.0f}deg > {p.one_jaw_max_open_deg:.0f}deg"
        elif asym > p.max_asymmetry_deg:
            # :g prints the float threshold as configured (2.5 on chopstick).
            reject = f"asymmetry {asym:.1f}deg > {p.max_asymmetry_deg:g}deg"

        self._record_fit(state, mode, info)
        if reject is not None:
            state.error = reject
            state.rejected_value = float(value)
        else:
            state.value = float(value)
        return state

    @staticmethod
    def _record_fit(state: GripperState, mode: str, info: dict) -> None:
        """Fit diagnostics into ``detail`` - for accepted AND gated frames
        (the long-gap rescue needs the tilts of gate-failed both-mode frames)."""
        d = state.detail
        d["mode"] = mode
        d["hinge_info"] = info
        if mode == "one_jaw":
            which = str(info["which_side"])
            d["which_side"] = which
            d["vis_angle_to_normal_deg"] = float(info["vis_angle_to_normal_deg"])
            d["n_vis"] = int(info[which]["n_dots"])
        else:
            d["left_angle_to_normal_deg"] = float(info["left"]["angle_to_normal_deg"])
            d["right_angle_to_normal_deg"] = float(info["right"]["angle_to_normal_deg"])
            d["asymmetry_deg"] = float(info["asymmetry_deg"])
            d["n_left"] = int(info["left"]["n_dots"])
            d["n_right"] = int(info["right"]["n_dots"])
            d["drop_dot_retry"] = info.get("drop_dot_retry")


def make_frame_estimator(ctx: "EstimatorContext") -> ChopsticksFrameEstimator:
    """Resolve the config, pick the plane-pose mode, precompute the warp.

    Mode precedence (the caller has already layered where the ``fixed_pose``
    comes from - unit calibration over tool default): ``live_pnp`` forces
    per-frame PnP; otherwise a resolved ``fixed_pose`` selects fixed-pose
    mode; with neither, PnP with a notice in ``warnings``.
    """
    params = resolve_params(config_from_block(ctx.session_block), ctx.overrides)
    if ctx.marker_size_m is None or not ctx.marker_size_m > 0:
        raise ValueError(
            "chopsticks needs the tool marker size in metres: name the arm's "
            "physical build as arms.<side>.gripper_unit in episode.yaml (its "
            "config/gripper/<tool>/units/<unit>.yaml carries "
            "build.marker.size), or pass --marker-size on a standalone run; "
            "it sets both the plane-warp extents and the PnP metric scale"
        )
    warnings: list[str] = []
    fixed_H = None
    pose_mode = "marker_pnp"
    pose_source = ""
    fp = None if ctx.live_pnp else ctx.fixed_pose
    if fp and fp.get("rvec") is not None:
        if (
            fp.get("fov_deg") is not None
            and abs(float(fp["fov_deg"]) - float(params.fov_deg)) > 1e-6
        ):
            warnings.append(
                f"fixed_pose was calibrated at fov_deg={fp['fov_deg']} but this "
                f"run rectifies at {params.fov_deg}; the homography may be off"
            )
        if fp.get("out_size") is not None and tuple(fp["out_size"]) != tuple(
            ctx.image_wh
        ):
            warnings.append(
                f"fixed_pose was calibrated at out_size={list(fp['out_size'])} "
                f"but this run rectifies to {list(ctx.image_wh)}; the "
                f"homography may be off"
            )
        rvec = np.asarray(fp["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(fp["tvec"], dtype=np.float64).reshape(3, 1)
        fixed_H = homography_from_aruco_pose(
            ctx.K, rvec, tvec, z_offset_m=DOT_PLANE_Z_OFFSET_M
        )
        pose_mode = "fixed_pose"
        pose_source = ctx.fixed_pose_source
    else:
        if ctx.marker_id is None:
            raise ValueError(
                "marker-PnP mode needs the tool marker id (a build fact: name "
                "a --gripper-unit whose build.marker.id states this build's "
                "tag, or pass --marker-id on a standalone run)"
            )
        if not ctx.live_pnp:
            warnings.append(
                "no fixed_pose resolved anywhere - falling back to per-frame "
                "marker PnP. Calibrate one with `kiwi eef calibrate`."
            )
    return ChopsticksFrameEstimator(
        params=params,
        marker_size_m=float(ctx.marker_size_m),
        image_wh=tuple(ctx.image_wh),
        pose_mode=pose_mode,
        pose_source=pose_source,
        fixed_H=fixed_H,
        marker_id=ctx.marker_id,
        axis_offset_deg=(
            float(params.symmetry_axis_offset_deg)
            if params.symmetry_axis_offset_deg is not None
            else 0.0
        ),
        warnings=warnings,
    )


def make_detector(ctx: "EstimatorContext"):
    """The per-frame marker detector for marker-PnP mode; ``None`` when the
    fixed pose makes detection unnecessary (the batch runner then feeds empty
    ``Detections``)."""
    if not ctx.live_pnp and ctx.fixed_pose and ctx.fixed_pose.get("rvec") is not None:
        return None
    dict_id = resolve_aruco_dict(ctx.marker_dictionary or _DEFAULT_DICTIONARY)
    allowed = {int(ctx.marker_id)} if ctx.marker_id is not None else None

    def detect(rectified: np.ndarray) -> Detections:
        from kiwi.core.markers import detect_aruco_markers

        corners_list, ids = detect_aruco_markers(
            rectified, marker_dict=dict_id, allowed_ids=allowed
        )
        if not corners_list:
            return _empty_detections()
        corners = np.asarray([c.reshape(4, 2) for c in corners_list], dtype=np.float32)
        return Detections(
            corners=corners, ids=np.asarray(ids, dtype=np.int32).reshape(-1)
        )

    return detect


# ---- whole-series post-passes --------------------------------------------- #


def _tilt_arrays(states: list[GripperState]) -> tuple[np.ndarray, np.ndarray]:
    tl = np.array(
        [float(s.detail.get("left_angle_to_normal_deg", np.nan)) for s in states]
    )
    tr = np.array(
        [float(s.detail.get("right_angle_to_normal_deg", np.nan)) for s in states]
    )
    return tl, tr


def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    ang = np.radians(angles_deg)
    return float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())))


def long_gap_min_frames(duration_s: float, fps: float) -> int:
    """``duration_s`` at ``fps`` as a whole frame count, at least one.

    The long-gap threshold is configured as a duration. A non-positive
    ``fps`` (a container that reports no frame rate) raises ``ValueError``.
    """
    if not fps > 0:
        raise ValueError(
            f"fps must be positive to size the long-gap threshold, got {fps!r}"
        )
    return max(1, int(round(duration_s * fps)))


def postprocess(
    states: list[GripperState], params: DotParams, fps: float
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Correct and rescue raw frame results in place.

    1. **Auto delta**: when ``symmetry_axis_offset_deg`` is unset, the
       one-jaw frames ran with delta=0 during the loop; recompute them
       from the clip's own both-side frames (the circular mean of every
       both-side jaw tilt IS the marker->axis offset). A recompute that
       crosses the plausibility cap revokes the value, so validity may
       change after the frame loop.
    2. **Long-gap rescue** (opt-in): recover rejected frames in runs longer
       than ``long_gap_min_s``, from the jaw consistent with the nearest
       originally-valid frame. Rescued raw estimates become anchors for
       the runner's later completion step.

    Mutates ``states`` (values revoked / rescued in place) and returns
    ``(detail_arrays, notes)`` where ``detail_arrays`` is the
    ``chopsticks/`` namespace of the artifact.
    """
    n = len(states)
    mode = np.array([str(s.detail.get("mode", "")) for s in states])
    tl, tr = _tilt_arrays(states)

    # delta: configured value, else the circular mean over both-side jaw
    # tilts (accepted AND gate-rejected both-mode frames).
    notes: list[str] = []
    both_tilts = np.concatenate([tl[mode == "both"], tr[mode == "both"]])
    both_tilts = both_tilts[np.isfinite(both_tilts)]
    n_both = both_tilts.size // 2
    if params.symmetry_axis_offset_deg is not None:
        delta = float(params.symmetry_axis_offset_deg)
    elif n_both >= DELTA_MIN_BOTH_FRAMES:
        delta = _circular_mean_deg(both_tilts)
        # Correct the accepted one-jaw frames, which ran at delta=0.
        n_corr = n_cap = 0
        for s in states:
            if s.value is None or s.detail.get("mode") != "one_jaw":
                continue
            vis = float(s.detail["vis_angle_to_normal_deg"])
            open_deg = 2.0 * abs(wrap180(vis - delta))
            if open_deg > params.one_jaw_max_open_deg:
                # The recompute pushed it past the cap; drop it exactly as
                # the frame loop would have.
                s.value = None
                s.error = (
                    f"one-jaw {open_deg:.0f}deg > {params.one_jaw_max_open_deg:.0f}deg"
                )
                n_cap += 1
            else:
                s.value = open_deg
                n_corr += 1
        if n_corr or n_cap:
            notes.append(
                f"one-jaw: auto axis offset delta={delta:+.2f}deg "
                f"({n_both} both-side frames) -> {n_corr} corrected, "
                f"{n_cap} capped"
            )
    else:
        delta = 0.0
        n_one_jaw = int(
            sum(
                s.value is not None and s.detail.get("mode") == "one_jaw"
                for s in states
            )
        )
        if n_one_jaw:
            notes.append(
                f"one-jaw: too few both-side frames ({n_both}) to "
                f"auto-calibrate delta; left {n_one_jaw} frames at delta=0"
            )

    min_frames = long_gap_min_frames(params.long_gap_min_s, fps)

    # Rescue raw estimates before completion. Reference tilts come only from
    # originally-valid frames, never from other rescues.
    rescued = np.zeros(n, dtype=bool)
    rescue_side = np.array([""] * n, dtype="U5")
    rescue_drift_deg = np.full(n, np.nan)
    rescue_ref_frame = np.full(n, -1, dtype=np.int32)
    if params.enable_long_gap_rescue:
        valid = np.array([s.value is not None for s in states])
        rescues = long_gap_rescue(
            valid,
            tl,
            tr,
            long_gap_min_frames=min_frames,
            axis_offset_deg=delta,
            max_jaw_drift_deg=params.long_gap_max_jaw_drift_deg,
            max_open_deg=params.one_jaw_max_open_deg,
        )
        for fidx, est, audit in rescues:
            s = states[fidx]
            s.value = float(est)
            s.error = None  # the failed fit stays inspectable in rejected_value
            rescued[fidx] = True
            rescue_side[fidx] = audit["side"]
            rescue_drift_deg[fidx] = audit["drift_deg"]
            rescue_ref_frame[fidx] = audit["ref_frame"]
        notes.append(
            f"long-gap rescue: {len(rescues)} frames recovered "
            f"from long rejected runs (delta={delta:+.2f}deg, "
            f"max jaw drift {params.long_gap_max_jaw_drift_deg}deg)"
        )

    detail = {
        "mode": mode.astype("U11"),
        "which_side": np.array(
            [str(s.detail.get("which_side", "")) for s in states], dtype="U5"
        ),
        "vis_angle_to_normal_deg": np.array(
            [float(s.detail.get("vis_angle_to_normal_deg", np.nan)) for s in states]
        ),
        "n_vis": np.array(
            [int(s.detail.get("n_vis", -1)) for s in states], dtype=np.int32
        ),
        "left_angle_to_normal_deg": tl,
        "right_angle_to_normal_deg": tr,
        "asymmetry_deg": np.array(
            [float(s.detail.get("asymmetry_deg", np.nan)) for s in states]
        ),
        "n_left": np.array(
            [int(s.detail.get("n_left", -1)) for s in states], dtype=np.int32
        ),
        "n_right": np.array(
            [int(s.detail.get("n_right", -1)) for s in states], dtype=np.int32
        ),
        "drop_dot_retry": np.array(
            [bool(s.detail.get("drop_dot_retry")) for s in states], dtype=bool
        ),
        "long_gap_rescued": rescued,
        "long_gap_rescue_side": rescue_side,
        "long_gap_rescue_drift_deg": rescue_drift_deg,
        "long_gap_rescue_ref_frame": rescue_ref_frame,
        "delta_deg": np.asarray(delta),
    }
    return detail, notes


def closed_open(session_block: Mapping) -> tuple[float, float]:
    """The normalization endpoints: hinge angle (deg) at closed and open.

    Session keys (tool default < unit measured override < episode-arm) that
    belong to the export side - ``(x - closed) / (open - closed)`` is how a
    hinge angle becomes the [0, 1] ``tool_state`` channel.
    """
    try:
        closed = float(session_block["angle_closed_deg"])
        opened = float(session_block["angle_open_deg"])
    except (KeyError, TypeError) as e:
        raise ValueError(
            "the resolved session config carries no angle_closed_deg / "
            "angle_open_deg - the tool YAML's session: block (or the gripper "
            "unit's measured override) must declare the hinge angles at "
            "closed and open, or the tool state cannot be normalized to [0, 1]"
        ) from e
    if opened == closed:
        raise ValueError(
            f"angle_open_deg == angle_closed_deg ({opened}); the "
            f"normalization would divide by zero"
        )
    return closed, opened


def _roi_paths(roi) -> list[np.ndarray]:
    """The ROI outline as plane-METER polylines.

    Path 0 is always the in-marker drop square; with a radius configured, the
    kept-half outer arc, the diameter chord along the keep boundary, and (for
    an annulus) the kept-half inner arc follow.
    """
    half = float(roi["half_marker_mm"])
    sq = (
        np.array(
            [[-half, half], [half, half], [half, -half], [-half, -half], [-half, half]]
        )
        / 1000.0
    )
    paths = [sq]
    r = roi.get("radius_mm")
    if r is None:
        return paths
    r = float(r)
    cx, cy = (float(v) for v in roi["center_mm"])
    lo, hi = {
        "+y": (0.0, np.pi),
        "-y": (np.pi, 2.0 * np.pi),
        "+x": (-np.pi / 2.0, np.pi / 2.0),
        "-x": (np.pi / 2.0, 1.5 * np.pi),
    }[str(roi["keep"])]
    th = np.linspace(lo, hi, 49)
    paths.append(np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)]) / 1000.0)
    if str(roi["keep"]) in ("+y", "-y"):
        paths.append(np.array([[cx - r, cy], [cx + r, cy]]) / 1000.0)
    else:
        paths.append(np.array([[cx, cy - r], [cx, cy + r]]) / 1000.0)
    inner = roi.get("inner_mm")
    if inner is not None:
        inner = float(inner)
        paths.append(
            np.column_stack([cx + inner * np.cos(th), cy + inner * np.sin(th)]) / 1000.0
        )
    return paths


def _draw_roi(canvas: np.ndarray, roi, plane_to_px) -> None:
    """Outline the ROI on ``canvas``: teal arcs/chord,
    gray in-marker square, tiny ROI label at the centre."""
    for i, path in enumerate(_roi_paths(roi)):
        pts = np.asarray(plane_to_px(path), dtype=np.float64)
        color = (110, 110, 110) if i == 0 else (200, 200, 0)
        cv2.polylines(
            canvas,
            [np.round(pts).astype(np.int32).reshape(-1, 1, 2)],
            False,
            color,
            1,
            cv2.LINE_AA,
        )
    if roi.get("radius_mm") is not None:
        cx, cy = (float(v) for v in roi["center_mm"])
        c = np.asarray(plane_to_px(np.array([[cx, cy]]) / 1000.0))[0]
        cv2.putText(
            canvas,
            "ROI",
            (int(round(c[0])) + 4, int(round(c[1])) + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 0),
            1,
            cv2.LINE_AA,
        )


def draw_overlay(canvas: np.ndarray, state: GripperState) -> None:
    """Annotate a rectified BGR frame in place, for the debug overlay video.

    The ROI outline draws first (teal arcs + chord, gray in-marker square),
    then kept dots ring green, dropped dots gray, and each fitted jaw draws
    its line segment - enough to judge detection, filter and fit at a
    glance. The tuner exposes the estimator's intermediate images for
    closer inspection.
    """
    d = state.detail
    roi = d.get("roi")
    if roi is not None and state.H_plane_to_img is not None:
        _draw_roi(
            canvas,
            roi,
            lambda pts_m: apply_homography_batch(state.H_plane_to_img, pts_m),
        )
    img_centers = d.get("img_centers")
    if img_centers is not None:
        kept = set(d.get("kept_indices") or [])
        for i, (cx, cy) in enumerate(np.asarray(img_centers)):
            c = (int(round(cx)), int(round(cy)))
            if i in kept:
                cv2.circle(canvas, c, 8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.circle(canvas, c, 4, (110, 110, 110), 1, cv2.LINE_AA)
    info = d.get("hinge_info")
    if info and state.H_plane_to_img is not None:
        for side in ("left", "right", d.get("which_side", "")):
            fit = info.get(side) if isinstance(info.get(side), dict) else None
            if fit is None:
                continue
            seg = apply_homography_batch(
                state.H_plane_to_img,
                np.vstack([fit["endpoint_a"], fit["endpoint_b"]]),
            )
            a = (int(round(seg[0, 0])), int(round(seg[0, 1])))
            b = (int(round(seg[1, 0])), int(round(seg[1, 1])))
            cv2.line(canvas, a, b, (0, 255, 255), 2, cv2.LINE_AA)

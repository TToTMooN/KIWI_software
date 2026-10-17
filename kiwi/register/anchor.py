"""Anchor a tracked trajectory to the table marker - the marker world frame.

The ``full_6dof`` convention (the only one that exists, enforced by the
manifest): solvePnP the anchor marker on early rectified frames, compose with
the tracked camera pose at each detection's timestamp, and re-express the
whole trajectory so the marker's own pose IS the world origin - all six DoF
from the marker, z along its normal. That is what co-registers the arms into
one world.

Transform chain, per arm::

    T_cam_marker      marker pose in the camera frame, from solvePnP on a
                      DS->pinhole rectified frame (identity-rotation
                      rectifier, so the pinhole frame == the camera frame).
    T_scene_cam(t*)   the state stage's trajectory, interpolated at the
                      detection's global-clock time. It is a camera-body
                      trajectory, so no T_imu_cam leg enters the chain.
    anchor:           T_scene_marker = T_scene_cam(t*) @ T_cam_marker
    re-expressed:     T_marker_cam(t) = inv(T_scene_marker) @ T_scene_cam(t)

The anchor is averaged (:func:`kiwi.register.pnp.avg_T`,
reprojection-weighted) over every detection that survives the
``max_marker_dist`` filter. Clock algebra: a detection at video time
``t_video`` sits at ``stream_offset_s + t_video`` on the episode global
clock. No IMU pre-roll term enters, because the input camera ``.tum`` is
already global-clock-native.

``T_cam_marker`` must come from the FRONT lens: this composes it directly
with the state pose of the front optical camera, and there is no
``T_camF_camB`` leg here. The file path is guarded as far as a
file can say - :func:`~kiwi.register.detections.load_detections` refuses one
that RECORDS a rear-lens scan, and the scan refuses the rear clip by name -
but this function takes observations, not a file, so a caller that builds
them itself owns the precondition.

The two flip checks (:mod:`kiwi.register.health`) run on every solve and the
result carries their verdict; whether a ``halt`` blocks the write is the
caller's policy (the stage refuses; the CLI has ``--no-flip-check``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.core.trajectory import PoseTrack, Trajectory, load_tum, read_tum_header
from kiwi.register.detections import MarkerObservation
from kiwi.register.health import reach_verdict, spread_verdict, worst
from kiwi.register.pnp import avg_T

__all__ = ["WORLD_FRAME", "AnchorError", "AnchorResult", "solve_anchor"]

#: The frame name every anchored trajectory declares. Export refuses two arms
#: whose frames disagree.
WORLD_FRAME = "marker"


def _weight(reproj: float) -> float:
    """The anchor weight: sharp head-on detections dominate oblique/blurry ones.

    The epsilon only guards a perfect 0.0 reprojection.
    """
    return 1.0 / (reproj**2 + 1e-6)


class AnchorError(ValueError):
    """The solve cannot produce an anchor (as opposed to producing a bad one)."""


@dataclasses.dataclass(frozen=True)
class AnchorResult:
    """One arm's anchored trajectory plus the evidence behind it."""

    trajectory: Trajectory  # frame=WORLD_FRAME, body="camera", global clock
    T_track_marker: np.ndarray  # the averaged anchor, in the track's own frame
    n_detected: int  # every detection the scan recorded
    n_too_far: int  # dropped by max_marker_dist (PnP flip risk)
    n_in_range: int  # survived max_marker_dist and fell inside track coverage
    spread: tuple[str, str]  # displacement flip check ('ok'|'warn'|'halt', msg)
    reach: tuple[str, str]  # reach flip check ('ok'|'warn'|'halt', msg)
    best_frame: int | None  # lowest-reproj detection used (sidecar diagnostics)
    best_reproj: float | None
    diagnostics: dict = dataclasses.field(default_factory=dict)

    @property
    def verdict(self) -> tuple[str, str]:
        """The combined flip verdict - the most severe of the two signals."""
        return worst(self.spread, self.reach)


def _require_track_header(tum: Path) -> None:
    """Refuse a trajectory whose header does not declare the solve's contract.

    The solve composes camera-body poses on the episode global clock; a file
    on any other convention is refused here.
    """
    h = read_tum_header(tum)
    if h is None:
        raise AnchorError(
            f"{tum}: no '(frame=<f> body=<b>; t on <clock>)' header - its "
            f"conventions are unknowable. Anchor the state stage's "
            f"<arm>/scene/state.tum, which declares them."
        )
    if h.body != "camera":
        raise AnchorError(
            f"{tum}: body is {h.body!r}, expected 'camera'. The solve composes "
            f"T_track_cam @ T_cam_marker, which is wrong for any other body."
        )
    if not h.clock.startswith("episode global clock"):
        raise AnchorError(
            f"{tum}: timestamps are on {h.clock!r}, not the episode global "
            f"clock - detections are placed on that clock, so the anchor "
            f"would sample the wrong poses. Re-run the state stage."
        )


def solve_anchor(
    track_tum: Path,
    observations: list[MarkerObservation],
    *,
    stream_offset_s: float,
    max_marker_dist: float = 1.0,
) -> AnchorResult:
    """Anchor ``track_tum`` to the marker seen in ``observations``.

    ``stream_offset_s`` is this stream's entry in the manifest's ``sync:``
    block (<= 0): where its video frame 0 sits on the episode global clock.

    ``max_marker_dist`` (metres, default 1.0) drops detections whose marker
    is farther from the camera. Pass ``inf`` to disable.
    """
    _require_track_header(track_tum)
    t, p, q_xyzw, _ = load_tum(track_tum)
    if len(t) < 2:
        raise AnchorError(f"{track_tum}: {len(t)} pose(s) - nothing to anchor")
    track = PoseTrack(t=t, rotations=Rotation.from_quat(q_xyzw), positions=p)

    candidates: list[np.ndarray] = []  # T_track_marker per detection
    weights: list[float] = []
    best: tuple[float, int] | None = None  # (reproj, frame)
    n_too_far = n_in_range = 0
    candidate_frames: list[int] = []
    for o in observations:
        # max_marker_dist is applied HERE, not at scan time.
        if o.distance_m > max_marker_dist:
            n_too_far += 1
            continue
        t_frame = stream_offset_s + o.t_video
        if not (track.t0 <= t_frame <= track.t1):
            # No tracked pose covers this frame (e.g. outside scene-state coverage).
            continue
        n_in_range += 1
        candidates.append(track.at(t_frame) @ o.T_cam_marker)
        candidate_frames.append(o.frame)
        weights.append(_weight(o.reproj))
        if best is None or o.reproj < best[0]:
            best = (o.reproj, o.frame)

    n_detected = len(observations)
    if not candidates:
        if n_detected and n_too_far == n_detected:
            # The marker WAS seen, just always too far.
            raise AnchorError(
                f"no usable anchor frames: all {n_detected} detection(s) were "
                f"beyond max_marker_dist ({max_marker_dist} m). The marker was "
                f"seen but only from too far (PnP flip risk). Re-capture with "
                f"the marker closer, or raise anchor.max_marker_dist if you "
                f"accept the flip risk."
            )
        raise AnchorError(
            "no usable anchor frames: marker never detected within the tracked "
            "time range. Try a larger anchor.scan_seconds, or check the marker "
            "is visible during scene-state coverage."
        )

    w = np.asarray(weights)
    T_track_marker = avg_T(candidates, weights=w)
    # Flip check #1 (dispersion): the marker is static, so every candidate
    # must agree. A flipped subset is bimodal here.
    diagnostics = {"candidate_frames": candidate_frames, "trajectory_t": t.tolist()}
    spread = spread_verdict(
        np.array([c[:3, 3] for c in candidates]), w, measurements=diagnostics
    )

    T_marker_track = np.linalg.inv(T_track_marker)
    poses = np.repeat(np.eye(4)[None], len(t), axis=0)
    poses[:, :3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    poses[:, :3, 3] = p
    world = Trajectory.from_poses(
        t, T_marker_track[None] @ poses, frame=WORLD_FRAME, body="camera"
    )
    # Flip check #2 (reach): catches the uniform flip that dispersion cannot -
    # every detection wrong the same way still agrees with itself.
    reach = reach_verdict(world.positions, measurements=diagnostics)

    return AnchorResult(
        trajectory=world,
        T_track_marker=T_track_marker,
        n_detected=n_detected,
        n_too_far=n_too_far,
        n_in_range=n_in_range,
        spread=spread,
        reach=reach,
        best_frame=None if best is None else best[1],
        best_reproj=None if best is None else best[0],
        diagnostics=diagnostics,
    )

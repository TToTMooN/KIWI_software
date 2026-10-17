"""Shared per-trajectory solve + pose/offset/continuity math (headless).

The single source of truth for solving one whole demonstrated trajectory
(``solve_trajectory``), plus the workspace transform, mocap-base placement,
rigid-offset and L0 continuity helpers that the batch machinery
(``kiwi.replay.batch``) and the viewer share.

Offsets are ``[dx, dy, dz, roll, pitch, yaw]`` (m, extrinsic-xyz degrees, the
``Mount.rpy_deg`` convention), applied AFTER the workspace transform and
pivoting about the shared workspace anchor, so they compose on top of the
workspace sliders and a gizmo drag feels local.
"""

from __future__ import annotations

import dataclasses
import threading

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.replay.collision import collide
from kiwi.replay.ik import Fr3MinkIK, TCP_SITE
from kiwi.replay.pose import _workspace_transform, _wrist_root_pose

SIDES = ("left", "right")
_YSIGN = {"left": 1.0, "right": -1.0}

# Feasibility thresholds on an episode's worst positional residual.
FEAS_GREEN_M = 0.005  # < 5 mm: tracks essentially everywhere
FEAS_YELLOW_M = 0.020  # < 20 mm: marginal; >= : infeasible as placed
_FEAS_COLORS = {
    "green": (26, 152, 80),
    "yellow": (217, 165, 32),
    "red": (215, 48, 31),
    "gray": (128, 128, 128),
}

# L0 continuity-score curve (VISTA's published breakpoints, arXiv 2606.04708
# Sec 3.3): per consecutive-frame delta, score 100 below the "perfect"
# threshold, linear down to 60 at the "soft" threshold, then an exponential
# tail toward 0. Thresholds are per step of a CONT_REF_FPS-rate stream;
# `continuity_score` scales them for other rates (a longer step covers a
# proportionally larger delta).
CONT_REF_FPS = 30.0
CONT_POS_PERFECT_M, CONT_POS_SOFT_M, CONT_POS_DECAY_M = 0.005, 0.045, 0.100
CONT_ANG_PERFECT_DEG, CONT_ANG_SOFT_DEG, CONT_ANG_DECAY_DEG = 1.0, 9.0, 20.0


# ---------------------------------------------------------------------------
# Shared pose/index helpers
# ---------------------------------------------------------------------------


def _quat_rpy_deg(quat_wxyz) -> np.ndarray:
    """wxyz quaternion -> (roll, pitch, yaw) degrees, extrinsic xyz."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz", degrees=True)


def _rpy_deg_wxyz(rpy_deg) -> np.ndarray:
    """(roll, pitch, yaw) degrees, extrinsic xyz -> wxyz quaternion (inverse
    of _quat_rpy_deg)."""
    x, y, z, w = Rotation.from_euler(
        "xyz", np.asarray(rpy_deg, dtype=np.float64), degrees=True
    ).as_quat()
    return np.array([w, x, y, z])


def _mat_rpy_deg(xmat9) -> np.ndarray:
    """row-major flattened 3x3 rotation -> (roll, pitch, yaw) degrees."""
    m = np.asarray(xmat9, dtype=np.float64).reshape(3, 3)
    return Rotation.from_matrix(m).as_euler("xyz", degrees=True)


def _base_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """RPY -> wxyz quaternion for a base mount: RotZ(yaw).RotX(roll).RotY(pitch).

    Unlike the workspace's ZYX order, pitch is the *innermost* axis here; the
    gimbal-lock singularity is at roll=+/-90. Single-axis slider moves match
    yaw=Z / pitch=Y / roll=X.
    """
    x, y, z, w = Rotation.from_euler(
        "ZXY", [yaw_deg, roll_deg, pitch_deg], degrees=True
    ).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _set_bases(
    d: mujoco.MjData,
    mocapids: dict,
    dist: float,
    z: float,
    roll_left: float,
    roll_right: float,
    pitch_left: float,
    pitch_right: float,
    yaw_left: float,
    yaw_right: float,
) -> None:
    """Write both mount poses into a MjData's mocap slots. Both bases sit at
    height ``z`` and are offset +/-dist/2 along Y; every rotation axis (roll,
    pitch, yaw) is independent per arm - nothing is mirrored."""
    half = dist / 2.0
    for side, y, q in (
        ("left", half, _base_quat(roll_left, pitch_left, yaw_left)),
        ("right", -half, _base_quat(roll_right, pitch_right, yaw_right)),
    ):
        mid = mocapids[side]
        d.mocap_pos[mid] = [0.0, y, z]
        d.mocap_quat[mid] = q


def _set_bases_dict(d: mujoco.MjData, mocapids: dict, base: dict) -> None:
    """``_set_bases`` from the live ``base`` dict the callers keep."""
    _set_bases(
        d,
        mocapids,
        base["dist"],
        base["z"],
        base["roll_left"],
        base["roll_right"],
        base["pitch_left"],
        base["pitch_right"],
        base["yaw_left"],
        base["yaw_right"],
    )


def _resolve_indices(
    model: mujoco.MjModel,
    side: str,
    arm_joints,
    tcp_site: str = TCP_SITE,
    gripper_joints=("rev_left", "rev_right"),
) -> dict:
    arm = [f"{side}_{j}" for j in arm_joints]
    arm_qadr = np.array(
        [model.jnt_qposadr[model.joint(n).id] for n in arm], dtype=np.int32
    )
    gripper_qadr = np.array(
        [model.jnt_qposadr[model.joint(f"{side}_{h}").id] for h in gripper_joints],
        dtype=np.int32,
    )
    site_id = model.site(f"{side}_{tcp_site}").id
    return {"arm_qadr": arm_qadr, "gripper_qadr": gripper_qadr, "site_id": site_id}


def _frame_world_target(
    frame7: np.ndarray,
    wrist_anchor: np.ndarray,
    anchor: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
):
    """World-frame ``(pos, quat)`` target for one trajectory frame: shift the
    wrist by ``wrist_anchor`` (the trajectory point that lands on the anchor),
    then apply the workspace transform to place/orient it at ``anchor``.

    Shared by the IK solve (``solve_trajectory``) and the batch-episode
    paths: overlaying an episode's frames through this reproduces exactly the
    path the arm is commanded to track."""
    sp, sq = _wrist_root_pose(frame7)
    return _workspace_transform(
        sp - wrist_anchor,
        sq,
        offset=anchor,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
    )


# ---------------------------------------------------------------------------
# Rigid offsets (batch-mode global + per-episode), pivoting about the anchor
# ---------------------------------------------------------------------------


def _offset_rot(offset6) -> Rotation:
    return Rotation.from_euler(
        "xyz", np.asarray(offset6, np.float64)[3:6], degrees=True
    )


def apply_offset(
    p: np.ndarray, q_wxyz: np.ndarray, offset6, anchor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one offset to a world pose: rotate about ``anchor``, then
    translate.

    ``p' = R(p - anchor) + anchor + t``; ``q' = R . q``."""
    off = np.asarray(offset6, np.float64)
    rot = _offset_rot(off)
    p2 = rot.apply(np.asarray(p, np.float64) - anchor) + anchor + off[:3]
    q = np.asarray(q_wxyz, np.float64)
    x, y, z, w = (rot * Rotation.from_quat([q[1], q[2], q[3], q[0]])).as_quat()
    return p2, np.array([w, x, y, z])


def compose_offsets(outer, inner) -> np.ndarray:
    """The single offset equivalent to
    ``apply_offset(outer) o apply_offset(inner)`` (both about the same
    anchor): ``R = R_o.R_i``, ``t = R_o(t_i) + t_o``."""
    outer = np.asarray(outer, np.float64)
    inner = np.asarray(inner, np.float64)
    r_o, r_i = _offset_rot(outer), _offset_rot(inner)
    t = r_o.apply(inner[:3]) + outer[:3]
    rpy = (r_o * r_i).as_euler("xyz", degrees=True)
    return np.concatenate([t, rpy])


def offset_segments(segs: np.ndarray, offset6, anchor: np.ndarray) -> np.ndarray:
    """Vectorized ``apply_offset`` (positions only) over a (M, 2, 3) segment
    array. Stage 2 of the path cache: runs on every gizmo drag."""
    off = np.asarray(offset6, np.float64)
    if not off.any():
        return segs
    flat = segs.reshape(-1, 3).astype(np.float64)
    out = _offset_rot(off).apply(flat - anchor) + anchor + off[:3]
    return out.reshape(segs.shape).astype(np.float32)


def shared_anchor(ws: dict) -> np.ndarray:
    """The shared-frame workspace anchor (offset pivot): ``[x, y, z]``."""
    return np.array([float(ws["x"]), float(ws["y"]), float(ws["z"])])


def feasibility_color(worst_pos_m: float | None) -> tuple[int, int, int]:
    """Episode path color from its worst positional residual (gray =
    unsolved)."""
    if worst_pos_m is None:
        return _FEAS_COLORS["gray"]
    if worst_pos_m < FEAS_GREEN_M:
        return _FEAS_COLORS["green"]
    if worst_pos_m < FEAS_YELLOW_M:
        return _FEAS_COLORS["yellow"]
    return _FEAS_COLORS["red"]


# ---------------------------------------------------------------------------
# L0: demo-quality continuity check (runs once at load; slider-invariant)
# ---------------------------------------------------------------------------


def _cont_curve(
    val: np.ndarray, perfect: float, soft: float, decay: float
) -> np.ndarray:
    """The VISTA piecewise score: 100 below ``perfect``, linear to 60 at
    ``soft``, then a continuous exponential tail with scale ``decay``."""
    v = np.asarray(val, np.float64)
    lin = 100.0 - 40.0 * (v - perfect) / (soft - perfect)
    tail = 60.0 * np.exp(-np.maximum(v - soft, 0.0) / decay)
    return np.where(v <= perfect, 100.0, np.where(v <= soft, lin, tail))


def continuity_score(qpos_by_side: dict, *, fps: float | None = None) -> dict:
    """Score the raw commanded demo stream for jumps (sensor dropout, tracking
    loss, abrupt human motion) - the L0 layer of the feasibility stack.

    Per consecutive frame pair and side: positional delta and orientation
    delta (rotation angle between the wrist frames), each mapped through the
    VISTA curve; the frame score is the min of the two, the episode score the
    min over frames and sides. Invariant to every rig parameter AND to the
    loader's traj-yaw / rot-offset (rigid transforms preserve consecutive
    deltas), so it is computed once, on the FULL-RATE data, before any stride
    subsampling.

    Returns ``{"min", "min_frame", "worst_pos_mm", "worst_ang_deg",
    "frame_scores", "pos_scores", "ang_scores", "pos_delta_mm",
    "ang_delta_deg"}`` (``min`` is None for degenerate <2-frame episodes).
    Per-frame arrays index the step into frame ``i+1``: scores are the min
    over sides, raw deltas the max over sides, and ``frame_scores`` is exactly
    ``min(pos_scores, ang_scores)`` - overlaying the three shows which
    component causes each dip.
    """
    n = min(len(qpos_by_side[s]) for s in SIDES)
    empty = np.empty(0, np.float32)
    if n < 2:
        return {
            "min": None,
            "min_frame": -1,
            "worst_pos_mm": 0.0,
            "worst_ang_deg": 0.0,
            "frame_scores": empty,
            "pos_scores": empty,
            "ang_scores": empty,
            "pos_delta_mm": empty,
            "ang_delta_deg": empty,
        }
    # Scale the per-step thresholds to the scored stream's rate: the VISTA
    # breakpoints are per CONT_REF_FPS step, and a slower (faster) stream's
    # step covers a proportionally larger (smaller) delta.
    # ``fps=None`` keeps the reference calibration.
    k = CONT_REF_FPS / float(fps) if fps else 1.0
    pos_scores = np.full(n - 1, 100.0)
    ang_scores = np.full(n - 1, 100.0)
    pos_mm = np.zeros(n - 1)
    ang_deg = np.zeros(n - 1)
    for s in SIDES:
        q = np.asarray(qpos_by_side[s], np.float64)
        dp = np.linalg.norm(np.diff(q[:n, 0:3], axis=0), axis=1)
        rot = Rotation.from_euler("XYZ", q[:n, 3:6])
        da = np.degrees((rot[:-1].inv() * rot[1:]).magnitude())
        pos_scores = np.minimum(
            pos_scores,
            _cont_curve(
                dp, k * CONT_POS_PERFECT_M, k * CONT_POS_SOFT_M, k * CONT_POS_DECAY_M
            ),
        )
        ang_scores = np.minimum(
            ang_scores,
            _cont_curve(
                da,
                k * CONT_ANG_PERFECT_DEG,
                k * CONT_ANG_SOFT_DEG,
                k * CONT_ANG_DECAY_DEG,
            ),
        )
        pos_mm = np.maximum(pos_mm, dp * 1000.0)
        ang_deg = np.maximum(ang_deg, da)
    frame_scores = np.minimum(pos_scores, ang_scores)
    i = int(np.argmin(frame_scores))
    return {
        "min": float(frame_scores[i]),
        "min_frame": i + 1,
        "worst_pos_mm": float(pos_mm.max()),
        "worst_ang_deg": float(ang_deg.max()),
        "frame_scores": frame_scores.astype(np.float32),
        "pos_scores": pos_scores.astype(np.float32),
        "ang_scores": ang_scores.astype(np.float32),
        "pos_delta_mm": pos_mm.astype(np.float32),
        "ang_delta_deg": ang_deg.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# The shared per-trajectory solve
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SolveResult:
    worst_pos: float = 0.0  # m
    worst_rot: float = 0.0  # rad
    mean_pos: float = 0.0  # m
    mean_rot: float = 0.0  # rad
    qp_failures: int = 0
    n: int = 0  # frames solved
    self_f: int = 0  # frames with same-arm self collision
    cross_f: int = 0  # frames with inter-arm link crossing
    grip_f: int = 0  # frames with gripper-vs-gripper overlap
    cancelled: bool = False


def solve_trajectory(
    ik: Fr3MinkIK,
    idx: dict,
    home_qpos: np.ndarray,
    trajs: dict,
    out_solved: dict,
    out_target: dict,
    out_actual: dict,
    *,
    shared_frame: bool,
    ws: dict,
    base_dist: float,
    ik_iters: int,
    start_frame: int = 0,
    offset6=None,
    collision_ctx: dict | None = None,
    model: mujoco.MjModel | None = None,
    cancel: threading.Event | None = None,
) -> SolveResult:
    """Solve differential IK for one whole trajectory pair, frame by frame.

    ``trajs`` maps side -> (N, >=7) replay qpos; results land in the
    ``out_*[side]`` arrays (solved: (N, dof); target/actual: (N, 6) pose
    logs). ``ws`` = {"x","y","z","yaw","pitch","roll"} workspace transform.
    Shared-frame mode always preserves the trajectory's own origin: the raw
    world/marker origin (e.g. the marker at (0,0,0)) is what maps onto the
    anchor. Per-arm mode instead anchors each arm to its own start, in front
    of its own base (inter-arm geometry then comes from the base layout).
    ``offset6`` (optional) is a rigid offset applied on top of the workspace
    transform about the shared anchor (see ``apply_offset``).
    ``collision_ctx`` (with ``model``, both from the same rig as ``ik``) adds
    the per-frame collision tally. The caller sets the mocap bases and calls
    ``ik.reset()`` first (that zeroes ``qp_failures``, which this reads back).
    """
    if collision_ctx is not None and model is None:
        raise ValueError("collision_ctx needs the matching model")
    nf = {s: trajs[s].shape[0] for s in SIDES}
    n_max = max(nf.values())
    sf = max(0, min(int(start_frame), n_max - 1))
    ws_x, ws_y, ws_z = float(ws["x"]), float(ws["y"]), float(ws["z"])
    yaw, pitch, roll = float(ws["yaw"]), float(ws["pitch"]), float(ws["roll"])

    # Per-arm clamped start frame, anchor (Y tracks the base distance) and the
    # wrist anchor used to make the motion relative.
    sfs = {s: min(sf, nf[s] - 1) for s in SIDES}
    if shared_frame:
        # One rigid transform for BOTH arms, re-planted at a single centered
        # anchor (y=0). Each arm's real offset from the common reference is
        # kept, so the co-registered inter-gripper geometry survives and the
        # workspace transform moves/rotates the whole pair as one body.
        #
        # The reference is ALWAYS the trajectory's own origin: the raw
        # world/marker frame origin maps onto the anchor, so absolute
        # placement relative to that origin survives the replay - and so do
        # episodes' positions relative to each other, since they all share
        # that one origin.
        common_ref = np.zeros(3)
        anchors = {s: np.array([ws_x, ws_y, ws_z]) for s in SIDES}
        wrist_anchor = {s: common_ref.copy() for s in SIDES}
    else:
        # Per-arm: each arm relative to its own start, anchored in front of
        # its own base (+/-dist/2, shifted by the shared y offset). Inter-arm
        # geometry comes from the base layout.
        anchors = {
            s: np.array([ws_x, ws_y + _YSIGN[s] * base_dist / 2.0, ws_z]) for s in SIDES
        }
        wrist_anchor = {s: trajs[s][sfs[s], 0:3].copy() for s in SIDES}
    off = None if offset6 is None else np.asarray(offset6, np.float64)
    if off is not None and not off.any():
        off = None
    warm = {s: home_qpos[idx[s]["arm_qadr"]].copy() for s in SIDES}

    res = SolveResult()
    sum_pos = sum_rot = 0.0
    count = 0
    for f in range(sf, n_max):
        if cancel is not None and cancel.is_set():
            res.cancelled = True
            break
        targets, grippers, seeds = [], [], []
        for s in SIDES:  # order matches ik.arms (built in SIDES order)
            fe = max(sfs[s], min(f, nf[s] - 1))  # shorter arm holds last frame
            tp, tq = _frame_world_target(
                trajs[s][fe], wrist_anchor[s], anchors[s], yaw, pitch, roll
            )
            if off is not None:
                tp, tq = apply_offset(tp, tq, off, anchors[s])
            targets.append((tp, tq))
            grippers.append(float(trajs[s][fe, 6]))
            seeds.append(warm[s] if f == sf else None)
        f_iters = max(ik_iters, 200) if f == sf else ik_iters
        results = ik.solve(targets, grippers, arm_seeds=seeds, max_iters=f_iters)
        dconf = ik.configuration.data
        for i, s in enumerate(SIDES):
            sol, pe, re = results[i]
            fi = min(f, nf[s] - 1)
            out_solved[s][fi] = sol
            # log commanded target and IK-achieved TCP pose for the plots
            tp, tq = targets[i]
            out_target[s][fi, :3] = tp
            out_target[s][fi, 3:] = _quat_rpy_deg(tq)
            site = dconf.site(idx[s]["site_id"])
            out_actual[s][fi, :3] = site.xpos
            out_actual[s][fi, 3:] = _mat_rpy_deg(site.xmat)
            warm[s] = sol.copy()
            res.worst_pos = max(res.worst_pos, pe)
            res.worst_rot = max(res.worst_rot, re)
            sum_pos += pe
            sum_rot += re
            count += 1
        res.n += 1
        if collision_ctx is not None:
            selfc, cross, grip = collide(model, dconf, collision_ctx)
            res.self_f += int(selfc)
            res.cross_f += int(cross)
            res.grip_f += int(grip)
    for s in SIDES:
        if sfs[s] > 0:
            out_solved[s][: sfs[s]] = out_solved[s][sfs[s]]
            out_target[s][: sfs[s]] = out_target[s][sfs[s]]
            out_actual[s][: sfs[s]] = out_actual[s][sfs[s]]
    res.mean_pos = sum_pos / max(count, 1)
    res.mean_rot = sum_rot / max(count, 1)
    res.qp_failures = int(ik.qp_failures)
    return res


# ---------------------------------------------------------------------------
# Path segments (two-stage cache: ws transform once, offsets vectorized)
# ---------------------------------------------------------------------------


def transform_episode_path(qpos_by_side: dict, *, ws: dict) -> dict:
    """Stage 1 of the batch path cache: one episode's shared-frame EE paths
    through the SAME workspace transform the arms track, WITHOUT any offsets,
    as {side: (M, 2, 3) float32 line segments or None}. Rebuild only when the
    workspace changes; apply offsets per draw with ``offset_segments``
    (stage 2).

    Anchored on the trajectory's own origin, exactly like
    ``solve_trajectory``'s shared-frame branch, so an overlaid episode lands
    on the path the arm is commanded to track."""
    n = min(len(qpos_by_side[s]) for s in SIDES)
    if n < 2:
        return {s: None for s in SIDES}
    common_ref = np.zeros(3)
    anchor = shared_anchor(ws)
    out = {}
    for s in SIDES:
        pts = np.array(
            [
                _frame_world_target(
                    qpos_by_side[s][f],
                    common_ref,
                    anchor,
                    float(ws["yaw"]),
                    float(ws["pitch"]),
                    float(ws["roll"]),
                )[0]
                for f in range(n)
            ]
        )
        out[s] = np.stack([pts[:-1], pts[1:]], axis=1).astype(np.float32)
    return out

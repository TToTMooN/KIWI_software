"""The interactive bimanual replay viewer (`kiwi replay run`).

Loads an episode set, solves both arms' IK, and serves a viser app: playback
with a timeline, live rig tuning (bases / workspace / gripper mount), direct
joint posing, draggable live-TCP gizmos, per-component tracking-error plots,
and the batch-solve controls (``kiwi.replay.batch_controls``, a folder in
the Playback tab) that solve the whole set on a private rig while the GUI
stays interactive.

Episodes are the only trajectory source, and
:func:`kiwi.replay.episodes.load_meta` refuses a set spanning more than one
world scene, so shared-frame anchoring rests on a CHECKED precondition.
"""

from __future__ import annotations

from kiwi.replay.gripper import jaw_qpos

import dataclasses
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from viser import uplot

from kiwi.replay.batch import BatchSolver, EpisodeSet
from kiwi.replay.ik import Fr3MinkIK, dual_arm_collision_pairs
from kiwi.replay.mjviser import ViserMujocoScene
from kiwi.replay.mount import Mount, model_mount, set_model_mount
from kiwi.replay.pose import _workspace_rot_quat, dashed_segments
from kiwi.replay.rig import FR3_CHOPSTICK_SA, build_dual
from kiwi.replay.solve import (
    SIDES,
    _resolve_indices,
    _rpy_deg_wxyz,
    _set_bases,
    solve_trajectory,
)

__all__ = ["ViewerParams", "launch"]


@dataclasses.dataclass(frozen=True)
class ViewerParams:
    """Everything `kiwi replay run` can set. Defaults match `kiwi replay solve`
    (a shared-defaults test enforces it)."""

    fps: float = 30.0
    stride: int = 1
    traj_yaw: float = 0.0
    rot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    workspace_x: float = 0.0
    workspace_y: float = 0.0
    workspace_z: float = 0.0
    ws_yaw: float = 0.0
    ws_pitch: float = 0.0
    ws_roll: float = 0.0
    base_distance: float = 0.805
    base_z: float = 0.0
    base_roll_left: float = 0.0
    base_roll_right: float = 0.0
    base_pitch_left: float = 0.0
    base_pitch_right: float = 0.0
    base_yaw_left: float = 0.0
    base_yaw_right: float = 0.0
    start_frame: int = 0
    solve_mode: str = "combined"
    avoid_collisions: bool = False
    collision_margin: float = 0.004
    ik_gain: float = 0.5
    ik_lm_damping: float = 1e-3
    ik_iters: int = 120
    posture_cost: float = 1e-3
    orientation_cost: float = 1.0
    offset: tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
    offsets: dict[int, tuple] = dataclasses.field(default_factory=dict)
    workers: int = 1
    port: int = 8080
    gui_width: str = "large"
    gui_layout: str = "fixed"


def launch(episode_dirs: list[Path], p: ViewerParams = ViewerParams()) -> None:
    """Build the rig, load + solve the episodes, and serve the viewer.

    Blocks serving until interrupted. Every episode dir must already hold a
    ``replay_state.npz`` (the CLI ensures that).
    """
    robot = FR3_CHOPSTICK_SA
    print(f"Building dual {robot.name} scene...")
    model = build_dual(robot, p.base_distance).compile()
    # Mounts as actually built (the composite's CAD flange attachment). The
    # batch solver snapshots the live model mounts per run, so Gripper-mount
    # slider tuning reaches it through the model, not through params.
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    home_qpos = data.qpos.copy()
    mocapids = {s: model.body(f"{s}_mount").mocapid[0] for s in SIDES}
    idx = {
        s: _resolve_indices(
            model, s, robot.arm_joints, robot.tcp_site, robot.gripper_joints
        )
        for s in SIDES
    }

    # --- Episodes: the one trajectory source. Episode 0 is the launch playback.
    episode_set = EpisodeSet.load(
        list(episode_dirs),
        fps=p.fps,
        rot_offset_deg=p.rot_offset,
        traj_yaw_deg=p.traj_yaw,
        stride=p.stride,
    )
    episode_set.offset_global = np.asarray(p.offset, np.float64)
    for ep, off in p.offsets.items():
        episode_set.offsets[int(ep)] = np.asarray(off, np.float64)
    launch_ep = episode_set.ep_ids[0]
    active_ep = [launch_ep]  # batch episode id currently swapped into playback
    print(
        f"Playback trajectory: episode {launch_ep} "
        f"({episode_set.episode_dir(launch_ep).name})"
    )
    trajs, nf = {}, {}
    for side in SIDES:
        q = np.array(episode_set.episodes[launch_ep][side], dtype=np.float64)
        trajs[side], nf[side] = q, q.shape[0]
        print(f"  {side}: {nf[side]} frames, {q.shape[1]} cols")

    n_max = max(nf.values())
    # One playback step = one frame of the loaded stream, so 1x plays in real
    # time and the time labels / plot axes are true seconds.
    sim_dt = 1.0 / float(episode_set.meta["fps"])
    duration = n_max * sim_dt
    start_frame = max(0, min(p.start_frame, n_max - 1))

    # One shared solver over both arms (one model). The solve mode decides
    # whether each arm gets its own QP or both share a single QP.
    combined = p.solve_mode == "combined"
    arm_specs = [idx[s] for s in SIDES]
    # Computed unconditionally: the IK/Workspace tab can toggle avoidance on
    # at runtime even when the launch flag left it off.
    avoid_pairs = dual_arm_collision_pairs(model)
    collision_pairs = avoid_pairs if p.avoid_collisions else None
    if p.avoid_collisions and not collision_pairs:
        print("  avoid-collisions: no arm/gripper geoms matched; avoidance disabled")
    ik = Fr3MinkIK(
        model,
        arm_specs,
        home_qpos,
        combined=combined,
        frame_gain=p.ik_gain,
        lm_damping=p.ik_lm_damping,
        posture_cost=p.posture_cost,
        orientation_cost=p.orientation_cost,
        collision_pairs=collision_pairs,
        collision_margin=p.collision_margin,
    )
    solved = {s: np.zeros((nf[s], robot.dof), dtype=np.float64) for s in SIDES}
    # Per-arm TCP pose logs for the tracking plots, columns [x,y,z,roll,pitch,
    # yaw] (m / deg): target = commanded pose, actual = pose achieved by IK.
    target_traj = {s: np.zeros((nf[s], 6), dtype=np.float64) for s in SIDES}
    actual_traj = {s: np.zeros((nf[s], 6), dtype=np.float64) for s in SIDES}
    print(
        f"IK backend: mink (daqp) - 2x {robot.dof}-dof {robot.name}, "
        f"solve-mode={p.solve_mode}, gain={p.ik_gain}, "
        f"orientation_cost={p.orientation_cost}"
        + (
            f", avoid-collisions ON ({len(collision_pairs)} geom-group pairs, "
            f"margin {p.collision_margin * 1000:.0f} mm)"
            if collision_pairs
            else ""
        )
    )

    def solve_all_frames(
        ws_x_v,
        ws_y_v,
        ws_z_v,
        yaw,
        pitch,
        roll,
        dist,
        b_z,
        b_roll_l,
        b_roll_r,
        b_pitch_l,
        b_pitch_r,
        b_yaw_l,
        b_yaw_r,
        ik_iters,
        sf,
    ):
        # Thin wrapper over the shared per-trajectory solver: place the bases,
        # reset, then solve the loaded trajectory pair into this GUI's
        # solved/target/actual arrays. Every loaded episode lives in ONE
        # co-registered world - load_meta refused the set otherwise - so
        # shared-frame anchoring holds by construction; the active episode's
        # offset keeps Recompute matching the Batch solve.
        _set_bases(
            data,
            mocapids,
            dist,
            b_z,
            b_roll_l,
            b_roll_r,
            b_pitch_l,
            b_pitch_r,
            b_yaw_l,
            b_yaw_r,
        )
        _set_bases(
            ik.configuration.data,
            mocapids,
            dist,
            b_z,
            b_roll_l,
            b_roll_r,
            b_pitch_l,
            b_pitch_r,
            b_yaw_l,
            b_yaw_r,
        )
        ik.reset()  # update(q=home) recomputes FK with the mocap bases set
        off = (
            episode_set.effective_offset(active_ep[0])
            if active_ep[0] is not None
            else None
        )
        r = solve_trajectory(
            ik,
            idx,
            home_qpos,
            trajs,
            solved,
            target_traj,
            actual_traj,
            shared_frame=True,
            ws={
                "x": ws_x_v,
                "y": ws_y_v,
                "z": ws_z_v,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
            },
            base_dist=dist,
            ik_iters=ik_iters,
            start_frame=sf,
            offset6=off,
        )
        return r.worst_pos, r.worst_rot

    # The batch's private-rig solver (Fr3MinkIK mutates the model it wraps, so
    # batch solves run on their own compile while this one keeps serving
    # Recompute / Live TCP).
    batch_solver = BatchSolver(
        robot,
        p.base_distance,
        None,  # snapshots the live app-model mounts per run instead
        home_qpos,
        combined=combined,
        lm_damping=p.ik_lm_damping,
        avoid_collisions=p.avoid_collisions,
        collision_margin=p.collision_margin,
        workers=p.workers,
    )

    print(f"Solving IK for both arms (frames from {start_frame})...")
    t0 = time.perf_counter()
    worst_pos, worst_rot = solve_all_frames(
        p.workspace_x,
        p.workspace_y,
        p.workspace_z,
        p.ws_yaw,
        p.ws_pitch,
        p.ws_roll,
        p.base_distance,
        p.base_z,
        p.base_roll_left,
        p.base_roll_right,
        p.base_pitch_left,
        p.base_pitch_right,
        p.base_yaw_left,
        p.base_yaw_right,
        p.ik_iters,
        start_frame,
    )
    qpf = ik.qp_failures
    n_solved = sum(nf[s] - min(start_frame, nf[s] - 1) for s in SIDES)
    print(
        f"  solved {n_solved} arm-frames in {time.perf_counter() - t0:.1f}s - "
        f"worst residual pos={worst_pos * 1000:.2f} mm, "
        f"rot={np.rad2deg(worst_rot):.2f} deg" + (f", {qpf} QP failures" if qpf else "")
    )

    mujoco.mj_resetDataKeyframe(model, data, home_id)

    server = viser.ViserServer(port=p.port)
    # viser's control panel is ONE panel on the right edge; its width is a
    # 3-way enum (small/medium/large = 256/320/384 px at the default font).
    server.gui.configure_theme(control_width=p.gui_width, control_layout=p.gui_layout)
    mj_scene = ViserMujocoScene(server, model, num_envs=1)
    mj_scene.camera_tracking_enabled = False
    server.scene.add_frame(
        "/fixed_bodies/world_origin",
        show_axes=True,
        axes_length=0.30,
        axes_radius=0.008,
        position=(0, 0, 0),
    )
    target_frames = {
        s: server.scene.add_frame(
            f"/fixed_bodies/target_{s}",
            show_axes=True,
            axes_length=0.12,
            axes_radius=0.004,
        )
        for s in SIDES
    }

    # --- target trajectories: the world-frame TCP path each arm tracks ------
    _PATH_COLOR = {"left": (31, 119, 180), "right": (214, 39, 40)}
    path_handles: dict = {s: None for s in SIDES}
    show_path = [True]
    target_markers = {
        s: server.scene.add_icosphere(
            f"/fixed_bodies/target_marker_{s}",
            radius=0.012,
            color=_PATH_COLOR[s],
            subdivisions=2,
        )
        for s in SIDES
    }
    # A small axis triad riding each ball = the COMMANDED TCP orientation.
    target_ori_frames = {
        s: server.scene.add_frame(
            f"/fixed_bodies/target_ori_{s}",
            show_axes=True,
            axes_length=0.06,
            axes_radius=0.0025,
        )
        for s in SIDES
    }

    def draw_target_paths() -> None:
        for s in SIDES:
            if path_handles[s] is not None:
                path_handles[s].remove()
                path_handles[s] = None
            if not show_path[0]:
                continue
            segments = dashed_segments(target_traj[s][:, :3])  # dotted (M, 2, 3)
            if len(segments) == 0:
                continue
            path_handles[s] = server.scene.add_line_segments(
                f"/fixed_bodies/target_path_{s}",
                segments,
                colors=_PATH_COLOR[s],
                line_width=3.0,
            )

    draw_target_paths()

    frame_idx = [start_frame]
    playing = [False]
    looping = [True]
    speed = [1.0]
    needs_render = [True]
    accumulator = [0.0]
    # Manual posing: when True the Joints sliders own qpos and the trajectory
    # is not written into it; scrubbing/playing/recomputing turns it back off.
    manual_pose = [False]
    joint_sliders: list = []  # [(slider, qpos_adr), ...] for live value sync
    # True while render_frame mirrors qpos into the Joints sliders: viser
    # fires on_update callbacks for server-side .value sets too, and the
    # slider callbacks return early while this is set.
    syncing_sliders = [False]
    # Live-IK ("drag the TCP"): when on, each arm follows a draggable 3D
    # target gizmo, re-solving on every drag.
    live_ik = [False]
    gizmos: dict = {}  # side -> viser TransformControlsHandle

    # Guards model.body_pos/quat (written by the mount sliders on a viser
    # callback thread) against the render loop's GIL-free mj_forward read.
    mount_lock = threading.Lock()
    mount_sliders: list = []
    # live base params (applied to the render data every frame; fed to
    # Recompute). roll and yaw are each driven by one shared slider (mirrored
    # L/R); pitch is per-arm. The dict still stores *_left/*_right
    # independently so an asymmetric launch value renders until dragged.
    base = {
        "dist": p.base_distance,
        "z": p.base_z,
        "roll_left": p.base_roll_left,
        "roll_right": p.base_roll_right,
        "pitch_left": p.base_pitch_left,
        "pitch_right": p.base_pitch_right,
        "yaw_left": p.base_yaw_left,
        "yaw_right": p.base_yaw_right,
    }
    batch_refresh = [lambda: None]  # set by the batch controls; on Recompute
    batch_stale = [lambda: None]  # set by the batch controls; rig moved -> stale

    # Build the tab group ourselves so the viewer opens on Playback: viser
    # activates the FIRST tab. mjviser's tabs are appended after ours (same
    # public builders) at the end of this GUI setup.
    tabs = server.gui.add_tab_group()
    playback_tab = tabs.add_tab("Playback", icon=viser.Icon.PLAYER_PLAY)
    with playback_tab:
        timeline = server.gui.add_slider(
            "Frame", min=0, max=n_max - 1, step=1, initial_value=start_frame
        )
        time_label = server.gui.add_html("")
        play_btn = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        speed_btns = server.gui.add_button_group(
            "Speed", options=["0.25x", "0.5x", "1x", "2x", "4x"]
        )
        loop_cb = server.gui.add_checkbox("Loop", initial_value=True)
        path_cb = server.gui.add_checkbox("Show target paths", initial_value=True)
        # The batch path-overlay controls live here with the rest of the view
        # toggles; the batch-solve controls receive the handles via
        # BatchControlsContext and wire their redraw to them. What is drawn is
        # still narrowed by the Recompute IK folder's Episode dropdown.
        with server.gui.add_folder("Episode paths"):
            episode_paths_cb = server.gui.add_checkbox(
                "Show all episodes", initial_value=True
            )
            episode_colorby = server.gui.add_dropdown(
                "Color by",
                options=["arm", "feasibility"],
                initial_value="arm",
            )
            episode_left_cb = server.gui.add_checkbox("left arm", initial_value=True)
            episode_right_cb = server.gui.add_checkbox("right arm", initial_value=True)
            episode_width_sl = server.gui.add_slider(
                "line width", min=0.5, max=5.0, step=0.5, initial_value=1.5
            )

        @path_cb.on_update
        def _(_) -> None:
            show_path[0] = bool(path_cb.value)
            for s in SIDES:
                target_markers[s].visible = show_path[0]
                target_ori_frames[s].visible = show_path[0]
            draw_target_paths()

        @play_btn.on_click
        def _(_) -> None:
            playing[0] = not playing[0]
            if playing[0]:
                manual_pose[0] = False  # let the trajectory drive again
                _drop_live()
            play_btn.label = "Pause" if playing[0] else "Play"
            play_btn.icon = (
                viser.Icon.PLAYER_PAUSE if playing[0] else viser.Icon.PLAYER_PLAY
            )

        @speed_btns.on_click
        def _(event) -> None:
            speed[0] = float(event.target.value.replace("x", ""))

        @loop_cb.on_update
        def _(_) -> None:
            looping[0] = bool(loop_cb.value)

        @timeline.on_update
        def _(_) -> None:
            frame_idx[0] = int(timeline.value)
            manual_pose[0] = False  # reload the trajectory pose for this frame
            _drop_live()
            needs_render[0] = True

    with tabs.add_tab("IK / Workspace", icon=viser.Icon.ADJUSTMENTS):
        ws_x = server.gui.add_slider(
            "x forward (m)",
            min=-2.0,
            max=2.0,
            step=0.005,
            initial_value=float(p.workspace_x),
        )
        ws_y = server.gui.add_slider(
            "y left (m)",
            min=-2.0,
            max=2.0,
            step=0.005,
            initial_value=float(p.workspace_y),
        )
        ws_z = server.gui.add_slider(
            "z height (m)",
            min=0.0,
            max=1.2,
            step=0.005,
            initial_value=float(p.workspace_z),
        )
        ws_yaw = server.gui.add_slider(
            "yaw (deg)",
            min=-180.0,
            max=180.0,
            step=1.0,
            initial_value=float(p.ws_yaw),
        )
        ws_pitch = server.gui.add_slider(
            "pitch (deg)",
            min=-180.0,
            max=180.0,
            step=1.0,
            initial_value=float(p.ws_pitch),
        )
        ws_roll = server.gui.add_slider(
            "roll (deg)",
            min=-180.0,
            max=180.0,
            step=1.0,
            initial_value=float(p.ws_roll),
        )
        ws_start_frame = server.gui.add_slider(
            "start frame", min=0, max=n_max - 1, step=1, initial_value=start_frame
        )
        ik_iters_slider = server.gui.add_slider(
            "ik iters", min=8, max=256, step=8, initial_value=int(p.ik_iters)
        )
        ik_gain_slider = server.gui.add_slider(
            "ik gain (step frac)",
            min=0.05,
            max=1.0,
            step=0.05,
            initial_value=float(p.ik_gain),
        )
        orient_slider = server.gui.add_slider(
            "orientation cost",
            min=0.0,
            max=5.0,
            step=0.1,
            initial_value=float(p.orientation_cost),
        )
        posture_slider = server.gui.add_slider(
            "posture cost",
            min=0.0,
            max=3.0,
            step=0.01,
            initial_value=float(p.posture_cost),
        )
        server.gui.add_markdown(
            "_How hard the IK holds the home posture vs. tracking the TCP "
            "(frame cost = 1.0). ~0 = free, up = stays near the posture._"
        )

        @posture_slider.on_update
        def _(_) -> None:
            ik.set_posture_cost(float(posture_slider.value))
            if live_ik[0]:
                _live_solve()  # re-solve immediately so the effect is visible

        # Active collision avoidance in the QP, live-togglable: the app solver
        # swaps its CollisionAvoidanceLimit in place, the batch solver
        # recompiles its private rig on the next run. Gripper-vs-gripper
        # contact is never constrained (task-inherent - where the demonstrated
        # hands meet); the counting in the metrics table is independent and
        # always on.
        avoid_cb = server.gui.add_checkbox(
            "avoid collisions", initial_value=bool(p.avoid_collisions)
        )
        margin_slider = server.gui.add_slider(
            "collision margin (mm)",
            min=1.0,
            max=20.0,
            step=0.5,
            initial_value=float(p.collision_margin) * 1000.0,
        )

        def _apply_avoidance(_) -> None:
            on = bool(avoid_cb.value)
            margin = float(margin_slider.value) / 1000.0
            ik.set_collision_avoidance(avoid_pairs if on else None, margin=margin)
            batch_solver.set_avoid_collisions(on, margin)
            batch_stale[0]()  # solved episodes no longer reflect the QP limits
            ik_status.content = (
                '<span style="font-size:0.85em;color:#c80">collision avoidance '
                f"{'ON' if on else 'off'} - press <b>Recompute IK</b> / re-run "
                "Solve to apply</span>"
            )

        avoid_cb.on_update(_apply_avoidance)
        margin_slider.on_update(_apply_avoidance)

        ik_status = server.gui.add_html("")
        recompute_btn = server.gui.add_button("Recompute IK", icon=viser.Icon.REFRESH)

    with tabs.add_tab("Bases", icon=viser.Icon.ROBOT):
        base_dist = server.gui.add_slider(
            "base-to-base (m)",
            min=0.05,
            max=1.4,
            step=0.01,
            initial_value=float(p.base_distance),
        )
        base_height = server.gui.add_slider(
            "base height z (m)",
            min=-0.2,
            max=2.0,
            step=0.005,
            initial_value=float(p.base_z),
        )
        # Roll is a single shared slider (mirrored): positive rolls both arms
        # outward / away from each other (right +v, left -v). Pitch stays
        # per-arm; yaw is shared/mirrored like roll. An asymmetric launch
        # value still renders until the slider is dragged.
        base_roll = server.gui.add_slider(
            "base roll (deg)",
            min=-90.0,
            max=90.0,
            step=1.0,
            initial_value=float(p.base_roll_right),
        )
        base_pitch_l = server.gui.add_slider(
            "base pitch L (deg)",
            min=-90.0,
            max=90.0,
            step=1.0,
            initial_value=float(p.base_pitch_left),
        )
        base_pitch_r = server.gui.add_slider(
            "base pitch R (deg)",
            min=-90.0,
            max=90.0,
            step=1.0,
            initial_value=float(p.base_pitch_right),
        )
        base_yaw = server.gui.add_slider(
            "base yaw (deg)",
            min=-180.0,
            max=180.0,
            step=1.0,
            initial_value=float(p.base_yaw_left),
        )
        server.gui.add_markdown(
            "_**Roll and yaw are single shared sliders** (mirrored: roll +v "
            "rolls the arms outward / right +v / left -v; yaw +v yaws left +v "
            "/ right -v); **pitch is per-arm**. Drag to move the bases live; "
            "click **Recompute IK** to re-solve the arms._"
        )

        def _on_base(_) -> None:
            base["dist"] = float(base_dist.value)
            base["z"] = float(base_height.value)
            base["roll_right"] = float(base_roll.value)
            base["roll_left"] = -float(base_roll.value)
            base["pitch_left"] = float(base_pitch_l.value)
            base["pitch_right"] = float(base_pitch_r.value)
            base["yaw_left"] = float(base_yaw.value)
            base["yaw_right"] = -float(base_yaw.value)
            update_target_frames()  # anchor Y tracks base distance
            batch_stale[0]()  # solved batch episodes no longer match the rig
            needs_render[0] = True

        for s in (
            base_dist,
            base_height,
            base_roll,
            base_pitch_l,
            base_pitch_r,
            base_yaw,
        ):
            s.on_update(_on_base)

    # --- Joints: pose each arm's joints directly (both arms + gripper) -------
    with tabs.add_tab("Joints", icon=viser.Icon.ADJUSTMENTS):
        server.gui.add_markdown(
            "_Pose joints by hand (pauses playback). Scrub the timeline or "
            "press **Play** to return to the trajectory._"
        )

        def _pose_joint(qadr: int, value: float, grip=None) -> None:
            if syncing_sliders[0]:
                return  # render_frame mirroring, not a user drag
            if playing[0]:
                playing[0] = False
                play_btn.label = "Play"
                play_btn.icon = viser.Icon.PLAYER_PLAY
            manual_pose[0] = True
            with mount_lock:
                if grip is not None:
                    data.qpos[list(grip)] = jaw_qpos(value)
                else:
                    data.qpos[qadr] = value
            needs_render[0] = True

        for side in SIDES:
            with server.gui.add_folder(side.capitalize()):
                for jn, qadr in zip(robot.arm_joints, idx[side]["arm_qadr"]):
                    jid = model.joint(f"{side}_{jn}").id
                    lo, hi = (float(v) for v in model.jnt_range[jid])
                    sl = server.gui.add_slider(
                        jn,
                        min=round(lo, 3),
                        max=round(hi, 3),
                        step=0.01,
                        initial_value=float(np.clip(data.qpos[qadr], lo, hi)),
                    )
                    joint_sliders.append((sl, int(qadr)))

                    @sl.on_update
                    def _(_, _q=int(qadr), _sl=sl) -> None:
                        _pose_joint(_q, float(_sl.value))

                gq = tuple(int(v) for v in idx[side]["gripper_qadr"])
                grid = model.joint(f"{side}_{robot.gripper_joints[0]}").id
                glo, ghi = (float(v) for v in model.jnt_range[grid])
                gsl = server.gui.add_slider(
                    "gripper",
                    min=glo,
                    max=ghi,
                    step=0.01,
                    initial_value=float(np.clip(data.qpos[gq[0]], glo, ghi)),
                )
                joint_sliders.append((gsl, gq[0]))

                @gsl.on_update
                def _(_, _gq=gq, _sl=gsl) -> None:
                    _pose_joint(_gq[0], float(_sl.value), grip=_gq)

    # --- Live TCP: a draggable 3D target per arm; IK re-solves on every move --
    def _tcp_world_pose(side: str):
        """Current world pose of an arm's TCP site as (pos(3,), quat_wxyz(4,))."""
        sid = idx[side]["site_id"]
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
        return data.site_xpos[sid].copy(), quat

    def _live_solve() -> None:
        """Solve both arms to their gizmo targets and show the result.
        Warm-starts from the previous solution, so repeated drags move
        smoothly."""
        if not live_ik[0]:
            return
        _set_bases(
            ik.configuration.data,
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
        targets, grippers = [], []
        for s in SIDES:  # SIDES order matches ik.arms
            gz = gizmos[s]
            targets.append(
                (
                    np.asarray(gz.position, dtype=np.float64),
                    np.asarray(gz.wxyz, dtype=np.float64),
                )
            )
            grippers.append(float(data.qpos[idx[s]["gripper_qadr"][0]]))
        results = ik.solve(targets, grippers, max_iters=int(ik_iters_slider.value))
        wp = max(pe for _, pe, _ in results)
        wr = max(re for _, _, re in results)
        with mount_lock:
            for i, s in enumerate(SIDES):
                data.qpos[idx[s]["arm_qadr"]] = results[i][0]
        live_status.content = (
            f'<span style="font-size:0.85em">tracking - worst '
            f"{wp * 1000:.0f} mm / {np.rad2deg(wr):.0f} deg</span>"
        )
        needs_render[0] = True

    def _set_live(on: bool) -> None:
        if on:
            playing[0] = False
            play_btn.label = "Play"
            play_btn.icon = viser.Icon.PLAYER_PLAY
            manual_pose[0] = True  # stop the trajectory from overwriting qpos
            ik.configuration.update(q=data.qpos.copy())  # seed from shown pose
            for s in SIDES:  # drop each gizmo onto that arm's current TCP
                pos, quat = _tcp_world_pose(s)
                gizmos[s].position = tuple(float(v) for v in pos)
                gizmos[s].wxyz = tuple(float(v) for v in quat)
                gizmos[s].visible = True
            live_ik[0] = True  # set last so gizmo placement can't mis-solve
            live_status.content = (
                '<span style="font-size:0.85em;color:#888">'
                "drag a gizmo to move that arm's TCP</span>"
            )
            needs_render[0] = True
        else:
            live_ik[0] = False
            for s in SIDES:
                if s in gizmos:
                    gizmos[s].visible = False
            manual_pose[0] = False  # hand control back to the trajectory
            live_status.content = ""
            needs_render[0] = True

    def _drop_live() -> None:
        """Leave live mode if active (playback/scrub/recompute takes over)."""
        if live_ik[0]:
            _set_live(False)
            live_cb.value = False

    with tabs.add_tab("Live TCP", icon=viser.Icon.FOCUS_2):
        server.gui.add_markdown(
            "_Drag the gizmo at each arm's tool tip to set a TCP target; the "
            "IK re-solves live. Translate to move the point, rotate the rings "
            "to reorient (set **orientation cost** to 0 for position-only). "
            "Uncheck to return to the trajectory._"
        )
        live_cb = server.gui.add_checkbox("Live TCP control", initial_value=False)
        live_status = server.gui.add_html("")

        for s in SIDES:  # created once, hidden until live mode turns on
            gz = server.scene.add_transform_controls(
                f"/live_target_{s}", scale=0.15, opacity=0.9, visible=False
            )
            gizmos[s] = gz

            @gz.on_update
            def _(_, _s=s) -> None:
                _live_solve()

        @live_cb.on_update
        def _(_) -> None:
            _set_live(bool(live_cb.value))

    def update_target_frames() -> None:
        q = _workspace_rot_quat(
            float(ws_yaw.value), float(ws_pitch.value), float(ws_roll.value)
        )
        for side in SIDES:
            # Episodes are shared-frame: one centered anchor for both arms.
            target_frames[side].position = (
                float(ws_x.value),
                float(ws_y.value),
                float(ws_z.value),
            )
            target_frames[side].wxyz = (
                float(q[0]),
                float(q[1]),
                float(q[2]),
                float(q[3]),
            )

    for s in (ws_x, ws_y, ws_z, ws_yaw, ws_pitch, ws_roll):

        @s.on_update
        def _(_) -> None:
            update_target_frames()

    update_target_frames()

    # --- Gripper mount: live flange->gripper transform sliders (both arms) ---
    # Mutates model.body_pos/quat of BOTH gripper-base bodies in place (the
    # adapter is the same physical part on each arm), no recompile; press
    # Recompute IK to re-solve both arms with the new mount.
    init_mount = model_mount(model, f"left_{robot.chopsticks_body}")
    with tabs.add_tab("Gripper mount", icon=viser.Icon.TOOL):
        server.gui.add_html(
            '<span style="font-size:0.8em;color:#888">Live mount-adapter '
            "(flange&rarr;gripper) transform, applied to <b>both</b> arms. "
            "Drag to re-pose, then press <b>Recompute IK</b> to re-track.</span>"
        )
        m_x = server.gui.add_slider(
            "x (m)",
            min=-0.1,
            max=0.1,
            step=0.001,
            initial_value=float(init_mount.xyz[0]),
        )
        m_y = server.gui.add_slider(
            "y (m)",
            min=-0.1,
            max=0.1,
            step=0.001,
            initial_value=float(init_mount.xyz[1]),
        )
        m_z = server.gui.add_slider(
            "z (m)",
            min=-0.1,
            max=0.1,
            step=0.001,
            initial_value=float(init_mount.xyz[2]),
        )
        m_roll = server.gui.add_slider(
            "roll (deg)",
            min=-180.0,
            max=180.0,
            step=1.0,
            initial_value=float(init_mount.rpy_deg[0]),
        )
        m_pitch = server.gui.add_slider(
            "pitch (deg)",
            min=-180.0,
            max=180.0,
            step=1.0,
            initial_value=float(init_mount.rpy_deg[1]),
        )
        m_yaw = server.gui.add_slider(
            "yaw (deg)",
            min=-180.0,
            max=180.0,
            step=1.0,
            initial_value=float(init_mount.rpy_deg[2]),
        )
        mount_status = server.gui.add_html("")
        reset_mount_btn = server.gui.add_button("Reset mount", icon=viser.Icon.RESTORE)
        mount_sliders.extend((m_x, m_y, m_z, m_roll, m_pitch, m_yaw))

        def apply_mount_sliders() -> None:
            mt = Mount(
                xyz=(float(m_x.value), float(m_y.value), float(m_z.value)),
                rpy_deg=(float(m_roll.value), float(m_pitch.value), float(m_yaw.value)),
            )
            with mount_lock:
                for side in SIDES:
                    set_model_mount(model, mt, f"{side}_{robot.chopsticks_body}")
            batch_stale[0]()  # solved batch episodes no longer match the rig
            needs_render[0] = True
            mount_status.content = (
                '<span style="font-size:0.85em;color:#c80">mount moved - press '
                "<b>Recompute IK</b> to re-track both arms</span>"
            )

        for _s in (m_x, m_y, m_z, m_roll, m_pitch, m_yaw):

            @_s.on_update
            def _(_) -> None:
                apply_mount_sliders()

        @reset_mount_btn.on_click
        def _(_) -> None:
            (m_x.value, m_y.value, m_z.value) = (float(v) for v in init_mount.xyz)
            (m_roll.value, m_pitch.value, m_yaw.value) = (
                float(v) for v in init_mount.rpy_deg
            )
            apply_mount_sliders()

    # --- Batch solve: the episode set, solved on a private rig. Lives in a
    # folder at the end of the Playback tab (re-entered here - the ws/ik
    # slider handles it needs are created by the tabs in between). Its "Load
    # into playback" / "Restore" buttons call the playback-swap machinery
    # (set_active_trajectory / _orig_active), defined further below; the
    # holders here are filled once those exist and read at click time.
    from kiwi.replay.batch_controls import BatchControlsContext, build_batch_controls

    set_active_holder = [None]
    orig_active_holder = [None]
    with playback_tab:
        build_batch_controls(
            BatchControlsContext(
                server=server,
                SIDES=SIDES,
                episode_set=episode_set,
                batch_solver=batch_solver,
                model=model,
                base=base,
                mount_lock=mount_lock,
                ws=(ws_x, ws_y, ws_z, ws_yaw, ws_pitch, ws_roll, ws_start_frame),
                ik=(ik_iters_slider, ik_gain_slider, orient_slider, posture_slider),
                set_active=set_active_holder,
                orig_active=orig_active_holder,
                batch_refresh=batch_refresh,
                batch_stale=batch_stale,
                paths=(
                    episode_paths_cb,
                    episode_colorby,
                    episode_left_cb,
                    episode_right_cb,
                    episode_width_sl,
                ),
                launch_ep=launch_ep,
            )
        )

    # --- Tracking plots: 6 stacked uPlots (x,y,z,yaw,pitch,roll vs time), each
    # with 4 lines - left/right arm target (dashed) vs IK-actual (solid). ---
    tracking_tab = tabs.add_tab("Tracking", icon=viser.Icon.CHART_LINE)
    with tracking_tab:
        server.gui.add_markdown(
            "_Per-frame tracking **error** (IK-achieved minus commanded "
            "target) per pose component: positions in mm, angles in deg. The "
            "dot on each curve marks the current frame._"
        )
        track_time = server.gui.add_slider(
            "time (frame)", min=0, max=n_max - 1, step=1, initial_value=start_frame
        )

    @track_time.on_update
    def _(_) -> None:
        frame_idx[0] = int(track_time.value)
        manual_pose[0] = False
        _drop_live()
        needs_render[0] = True

    # mjviser's Scene tab is not carried, but its initial framing is: put each
    # connecting client's camera where create_scene_gui would have - distance
    # from the model extent, the stock azimuth/elevation - aimed at the model
    # center.
    _center = model.stat.center.copy()
    _cam_dist = 3.0 * model.stat.extent
    _az, _el = np.deg2rad(120.0), np.deg2rad(20.0)
    _cam_pos = (
        np.array([-np.cos(_el) * np.cos(_az), -np.cos(_el) * np.sin(_az), np.sin(_el)])
        * _cam_dist
    )

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.fov = np.radians(60.0)
        client.camera.position = _cam_pos
        client.camera.look_at = _center

    # mjviser's render-inspection tabs go last, after the workflow tabs.
    with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
        mj_scene.create_overlay_gui()
    with tabs.add_tab("Groups", icon=viser.Icon.LAYERS_INTERSECT):
        mj_scene.create_groups_gui()

    # (title, column index into the [x,y,z,roll,pitch,yaw] pose logs)
    _PLOTS = [
        ("x err (mm)", 0),
        ("y err (mm)", 1),
        ("z err (mm)", 2),
        ("yaw err (deg)", 5),
        ("pitch err (deg)", 4),
        ("roll err (deg)", 3),
    ]
    _plot_handles = []
    _plot_state = []  # (handle, t, l_err, r_err, kept) per plot, for the cursor
    _cursor_frame = [-1]  # frame the plot cursor dots currently mark
    _cursor_ts = [0.0]

    def _pad(col: np.ndarray) -> np.ndarray:
        # pad a per-arm column to the shared n_max length (hold last value)
        return (
            col
            if len(col) >= n_max
            else np.concatenate([col, np.full(n_max - len(col), col[-1])])
        )

    def _err(side: str, c: int) -> np.ndarray:
        # per-frame tracking error, actual - target: mm for position columns,
        # deg wrapped to +/-180 for angle columns (so the +/-180 branch cut of
        # the logged Euler angles never shows up as a +/-360 spike)
        err = _pad(actual_traj[side][:, c]) - _pad(target_traj[side][:, c])
        if c >= 3:
            return (err + 180.0) % 360.0 - 180.0
        return err * 1000.0

    # Cap the points per plot: every cursor-dot move re-sends each plot's full
    # data tuple. Decimation keeps the per-bin peak-|error| frame.
    _PLOT_MAX_PTS = 400

    def _decimate(l_err: np.ndarray, r_err: np.ndarray) -> np.ndarray:
        n = len(l_err)
        if n <= _PLOT_MAX_PTS:
            return np.arange(n)
        edges = np.linspace(0, n, _PLOT_MAX_PTS + 1).astype(int)
        mag = np.maximum(np.abs(l_err), np.abs(r_err))
        return np.unique(
            [
                a + int(np.argmax(mag[a:b])) if b > a else a
                for a, b in zip(edges[:-1], edges[1:])
            ]
        )

    def _cursor_cols(
        l_dec: np.ndarray, r_dec: np.ndarray, kept: np.ndarray, f: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # all-NaN columns with a single sample at (the kept frame nearest to)
        # ``f``: uPlot renders them as one dot per arm on the error curves
        j = int(np.argmin(np.abs(kept - f)))
        ml = np.full(l_dec.shape, np.nan)
        mr = np.full(r_dec.shape, np.nan)
        ml[j] = l_dec[j]
        mr[j] = r_dec[j]
        return ml, mr

    def rebuild_plots() -> None:
        for h in _plot_handles:
            h.remove()
        _plot_handles.clear()
        _plot_state.clear()
        f0 = min(frame_idx[0], n_max - 1)
        t = np.arange(n_max, dtype=np.float64) * sim_dt
        with tracking_tab:
            for title, c in _PLOTS:
                l_err, r_err = _err("left", c), _err("right", c)
                kept = _decimate(l_err, r_err)
                t_dec, l_err, r_err = t[kept], l_err[kept], r_err[kept]
                ml, mr = _cursor_cols(l_err, r_err, kept, f0)
                plot_data = (t_dec, l_err, r_err, ml, mr)
                # Pin the y-range: viser's uplot client never re-runs y
                # autoscale when `handle.data` is reassigned (the cursor-dot
                # updates).
                lo = float(min(l_err.min(), r_err.min()))
                hi = float(max(l_err.max(), r_err.max()))
                pad = max(0.05 * (hi - lo), 1e-3)
                y_scale = uplot.Scale(range=(lo - pad, hi + pad))
                series = (
                    uplot.Series(label="t (s)"),
                    uplot.Series(label="L", stroke="#1f77b4", width=1.75),
                    uplot.Series(label="R", stroke="#d62728", width=1.75),
                    uplot.Series(
                        label="L @frame",
                        stroke="#1f77b4",
                        width=0.0,
                        points=uplot.Series_Points(show=True, size=9, fill="#1f77b4"),
                    ),
                    uplot.Series(
                        label="R @frame",
                        stroke="#d62728",
                        width=0.0,
                        points=uplot.Series_Points(show=True, size=9, fill="#d62728"),
                    ),
                )
                h = server.gui.add_uplot(
                    data=plot_data,
                    series=series,
                    title=title,
                    mode=1,
                    aspect=2.0,
                    scales={"y": y_scale},
                    legend=uplot.Legend(show=True),
                )
                _plot_handles.append(h)
                _plot_state.append((h, t_dec, l_err, r_err, kept))
        _cursor_frame[0] = f0

    def update_plot_cursor(f: int) -> None:
        # move the frame dots onto frame ``f``; re-sends every plot's data
        # tuple, so the render loop throttles calls to >= 0.1 s apart
        for h, t_dec, l_err, r_err, kept in _plot_state:
            ml, mr = _cursor_cols(l_err, r_err, kept, f)
            h.data = (t_dec, l_err, r_err, ml, mr)
        _cursor_frame[0] = f
        _cursor_ts[0] = time.perf_counter()

    rebuild_plots()

    # --- Active-trajectory swap: point the playback machinery at another
    # (trajs, solved, target, actual) set, e.g. a solved batch episode. The
    # render loop, plots and labels read nf/n_max/duration dynamically, so
    # rebinding the dict entries + the two scalars is enough.
    # Deep-copied: Recompute IK writes into the live playback arrays in place,
    # so the launch snapshot (and any swapped-in batch result, below) must own
    # its own buffers.
    _orig_active = tuple(
        {s: np.array(d[s]) for s in SIDES}
        for d in (trajs, solved, target_traj, actual_traj)
    )

    def set_active_trajectory(
        new_trajs,
        new_solved,
        new_target,
        new_actual,
        label: str,
        ep: int | None = None,
    ) -> None:
        nonlocal n_max, duration
        playing[0] = False
        play_btn.label = "Play"
        play_btn.icon = viser.Icon.PLAYER_PLAY
        manual_pose[0] = False
        _drop_live()
        for s in SIDES:
            # Copies, not views: the stored batch result / launch snapshot
            # must survive a later Recompute IK, which mutates these arrays.
            trajs[s] = np.array(new_trajs[s])
            solved[s] = np.array(new_solved[s])
            target_traj[s] = np.array(new_target[s])
            actual_traj[s] = np.array(new_actual[s])
            nf[s] = new_trajs[s].shape[0]
        n_max = max(nf.values())
        duration = n_max * sim_dt
        active_ep[0] = ep
        frame_idx[0] = 0
        accumulator[0] = 0.0
        for sl in (timeline, track_time, ws_start_frame):
            sl.value = min(int(sl.value), n_max - 1)
            sl.max = n_max - 1
        timeline.value = 0
        rebuild_plots()
        draw_target_paths()
        print(f"Playback source -> {label} ({n_max} frames)")
        needs_render[0] = True

    # Wire the batch-solve buttons to the swap machinery now that it exists
    # (the controls were built above and read these holders at click time).
    set_active_holder[0] = set_active_trajectory
    orig_active_holder[0] = _orig_active

    @recompute_btn.on_click
    def _(_) -> None:
        recompute_btn.disabled = True
        for _ms in mount_sliders:
            _ms.disabled = True
        ik_status.content = (
            '<span style="font-size:0.85em;color:#888">Solving...</span>'
        )
        try:
            t1 = time.perf_counter()
            ik.set_gains(float(ik_gain_slider.value), float(orient_slider.value))
            ik.set_posture_cost(float(posture_slider.value))
            sf_idx = int(ws_start_frame.value)
            wp, wr = solve_all_frames(
                float(ws_x.value),
                float(ws_y.value),
                float(ws_z.value),
                float(ws_yaw.value),
                float(ws_pitch.value),
                float(ws_roll.value),
                base["dist"],
                base["z"],
                base["roll_left"],
                base["roll_right"],
                base["pitch_left"],
                base["pitch_right"],
                base["yaw_left"],
                base["yaw_right"],
                int(ik_iters_slider.value),
                sf_idx,
            )
            rebuild_plots()  # refresh the tracking plots with the new solution
            draw_target_paths()  # targets moved with the workspace sliders
            batch_refresh[0]()  # batch paths re-transform; solved eps go stale
            dt = time.perf_counter() - t1
            qpf2 = ik.qp_failures
            ik_status.content = (
                f'<span style="font-size:0.85em">'
                f"re-solved both arms in {dt:.2f}s<br>"
                f"worst {wp * 1000:.1f} mm / {np.rad2deg(wr):.1f} deg"
                + (f" - {qpf2} QP fails" if qpf2 else "")
                + "</span>"
            )
            frame_idx[0] = sf_idx
            timeline.value = sf_idx
            accumulator[0] = 0.0
            manual_pose[0] = False  # show the freshly-solved trajectory
            _drop_live()
            if playing[0]:
                playing[0] = False
                play_btn.label = "Play"
                play_btn.icon = viser.Icon.PLAYER_PLAY
            needs_render[0] = True
        finally:
            recompute_btn.disabled = False
            for _ms in mount_sliders:
                _ms.disabled = False

    def render_frame(f: int) -> None:
        _set_bases(
            data,
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
        grips = {}
        for side in SIDES:
            fe = min(f, nf[side] - 1)  # shorter arm holds its last frame
            if not manual_pose[0]:  # trajectory drives qpos unless hand-posed
                data.qpos[idx[side]["arm_qadr"]] = solved[side][fe]
                g = float(trajs[side][fe, 6])
                data.qpos[idx[side]["gripper_qadr"]] = jaw_qpos(g)
            grips[side] = float(data.qpos[idx[side]["gripper_qadr"][0]])
            target_markers[side].position = tuple(
                float(v) for v in target_traj[side][fe, :3]
            )
            target_ori_frames[side].position = tuple(
                float(v) for v in target_traj[side][fe, :3]
            )
            target_ori_frames[side].wxyz = tuple(
                float(v) for v in _rpy_deg_wxyz(target_traj[side][fe, 3:])
            )
        with mount_lock:
            mujoco.mj_forward(model, data)
        mj_scene.update_from_mjdata(data)
        update_target_frames()
        # Keep both time sliders in sync with the rendered frame (server-set
        # .value doesn't re-fire on_update, so no feedback loop).
        timeline.value = f
        track_time.value = f
        # Mirror the rendered pose into the Joints sliders when paused (skipped
        # during playback). The guard
        # keeps the sliders' own on_update callbacks - which viser fires for
        # server-side value sets too - from treating the mirror as a user drag.
        if not playing[0]:
            syncing_sliders[0] = True
            try:
                for sl, qadr in joint_sliders:
                    sl.value = round(float(data.qpos[qadr]), 3)
            finally:
                syncing_sliders[0] = False
        time_label.content = (
            f'<span style="font-size:0.85em">{f * sim_dt:.2f}s / {duration:.2f}s'
            f" &nbsp;&middot;&nbsp; frame {f}/{n_max - 1} &nbsp;&middot;&nbsp; "
            f"grip L<tt>{grips['left']:+.2f}</tt> R<tt>{grips['right']:+.2f}</tt>"
            f" &nbsp;&middot;&nbsp; base {base['dist']:.2f} m</span>"
        )

    render_frame(start_frame)

    last = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            wall_dt, last = now - last, now
            if playing[0]:
                accumulator[0] += wall_dt * speed[0]
                adv = int(accumulator[0] / sim_dt)
                if adv > 0:
                    accumulator[0] -= adv * sim_dt
                    new_idx = frame_idx[0] + adv
                    if new_idx >= n_max:
                        if looping[0]:
                            new_idx %= n_max
                        else:
                            new_idx = n_max - 1
                            playing[0] = False
                            play_btn.label = "Play"
                            play_btn.icon = viser.Icon.PLAYER_PLAY
                    frame_idx[0] = new_idx
                    timeline.value = new_idx
                    render_frame(new_idx)
            elif needs_render[0]:
                render_frame(frame_idx[0])
                needs_render[0] = False
            # Sweep the tracking-plot frame dots after the rendered frame
            # (throttled; the != check makes the last move always land, so the
            # dots never stay stale after a scrub or pause).
            if _cursor_frame[0] != frame_idx[0] and now - _cursor_ts[0] >= 0.1:
                update_plot_cursor(frame_idx[0])
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("\nStopped.")
        server.stop()

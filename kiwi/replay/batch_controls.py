"""The batch-solve controls of the replay viewer's Playback tab.

Solve-all / cancel with progress, the per-episode metrics table, the
arm- or feasibility-colored path overlay, and the load-into-playback /
restore buttons - built as a "Recompute IK" folder inside whatever GUI
container the caller has entered. The engine underneath is
``kiwi.replay.batch``; this module is GUI only and is imported by
``kiwi.replay.viewer`` alone.

This module must never import the viewer (state flows in via
``BatchControlsContext``, and the playback-swap callables arrive late through
single-element holders the viewer fills after building its own machinery).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import viser

from kiwi.replay.batch import BatchSolver, EpisodeSet
from kiwi.replay.solve import (
    feasibility_color,
    offset_segments,
    shared_anchor,
    transform_episode_path,
)

EE_FRAME_NOTE = (
    "EE frame: verified from the npz (ee_frame='tcp'), so these poses are at "
    "the tool grasp tip the IK targets."
)


@dataclass
class BatchControlsContext:
    """Everything ``build_batch_controls`` needs from the viewer's ``launch`` scope.

    ``set_active`` / ``orig_active`` are single-element holders for the
    playback swap machinery, which the viewer defines *after* the tab is
    built; the tab reads them at button-click time (runtime), and the viewer
    fills them before serving. ``batch_refresh`` / ``batch_stale`` are the
    holders the tab assigns its refresh/stale callbacks into (called from
    Recompute / rig moves).
    """

    server: Any
    SIDES: tuple
    episode_set: EpisodeSet
    batch_solver: BatchSolver
    model: Any
    base: dict
    mount_lock: Any
    ws: tuple  # (ws_x, ws_y, ws_z, ws_yaw, ws_pitch, ws_roll, ws_start_frame)
    ik: tuple  # (ik_iters_slider, ik_gain_slider, orient_slider, posture_slider)
    set_active: list  # holder -> set_active_trajectory(new_trajs, new_solved, ...)
    orig_active: list  # holder -> the launch (trajs, solved, target, actual)
    batch_refresh: list  # holder the tab writes its refresh callback into
    batch_stale: list  # holder the tab writes its stale callback into
    # The path-overlay control handles, created by the viewer in the Playback
    # tab's "Episode paths" folder: (show_all, colorby, left, right, width).
    # This tab only reads them and wires its redraw to them.
    paths: tuple | None = None
    launch_ep: int | None = None  # episode the app launched on. Restore
    # replays it WITH this id, so the restored trajectory keeps the episode's
    # offset + shared-frame anchoring.


def build_batch_controls(ctx: BatchControlsContext) -> None:
    """Build the "Recompute IK" folder: the episode set, solved on a private
    rig.

    Paths use a two-stage cache: the expensive workspace transform is rebuilt
    only on Recompute; offsets are applied vectorized on every draw. The
    "Episode" dropdown selects what Solve / the drawn path / Load into
    playback act on.
    """
    server = ctx.server
    SIDES = ctx.SIDES
    episode_set = ctx.episode_set
    batch_solver = ctx.batch_solver
    model = ctx.model
    base = ctx.base
    mount_lock = ctx.mount_lock
    ws_x, ws_y, ws_z, ws_yaw, ws_pitch, ws_roll, ws_start_frame = ctx.ws
    ik_iters_slider, ik_gain_slider, orient_slider, posture_slider = ctx.ik

    with server.gui.add_folder("Recompute IK"):
        bset = episode_set
        b_ids = bset.ep_ids
        _ALL = "all (batch)"
        # Identity mode colors by ARM, not by episode: solid blue = left,
        # solid red = right - the same L/R convention the playback target
        # path and the Tracking plots use. The selected episode is drawn
        # 2.5x thicker.
        _ARM_COLOR = {"left": (31, 119, 180), "right": (214, 39, 40)}
        b_segs = {ep: {s: None for s in SIDES} for ep in b_ids}  # stage 1
        b_handles = {ep: {s: None for s in SIDES} for ep in b_ids}

        # The EE frame is a verified fact here: the episode source hard-refuses
        # a non-tcp replay_state.npz.
        server.gui.add_html(
            f'<span style="font-size:0.75em;color:#690">{EE_FRAME_NOTE}</span>'
        )

        b_target = server.gui.add_dropdown(
            "Episode", options=[_ALL] + [str(e) for e in b_ids], initial_value=_ALL
        )
        # Offsets from --offset / --offsets-json shift the solve; they are
        # shown here read-only.
        _off_eps = [ep for ep in b_ids if bset.offsets[ep].any()]
        if bset.offset_global.any() or _off_eps:
            _g = np.round(bset.offset_global, 3).tolist()
            _per = f", per-episode: {_off_eps}" if _off_eps else ""
            server.gui.add_html(
                f'<span style="font-size:0.75em;color:#c80">Batch offsets active '
                f"(global {_g}{_per}), applied on top of the workspace transform. "
                f"Set via --offset / --offsets-json; not editable here - episodes "
                f"carrying one are marked * in the table.</span>"
            )
        b_solve_btn = server.gui.add_button(
            f"Solve all ({len(b_ids)})", icon=viser.Icon.CALCULATOR
        )
        b_cancel_btn = server.gui.add_button("Cancel", icon=viser.Icon.X, disabled=True)
        b_progress = server.gui.add_progress_bar(0.0)
        b_status = server.gui.add_html("")

        b_play_btn = server.gui.add_button(
            "Load into playback", icon=viser.Icon.PLAYLIST_ADD, disabled=True
        )
        b_restore_btn = server.gui.add_button(
            "Restore launch trajectory", icon=viser.Icon.ARROW_BACK_UP
        )

        # The path-overlay controls live in the Playback tab's "Episode paths"
        # folder (the viewer passes their handles); fall back to building the
        # folder here when none were given.
        if ctx.paths is not None:
            b_show, b_colorby, b_left, b_right, b_width = ctx.paths
        else:
            with server.gui.add_folder("Episode paths"):
                b_show = server.gui.add_checkbox(
                    "Show all episodes", initial_value=True
                )
                b_colorby = server.gui.add_dropdown(
                    "Color by",
                    options=["arm", "feasibility"],
                    initial_value="arm",
                )
                b_left = server.gui.add_checkbox("left arm", initial_value=True)
                b_right = server.gui.add_checkbox("right arm", initial_value=True)
                b_width = server.gui.add_slider(
                    "line width", min=0.5, max=5.0, step=0.5, initial_value=1.5
                )
        b_table = server.gui.add_html("")

        def _b_ws() -> dict:
            return {
                "x": float(ws_x.value),
                "y": float(ws_y.value),
                "z": float(ws_z.value),
                "yaw": float(ws_yaw.value),
                "pitch": float(ws_pitch.value),
                "roll": float(ws_roll.value),
            }

        _STATUS_COLOR = {
            "done": "#1a9850",
            "stale": "#c80",
            "solving": "#3182bd",
            "error": "#d7301f",
        }

        def _b_cont_cell(ep: int) -> str:
            # L0 continuity (static: computed at load, immune to sliders).
            c = bset.continuity.get(ep, {}).get("min")
            if c is None:
                return "<td style='color:#888'>-</td>"
            col = "#1a9850" if c >= 90.0 else "#c80" if c >= 60.0 else "#d7301f"
            return f"<td style='color:{col}'>{c:.0f}</td>"

        def b_table_html() -> None:
            with batch_solver.lock:
                rows = [(ep, bset.status[ep], bset.metrics(ep)) for ep in b_ids]
            cells = [
                "<table style='font-size:0.8em;border-spacing:6px 1px'>"
                "<tr><th>ep</th><th>status</th><th>cont</th>"
                "<th>worst mm</th><th>mean mm</th><th>worst °</th>"
                "<th>qp</th><th>cross/self/grip</th></tr>"
            ]
            for ep, st, m in rows:
                c = _STATUS_COLOR.get(st, "#888")
                off_mark = "*" if bset.offsets[ep].any() else ""
                if m is None:
                    cells.append(
                        f"<tr><td>{ep}{off_mark}</td>"
                        f"<td style='color:{c}'>{st}</td>"
                        + _b_cont_cell(ep)
                        + "<td colspan=5 style='color:#888'>-</td></tr>"
                    )
                    continue
                cells.append(
                    f"<tr><td>{ep}{off_mark}</td>"
                    f"<td style='color:{c}'>{st}</td>"
                    + _b_cont_cell(ep)
                    + f"<td>{m.worst_pos * 1000:.1f}</td>"
                    f"<td>{m.mean_pos * 1000:.2f}</td>"
                    f"<td>{np.rad2deg(m.worst_rot):.1f}</td>"
                    f"<td>{m.qp_failures}</td>"
                    f"<td>{m.cross_f}/{m.self_f}/{m.grip_f}</td></tr>"
                )
            cells.append("</table>")
            b_table.content = "".join(cells)

        def b_retransform() -> None:  # stage 1: expensive, per-frame loop
            ws = _b_ws()
            for ep in b_ids:
                b_segs[ep] = transform_episode_path(bset.episodes[ep], ws=ws)

        def b_draw() -> None:  # stage 2: cached segs + vectorized offsets
            anchor = shared_anchor(_b_ws())
            sel = b_target.value
            by_feas = b_colorby.value == "feasibility"
            for ep in b_ids:
                # The Episode dropdown IS the view filter: "all (batch)" draws
                # the whole set, picking an episode draws only that one.
                selected = sel == str(ep)
                visible = b_show.value and (sel == _ALL or selected)
                eff = bset.effective_offset(ep)
                feas = None
                if by_feas:
                    m = bset.metrics(ep)
                    feas = feasibility_color(None if m is None else m.worst_pos)
                lw = float(b_width.value) * (2.5 if selected else 1.0)
                for s in SIDES:
                    if b_handles[ep][s] is not None:
                        b_handles[ep][s].remove()
                        b_handles[ep][s] = None
                    segs = b_segs[ep][s]
                    if (
                        not visible
                        or segs is None
                        or not (b_left.value if s == "left" else b_right.value)
                    ):
                        continue
                    b_handles[ep][s] = server.scene.add_line_segments(
                        f"/batch/ep{ep}_{s}",
                        offset_segments(segs, eff, anchor),
                        colors=feas if feas is not None else _ARM_COLOR[s],
                        line_width=lw,
                    )

        def b_mark_stale(only_ep: int | None = None) -> None:
            with batch_solver.lock:
                for ep in b_ids if only_ep is None else (only_ep,):
                    if bset.status[ep] == "done":
                        bset.status[ep] = "stale"
            b_table_html()
            _b_update_play()

        def _b_update_play() -> None:
            sel = b_target.value
            if sel == _ALL:
                b_play_btn.disabled = True
                return
            with batch_solver.lock:
                st = bset.status[int(sel)]
            b_play_btn.disabled = st not in ("done", "stale")

        @b_target.on_update
        def _(_) -> None:
            sel = b_target.value
            b_solve_btn.label = (
                f"Solve all ({len(b_ids)})"
                if sel == _ALL
                else f"Re-solve episode {sel}"
            )
            _b_update_play()
            b_draw()  # only the selected episode's path stays drawn

        # -- solving (worker thread; viser handle writes are queued/safe) --
        def _b_progress_cb(k: int, total: int, ep: int) -> None:
            b_progress.value = 100.0 * k / max(total, 1)
            # serial: fires before solving ep (k done so far); parallel:
            # fires as each ep completes (k done, many still in flight).
            msg = (
                f"{k}/{total} episodes solved (ep {ep} done)..."
                if batch_solver.parallel
                else f"solving ep {ep} ({k + 1}/{total})..."
            )
            b_status.content = f'<span style="font-size:0.85em">{msg}</span>'
            b_table_html()

        def _b_ep_done(ep: int, r) -> None:
            b_table_html()
            if b_colorby.value == "feasibility":
                b_draw()  # recolor as results land
            _b_update_play()

        def _b_finished() -> None:
            b_progress.value = 100.0
            b_solve_btn.disabled = False
            b_cancel_btn.disabled = True
            b_status.content = (
                '<span style="font-size:0.85em;color:#888">batch idle</span>'
            )
            b_table_html()
            _b_update_play()

        @b_solve_btn.on_click
        def _(_) -> None:
            sel = b_target.value
            ids = b_ids if sel == _ALL else [int(sel)]
            b_solve_btn.disabled = True
            b_cancel_btn.disabled = False
            b_progress.value = 0.0
            started = batch_solver.start(
                bset,
                ids,
                ws=_b_ws(),
                base=dict(base),
                ik_iters=int(ik_iters_slider.value),
                frame_gain=float(ik_gain_slider.value),
                orientation_cost=float(orient_slider.value),
                posture_cost=float(posture_slider.value),
                start_frame=int(ws_start_frame.value),
                app_model=model,
                app_mount_lock=mount_lock,
                on_episode_done=_b_ep_done,
                on_progress=_b_progress_cb,
                on_finished=_b_finished,
            )
            if not started:  # already running
                b_solve_btn.disabled = False
                b_cancel_btn.disabled = True

        @b_cancel_btn.on_click
        def _(_) -> None:
            batch_solver.cancel()

        @b_play_btn.on_click
        def _(_) -> None:
            if ctx.set_active[0] is None:
                return  # viewer still starting up; holders not filled yet
            sel = b_target.value
            if sel == _ALL:
                return
            ep = int(sel)
            with batch_solver.lock:
                res = bset.results.get(ep)
            if not res or "solved" not in res:
                return
            ctx.set_active[0](
                bset.episodes[ep],
                res["solved"],
                res["target"],
                res["actual"],
                f"batch episode {ep}",
                ep=ep,
            )

        @b_restore_btn.on_click
        def _(_) -> None:
            if ctx.set_active[0] is None or ctx.orig_active[0] is None:
                return  # viewer still starting up; holders not filled yet
            lep = ctx.launch_ep
            label = (
                "launch trajectory" if lep is None else f"batch episode {lep} (launch)"
            )
            ctx.set_active[0](*ctx.orig_active[0], label, ep=lep)

        for _c in (b_show, b_colorby, b_left, b_right, b_width):
            _c.on_update(lambda _e: b_draw())  # cheap: cached segments

        def b_refresh() -> None:
            # Recompute pressed: workspace/start-frame may have moved, so
            # rebuild stage 1 and flag results stale.
            b_retransform()
            b_mark_stale()
            b_draw()

        ctx.batch_refresh[0] = b_refresh
        ctx.batch_stale[0] = b_mark_stale
        b_retransform()
        b_table_html()
        b_draw()
        _b_finished()  # seed button/progress/status idle state

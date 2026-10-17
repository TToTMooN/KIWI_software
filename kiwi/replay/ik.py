"""mink differential IK for FR3 + chopsticks arms.

``Fr3MinkIK`` solves one or more 7-dof FR3 arms sharing one MuJoCo model, in
either independent (one QP per arm) or combined (one shared QP) mode.
Robot-agnostic geometry helpers live in ``kiwi.replay.pose``; this module only
needs mink / mujoco / numpy.

Every solver must own its compiled model (see ``Fr3MinkIK.__init__``).
"""

from __future__ import annotations

from kiwi.replay.gripper import jaw_qpos

import mink
import mujoco
import numpy as np

from kiwi.replay.mount import TCP_SITE  # re-export; mount.py owns the name

__all__ = ["MINK_SOLVER", "TCP_SITE", "Fr3MinkIK", "dual_arm_collision_pairs"]

MINK_SOLVER = "daqp"  # the qpsolvers backend the replay extra installs


def dual_arm_collision_pairs(
    model: mujoco.MjModel, sides: tuple[str, str] = ("left", "right")
) -> list:
    """Geom-pair spec for ``mink.CollisionAvoidanceLimit`` covering the
    AVOIDABLE collisions of a dual FR3 + chopsticks rig: inter-arm crossings
    and per-arm self-collision in which an arm LINK is involved.

    Mirrors the collision counter's "avoidable" definition (see
    ``kiwi.replay.collision``), so it does NOT constrain the two
    grippers meeting (task-inherent, where the demonstrated hands meet) or a
    gripper's own jaws (designed to touch). The link/gripper split follows the explicitly named SA root subtree.
    mink itself drops welded / parent-child / contype-incompatible pairs, so
    adjacent links and the fixed flange->gripper attachment need no
    special-casing.

    Returns a list of ``(group_a, group_b)`` geom-id lists; empty groups are
    pruned (e.g. a rig with no matching geoms yields ``[]`` -> no limit)."""

    from kiwi.replay.collision import geom_roles

    roles = geom_roles(model, sides)
    links = {s: roles[s]["links"] for s in sides}
    grips = {s: roles[s]["gripper"] for s in sides}
    lt, rt = sides
    pairs = [
        (links[lt], links[rt]),  # inter-arm: the arms' links cross
        (links[lt], grips[rt]),  # inter-arm: left link into the right gripper
        (links[rt], grips[lt]),  # inter-arm: right link into the left gripper
        (links[lt], links[lt]),  # self: left arm folds its own links together
        (links[lt], grips[lt]),  # self: left link into its own gripper
        (links[rt], links[rt]),  # self: right arm folds its own links together
        (links[rt], grips[rt]),  # self: right link into its own gripper
    ]
    return [(a, b) for a, b in pairs if a and b]


class Fr3MinkIK:
    """mink differential IK for one or more FR3 arms sharing one model.

    ``arms`` is a list of dicts, one per arm, each with int arrays/ids:
    ``{"arm_qadr": (7,), "gripper_qadr": (2,), "site_id": int}``. The whole
    model is wrapped in a single ``mink.Configuration``; a low-cost PostureTask
    soaks up each 7-dof arm's redundant nullspace. Two solve modes:

      combined=False (default): each arm is solved in its own QP, freezing
        every other dof (the other arms + all grippers). The arms never see
        each other - this is two independent solves for a dual rig.

      combined=True: all arms' FrameTasks go into ONE QP that moves every arm
        joint at once (only the grippers are frozen). The arms share one
        configuration, so they can be coordinated - pass ``collision_pairs``
        (see ``dual_arm_collision_pairs``) to add active inter-arm + self
        collision avoidance to the shared QP.

    ``collision_pairs`` also works in independent mode (each arm dodges the
    other at its frozen pose), but combined lets both arms move to resolve a
    clash.

    A single arm is just ``arms=[one]`` - the two modes coincide there. Each
    arm's coupled jaw is written straight in every frame (rev_left = angle,
    rev_right = -angle) and held frozen.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        arms: list[dict],
        home_qpos: np.ndarray,
        *,
        combined: bool = False,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        posture_cost: float = 1e-3,
        frame_gain: float = 0.5,
        lm_damping: float = 1e-3,
        config_limit_gain: float = 0.95,
        collision_pairs: list | None = None,
        collision_margin: float = 0.004,
        collision_detect: float = 0.01,
        collision_gain: float = 0.85,
    ) -> None:
        self.model = model
        self.combined = bool(combined)
        self.home_qpos = np.asarray(home_qpos, dtype=np.float64).copy()

        self.configuration = mink.Configuration(model, q=self.home_qpos)
        self.posture_task = mink.PostureTask(model, cost=posture_cost)
        self.posture_task.set_target(self.home_qpos)
        self.config_limit = mink.ConfigurationLimit(model, gain=config_limit_gain)
        # Optional active collision avoidance (inter-arm + self). When enabled
        # it is added to the QP's velocity limits in BOTH solve modes; in
        # `combined` mode both arms can move to dodge each other, in
        # `independent` mode each arm dodges the other at its frozen pose.
        # Off by default; live-togglable via
        # `set_collision_avoidance`.
        self._collision_detect = float(collision_detect)
        self._collision_gain = float(collision_gain)
        self.collision_limit = None
        self._limits = [self.config_limit]
        self.set_collision_avoidance(
            collision_pairs or None, margin=float(collision_margin)
        )

        self.arms = []
        for a in arms:
            sid = int(a["site_id"])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
            arm_qadr = np.asarray(a["arm_qadr"], dtype=np.int32)
            self.arms.append(
                {
                    "arm_qadr": arm_qadr,
                    "gripper_qadr": np.asarray(a["gripper_qadr"], dtype=np.int32),
                    "site_id": sid,
                    "arm_dof": [int(d) for d in arm_qadr],
                    "frame_task": mink.FrameTask(
                        frame_name=name,
                        frame_type="site",
                        position_cost=position_cost,
                        orientation_cost=orientation_cost,
                        gain=frame_gain,
                        lm_damping=lm_damping,
                    ),
                }
            )

        active = {d for a in self.arms for d in a["arm_dof"]}
        # combined: freeze only the non-arm dofs (grippers).
        self._combined_frozen = np.asarray(
            [d for d in range(model.nv) if d not in active], dtype=np.int32
        )
        self._combined_freeze = mink.DofFreezingTask(
            model=model, dof_indices=[int(d) for d in self._combined_frozen]
        )
        # independent: per arm, freeze everything except that arm's 7 dofs.
        for a in self.arms:
            fr = [d for d in range(model.nv) if d not in set(a["arm_dof"])]
            a["frozen_dof"] = np.asarray(fr, dtype=np.int32)
            a["freeze"] = mink.DofFreezingTask(model=model, dof_indices=fr)
        self.qp_failures = 0

    def set_gains(self, gain: float, orientation_cost: float) -> None:
        """Live-tune the FrameTask gain + orientation cost on every arm."""
        for a in self.arms:
            a["frame_task"].gain = float(gain)
            a["frame_task"].set_orientation_cost(float(orientation_cost))

    def set_posture_cost(self, cost: float) -> None:
        """Live-tune the PostureTask weight: how hard the IK is pulled toward
        the posture reference (the home/preset configuration) relative to TCP
        tracking. Higher = holds the posture closer, sacrificing tracking."""
        self.posture_task.set_cost(float(cost))

    def set_collision_avoidance(
        self, pairs: list | None, *, margin: float = 0.004
    ) -> None:
        """Enable (``pairs`` given, see ``dual_arm_collision_pairs``) or
        disable (``None``/empty) active collision avoidance, live.

        Rebuilds the ``CollisionAvoidanceLimit`` so a margin change takes
        effect too; the next ``solve`` uses the new limit set."""
        if pairs:
            self.collision_limit = mink.CollisionAvoidanceLimit(
                self.model,
                geom_pairs=pairs,
                gain=self._collision_gain,
                minimum_distance_from_collisions=float(margin),
                collision_detection_distance=self._collision_detect,
            )
        else:
            self.collision_limit = None
        self._limits = [self.config_limit] + (
            [self.collision_limit] if self.collision_limit is not None else []
        )

    def reset(self, arm_seeds: list | None = None) -> None:
        """Park at home; optionally seed each arm (``arm_seeds[i]`` may be
        None)."""
        q = self.home_qpos.copy()
        if arm_seeds is not None:
            for a, seed in zip(self.arms, arm_seeds):
                if seed is not None:
                    q[a["arm_qadr"]] = seed
        self.configuration.update(q=q)
        self.qp_failures = 0

    def _errs(self, a: dict) -> tuple[float, float]:
        err = a["frame_task"].compute_error(self.configuration)
        pe = float(
            np.linalg.norm(a["_tpos"] - self.configuration.data.site(a["site_id"]).xpos)
        )
        return pe, float(np.linalg.norm(err[3:]))

    def solve(
        self,
        targets: list,
        grippers: list,
        *,
        arm_seeds: list | None = None,
        max_iters: int = 120,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
        step_tol: float = 1e-6,
        dt: float = 0.1,
    ) -> list[tuple[np.ndarray, float, float]]:
        """Solve one frame for all arms. ``targets[i]`` is
        (pos(3,), quat_wxyz(4,)) and ``grippers[i]`` the rev_left angle, one
        per arm. Returns a list of (arm_qpos(7,), pos_err_m, rot_err_rad), one
        per arm, in ``arms`` order."""
        data = self.configuration.data
        for i, a in enumerate(self.arms):
            if arm_seeds is not None and arm_seeds[i] is not None:
                data.qpos[a["arm_qadr"]] = arm_seeds[i]
            g = float(grippers[i])
            data.qpos[a["gripper_qadr"]] = jaw_qpos(g)
            pos, quat = targets[i]
            rot = mink.SO3(wxyz=np.asarray(quat, dtype=np.float64)).normalize()
            a["frame_task"].set_target(
                mink.SE3.from_rotation_and_translation(
                    rotation=rot, translation=np.asarray(pos, dtype=np.float64)
                )
            )
            a["_tpos"] = np.asarray(pos, dtype=np.float64)
        self.configuration.update()

        if self.combined:
            return self._solve_combined(max_iters, tol_pos, tol_rot, step_tol, dt)
        return self._solve_independent(max_iters, tol_pos, tol_rot, step_tol, dt)

    def _solve_combined(self, max_iters, tol_pos, tol_rot, step_tol, dt):
        tasks = [a["frame_task"] for a in self.arms] + [self.posture_task]
        active = [d for a in self.arms for d in a["arm_dof"]]
        for _ in range(max_iters):
            if all(
                pe < tol_pos and re < tol_rot
                for pe, re in (self._errs(a) for a in self.arms)
            ):
                break
            try:
                vel = mink.solve_ik(
                    self.configuration,
                    tasks,
                    dt,
                    MINK_SOLVER,
                    limits=self._limits,
                    constraints=[self._combined_freeze],
                )
            except mink.exceptions.NoSolutionFound:
                self.qp_failures += 1
                break
            vel[self._combined_frozen] = 0.0
            self.configuration.integrate_inplace(vel, dt)
            if float(np.linalg.norm(vel[active])) * dt < step_tol:
                break
        data = self.configuration.data
        return [(data.qpos[a["arm_qadr"]].copy(), *self._errs(a)) for a in self.arms]

    def _solve_independent(self, max_iters, tol_pos, tol_rot, step_tol, dt):
        data = self.configuration.data
        out = []
        for a in self.arms:
            tasks = [a["frame_task"], self.posture_task]
            for _ in range(max_iters):
                pe, re = self._errs(a)
                if pe < tol_pos and re < tol_rot:
                    break
                try:
                    vel = mink.solve_ik(
                        self.configuration,
                        tasks,
                        dt,
                        MINK_SOLVER,
                        limits=self._limits,
                        constraints=[a["freeze"]],
                    )
                except mink.exceptions.NoSolutionFound:
                    self.qp_failures += 1
                    break
                vel[a["frozen_dof"]] = 0.0
                self.configuration.integrate_inplace(vel, dt)
                if float(np.linalg.norm(vel[a["arm_dof"]])) * dt < step_tol:
                    break
            out.append((data.qpos[a["arm_qadr"]].copy(), *self._errs(a)))
        return out

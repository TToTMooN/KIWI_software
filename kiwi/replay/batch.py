"""Batch-episode machinery: load an episode set, solve it on a private rig.

``EpisodeSet`` holds a loaded batch of episodes with their global +
per-episode offsets and solve state; ``BatchSolver`` solves them on a PRIVATE
compiled model + Fr3MinkIK, optionally in a daemon thread so a GUI never
blocks, and nothing mutable is shared with any other solver (the Fr3MinkIK
constructor mutates the model it wraps). With ``workers > 1`` the episodes -
which are fully independent solves - fan out over spawn-context worker
processes, each with its own rig (BLAS pinned to one thread per worker). The
shared per-trajectory solve + pose/offset/continuity math it builds on lives
in ``kiwi.replay.solve``.

The one input source is episode directories (``kiwi.replay.episodes``).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import multiprocessing
import os
import threading
import traceback
from pathlib import Path

import mujoco
import numpy as np

from kiwi.replay.ik import Fr3MinkIK, dual_arm_collision_pairs
from kiwi.replay.mount import model_mount, set_model_mount
from kiwi.replay.solve import (
    SIDES,
    SolveResult,
    _resolve_indices,
    _set_bases_dict,
    compose_offsets,
    continuity_score,
    solve_trajectory,
)

# ---------------------------------------------------------------------------
# Episode set + background batch solver
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EpisodeSet:
    """A loaded batch of episodes with offsets + solve state.

    ``offset_global`` is the batch-mode target ("move ALL trajectories");
    ``offsets[ep]`` the individual-mode target. ``continuity[ep]`` is the L0
    demo-quality check, computed once at load on the full-rate stream (see
    ``continuity_score``) - no later parameter change can invalidate it.
    Every episode's arrays are retained (results: solved/target/actual +
    metrics) so playback of any solved episode is an instant swap."""

    meta: dict
    episodes: dict[int, dict[str, np.ndarray]]
    offset_global: np.ndarray
    offsets: dict[int, np.ndarray]
    results: dict[int, dict]
    status: dict[int, str]  # pending | solving | done | stale | cancelled | error
    continuity: dict[int, dict]
    # Total capture-clock decimation behind `episodes` (meta decim x stride),
    # so a consumer can map a replay frame back onto the raw stream.
    decim: int = 1

    @classmethod
    def load(
        cls,
        dirs: list[Path],
        *,
        fps: float | None = None,
        rot_offset_deg=(0.0, 0.0, 0.0),
        traj_yaw_deg: float = 0.0,
        stride: int = 1,
        quiet: bool = False,
    ) -> "EpisodeSet":
        """Load the given episode dirs (each holding a ``replay_state.npz``).

        Episode ids are positions in ``dirs``. ``fps`` decimates the capture
        clock (None keeps ``episodes.DEFAULT_FPS``; 0 keeps the native rate).
        """
        from kiwi.replay import episodes as source

        meta = source.load_meta(
            dirs, fps=source.DEFAULT_FPS if fps is None else fps, quiet=quiet
        )
        ep_ids = list(range(meta["total_episodes"]))
        eps, continuity = {}, {}
        for ep in ep_ids:
            # Load at FULL rate: the L0 continuity thresholds are per-step of
            # the native ~30 fps stream, so score before stride subsampling.
            q, _ = source.episode_to_qpos(
                meta,
                ep,
                rot_offset_deg=rot_offset_deg,
                traj_yaw_deg=traj_yaw_deg,
                stride=1,
            )
            # Scored at the decimated rate the solve consumes, with the
            # thresholds scaled to that rate (see continuity_score).
            continuity[ep] = continuity_score(q, fps=meta["fps"])
            if stride > 1:
                q = {s: arr[::stride] for s, arr in q.items()}
            eps[ep] = q
        return cls(
            meta=meta,
            episodes=eps,
            offset_global=np.zeros(6),
            offsets={ep: np.zeros(6) for ep in ep_ids},
            results={},
            status={ep: "pending" for ep in ep_ids},
            continuity=continuity,
            decim=int(meta.get("decim", 1)) * max(int(stride), 1),
        )

    @property
    def ep_ids(self) -> list[int]:
        return sorted(self.episodes)

    def episode_dir(self, ep: int) -> Path:
        return self.meta["dirs"][ep]

    def effective_offset(self, ep: int) -> np.ndarray:
        """The one composed offset the solver/paths see for this episode."""
        return compose_offsets(self.offset_global, self.offsets[ep])

    def metrics(self, ep: int) -> SolveResult | None:
        r = self.results.get(ep)
        return None if r is None else r.get("metrics")


@dataclasses.dataclass
class _Rig:
    model: mujoco.MjModel
    ik: Fr3MinkIK
    idx: dict
    mocapids: dict
    home_qpos: np.ndarray
    coll: dict | None
    chopsticks_body: str


def _build_rig(
    robot,
    base_distance: float,
    mount,
    home_qpos,
    *,
    combined: bool,
    lm_damping: float,
    avoid_collisions: bool,
    collision_margin: float,
    count_collisions: bool,
) -> _Rig:
    """Compile a private dual rig + Fr3MinkIK. Used once by the in-process
    BatchSolver and once per pool worker: Fr3MinkIK mutates the model it
    wraps, so every solver needs its own compile. ``home_qpos=None`` reads the
    compiled model's ``home`` keyframe."""
    from kiwi.replay.collision import build_collision_ctx
    from kiwi.replay.rig import build_dual

    model = build_dual(robot, base_distance, mount).compile()
    if home_qpos is None:
        home_qpos = model.key("home").qpos
    home_qpos = np.asarray(home_qpos, np.float64).copy()
    idx = {
        s: _resolve_indices(
            model, s, robot.arm_joints, robot.tcp_site, robot.gripper_joints
        )
        for s in SIDES
    }
    mocapids = {s: model.body(f"{s}_mount").mocapid[0] for s in SIDES}
    arm_specs = [idx[s] for s in SIDES]
    pairs = dual_arm_collision_pairs(model) if avoid_collisions else None
    ik = Fr3MinkIK(
        model,
        arm_specs,
        home_qpos,
        combined=combined,
        lm_damping=lm_damping,
        collision_pairs=pairs,
        collision_margin=collision_margin,
    )
    coll = build_collision_ctx(model) if count_collisions else None
    return _Rig(
        model=model,
        ik=ik,
        idx=idx,
        mocapids=mocapids,
        home_qpos=home_qpos,
        coll=coll,
        chopsticks_body=robot.chopsticks_body,
    )


def _apply_run_setup(
    rig: _Rig,
    mounts,
    base: dict,
    frame_gain: float,
    orientation_cost: float,
    posture_cost: float,
) -> None:
    """Write one run's rig snapshot (live mounts, bases, gains) onto ``rig``."""
    if mounts is not None:
        for s in SIDES:
            set_model_mount(rig.model, mounts[s], f"{s}_{rig.chopsticks_body}")
    _set_bases_dict(rig.ik.configuration.data, rig.mocapids, base)
    rig.ik.set_gains(frame_gain, orientation_cost)
    rig.ik.set_posture_cost(posture_cost)


def _solve_episode(
    rig: _Rig,
    trajs: dict,
    *,
    ws: dict,
    base_dist: float,
    ik_iters: int,
    start_frame: int,
    offset6,
    cancel: threading.Event | None = None,
) -> dict:
    """Solve one episode on ``rig``; returns the EpisodeSet results entry
    ``{"solved", "target", "actual", "metrics"}``."""
    nf = {s: trajs[s].shape[0] for s in SIDES}
    dof = len(rig.idx["left"]["arm_qadr"])
    solved = {s: np.zeros((nf[s], dof)) for s in SIDES}
    target = {s: np.zeros((nf[s], 6)) for s in SIDES}
    actual = {s: np.zeros((nf[s], 6)) for s in SIDES}
    rig.ik.reset()
    r = solve_trajectory(
        rig.ik,
        rig.idx,
        rig.home_qpos,
        trajs,
        solved,
        target,
        actual,
        shared_frame=True,
        ws=ws,
        base_dist=base_dist,
        ik_iters=ik_iters,
        start_frame=start_frame,
        offset6=offset6,
        collision_ctx=rig.coll,
        model=rig.model,
        cancel=cancel,
    )
    return {"solved": solved, "target": target, "actual": actual, "metrics": r}


# --- pool workers (workers > 1): one rig per process, episodes as jobs -------

_POOL_RIG: dict = {}  # per-process context, populated by _pool_init


def _pool_init(cfg: dict) -> None:
    """Build this worker's rig once (spawn start). A failure is kept, not
    raised; the jobs report it per-episode instead."""
    try:
        rig = _build_rig(
            cfg["robot"],
            cfg["base_distance"],
            cfg["mount"],
            cfg["home_qpos"],
            combined=cfg["combined"],
            lm_damping=cfg["lm_damping"],
            avoid_collisions=cfg["avoid_collisions"],
            collision_margin=cfg["collision_margin"],
            count_collisions=cfg["count_collisions"],
        )
        _apply_run_setup(
            rig,
            cfg["mounts"],
            cfg["base"],
            cfg["frame_gain"],
            cfg["orientation_cost"],
            cfg["posture_cost"],
        )
        _POOL_RIG["rig"] = rig
    except Exception as e:
        traceback.print_exc()
        _POOL_RIG["error"] = f"{type(e).__name__}: {e}"


def _pool_solve_job(job: tuple) -> tuple[int, dict]:
    """Solve one ``(ep, trajs, kwargs)`` job on this process's rig. Exceptions
    come back as an ``{"error": ...}`` result."""
    ep, trajs, kw = job
    if "error" in _POOL_RIG:
        return ep, {"error": f"worker init failed: {_POOL_RIG['error']}"}
    try:
        return ep, _solve_episode(_POOL_RIG["rig"], trajs, **kw)
    except Exception as e:
        traceback.print_exc()
        return ep, {"error": f"{type(e).__name__}: {e}"}


class BatchSolver:
    """Batch IK over an EpisodeSet on a private rig, optionally in a daemon
    thread so a GUI stays interactive. The rig (build_dual + Fr3MinkIK) is
    compiled lazily on first use - on the worker thread, so a GUI never blocks
    on it - and shares NOTHING mutable with any other solver. At the start of
    every run the live gripper mounts are copied off the app model (under its
    mount lock, when one is given) and the live bases are written to the
    private MjData, so results always reflect the rig as currently tuned.

    ``workers > 1`` solves multi-episode runs in that many spawn-context
    worker processes instead (single-episode runs stay in-process). Episodes
    are independent solves, so the results are identical either way."""

    def __init__(
        self,
        robot,
        base_distance: float,
        build_mount,
        home_qpos=None,
        *,
        combined: bool = False,
        lm_damping: float = 1e-3,
        avoid_collisions: bool = False,
        collision_margin: float = 0.004,
        count_collisions: bool = True,
        workers: int = 1,
    ):
        self._robot = robot
        self._base_distance = float(base_distance)
        self._build_mount = build_mount
        self._home_qpos = (
            None if home_qpos is None else np.asarray(home_qpos, np.float64).copy()
        )
        self._combined = bool(combined)
        self._lm_damping = float(lm_damping)
        self._avoid_collisions = bool(avoid_collisions)
        self._collision_margin = float(collision_margin)
        self._count_collisions = bool(count_collisions)
        self.workers = max(1, int(workers))
        self.parallel = False  # set per run: workers > 1 and > 1 episode
        self._rig: _Rig | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.lock = threading.Lock()  # guards EpisodeSet results/status

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        self._cancel.set()

    def set_avoid_collisions(self, avoid: bool, margin: float | None = None) -> None:
        """Change the QP collision-avoidance setting for FUTURE runs.

        Drops the cached private rig so the next run recompiles with the new
        setting (avoidance is baked into the rig's Fr3MinkIK at build time).
        A run already in flight keeps the rig it started with.
        """
        self._avoid_collisions = bool(avoid)
        if margin is not None:
            self._collision_margin = float(margin)
        self._rig = None

    def _ensure_rig(self) -> _Rig:
        if self._rig is None:
            self._rig = _build_rig(
                self._robot,
                self._base_distance,
                self._build_mount,
                self._home_qpos,
                combined=self._combined,
                lm_damping=self._lm_damping,
                avoid_collisions=self._avoid_collisions,
                collision_margin=self._collision_margin,
                count_collisions=self._count_collisions,
            )
        return self._rig

    def start(
        self,
        episode_set: EpisodeSet,
        ep_ids,
        *,
        ws: dict,
        base: dict,
        ik_iters: int,
        frame_gain: float,
        orientation_cost: float,
        posture_cost: float,
        start_frame: int = 0,
        app_model: mujoco.MjModel | None = None,
        app_mount_lock: threading.Lock | None = None,
        on_episode_done=None,
        on_progress=None,
        on_finished=None,
        background: bool = True,
    ) -> bool:
        """Solve ``ep_ids`` against the given rig snapshot. Returns False if a
        run is already active. Callbacks (all optional) fire on the worker
        thread: ``on_progress(k, total, ep)`` with ``k`` episodes finished
        (serial: before solving ``ep``; parallel: after ``ep`` completed -
        check ``self.parallel``), ``on_episode_done(ep, SolveResult)`` after
        each, ``on_finished()`` at the end (also after a cancel or an
        error)."""
        if self.running:
            return False
        self._cancel.clear()
        ep_ids = list(ep_ids)
        self.parallel = self.workers > 1 and len(ep_ids) > 1
        # Snapshot the app's live mounts NOW (caller thread, under its lock):
        # the worker applies them to the private model before solving.
        mounts = None
        if app_model is not None:
            lock = app_mount_lock if app_mount_lock is not None else threading.Lock()
            with lock:
                mounts = {
                    s: model_mount(app_model, f"{s}_{self._robot.chopsticks_body}")
                    for s in SIDES
                }
        run_args = (
            episode_set,
            ep_ids,
            dict(ws),
            dict(base),
            int(ik_iters),
            float(frame_gain),
            float(orientation_cost),
            float(posture_cost),
            int(start_frame),
            mounts,
            on_episode_done,
            on_progress,
            on_finished,
        )
        if not background:
            self._run(*run_args)
            return True
        self._thread = threading.Thread(
            target=self._run, args=run_args, daemon=True, name="batch-ik"
        )
        self._thread.start()
        return True

    def _run(
        self,
        es: EpisodeSet,
        ep_ids,
        ws,
        base,
        ik_iters,
        frame_gain,
        orientation_cost,
        posture_cost,
        start_frame,
        mounts,
        on_episode_done,
        on_progress,
        on_finished,
    ) -> None:
        try:
            run = self._run_parallel if self.parallel else self._run_serial
            run(
                es,
                ep_ids,
                ws,
                base,
                ik_iters,
                frame_gain,
                orientation_cost,
                posture_cost,
                start_frame,
                mounts,
                on_episode_done,
                on_progress,
            )
        except Exception as e:  # surface, don't kill the caller's process
            traceback.print_exc()
            with self.lock:
                for ep in ep_ids:
                    if es.status.get(ep) == "solving":
                        es.status[ep] = "error"
                        es.results[ep] = {"error": str(e)}
        finally:
            if on_finished is not None:
                on_finished()

    def _run_serial(
        self,
        es: EpisodeSet,
        ep_ids,
        ws,
        base,
        ik_iters,
        frame_gain,
        orientation_cost,
        posture_cost,
        start_frame,
        mounts,
        on_episode_done,
        on_progress,
    ) -> None:
        rig = self._ensure_rig()
        _apply_run_setup(rig, mounts, base, frame_gain, orientation_cost, posture_cost)
        for k, ep in enumerate(ep_ids):
            if self._cancel.is_set():
                with self.lock:
                    for e2 in ep_ids[k:]:
                        if es.status.get(e2) == "pending":
                            es.status[e2] = "cancelled"
                break
            with self.lock:
                es.status[ep] = "solving"
            if on_progress is not None:
                on_progress(k, len(ep_ids), ep)
            result = _solve_episode(
                rig,
                es.episodes[ep],
                ws=ws,
                base_dist=base["dist"],
                ik_iters=ik_iters,
                start_frame=start_frame,
                offset6=es.effective_offset(ep),
                cancel=self._cancel,
            )
            r = result["metrics"]
            with self.lock:
                es.results[ep] = result
                es.status[ep] = "cancelled" if r.cancelled else "done"
            if on_episode_done is not None:
                on_episode_done(ep, r)

    def _run_parallel(
        self,
        es: EpisodeSet,
        ep_ids,
        ws,
        base,
        ik_iters,
        frame_gain,
        orientation_cost,
        posture_cost,
        start_frame,
        mounts,
        on_episode_done,
        on_progress,
    ) -> None:
        """Fan the episodes out over spawn-context pool workers (episodes are
        fully independent solves, so results are identical to serial). Each
        worker compiles its own rig once (``_pool_init``); the collection loop
        polls so Cancel terminates in-flight solves within ~0.2 s."""
        # The parent's BLAS is long since initialized (no effect on it), but
        # spawned children inherit os.environ and their numpy reads these at
        # import. setdefault respects an explicit user override.
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(var, "1")
        cfg = {
            "robot": self._robot,
            "base_distance": self._base_distance,
            "mount": self._build_mount,
            "home_qpos": self._home_qpos,
            "combined": self._combined,
            "lm_damping": self._lm_damping,
            "avoid_collisions": self._avoid_collisions,
            "collision_margin": self._collision_margin,
            "count_collisions": self._count_collisions,
            "mounts": mounts,
            "base": base,
            "frame_gain": float(frame_gain),
            "orientation_cost": float(orientation_cost),
            "posture_cost": float(posture_cost),
        }
        with self.lock:
            for ep in ep_ids:
                es.status[ep] = "solving"
        jobs = [
            (
                ep,
                es.episodes[ep],
                {
                    "ws": ws,
                    "base_dist": base["dist"],
                    "ik_iters": ik_iters,
                    "start_frame": start_frame,
                    "offset6": es.effective_offset(ep),
                },
            )
            for ep in ep_ids
        ]
        total, done = len(jobs), 0
        ctx = multiprocessing.get_context("spawn")
        # A worker that dies HARD (segfault/OOM-kill in native MuJoCo/QP code)
        # breaks the executor and fails every outstanding future with
        # BrokenProcessPool.
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(self.workers, total),
            mp_context=ctx,
            initializer=_pool_init,
            initargs=(cfg,),
        )
        try:
            futures = {executor.submit(_pool_solve_job, job): job[0] for job in jobs}
            pending = set(futures)
            while pending and not self._cancel.is_set():
                finished, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.2,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in finished:
                    ep = futures[fut]
                    try:
                        ep, result = fut.result()
                    except Exception as e:  # BrokenProcessPool et al.
                        result = {"error": f"{type(e).__name__}: {e}"}
                    with self.lock:
                        es.results[ep] = result
                        es.status[ep] = "error" if "error" in result else "done"
                    done += 1
                    if on_progress is not None:
                        on_progress(done, total, ep)
                    if on_episode_done is not None and "metrics" in result:
                        on_episode_done(ep, result["metrics"])
        finally:
            # On cancel: queued jobs are dropped; solves already in flight
            # finish in the background and are discarded (the executor cannot
            # kill a running process).
            executor.shutdown(wait=False, cancel_futures=True)
        if self._cancel.is_set():
            with self.lock:
                for ep in ep_ids:
                    if es.status.get(ep) == "solving":
                        es.status[ep] = "cancelled"

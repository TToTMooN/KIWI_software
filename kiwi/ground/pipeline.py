"""decode -> windows -> (envelope) -> sweep + stereo -> identify, per camera; then pool."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from kiwi.ground import decode
from kiwi.ground.camera import kb4_scaled
from kiwi.ground.gripper import envelope
from kiwi.ground.identify import identify
from kiwi.ground.pool import Result, pool
from kiwi.ground.stereo import sgbm, stereo_window
from kiwi.ground.sweep import sweep_window
from kiwi.ground.windows import keep_frames, make_windows, omega

ENV_FRAMES = 300  # frames for the static envelope
ENV_MIN = 100  # fewer keyframes than this: sample the video on a stride instead


def _print(*a):
    print(*a, flush=True)


def rig_envelope(cam, cfg):
    kb4 = kb4_scaled(cam.kb4, cfg.decode_size / cam.image_size)
    kf = decode.keyframes(cam.video, cfg.decode_size, cam.image_size, limit=ENV_FRAMES)
    env, n, frac = envelope(kf, kb4)
    if n < ENV_MIN:  # long-GOP encodings
        step = max(1, len(cam.t) // ENV_FRAMES)
        strided = (
            f
            for _, f in decode.frames(
                cam.video, range(0, len(cam.t), step), cfg.decode_size, cam.image_size
            )
        )
        env, n, frac = envelope(strided, kb4)
    return env, n, frac


def camera_windows(cam, cfg, disparity=sgbm, log=_print):
    """All window records of one camera and the stage timings [s].

    Windows failing the pose gate are recorded with flag 'pose_gate', windows
    thinned by window_stride with 'skipped'.
    """
    t_start = time.time()
    T = dict(envelope=0.0, decode=0.0, sweep=0.0, stereo=0.0, identify=0.0, window=0.0)
    env, n_env, frac = rig_envelope(cam, cfg)
    T["envelope"] = time.time() - t_start
    warn = (
        "  -- more than half the cone is static: too little motion or texture"
        if frac > 0.5
        else ""
    )
    log(
        f"[{cam.name}] static envelope from {n_env} frames: {frac:.0%} of the lens cone{warn}"
    )
    wins = make_windows(cam, cfg)
    om = omega(cam)
    gated = [w for w in wins if w["gate"]]
    todo = gated[:: cfg.window_stride]
    log(
        f"[{cam.name}] {len(gated)}/{len(wins)} windows pass the pose gate (median off-nadir "
        f"{np.median(cam.off_nadir_deg):.0f} deg, median span {np.median([w['span_m'] for w in wins]):.2f} m)"
    )
    records = [
        dict(
            t0=w["t0"],
            t1=w["t1"],
            z=np.nan,
            sigma=np.nan,
            n_support=0,
            channel=None,
            flags=["pose_gate" if not w["gate"] else "skipped"],
            cam_z=float(np.median(cam.pos[w["frames"], 2])),
        )
        for w in wins
        if w not in todo
    ]
    need = sorted({f for w in todo for f in w["frames"]})
    last_use = {}
    for k, w in enumerate(todo):
        for f in w["frames"]:
            last_use[f] = k
    buf = {}
    p = 0
    t0 = time.time()
    t_dec = time.time()
    ex = ThreadPoolExecutor(1)

    def timed_stereo(*a):
        t = time.time()
        return stereo_window(*a), time.time() - t

    stream = (
        decode.prefetch(decode.frames(cam.video, need, cfg.decode_size, cam.image_size))
        if todo
        else ()
    )
    for i, img in stream:
        T["decode"] += time.time() - t_dec
        buf[i] = img
        while p < len(todo) and all(f in buf for f in todo[p]["frames"]):
            w = todo[p]
            idx = keep_frames(w["frames"], om)
            t1 = time.time()
            # CPU (SGBM) concurrently with the GPU sweep
            fut = ex.submit(
                timed_stereo, {f: buf[f] for f in idx}, cam, idx, env, cfg, disparity
            )
            sw = sweep_window(np.stack([buf[f] for f in idx]), cam, idx, env, cfg)
            t2 = time.time()
            st, dt_st = fut.result()
            t3 = time.time()
            rec = identify(sw, st, w, cfg)
            T["sweep"] += t2 - t1
            T["stereo"] += dt_st
            T["window"] += t3 - t1
            T["identify"] += time.time() - t3
            use_st = rec["channel"] in ("stereo", "both") and st["points"] is not None
            rec["points"] = st["points"] if use_st else sw["points"]
            records.append(rec)
            p += 1
            for f in [f for f in buf if last_use.get(f, -1) < p]:
                del buf[f]
            if p % 10 == 0 or p == len(todo):
                dt = (time.time() - t0) / p
                log(
                    f"[{cam.name}] {p}/{len(todo)} windows, {dt:.2f} s/window, last z={rec['z']:+.3f} {rec['channel'] or rec['flags']}"
                )
        t_dec = time.time()
    ex.shutdown()
    if p != len(todo):
        raise RuntimeError(
            f"{cam.video}: the decoder stopped early, {len(todo) - p} of {len(todo)} windows never received their frames"
        )
    records.sort(key=lambda r: r["t0"])
    T.update(n_windows=len(todo), n_frames=len(need), total=time.time() - t_start)
    return records, T


def estimate(cameras, cfg, disparity=sgbm, log=_print) -> Result:
    """The whole estimate for one episode: every camera, then the pooled level."""
    t0 = time.time()
    by_cam, timing = {}, {}
    for cam in cameras:
        by_cam[cam.name], timing[cam.name] = camera_windows(cam, cfg, disparity, log)
        n = sum(not r["flags"] for r in by_cam[cam.name])
        log(
            f"[{cam.name}] {n}/{len(by_cam[cam.name])} windows usable, {timing[cam.name]['total']:.0f} s"
        )
    t1 = time.time()
    res = pool(by_cam, cfg, time.time() - t0)
    timing["pool"] = time.time() - t1
    res.timing = timing
    return res

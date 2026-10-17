"""Windows of window_s seconds on the pose clock: frame grid, pose-only gate, angular-rate filter."""

from __future__ import annotations

import numpy as np

OMEGA_MAX = 1.5  # rad/s: faster-turning frames are dropped ...
MIN_KEEP = 10  # ... if at least this many remain


def omega(cam) -> np.ndarray:
    """Angular rate per frame [rad/s] from consecutive poses."""
    dR = np.einsum("nji,njk->nik", cam.R[:-1], cam.R[1:])
    tr = np.trace(dR, axis1=1, axis2=2)
    a = np.arccos(np.clip((tr - 1) / 2, -1, 1)) * cam.fps
    return np.append(a, a[-1])


def frame_stride(cam, cfg) -> int:
    """Frames per step so that n_frames frames span window_s (6 at 120 fps, 1 at 20 fps)."""
    return max(1, round(cam.fps * cfg.window_s / cfg.n_frames))


def make_windows(cam, cfg) -> list[dict]:
    """dict(t0, t1, frames, span_m, off_nadir_deg, gate) per window."""
    t = cam.t
    offn = cam.off_nadir_deg
    L, S, N = cfg.window_s, frame_stride(cam, cfg), cfg.n_frames
    out = []
    for k in range(int(np.ceil(t[0] / L)), int(np.floor((t[-1] - L) / L)) + 1):
        t0 = k * L
        i0 = int(np.searchsorted(t, t0))
        idx = np.arange(-(-i0 // S) * S, i0 + N * S, S)[:N]
        if idx[-1] >= len(t):
            break
        xy = cam.pos[idx, :2]
        span = float(np.max(np.linalg.norm(xy[:, None] - xy[None], axis=-1)))
        o = float(np.median(offn[idx]))
        out.append(
            dict(
                t0=float(t0),
                t1=float(t0 + L),
                frames=idx.tolist(),
                span_m=span,
                off_nadir_deg=o,
                gate=bool(span >= cfg.min_span_m and o <= cfg.max_off_nadir_deg),
            )
        )
    return out


def keep_frames(idx, om) -> np.ndarray:
    """Drop frames turning faster than OMEGA_MAX if at least MIN_KEEP remain."""
    idx = np.asarray(idx)
    k = om[idx] <= OMEGA_MAX
    return idx[k] if k.sum() >= MIN_KEEP else idx

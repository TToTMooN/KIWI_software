"""Rectified intra-camera stereo: metric depth from the pose baseline, geometric floor pick.

Two frames of the same camera 0.15-0.35 m apart are rendered into an exactly
rectified pair (:func:`kiwi.ground.views.rectified_pair`); ``disparity(left,
right, n_disp)`` is the swappable matcher (SGBM by default; a learned net has
the same signature). Depth Z = f |b| / d is metric because |b| comes from the
poses, so there is no scale step. Floor pick: slope-gated pixels -> 5 cm z
histogram -> per-peak Huber plane -> gates -> lowest survivor inside the
camera-height band. Up to ``stereo_pairs`` disjoint pairs per window, median.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from kiwi.ground.camera import kb4_scaled
from kiwi.ground.gripper import scene_valid
from kiwi.ground.views import rectified_pair, render

BIN = 0.05  # z histogram bin [m]
SLOPE_MAX_DEG = 25.0  # local slope gate before the histogram
MAD_PX_MAX = 2.0  # plane residual MAD in disparity pixels
LR_TOL = 1.0  # left-right consistency [px]
LOW_TEX = 0.10  # LR-consistent fraction below this -> 'low_texture'
H_DISP_MIN = 0.35  # nearest surface the disparity range must cover [m]
VERT_MIN_DEG = (
    45.0  # baseline angle from vertical: near-vertical pairs cannot be rectified
)


def sgbm(left, right, n_disp):
    """Default matcher: left disparity in px (x_left - x_right), <= 0 where invalid."""
    gl, gr = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (left, right))
    bs = 5
    m = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=n_disp,
        blockSize=bs,
        P1=8 * bs * bs,
        P2=32 * bs * bs,
        uniquenessRatio=10,
        speckleWindowSize=200,
        speckleRange=2,
        disp12MaxDiff=-1,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )
    return m.compute(gl, gr).astype(np.float32) / 16


def lr_consistent(disparity, left, right, n_disp):
    """Left disparity and the left-right-consistent mask (matcher run on the mirrored pair)."""
    dL = disparity(left, right, n_disp)
    dR = disparity(
        np.ascontiguousarray(right[:, ::-1]),
        np.ascontiguousarray(left[:, ::-1]),
        n_disp,
    )
    dR = dR[:, ::-1]  # x_right - x_left = -d
    h, w = dL.shape
    u = np.arange(w)[None].repeat(h, 0)
    ur = np.clip(np.round(u - dL).astype(int), 0, w - 1)
    ok = (
        (dL > 0.5)
        & (u - dL >= 0)
        & (np.abs(dL - dR[np.arange(h)[:, None], ur]) <= LR_TOL)
    )
    return dL, ok, ur


def lift(view, d, pos, bl):
    Z = view.fx * bl / np.where(d > 0.5, d, np.nan)
    u, v = np.meshgrid(np.arange(view.w), np.arange(view.h))
    P = np.stack([(u - view.cx) / view.fx * Z, (v - view.cy) / view.fy * Z, Z], -1)
    return P @ view.R.T + pos


def local_slope(P, m, k):
    """Horizontal slope of a k x k masked least-squares plane around every pixel."""
    w = m.astype(np.float64)
    c = np.nanmean(np.where(m[..., None], P, np.nan), (0, 1))
    x, y, z = ((np.nan_to_num(P[..., i] - c[i]) * w) for i in range(3))

    def f(a):
        return cv2.boxFilter(a, -1, (k, k), normalize=False)

    n = f(w).clip(1e-9)
    mx, my, mz = f(x) / n, f(y) / n, f(z) / n
    cxx, cyy, cxy = (
        f(x * x) / n - mx * mx,
        f(y * y) / n - my * my,
        f(x * y) / n - mx * my,
    )
    cxz, cyz = f(x * z) / n - mx * mz, f(y * z) / n - my * mz
    det = cxx * cyy - cxy * cxy
    ok = (n >= 0.5 * k * k) & (det > 1e-12)
    dd = np.where(ok, det, 1)
    a = np.where(ok, (cxz * cyy - cyz * cxy) / dd, np.inf)
    b = np.where(ok, (cyz * cxx - cxz * cxy) / dd, np.inf)
    return np.hypot(a, b)


def huber_plane(X, Y, Z, iters=6):
    """z = a x + b y + c by Huber IRLS -> (a, b, c), MAD scale, inliers."""
    A = np.stack([X, Y, np.ones_like(X)], 1)
    w = np.ones(len(X))
    p = np.zeros(3)
    s = 1.0
    r = Z
    for _ in range(iters):
        p = np.linalg.lstsq(A * w[:, None], Z * w, rcond=None)[0]
        r = Z - A @ p
        s = 1.4826 * np.median(np.abs(r)) + 1e-4
        w = np.sqrt(np.minimum(1, 1.345 * s / np.abs(r).clip(1e-9)))
    return p, s, np.abs(r) < 3 * s


def surfaces(P, m, n_sv, d, f, bl, cfg):
    """Horizontal surfaces in the lifted points, lowest first: dict(z, tilt_deg, mad, support, mask).

    The tilt gate is the FIXED ``cfg.stereo_tilt_deg``.
    """
    g = m & np.isfinite(P[..., 2])
    g &= local_slope(P, g, 21) < math.tan(math.radians(SLOPE_MAX_DEG))
    z = P[..., 2]
    zz = z[g]
    out = []
    smin = cfg.stereo_support_min * n_sv
    if zz.size < smin:
        return out
    edges = np.arange(np.floor(zz.min() / BIN) * BIN - BIN, zz.max() + 2 * BIN, BIN)
    H, _ = np.histogram(zz, edges)
    Hp = np.pad(H, 1)
    peaks = [
        k
        for k in range(len(H))
        if Hp[k] + H[k] + Hp[k + 2] >= smin and H[k] >= Hp[k] and H[k] >= Hp[k + 2]
    ]
    for k in peaks:
        sel = g & (np.abs(z - (edges[k] + BIN / 2)) < 1.5 * BIN)
        if sel.sum() < smin:
            continue
        p, s, inl = huber_plane(P[..., 0][sel], P[..., 1][sel], z[sel])
        sup = np.zeros(sel.shape, bool)
        sup[sel] = inl
        n = int(sup.sum())
        tilt = math.degrees(math.atan(math.hypot(p[0], p[1])))
        mad_px = s * np.median(d[sup]) ** 2 / (f * bl) if n else np.inf
        if n < smin or tilt > cfg.stereo_tilt_deg or mad_px > MAD_PX_MAX:
            continue
        zl = float(p[0] * P[..., 0][sup].mean() + p[1] * P[..., 1][sup].mean() + p[2])
        dup = [o for o in out if abs(zl - o["z"]) < BIN]
        if dup:
            if n <= dup[0]["support"]:
                continue
            out.remove(dup[0])
        out.append(dict(z=zl, tilt_deg=tilt, mad=float(s), support=n, mask=sup))
    return sorted(out, key=lambda o: o["z"])


def pick_floor(surf, cam_z, band):
    """Lowest surface inside the camera-height band; else the lowest overall (flagged by the caller)."""
    inb = [o for o in surf if band[0] <= cam_z - o["z"] <= band[1]]
    return (inb[0], True) if inb else (surf[0], False)


def split_half(P, sup, B=32):
    v, u = np.nonzero(sup)
    par = ((u // B) + (v // B)) % 2 == 0
    X, Y, Z = (P[..., i][sup] for i in range(3))
    zs = []
    for h in (par, ~par):
        if h.sum() < 50:
            return float("nan")
        p = huber_plane(X[h], Y[h], Z[h])[0]
        zs.append(p[0] * X.mean() + p[1] * Y.mean() + p[2])
    return abs(zs[0] - zs[1]) / 2


def pick_pairs(cam, idx, cfg):
    """Up to cfg.stereo_pairs disjoint frame pairs with admissible baseline, smallest first."""
    idx = list(idx)
    cands = []
    for a, i in enumerate(idx):
        for j in idx[a + 1 :]:
            b = cam.pos[j] - cam.pos[i]
            n = np.linalg.norm(b)
            if not cfg.stereo_b_range[0] <= n <= cfg.stereo_b_range[1]:
                continue
            if math.degrees(math.acos(min(1, abs(b[2]) / n))) > VERT_MIN_DEG:
                cands.append((n, i, j))
    used: set = set()
    pairs = []
    for n, i, j in sorted(cands):
        if i in used or j in used:
            continue
        pairs.append((i, j))
        used |= {i, j}
        if len(pairs) == cfg.stereo_pairs:
            break
    return pairs


def _no_surface(flags, cam_z, bl):
    return dict(
        z=np.nan,
        sigma=np.nan,
        n_support=0,
        surfaces=[],
        flags=flags,
        cam_z=cam_z,
        b_m=bl,
        points=None,
        qc=None,
    )


def stereo_pair(imgs, cam, i, j, env, cfg, disparity):
    """One rectified pair -> dict(z, sigma, n_support, surfaces, flags, ..., points, qc)."""
    S = imgs[i].shape[0]
    kb4 = kb4_scaled(cam.kb4, S / cam.image_size)
    dev = cfg.resolved_device()
    view, bl = rectified_pair(cam.pos[i], cam.pos[j], cfg.view_size)
    f = view.fx
    rgb, valid = render(np.stack([imgs[i], imgs[j]]), kb4, cam.R[[i, j]], view, dev)
    sv = scene_valid(valid, env, kb4, cam.R[[i, j]], view, dev)
    n_disp = int(16 * math.ceil(1.1 * f * bl / H_DISP_MIN / 16))
    dL, ok, ur = lr_consistent(disparity, rgb[0], rgb[1], n_disp)
    ok &= sv[0] & sv[1][np.arange(view.h)[:, None], ur]
    P = lift(view, dL, cam.pos[i], bl)
    n_sv = int(sv[0].sum())
    lr_frac = ok.sum() / max(n_sv, 1)
    surf = surfaces(P, ok, n_sv, dL, f, bl, cfg)
    cam_z = float(cam.pos[i, 2])
    flags = []
    if lr_frac < LOW_TEX:
        flags.append("low_texture")
    if not surf:
        flags.append("no_floor")
        return _no_surface(flags, cam_z, bl)
    fl, plausible = pick_floor(surf, cam_z, cfg.cam_height_band)
    h = cam_z - fl["z"]
    sup = fl["mask"]
    if not plausible:
        flags.append("implausible_height")
    sh = split_half(P, sup)  # nan on a compact support: the pose term alone remains
    sigma = math.hypot(sh if np.isfinite(sh) else 0.0, h * cfg.pose_sigma_m / bl)
    v, u = np.nonzero(sup)
    k = max(1, len(u) // 4000)
    pts = P[v[::k], u[::k]].astype(np.float32)
    return dict(
        z=fl["z"],
        sigma=sigma,
        n_support=fl["support"],
        surfaces=[(o["z"], o["support"] / n_sv) for o in surf],
        flags=flags,
        cam_z=cam_z,
        b_m=bl,
        lr_frac=float(lr_frac),
        tilt_deg=fl["tilt_deg"],
        points=pts,
        qc=dict(left=rgb[0], ok=ok, floor=sup),
    )


def stereo_window(imgs, cam, idx, env, cfg, disparity=sgbm):
    """imgs: dict frame_index -> rgb (S, S, 3); idx: window frames. z = median over pairs."""
    pairs = pick_pairs(cam, idx, cfg)
    cam_z = float(np.median(cam.pos[list(idx), 2]))
    if not pairs:
        return dict(
            z=np.nan,
            sigma=np.nan,
            n_support=0,
            flags=["no_pair"],
            planes=[],
            cam_z=cam_z,
            extra={},
            points=None,
            qc=None,
        )
    res = [stereo_pair(imgs, cam, i, j, env, cfg, disparity) for i, j in pairs]
    good = [r for r in res if not r["flags"]]
    pl: dict = {}
    for r in res:
        for z, s in r["surfaces"]:
            pl[round(z, 2)] = max(pl.get(round(z, 2), 0.0), round(s, 4))
    planes = sorted(pl.items())
    if not good:
        flags = sorted({f for r in res for f in r["flags"]})
        r = min(res, key=lambda r: r["z"] if np.isfinite(r["z"]) else np.inf)
        return dict(
            z=r["z"],
            sigma=r["sigma"],
            n_support=r["n_support"],
            flags=flags,
            planes=planes,
            cam_z=cam_z,
            extra=dict(n_pairs=len(pairs), n_good=0),
            points=None,
            qc=res[0]["qc"],
        )
    # no / sqrt(n_good): the pairs share the window's poses and floor patch,
    # so their errors are not independent
    z = float(np.median([r["z"] for r in good]))
    sigma = float(np.median([r["sigma"] for r in good]))
    pts = np.concatenate([r["points"] for r in good])
    extra = dict(
        n_pairs=len(pairs),
        n_good=len(good),
        b_m=float(np.median([r["b_m"] for r in good])),
        lr_frac=float(np.median([r["lr_frac"] for r in good])),
        tilt_deg=float(np.median([r["tilt_deg"] for r in good])),
        pair_spread_cm=float(
            100 * (max(r["z"] for r in good) - min(r["z"] for r in good))
        ),
    )
    return dict(
        z=z,
        sigma=sigma,
        n_support=int(np.median([r["n_support"] for r in good])),
        flags=[],
        planes=planes,
        cam_z=cam_z,
        extra=extra,
        points=pts,
        qc=good[0]["qc"],
    )

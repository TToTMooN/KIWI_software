"""Gravity-fixed plane sweep over the floor height (no learned model).

Hypothesise the floor z on a grid of camera heights; build one metric ground
grid (scene frame) over the window's footprint; project every cell into
every frame (pose -> kb4) and gather the grey fisheye frame directly. Per
cell the cost is the best-half (over frames) mean of 1 - ZNCC between the
patch of a frame and the patch of the median over the OTHER half of the
trajectory (so every comparison has a baseline of about half the window
span). Planes are peaks of the photo-consistent image fraction S(h) and
minima of the mean cost C(h); the floor is the deepest plane inside the
camera-height band with enough support. Refined on a 2 mm grid over the
plane's own cells; sigma = block bootstrap over cells (+) pose term.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

from kiwi.ground.camera import kb4_scaled
from kiwi.ground.fisheye import outside_cone, project_torch
from kiwi.ground.gripper import scene_valid
from kiwi.ground.views import rays_world, sampling_grid, window_view

CELL_M = 0.006  # ground cell [m]
BLUR_PX = 0.6  # anti-alias sigma in decoded pixels (0 = off)
CELL_PATCH = 11  # ZNCC patch [cells]
VAR_MIN = 4.0  # min patch variance [grey^2]
MIN_FRAMES = 6  # min textured frames per cell
CHUNK = 4  # max hypotheses per gather pass
FCHUNK = 5  # frames per gather launch
ELEM_MAX = 32e6  # frames x hypotheses x cells per pass (~3 GB of fp32 statistics)
CONTRAST_MAX, CMED_MIN = (
    0.5,
    0.5,
)  # per-cell consistency: cost < 0.5 x own median (median >= 0.5)
C_PROM = 0.04  # C(h) dip prominence, relative to the median cost
H0, OFFN_MAX = 1.3, 60.0  # footprint height [m] and off-nadir cap [deg]
FINE_HALF = 0.03  # fine sweep half-width [m]
PEAK_SEP, RECENTRE = (
    0.06,
    0.08,
)  # min plane separation [m]; recentre the pick within +- this [m]
BLOCK, N_BOOT = 16, 64  # bootstrap block [cells] and resamples


def prep_frames(rgb, bad, dev):
    """(F, S, S, 3) uint8 -> (F, 1, S, S) float32 grey on dev, anti-aliased, NaN on bad pixels."""
    lum = torch.tensor([0.299, 0.587, 0.114], device=dev)
    g = (torch.from_numpy(np.ascontiguousarray(rgb)).to(dev).float() @ lum)[:, None]
    if BLUR_PX > 0:
        x = torch.arange(-3, 4, device=dev, dtype=torch.float32)
        k = torch.exp(-(x**2) / (2 * BLUR_PX**2))
        k /= k.sum()
        g = F.pad(g, (3, 3, 3, 3), mode="replicate")
        g = F.conv2d(F.conv2d(g, k.view(1, 1, 1, 7)), k.view(1, 1, 7, 1))
    g[:, :, F.max_pool2d(bad[None, None].float(), 5, 1, 2)[0, 0] > 0] = torch.nan
    return g


def ground_grid(cam, idx, view, env, kb4, size, dev):
    """Metric XY grid over the union footprint (at height H0) of the window's scene-valid rays."""
    R = cam.R[idx]
    c = cam.pos[idx]
    _, valid, _, _ = sampling_grid(view, kb4, R, size, dev)
    sv = scene_valid(valid.cpu().numpy(), env, kb4, R, view, dev)[:, ::4, ::4]
    r = rays_world(view)[::4, ::4]
    down = r[..., 2] < -math.cos(math.radians(OFFN_MAX))
    P = np.concatenate(
        [
            c[i, :2] + r[m, :2] * (-H0 / r[m, 2])[:, None]
            for i, m in enumerate(sv & down)
        ]
    )
    lo, hi = P.min(0), P.max(0)
    return np.arange(lo[0], hi[0], CELL_M), np.arange(lo[1], hi[1], CELL_M)


def _gather(nf, kb4, R, c, X, Y, z, mid=None):
    """Sample every frame on the ground grid at floor-z hypotheses z (Kc, 1, 1):
    (F, Kc, Ny, Nx) fp16, NaN where invalid; plus the pixel area of every cell in
    frame ``mid`` (the image-fraction weight)."""
    dev = nf.device
    Fn, Ny, Nx = nf.shape[0], len(Y), len(X)
    W = nf.shape[-1]
    Kc = z.shape[0]
    Rt = torch.as_tensor(R, device=dev, dtype=torch.float32)
    ct = torch.as_tensor(c, device=dev, dtype=torch.float32)
    gx, gy = torch.meshgrid(
        torch.as_tensor(X, device=dev, dtype=torch.float32),
        torch.as_tensor(Y, device=dev, dtype=torch.float32),
        indexing="xy",
    )
    obs = torch.empty(Fn, Kc, Ny, Nx, device=dev, dtype=torch.float16)
    area = None
    for i0 in range(0, Fn, FCHUNK):
        i1 = min(i0 + FCHUNK, Fn)
        Ri = Rt[i0:i1, :, :, None, None, None]
        ci = ct[i0:i1, :, None, None, None]
        dx, dy, dz = (
            gx[None, None] - ci[:, 0],
            gy[None, None] - ci[:, 1],
            z[None] - ci[:, 2],
        )
        xc = Ri[:, 0, 0] * dx + Ri[:, 1, 0] * dy + Ri[:, 2, 0] * dz  # cam = R^T (X - c)
        yc = Ri[:, 0, 1] * dx + Ri[:, 1, 1] * dy + Ri[:, 2, 1] * dz
        zc = Ri[:, 0, 2] * dx + Ri[:, 1, 2] * dy + Ri[:, 2, 2] * dz
        px, py, ok = project_torch(kb4, xc, yc, zc, W)
        f = i1 - i0
        grid = torch.stack([2 * px / (W - 1) - 1, 2 * py / (W - 1) - 1], -1)
        grid = grid.view(f, Kc * Ny, Nx, 2)
        o = F.grid_sample(
            nf[i0:i1], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        o = o.view(f, Kc, Ny, Nx)
        obs[i0:i1] = torch.where(ok & ~torch.isnan(o), o, torch.nan).half()
        if mid is not None and i0 <= mid < i1:
            (dpx_y, dpx_x) = torch.gradient(px[mid - i0], dim=(1, 2))
            (dpy_y, dpy_x) = torch.gradient(py[mid - i0], dim=(1, 2))
            area = (dpx_x * dpy_y - dpx_y * dpy_x).abs().clamp(0, 100).half()
    return obs, area


def split_halves(c, dev):
    """Frames on the far side of the principal axis of the horizontal positions."""
    xy = c[:, :2] - c[:, :2].mean(0)
    e = np.linalg.svd(xy, full_matrices=False)[2][0]
    p = xy @ e
    return torch.as_tensor(p > np.median(p), device=dev)


def costs(nf, kb4, R, c, X, Y, zs, grp, mid):
    """Cost volume (K, Ny, Nx) fp16 (NaN below MIN_FRAMES textured frames) and the area weights."""
    dev = nf.device
    Ny, Nx, K = len(Y), len(X), len(zs)
    cost = torch.empty(K, Ny, Nx, device=dev, dtype=torch.float16)
    wgt = torch.empty(K, Ny, Nx, device=dev, dtype=torch.float16)
    chunk = max(1, min(CHUNK, int(ELEM_MAX // (nf.shape[0] * Ny * Nx))))
    P, h = CELL_PATCH, CELL_PATCH // 2

    def pool(
        x,
    ):  # separable box mean (11 + 11 taps), zero padding counted like avg_pool2d
        y = F.avg_pool2d(x.reshape(-1, 1, Ny, Nx), (P, 1), 1, (h, 0))
        return F.avg_pool2d(y, (1, P), 1, (0, h)).view(x.shape)

    for k0 in range(0, K, chunk):
        z = torch.as_tensor(zs[k0 : k0 + chunk], device=dev, dtype=torch.float32)[
            :, None, None
        ]
        Kc = z.shape[0]
        obs, area = _gather(nf, kb4, R, c, X, Y, z, mid)
        rA = torch.nanmedian(obs[grp], 0).values
        rB = torch.nanmedian(obs[~grp], 0).values
        ref = torch.where(grp[:, None, None, None], rB[None], rA[None]).float()
        M = ~torch.isnan(obs) & ~torch.isnan(ref)
        ref = ref.nan_to_num(0)
        Mf = M.float()
        Of = torch.where(M, obs, 0).float()
        cnt = pool(Mf).clamp_min(1 / CELL_PATCH**2)
        mo = pool(Of) / cnt
        mr = pool(ref * Mf) / cnt
        vo = pool(Of * Of) / cnt - mo * mo
        vr = pool(ref * ref * Mf) / cnt - mr * mr
        cov = pool(Of * ref) / cnt - mo * mr
        ok = M & (cnt >= 0.5) & (vo >= VAR_MIN) & (vr >= VAR_MIN)
        Dp = torch.where(
            ok, 1 - cov / (vo * vr).clamp_min(VAR_MIN**2).sqrt(), torch.inf
        )
        n = ok.sum(0)
        cs = Dp.sort(0).values.cumsum(0)
        kb = (n // 2).clamp_min(3).long()  # best half, at least 3 frames
        cst = cs.gather(0, (kb - 1)[None]).squeeze(0) / kb
        cost[k0 : k0 + Kc] = torch.where(n >= MIN_FRAMES, cst, torch.nan).half()
        wgt[k0 : k0 + Kc] = torch.where(n >= MIN_FRAMES, area, 0)
    return cost, wgt


def planes(cost, wgt, support_min, dist=3):
    """Area-weighted mean cost C(h), consistent image fraction S(h), planes
    [(k, support excess)] sorted by support, and the per-cell consistency volume."""
    cst = cost.float().flatten(1)
    valid = ~torch.isnan(cst)
    wv = wgt.float().flatten(1)
    ws = wv.sum(1).clamp_min(1e-6)
    C = ((wv * cst.nan_to_num(0)).sum(1) / ws).cpu().numpy()
    cmed = torch.nanmedian(cst, 0).values
    cons = valid & (cst < CONTRAST_MAX * cmed) & (cmed >= CMED_MIN)[None]
    S = ((wv * cons).sum(1) / ws).cpu().numpy()
    base = float(np.median(S))
    ks, _ = find_peaks(S, prominence=support_min, distance=dist)
    pk = [(int(k), float(S[k] - base)) for k in ks if S[k] - base >= support_min]
    kc, _ = find_peaks(-C, prominence=C_PROM * float(np.nanmedian(C)), distance=dist)
    pk += [
        (int(k), float(S[k] - base))
        for k in kc
        if all(abs(k - q) > dist for q, _ in pk)
    ]
    return C, S, sorted(pk, key=lambda q: -q[1]), cons.view(cost.shape)


def parabola(H, C):
    j = int(np.nanargmin(C))
    j = min(max(j, 1), len(C) - 2)
    y0, y1, y2 = C[j - 1], C[j], C[j + 1]
    dh = H[1] - H[0]
    a = (y0 - 2 * y1 + y2) / (2 * dh**2)
    d = (y0 - y2) / (4 * a * dh) if a > 0 else 0.0
    return float(H[j] + np.clip(d, -dh, dh))


def _empty(flags, cands, zc, extra, qc):
    return dict(
        z=np.nan,
        sigma=np.nan,
        n_support=0,
        flags=sorted(set(flags) | {"no_floor"}),
        planes=[(zc - H, s) for H, s in cands],
        cam_z=zc,
        extra=extra,
        points=None,
        qc=qc,
    )


def sweep_window(rgb, cam, idx, env, cfg):
    """rgb: (F, S, S, 3) uint8 decoded frames of the window, idx their frame indices.

    Returns dict(z, sigma, n_support, flags, planes=[(z, support)], cam_z, extra, points, qc).
    """
    dev = cfg.resolved_device()
    S = rgb.shape[1]
    kb4 = kb4_scaled(cam.kb4, S / cam.image_size)
    idx = np.asarray(idx)
    c, R = cam.pos[idx], cam.R[idx]
    zc = float(np.median(c[:, 2]))
    mid = len(idx) // 2
    dh = cfg.sweep_dh_coarse
    HS = np.round(np.arange(cfg.sweep_h_range[0], cfg.sweep_h_range[1] + 1e-9, dh), 4)
    nb = max(1, round(RECENTRE / dh))
    view, pitch = window_view(cam, idx, cfg)
    band = cfg.cam_height_band
    bad = torch.from_numpy(env | outside_cone(kb4, S)).to(dev)
    xy = c[:, :2] - c[:, :2].mean(0)
    b_eff = 2 * float(np.sqrt((xy**2).sum(1).mean()))
    with torch.no_grad():
        X, Y = ground_grid(cam, idx, view, env, kb4, S, dev)
        nf = prep_frames(rgb, bad, dev)
        grp = split_halves(c, dev)
        cost, wgt = costs(nf, kb4, R, c, X, Y, zc - HS, grp, mid)
        C, Sh, pk, cons = planes(
            cost, wgt, cfg.sweep_support_min, max(1, round(PEAK_SEP / dh))
        )
        cands = [(float(HS[k]), s) for k, s in pk]
        inband = [q for q in cands if band[0] <= q[0] <= band[1]]
        flags = []
        if not cands:
            flags.append("no_floor")
            Hc, sc = float(HS[np.nanargmin(C)]), 0.0
        elif not inband:
            flags.append("implausible_height")
            Hc, sc = cands[0]
        else:
            strong = [q for q in inband if q[1] >= cfg.sweep_support_min]
            Hc, sc = max(strong) if strong else max(inband)  # deepest plausible plane
            if not strong:
                flags.append("low_support")
            if any(s >= sc and abs(H - Hc) > PEAK_SEP for H, s in inband):
                flags.append("multi_minima")
        kc = int(np.argmin(np.abs(HS - Hc)))
        mask = cons[kc]
        # erode by the patch half-width unless that empties a thin region
        er = (
            -F.max_pool2d(-mask[None, None].float(), CELL_PATCH, 1, CELL_PATCH // 2)[
                0, 0
            ]
            > 0.5
        )
        mask = er if er.any() else mask
        extra = dict(
            pitch_deg=pitch,
            n_frames=len(idx),
            b_eff_m=b_eff,
            support=sc,
            n_planes=len(cands),
        )
        qc = dict(HS=HS, C=C, S=Sh, band=band, frame=rgb[mid][::4, ::4].copy())
        if not mask.any():  # nothing photo-consistent at the pick: no level
            return _empty(flags, cands, zc, extra, qc)
        wm = torch.where(mask[None] & ~torch.isnan(cost), wgt.float(), 0).flatten(1)
        Cm = (wm * cost.float().nan_to_num(0).flatten(1)).sum(1) / wm.sum(1).clamp_min(
            1e-6
        )
        Cm = Cm.cpu().numpy()
        lo, hi = max(kc - nb, 0), min(kc + nb + 1, len(HS))
        kc = lo + int(np.nanargmin(Cm[lo:hi]))
        Hf = np.round(
            HS[kc] + np.arange(-FINE_HALF, FINE_HALF + 1e-9, cfg.sweep_dh_fine), 4
        )
        cf, wf = costs(nf, kb4, R, c, X, Y, zc - Hf, grp, mid)
        m = mask[None] & ~torch.isnan(cf)
        mf = torch.where(m, wf.float(), 0)
        cfz = mf * cf.float().nan_to_num(0)
        Cf = (
            (cfz.flatten(1).sum(1) / mf.flatten(1).sum(1).clamp_min(1e-6)).cpu().numpy()
        )
        Hstar = parabola(Hf, Cf)
        py_, px_ = (-cf.shape[1]) % BLOCK, (-cf.shape[2]) % BLOCK
        bs = F.avg_pool2d(F.pad(cfz, (0, px_, 0, py_)), BLOCK).flatten(1)
        bn = F.avg_pool2d(F.pad(mf, (0, px_, 0, py_)), BLOCK).flatten(1)
        use = bn.sum(0) > 0
        bs, bn = bs[:, use], bn[:, use]
        nblk = int(use.sum())
        if nblk == 0:  # the picked cells gather nothing at the fine heights
            return _empty(flags, cands, zc, extra, qc)
        gen = torch.Generator(device=dev).manual_seed(0)  # reproducible sigma
        ridx = torch.randint(0, nblk, (N_BOOT, nblk), device=dev, generator=gen)
        Cb = (bs[:, ridx].sum(-1) / bn[:, ridx].sum(-1).clamp_min(1e-6)).cpu().numpy()
        sig_boot = float(np.std([parabola(Hf, Cb[:, b]) for b in range(N_BOOT)]))
        mask_np = mask.cpu().numpy()
        mask_frac = float(wm[kc].sum() / wgt[kc].float().sum().clamp_min(1e-6))
    sig_pose = Hstar * cfg.pose_sigma_m / b_eff
    fine_depth = 1 - float(np.nanmin(Cf) / np.nanmax(Cf))
    if sig_boot > 0.02 or fine_depth < 0.05:
        flags.append("low_texture")
    z = zc - Hstar
    n_sup = int(mask_np.sum())
    iy, ix = np.nonzero(mask_np)
    k = max(1, n_sup // 4000)
    points = np.stack([X[ix[::k]], Y[iy[::k]], np.full(len(ix[::k]), z)], 1).astype(
        np.float32
    )
    extra.update(
        H_m=Hstar,
        c_dip=1 - float(C[kc]) / float(np.nanmedian(C)),
        fine_depth=fine_depth,
        sigma_boot=sig_boot,
        sigma_pose=sig_pose,
        grid=(len(Y), len(X)),
        mask_frac=mask_frac,
    )
    return dict(
        z=z,
        sigma=float(math.hypot(sig_boot, sig_pose)),
        n_support=n_sup,
        flags=flags,
        planes=[(zc - H, s) for H, s in cands],
        cam_z=zc,
        extra=extra,
        points=points,
        qc=dict(qc, Hf=Hf, Cf=Cf),
    )

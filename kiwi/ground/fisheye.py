"""Kannala-Brandt (kb4) fisheye: d(theta) = theta (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)."""

from __future__ import annotations

import math

import numpy as np

THETA_MAX_DEG = 85.0
"""Rays beyond the fitted cone of the kb4 calibration are never sampled (KIWI fits to 85 deg)."""


def _distort(kb4, th):
    t2 = th * th
    return th * (
        1 + t2 * (kb4["k1"] + t2 * (kb4["k2"] + t2 * (kb4["k3"] + t2 * kb4["k4"])))
    )


def project(kb4, pts_cam, theta_max_deg=THETA_MAX_DEG, size=None):
    """(N, 3) camera-frame points -> (N, 2) pixels and a validity mask."""
    p = np.asarray(pts_cam, float)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.hypot(x, y)
    th = np.arctan2(r, z)
    d = _distort(kb4, th)
    s = np.where(r > 1e-12, d / np.maximum(r, 1e-12), 0.0)
    u = kb4["fx"] * s * x + kb4["cx"]
    v = kb4["fy"] * s * y + kb4["cy"]
    ok = th <= math.radians(theta_max_deg)
    if size:
        ok &= (u >= 0) & (u <= size - 1) & (v >= 0) & (v <= size - 1)
    return np.stack([u, v], 1), ok


def unproject(kb4, uv, theta_max_deg=THETA_MAX_DEG, iters=12):
    """(N, 2) pixels -> unit rays (N, 3), valid inside the cone (Newton inversion)."""
    uv = np.asarray(uv, float)
    mx = (uv[:, 0] - kb4["cx"]) / kb4["fx"]
    my = (uv[:, 1] - kb4["cy"]) / kb4["fy"]
    d = np.hypot(mx, my)
    th = d.copy()
    for _ in range(iters):
        t2 = th * th
        f = _distort(kb4, th) - d
        fp = (
            1
            + 3 * kb4["k1"] * t2
            + 5 * kb4["k2"] * t2**2
            + 7 * kb4["k3"] * t2**3
            + 9 * kb4["k4"] * t2**4
        )
        th = th - f / fp
    ok = (th >= 0) & (th <= math.radians(theta_max_deg)) & np.isfinite(th)
    c = np.where(d > 1e-12, 1 / np.maximum(d, 1e-12), 0.0)
    return np.stack([np.sin(th) * mx * c, np.sin(th) * my * c, np.cos(th)], 1), ok


def project_torch(kb4, x, y, z, size, theta_max_deg=THETA_MAX_DEG):
    """Torch version for the samplers: camera-frame components -> (px, py, valid)."""
    import torch

    r = torch.hypot(x, y)
    th = torch.atan2(r, z)
    d = _distort(kb4, th)
    s = d / r.clamp_min(1e-9)
    px = kb4["fx"] * s * x + kb4["cx"]
    py = kb4["fy"] * s * y + kb4["cy"]
    ok = (th <= math.radians(theta_max_deg)) & (px >= 0) & (px <= size - 1)
    ok &= (py >= 0) & (py <= size - 1)
    return px, py, ok


def outside_cone(kb4, size, theta_max_deg=THETA_MAX_DEG):
    """Bool (size, size): pixels beyond the fitted cone (never sampled)."""
    v, u = np.mgrid[0:size, 0:size]
    r = np.hypot((u - kb4["cx"]) / kb4["fx"], (v - kb4["cy"]) / kb4["fy"])
    return r > _distort(kb4, math.radians(theta_max_deg))

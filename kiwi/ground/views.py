"""GPU virtual views over kb4 fisheye frames (torch grid_sample).

Gravity-aligned pitched views for the sweep footprint and QC, and exactly
rectified intra-camera pairs for the stereo channel. Conventions:
R_world_from_x has the camera axes as COLUMNS; camera frames are OpenCV (x
right, y down, z forward); the world frame is z-up. A view is (K,
R_world_from_virtual) placed at the camera centre of the frame it renders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from kiwi.ground.fisheye import THETA_MAX_DEG, project_torch

VIEW_FOV_DEG = 90.0
# pitch = clip(off_nadir - offset, 0, cap): the offset is the off-nadir angle
# at which the corner of a square VIEW_FOV_DEG view reaches the THETA_MAX_DEG
# cone (85 - atan(sqrt(2) tan 45) = 30.3 deg)
FOLLOW_OFFSET_DEG = THETA_MAX_DEG - math.degrees(
    math.atan(math.sqrt(2) * math.tan(math.radians(VIEW_FOV_DEG / 2)))
)


@dataclass(frozen=True)
class View:
    fx: float
    fy: float
    cx: float
    cy: float
    w: int
    h: int
    R: np.ndarray
    """(3, 3) R_world_from_virtual."""


def view_from_fov(fov_deg, size, R) -> View:
    f = (size / 2) / math.tan(math.radians(fov_deg) / 2)
    c = (size - 1) / 2
    return View(f, f, c, c, size, size, np.asarray(R, float))


def R_world_from_axis(axis, up_hint) -> np.ndarray:
    """Camera looking along ``axis``, image-up along ``up_hint`` projected orthogonal to it."""
    d = np.asarray(axis, float)
    d = d / np.linalg.norm(d)
    up = np.asarray(up_hint, float)
    up = up - (up @ d) * d
    up = up / np.linalg.norm(up)
    yv = -up
    xv = np.cross(yv, d)
    return np.stack([xv, yv, d], 1)


def gravity_view(size, azimuth, pitch_deg) -> View:
    """Axis pitched ``pitch_deg`` from nadir toward ``azimuth``; image-up = the horizon side."""
    p = math.radians(pitch_deg)
    axis = (
        math.sin(p) * math.cos(azimuth),
        math.sin(p) * math.sin(azimuth),
        -math.cos(p),
    )
    up = (math.cos(azimuth), math.sin(azimuth), 0.0)
    return view_from_fov(VIEW_FOV_DEG, size, R_world_from_axis(axis, up))


def window_view(cam, frames, cfg) -> tuple[View, float]:
    """One rigid view per window: circular-mean azimuth, pitch following the median off-nadir."""
    az = cam.azimuth[frames]
    a = math.atan2(np.mean(np.sin(az)), np.mean(np.cos(az)))
    p = float(
        np.clip(
            np.median(cam.off_nadir_deg[frames]) - FOLLOW_OFFSET_DEG,
            0.0,
            cfg.pitch_cap_deg,
        )
    )
    return gravity_view(cfg.view_size, a, p), p


def rectified_pair(pos_i, pos_j, size) -> tuple[View, float]:
    """Rectified view for two frames of ONE camera: x along the baseline, z =
    gravity-down orthogonalised, same K. Frame i is the left image; a point at
    depth Z along z has disparity d = f |b| / Z (x_i - x_j)."""
    b = np.asarray(pos_j, float) - np.asarray(pos_i, float)
    x = b / np.linalg.norm(b)
    g = np.array([0.0, 0.0, -1.0])
    z = g - (g @ x) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return view_from_fov(VIEW_FOV_DEG, size, np.stack([x, y, z], 1)), float(
        np.linalg.norm(b)
    )


def _batch(a, nd):
    a = np.ascontiguousarray(a)
    t = torch.from_numpy(
        a if a.flags.writeable else a.copy()
    )  # decoded frames are read-only
    return t[None] if t.ndim == nd else t


def sampling_grid(view, kb4, R_wf, size, dev):
    """Fisheye pixel of every virtual pixel per pose: grid (N, h, w, 2) for grid_sample, valid, px, py."""
    R_wf = _batch(R_wf, 2).to(dev, torch.float64)
    u = torch.arange(view.w, device=dev, dtype=torch.float64)
    v = torch.arange(view.h, device=dev, dtype=torch.float64)
    gv, gu = torch.meshgrid(v, u, indexing="ij")
    d_v = torch.stack(
        [(gu - view.cx) / view.fx, (gv - view.cy) / view.fy, torch.ones_like(gu)], -1
    )
    d_w = d_v @ torch.as_tensor(view.R, device=dev, dtype=torch.float64).T
    d_f = torch.einsum("hwk,nkj->nhwj", d_w, R_wf)  # R_wf^T d_w
    px, py, valid = project_torch(kb4, d_f[..., 0], d_f[..., 1], d_f[..., 2], size)
    grid = torch.stack([2 * px / (size - 1) - 1, 2 * py / (size - 1) - 1], -1)
    return grid.float(), valid, px, py


def render(img, kb4, R_wf, view, dev):
    """img uint8 (H, W, 3) or (N, H, W, 3) with R_wf (3, 3) or (N, 3, 3); kb4 at the image size.
    Returns rgb uint8 (..., h, w, 3) and valid bool (..., h, w) as numpy."""
    x = _batch(img, 3).to(dev).permute(0, 3, 1, 2).float()
    W = x.shape[-1]
    grid, valid, _, _ = sampling_grid(view, kb4, R_wf, W, dev)
    out = F.grid_sample(
        x, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    out = out.permute(0, 2, 3, 1)
    out = (
        torch.where(valid[..., None], out, torch.zeros_like(out)).round().clamp(0, 255)
    )
    rgb, valid = out.to(torch.uint8).cpu().numpy(), valid.cpu().numpy()
    return (rgb, valid) if img.ndim == 4 else (rgb[0], valid[0])


def render_mask(mask, kb4, R_wf, view, dev):
    """Nearest-neighbour resampling of a fisheye-space bool mask into the view."""
    m = _batch(np.asarray(mask) > 0, 2).to(dev).float()[:, None]
    size = m.shape[-1]
    grid, valid, _, _ = sampling_grid(view, kb4, R_wf, size, dev)
    m = m.expand(grid.shape[0], -1, -1, -1)
    out = F.grid_sample(
        m, grid, mode="nearest", padding_mode="zeros", align_corners=True
    )
    out = (out[:, 0] > 0.5) & valid
    return out.cpu().numpy() if np.ndim(R_wf) == 3 else out[0].cpu().numpy()


def rays_world(view) -> np.ndarray:
    """Unit rays of every virtual pixel in the world frame, (h, w, 3)."""
    u, v = np.meshgrid(np.arange(view.w), np.arange(view.h))
    d = np.stack(
        [(u - view.cx) / view.fx, (v - view.cy) / view.fy, np.ones_like(u, float)], -1
    )
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d @ view.R.T

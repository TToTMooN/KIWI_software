"""Static rig envelope per camera: pixels whose grey barely changes over the video are the gripper."""

from __future__ import annotations

import cv2
import numpy as np

from kiwi.ground.fisheye import outside_cone
from kiwi.ground.views import render_mask

STD_MAX = 35.0  # grey levels: temporal std below this = rigid to the camera
MIN_FRAC = 0.005  # components smaller than this fraction of the lens cone are noise


def envelope(frames_rgb, kb4):
    """frames_rgb: iterable of (S, S, 3) uint8 frames sampled across the video.

    Returns (bool (S, S), n_frames, cone_fraction): static pixels inside the
    cone, morphologically closed, components larger than MIN_FRAC of the cone.
    The temporal std is accumulated frame by frame.
    """
    acc = acc2 = None
    nf = 0
    for f in frames_rgb:
        g = f.astype(np.float32).mean(-1)
        if acc is None:
            acc = np.zeros(g.shape, np.float64)
            acc2 = np.zeros(g.shape, np.float64)
        acc += g
        acc2 += g * g
        nf += 1
    if nf == 0:
        raise RuntimeError("no frames decoded for the rig envelope")
    S = acc.shape[0]
    cone = ~outside_cone(kb4, S)
    std = np.sqrt(np.maximum(acc2 / nf - (acc / nf) ** 2, 0))
    static = (std < STD_MAX) & cone
    env = cv2.morphologyEx(
        static.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(env)
    keep = np.zeros(env.shape, bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] > MIN_FRAC * cone.sum():
            keep |= lab == k
    return keep, nf, float(keep.sum() / cone.sum())


def scene_valid(valid, env, kb4, R_wf, view, dev):
    """valid & ~envelope rendered into the view, for one pose or a batch."""
    return valid & ~render_mask(env, kb4, R_wf, view, dev)

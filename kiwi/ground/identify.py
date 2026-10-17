"""Combine the two channels of one window: agreement gate, plausibility band, flags."""

from __future__ import annotations

import math

import numpy as np

SOFT = {"low_support", "low_texture"}
"""Channel flags that still allow the value to CONFIRM the other channel."""


def identify(sw, st, w, cfg) -> dict:
    """sw, st: channel dicts (z, sigma, flags, ...). Returns the window record."""
    cam_z = sw["cam_z"]

    def hard(c):
        return np.isfinite(c["z"]) and not c["flags"]

    def soft(c):
        return np.isfinite(c["z"]) and set(c["flags"]) <= SOFT

    def agree(a, b):
        return abs(a["z"] - b["z"]) <= cfg.agree_cm / 100

    rec = dict(
        t0=w["t0"],
        t1=w["t1"],
        cam_z=cam_z,
        pitch_deg=sw["extra"].get("pitch_deg"),
        flags=[],
        channel=None,
        sweep_z=sw["z"],
        sweep_sigma=sw["sigma"],
        sweep_flags=list(sw["flags"]),
        stereo_z=st["z"],
        stereo_sigma=st["sigma"],
        stereo_flags=list(st["flags"]),
        planes_sweep=sw["planes"],
        planes_stereo=st["planes"],
        extra=dict(sweep=sw["extra"], stereo=st["extra"]),
    )
    usable = (hard(sw) or soft(sw)) and (hard(st) or soft(st))
    if usable and agree(sw, st) and (hard(sw) or hard(st)):
        z = 0.5 * (sw["z"] + st["z"])
        sigma = 0.5 * math.hypot(sw["sigma"], st["sigma"])
        ch = "both"
    elif hard(sw) and hard(st):
        z, sigma, ch = st["z"], st["sigma"], "stereo"
        rec["flags"].append("channels_disagree")
    elif hard(st):
        z, sigma, ch = st["z"], st["sigma"], "stereo"
    elif hard(sw):
        z, sigma, ch = sw["z"], sw["sigma"], "sweep"
    else:
        f = (set(sw["flags"]) | set(st["flags"])) - {"no_pair"}
        rec["flags"] = sorted(f or {"no_floor"})
        rec.update(z=np.nan, sigma=np.nan, n_support=0)
        return rec
    h = cam_z - z
    if not cfg.cam_height_band[0] <= h <= cfg.cam_height_band[1]:
        rec["flags"].append("implausible_height")
    rec.update(
        z=float(z),
        sigma=float(sigma),
        channel=ch,
        n_support=int(max(sw["n_support"], st["n_support"])),
        cam_height=float(h),
    )
    return rec

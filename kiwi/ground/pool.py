"""Pool per-window floor estimates: temporal gate, weighted median per camera, cross-camera pairs, verdict."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields

import numpy as np

PAIR_TOL_S = 0.5
VERSION = "ground_v1"
DROP = ("extra", "qc", "points")
"""Per-window payloads that stay in memory (images, lifted points, channel internals)."""


def weighted_median(z, sigma) -> float:
    """Weights 1/sigma^2 clipped to [0.1, 10] x their median."""
    z = np.asarray(z, float)
    w = 1 / np.asarray(sigma, float) ** 2
    ok = np.isfinite(w)
    z, w = z[ok], w[ok]
    w = np.clip(w, 0.1 * np.median(w), 10 * np.median(w))
    o = np.argsort(z)
    c = np.cumsum(w[o])
    return float(z[o][np.searchsorted(c, 0.5 * c[-1])])


def mad(z) -> float:
    z = np.asarray(z, float)
    return (
        float(1.4826 * np.median(np.abs(z - np.median(z)))) if len(z) else float("nan")
    )


def temporal_gate(wins, cfg, half=10):
    """Flag windows farther than min(outlier_mad x MAD (floor 1 cm), outlier_cap_cm) from the
    running median of their +-half accepted neighbours. A run of more than
    ``half`` consecutive wrong windows owns its own median and passes: ``pool`` reports that."""
    acc = [i for i, w in enumerate(wins) if not w["flags"]]
    z = np.array([wins[i]["z"] for i in acc])
    for k, i in enumerate(acc):
        nb = z[max(0, k - half) : k + half + 1]
        if len(nb) < 5:
            continue
        thr = min(cfg.outlier_mad * max(mad(nb), 0.01), cfg.outlier_cap_cm / 100)
        if abs(z[k] - np.median(nb)) > thr:
            wins[i]["flags"].append("temporal_outlier")
    return wins


def clean(o):
    """JSON-safe: numpy -> python, non-finite floats -> None (bare NaN tokens are not JSON)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


@dataclass
class Result:
    floor_z_m: float
    sigma_m: float
    verdict: str
    reasons: list
    cameras: dict
    """name -> dict(z, mad_cm, n_used, n_total, n_gated, n_processed, rejected, sigma_m, channel_counts)."""
    cross: dict
    """gap_cm, pair_mean_cm, pair_sd_cm (robust), pair_sd_raw_cm, n_pairs, channel_offset_cm; {} for one camera."""
    windows: dict
    """name -> list of window records."""
    runtime_s: float = 0.0
    config: dict = field(default_factory=dict)
    version: str = VERSION
    timing: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "windows"}
        d["windows"] = {
            n: [{k: v for k, v in w.items() if k not in DROP} for w in ws]
            for n, ws in self.windows.items()
        }
        return clean(d)

    def to_json(self, path=None) -> str:
        s = json.dumps(self.to_dict(), indent=1, allow_nan=False)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(s)
        return s


def pool(windows_by_cam, cfg, runtime_s=0.0) -> Result:
    cams = {}
    used = {}
    for name, wins in windows_by_cam.items():
        wins = temporal_gate(wins, cfg)
        u = [w for w in wins if not w["flags"]]
        used[name] = u
        rej: dict = {}
        for w in wins:
            for f in w["flags"]:
                rej[f] = rej.get(f, 0) + 1
        z = [w["z"] for w in u]
        s = [w["sigma"] for w in u]
        n_gated = len(wins) - rej.get(
            "pose_gate", 0
        )  # the camera moved and looked down
        cams[name] = dict(
            z=weighted_median(z, s) if u else float("nan"),
            mad_cm=100 * mad(z),
            n_used=len(u),
            n_total=len(wins),
            n_gated=n_gated,
            n_processed=n_gated
            - rej.get("skipped", 0),  # ... of which the estimators ran
            sigma_m=mad(z) / np.sqrt(max(len(u), 1)) if u else float("nan"),
            rejected=rej,
            channel_counts={
                c: sum(w["channel"] == c for w in u)
                for c in ("both", "stereo", "sweep")
            },
        )
    allz = [w["z"] for u in used.values() for w in u]
    alls = [w["sigma"] for u in used.values() for w in u]
    z = weighted_median(allz, alls) if allz else float("nan")
    sig_stat = mad(allz) / np.sqrt(max(len(allz), 1)) if allz else float("nan")
    # half the median sweep-stereo offset enters the pooled sigma
    both = [w for u in used.values() for w in u if w["channel"] == "both"]
    ch_off = (
        float(np.median([w["sweep_z"] - w["stereo_z"] for w in both])) if both else 0.0
    )
    cross: dict = {}
    names = list(cams)
    if len(names) == 2 and all(cams[n]["n_used"] for n in names):
        a, b = used[names[0]], used[names[1]]
        tb = np.array([w["t0"] for w in b])
        d = []
        for w in a:
            j = int(np.argmin(np.abs(tb - w["t0"])))
            if abs(tb[j] - w["t0"]) <= PAIR_TOL_S:
                d.append(w["z"] - b[j]["z"])
        cross = dict(
            gap_cm=100 * (cams[names[0]]["z"] - cams[names[1]]["z"]),
            pair_mean_cm=100 * float(np.mean(d)) if d else float("nan"),
            pair_sd_cm=100 * mad(d) if len(d) > 1 else float("nan"),  # robust SD
            pair_sd_raw_cm=100 * float(np.std(d)) if len(d) > 1 else float("nan"),
            n_pairs=len(d),
            channel_offset_cm=100 * ch_off,
        )
    gap_term = abs(cross["gap_cm"]) / 200 if cross else 0.0
    sigma = float(
        np.hypot(
            np.hypot(sig_stat, gap_term), np.hypot(abs(ch_off) / 2, cfg.sigma_floor_m)
        )
    )
    reasons = []
    n_used = len(allz)
    n_proc = sum(c["n_processed"] for c in cams.values())
    frac = n_used / max(n_proc, 1)
    if frac < cfg.green_used_frac:
        reasons.append(
            f"only {frac:.0%} of the windows that passed the pose gate are usable"
        )
    low = [n for n, c in cams.items() if c["n_used"] < cfg.min_used_per_cam]
    if low:
        reasons.append(
            f"fewer than {cfg.min_used_per_cam} usable windows on {', '.join(low)}"
        )
    far = sum(abs(v - z) > cfg.outlier_cap_cm / 100 for v in allz)
    if allz and far > 0.05 * n_used:
        reasons.append(
            f"{far} of {n_used} used windows more than {cfg.outlier_cap_cm:.0f} cm from the level"
        )
    if cross:
        if cross["n_pairs"] < 2:
            reasons.append("no time-matched cross-camera pairs")
        elif cross["pair_sd_cm"] > cfg.green_pair_sd_cm:
            reasons.append(f"cross-camera pair SD {cross['pair_sd_cm']:.1f} cm")
        if abs(cross["gap_cm"]) > cfg.green_gap_cm:
            reasons.append(f"cross-camera gap {cross['gap_cm']:+.1f} cm")
    elif any(c["mad_cm"] > cfg.green_pair_sd_cm for c in cams.values() if c["n_used"]):
        reasons.append("window scatter above the green gate")
    if not allz:
        verdict, reasons = "RED", ["no usable window"]
    elif not reasons:
        verdict = "GREEN"
    elif frac >= cfg.green_used_frac / 2 and (
        not cross or abs(cross["gap_cm"]) <= 2.5 * cfg.green_gap_cm
    ):
        verdict = "YELLOW"
    else:
        verdict = "RED"
    return Result(
        z,
        sigma,
        verdict,
        reasons,
        cams,
        cross,
        windows_by_cam,
        runtime_s,
        cfg.as_dict(),
    )

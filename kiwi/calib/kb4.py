"""Fit KB4 (equidistant) to DoubleSphere for scene mapping and ground.

Sample rays across the calibrated field of view and fit KB4's eight
parameters to their DoubleSphere projections. Report error versus field
of view, including the rim beyond the fit region. ORB-SLAM3 localization
and ground geometry consume the lens-aware kb4_intrinsics registry JSON;
its reader lives in kiwi.core.camera.kb4.

    theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
    u = fx*theta_d*cos(phi) + cx,   v = fy*theta_d*sin(phi) + cy
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from kiwi.core.camera.kb4 import KB4_SCHEMA, LENSES

__all__ = [
    "KB4_SCHEMA",
    "Kb4Fit",
    "fit_kb4",
    "project_ds",
    "project_kb4",
    "write_kb4_json",
]


# Error-report bands (half-angle, degrees).
_FOV_BANDS = ((0, 40), (40, 60), (60, 75), (75, 85), (85, 95))


def project_ds(
    rays: np.ndarray,
    xi: float,
    alpha: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Double-sphere projection of Nx3 rays -> Nx2 pixels (Usenko 2018)."""
    x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    d2 = np.sqrt(x * x + y * y + (xi * d1 + z) ** 2)
    denom = alpha * d2 + (1.0 - alpha) * (xi * d1 + z)
    u = fx * x / denom + cx
    v = fy * y / denom + cy
    return np.column_stack([u, v])


def project_kb4(theta: np.ndarray, phi: np.ndarray, p: np.ndarray) -> np.ndarray:
    """KB4 (equidistant) projection from ray angle/azimuth -> Nx2 pixels."""
    fx, fy, cx, cy, k1, k2, k3, k4 = p
    t = theta
    theta_d = t + k1 * t**3 + k2 * t**5 + k3 * t**7 + k4 * t**9
    u = fx * theta_d * np.cos(phi) + cx
    v = fy * theta_d * np.sin(phi) + cy
    return np.column_stack([u, v])


@dataclasses.dataclass(frozen=True)
class Kb4Fit:
    """One fit: the 8 params plus the error-vs-FOV evidence to judge them by."""

    params: np.ndarray  # (8,) [fx, fy, cx, cy, k1, k2, k3, k4]
    source_ds: np.ndarray  # (6,) [xi, alpha, fx, fy, cx, cy]
    resolution: tuple[int, int]  # (W, H)
    fit_max_deg: float
    eval_max_deg: float
    rms_by_fov: tuple[tuple[float, float, float, float], ...]  # (lo, hi, rms, max) px
    # First half-angle where per-degree RMS crosses the threshold, else None.
    crossings_deg: dict[float, float | None]

    def rms_within_fit_fov(self) -> float:
        """Worst band RMS among the bands fully inside the fit FOV."""
        inside = [r for r in self.rms_by_fov if r[1] <= self.fit_max_deg]
        if not inside:
            return float("nan")
        return float(max(r[2] for r in inside))


def fit_kb4(
    ds: np.ndarray,
    resolution: tuple[int, int],
    *,
    fit_max_deg: float = 85.0,
    eval_max_deg: float = 95.0,
    n_theta: int = 120,
    n_phi: int = 72,
) -> Kb4Fit:
    """Least-squares fit KB4 to a DS model over ``fit_max_deg``, judged wider.

    ``ds`` is the registry 6-vector [xi, alpha, fx, fy, cx, cy]. Rays sample
    a theta/phi grid out to ``eval_max_deg``; only rays DS projects to
    finite, in-bounds pixels participate, and only those within the fit FOV
    constrain the solve; the rim rows of ``rms_by_fov`` report the error
    beyond it.
    """
    xi, alpha, fx, fy, cx, cy = (float(v) for v in ds)
    W, H = resolution
    thetas = np.linspace(1e-4, np.deg2rad(eval_max_deg), n_theta)
    phis = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis)
    th, ph = TH.ravel(), PH.ravel()
    rays = np.column_stack(
        [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    )
    px_ds = project_ds(rays, xi, alpha, fx, fy, cx, cy)
    good = (
        np.isfinite(px_ds).all(axis=1)
        & (px_ds[:, 0] >= 0)
        & (px_ds[:, 0] < W)
        & (px_ds[:, 1] >= 0)
        & (px_ds[:, 1] < H)
    )
    th, ph, px_ds = th[good], ph[good], px_ds[good]
    if not len(th):
        raise ValueError("no valid rays - DS parameters or resolution are wrong")
    fit_mask = th <= np.deg2rad(fit_max_deg)

    def resid(p: np.ndarray) -> np.ndarray:
        return (project_kb4(th[fit_mask], ph[fit_mask], p) - px_ds[fit_mask]).ravel()

    p0 = np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0])
    sol = least_squares(resid, p0, method="lm", max_nfev=20000)
    p = sol.x

    err = np.linalg.norm(project_kb4(th, ph, p) - px_ds, axis=1)
    deg = np.rad2deg(th)
    rows = []
    for lo, hi in _FOV_BANDS:
        m = (deg >= lo) & (deg < hi)
        if m.any():
            rows.append(
                (
                    float(lo),
                    float(hi),
                    float(np.sqrt((err[m] ** 2).mean())),
                    float(err[m].max()),
                )
            )
    crossings: dict[float, float | None] = {}
    bins = np.arange(0, eval_max_deg + 1)
    idx = np.digitize(deg, bins)
    for thr in (0.5, 1.0, 2.0):
        crossed = None
        for b in range(1, len(bins)):
            m = idx == b
            if m.any() and np.sqrt((err[m] ** 2).mean()) > thr:
                crossed = float(bins[b - 1])
                break
        crossings[thr] = crossed
    return Kb4Fit(
        params=p,
        source_ds=np.asarray(ds, dtype=np.float64),
        resolution=(int(W), int(H)),
        fit_max_deg=float(fit_max_deg),
        eval_max_deg=float(eval_max_deg),
        rms_by_fov=tuple(rows),
        crossings_deg=crossings,
    )


def write_kb4_json(path: Path, fit: Kb4Fit, *, unit: str, lens: str) -> Path:
    """Write the lens-aware ``kb4_intrinsics`` artifact - the single writer.

    ``unit`` and ``lens`` are required.
    """
    if lens not in LENSES:
        raise ValueError(f"lens must be one of {LENSES}, got {lens!r}")
    if not unit:
        raise ValueError("a lens-aware KB4 artifact must record its camera unit")
    fx, fy, cx, cy, k1, k2, k3, k4 = (float(v) for v in fit.params)
    xi, alpha, dfx, dfy, dcx, dcy = (float(v) for v in fit.source_ds)
    doc = {
        "schema": KB4_SCHEMA,
        "unit": unit,
        "lens": lens,
        "kb4": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
        },
        "source_ds": {
            "xi": xi,
            "alpha": alpha,
            "fx": dfx,
            "fy": dfy,
            "cx": dcx,
            "cy": dcy,
        },
        "resolution": list(fit.resolution),
        "fit_max_deg": fit.fit_max_deg,
        "rms_by_fov": [
            {"lo": lo, "hi": hi, "rms_px": rms, "max_px": mx}
            for lo, hi, rms, mx in fit.rms_by_fov
        ],
        "rms_exceeds_deg": {str(t): c for t, c in fit.crossings_deg.items()},
    }
    path = Path(path)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path

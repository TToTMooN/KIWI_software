"""IMU noise densities from a stationary log -> the ``imu_noise_v1`` artifact.

The four Kalibr noise parameters (accelerometer/gyroscope noise density and
random walk) of the *camera's IMU* live as a per-unit registry artifact
(``config/cameras/<unit>/imu-noise-<date>.yaml``), not inside any
estimator's config tree. State estimation reads this artifact
through the registry. Any noise inflation belongs to the named estimator
profile, while the measured densities stay raw.

Measurement protocol:

* record ~2 h stationary **on-camera** (not USB live),
* discard a warmup window (the protocol uses 20 min; explicit, like the tail),
* **trim the tail**: the log ends with the operator physically stopping the
  recording. ``trim_end_s`` defaults to 0 - pass it explicitly, checked per
  unit and per end. The warmup discard also defaults to 0.

Deploy-side note: estimator configs conventionally inflate
the raw densities (X5 uses raw x10); the multiplier belongs to the
estimator config, never to this artifact.

Kept importable without matplotlib; :func:`plot_allan_curves` imports it
lazily and is optional diagnostics.
"""

from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path

import numpy as np
import yaml

from kiwi.calib.allan import (
    fit_random_walk,
    fit_white_noise,
    overlapping_allan,
    stitch_segments,
)
from kiwi.core.camera.registry import ARTIFACT_SCHEMAS

__all__ = [
    "AXES",
    "IMU_NOISE_SCHEMA",
    "ImuNoise",
    "ImuNoiseResult",
    "assemble_segments",
    "noise_from_series",
    "plot_allan_curves",
    "read_imu_noise",
    "write_imu_noise",
]

IMU_NOISE_SCHEMA = ARTIFACT_SCHEMAS["imu_noise"]
AXES = ("gx", "gy", "gz", "ax", "ay", "az")


@dataclasses.dataclass(frozen=True)
class ImuNoise:
    """The four Kalibr noise parameters, in Kalibr's exact key spelling.

    Units for a rad/s gyro and m/s^2 accel input: noise densities in
    <unit>/sqrt(Hz), random walks in <unit>*sqrt(Hz).
    """

    accelerometer_noise_density: float
    accelerometer_random_walk: float
    gyroscope_noise_density: float
    gyroscope_random_walk: float

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ImuNoiseResult:
    """One fit: the aggregated parameters plus everything needed to judge them."""

    noise: ImuNoise
    per_axis_noise_density: dict[str, float]  # keyed by AXES
    per_axis_random_walk: dict[str, float]
    fs_hz: float
    n_samples: int  # after trimming
    duration_s: float  # after trimming
    discard_s: float
    trim_end_s: float
    taus_s: np.ndarray  # the tau grid, for plotting
    allan_dev: np.ndarray  # (6, n_taus) per-axis Allan deviation, AXES order


def assemble_segments(
    ts: list[np.ndarray], vals: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitch per-part ``(t, [gyro|accel])`` segments into (t, gyro, accel)."""
    if len(ts) == 1:
        t, v = ts[0], np.asarray(vals[0])
    else:
        t, v = stitch_segments(ts, vals)
    return t, v[:, :3], v[:, 3:]


def noise_from_series(
    t: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    *,
    discard_s: float = 0.0,
    trim_end_s: float = 0.0,
    n_taus: int = 60,
) -> ImuNoiseResult:
    """Fit the noise parameters from a continuous stationary series.

    ``t`` in seconds, ``gyro`` (N, 3) rad/s, ``accel`` (N, 3) m/s^2.
    Discards ``discard_s`` of warmup and ``trim_end_s`` of tail, computes the
    overlapping Allan deviation per axis on a log tau grid (4 samples up to
    1/10 of the log), and aggregates each sensor's N/K as the mean of its
    three axes. Raises ``ValueError`` when trimming leaves no samples.
    """
    keep = (t - t[0]) >= discard_s
    if trim_end_s > 0:
        keep &= (t[-1] - t) >= trim_end_s
    if not keep.any():
        raise ValueError(
            f"nothing left to fit: discarding {discard_s:.0f} s of warmup and "
            f"{trim_end_s:.0f} s of tail from a {t[-1] - t[0]:.0f} s log removes "
            "every sample - lower the trims or use a longer log"
        )
    t = t[keep]
    gyro = np.asarray(gyro)[keep]
    accel = np.asarray(accel)[keep]

    fs = 1.0 / float(np.median(np.diff(t)))
    tau_min = 4.0 / fs
    tau_max = (t[-1] - t[0]) / 10.0
    taus = np.logspace(np.log10(tau_min), np.log10(tau_max), n_taus)

    series = [gyro[:, 0], gyro[:, 1], gyro[:, 2], accel[:, 0], accel[:, 1], accel[:, 2]]
    # Subtract per-axis mean so the static bias doesn't leak into the long-tau
    # random-walk region.
    ads = np.stack([overlapping_allan(s - s.mean(), fs, taus) for s in series])

    n_axes = [fit_white_noise(taus, ad) for ad in ads]
    k_axes = [fit_random_walk(taus, ad) for ad in ads]
    noise = ImuNoise(
        accelerometer_noise_density=float(f"{np.nanmean(n_axes[3:]):.6g}"),
        accelerometer_random_walk=float(f"{np.nanmean(k_axes[3:]):.6g}"),
        gyroscope_noise_density=float(f"{np.nanmean(n_axes[:3]):.6g}"),
        gyroscope_random_walk=float(f"{np.nanmean(k_axes[:3]):.6g}"),
    )
    return ImuNoiseResult(
        noise=noise,
        per_axis_noise_density={a: float(v) for a, v in zip(AXES, n_axes)},
        per_axis_random_walk={a: float(v) for a, v in zip(AXES, k_axes)},
        fs_hz=float(fs),
        n_samples=int(len(t)),
        duration_s=float(t[-1] - t[0]),
        discard_s=float(discard_s),
        trim_end_s=float(trim_end_s),
        taus_s=taus,
        allan_dev=ads,
    )


def write_imu_noise(
    path: Path,
    result: ImuNoiseResult,
    *,
    source: str,
    unit: str | None = None,
) -> None:
    """Write the ``imu_noise_v1`` YAML - the single writer of this artifact.

    ``source`` names the recording(s) the fit came from.
    """
    body = {
        "schema": IMU_NOISE_SCHEMA,
        **({"unit": unit} if unit else {}),
        "source": source,
        "measured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "fs_hz": round(result.fs_hz, 3),
        "n_samples": result.n_samples,
        "duration_s": round(result.duration_s, 1),
        "discard_s": result.discard_s,
        "trim_end_s": result.trim_end_s,
        "noise": result.noise.as_dict(),
        "per_axis_noise_density": result.per_axis_noise_density,
        "per_axis_random_walk": result.per_axis_random_walk,
    }
    Path(path).write_text(yaml.safe_dump(body, sort_keys=False))


def read_imu_noise(path: Path) -> tuple[ImuNoise, dict]:
    """Read an ``imu_noise_v1`` YAML back: ``(noise, full_document)``.

    Raises ``ValueError`` on a wrong/missing schema tag or missing noise
    keys.
    """
    path = Path(path)
    doc = yaml.safe_load(path.read_text()) or {}
    if doc.get("schema") != IMU_NOISE_SCHEMA:
        raise ValueError(
            f"{path}: schema {doc.get('schema')!r}, expected {IMU_NOISE_SCHEMA!r}"
        )
    try:
        noise = ImuNoise(
            **{k: float(doc["noise"][k]) for k in ImuNoise.__annotations__}
        )
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path}: incomplete noise block ({e})") from e
    return noise, doc


def plot_allan_curves(result: ImuNoiseResult, path: Path) -> Path | None:
    """Write the per-axis Allan-deviation log-log plot, or ``None`` without matplotlib.

    Diagnostics only - the YAML artifact is complete without it.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    taus, ads = result.taus_s, result.allan_dev
    fig, (axg, axa) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for ax, block, labels, title, units, n_agg, k_agg in [
        (axg, ads[:3], AXES[:3], "Gyro", "rad/s",
         result.noise.gyroscope_noise_density, result.noise.gyroscope_random_walk),
        (axa, ads[3:], AXES[3:], "Accel", "m/s^2",
         result.noise.accelerometer_noise_density, result.noise.accelerometer_random_walk),
    ]:  # fmt: skip
        for ad, lbl, c in zip(block, labels, colors):
            ax.loglog(taus, ad, "-", color=c, label=lbl, lw=1.2)
        tref = np.array([taus[0], taus[-1]])
        ad = block[-1]
        ax.loglog(tref, ad[~np.isnan(ad)][0] * np.sqrt(taus[0] / tref),
                  "k--", lw=0.6, alpha=0.4, label="slope -1/2 (N)")  # fmt: skip
        ax.loglog(tref, ad[~np.isnan(ad)][-1] * np.sqrt(tref / taus[-1]),
                  "k:", lw=0.6, alpha=0.4, label="slope +1/2 (K)")  # fmt: skip
        ax.set_xlabel("tau [s]")
        ax.set_ylabel(f"Allan deviation [{units}]")
        ax.set_title(f"{title}  (N={n_agg:.3g}, K={k_agg:.3g})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)

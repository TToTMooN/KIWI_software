"""Allan-deviation estimation of IMU noise parameters (pure numpy).

Overlapping Allan deviation plus slope-band regression fits for the
white-noise density N and rate random walk K, using Kalibr conventions
(Stockwell / IEEE 952-1997 regimes):

  sigma(tau) = N / sqrt(tau)          white-noise region (log-log slope -1/2)
  sigma(tau) ~ 0.664 * B              bias-instability floor
  sigma(tau) = K * sqrt(tau / 3)      random-walk region (log-log slope +1/2)

N and K come back in the units of the input series per Kalibr's conventions
(e.g. rad/s input -> N in rad/s/sqrt(Hz), K in rad/s^2/sqrt(Hz)).

Both fits locate the region where the local log-log slope matches the
theoretical value and take the median of the per-point closed-form
estimates over that band.

Also owns ``stitch_segments``: lossless reassembly of an auto-split
stationary recording (the X5 splits long ``.insv`` runs into ~30-min files
whose telemetry clocks restart per file but whose tails/heads carry a few
seconds of bit-identical duplicated samples).

Driven by the ``kiwi calib imu`` verbs. Kept free of cv2/ROS/matplotlib
imports.
"""

from __future__ import annotations

import numpy as np

# Half-width of the accepted local-slope band around the theoretical
# log-log slopes of -1/2 (white noise) and +1/2 (random walk).
SLOPE_TOL = 0.2

# Search window for the duplicated overlap at a segment boundary, in samples.
_MAX_OVERLAP_SAMPLES = 30_000


def overlapping_allan(y: np.ndarray, fs: float, taus_s: np.ndarray) -> np.ndarray:
    """Overlapping Allan deviation sigma(tau) for a 1-D signal sampled at fs Hz.

    Implements sigma^2(m*tau0) = 1/(2 (N - 2m) m^2) * sum (S_{i+2m} - 2 S_{i+m} + S_i)^2
    where S_k = sum_{j<k} y_j is the cumulative sum. This is the classic
    overlapping estimator from Riley 2008 section 5.2.4 - O(N) per tau.
    """
    N = len(y)
    tau0 = 1.0 / fs
    S = np.concatenate(([0.0], np.cumsum(y)))  # length N+1
    ad = np.empty_like(taus_s, dtype=np.float64)
    for i, tau in enumerate(taus_s):
        m = int(round(tau / tau0))
        if m < 1 or 2 * m >= N:
            ad[i] = np.nan
            continue
        # S[i+2m] - 2 S[i+m] + S[i], for i in [0, N - 2m)
        d = S[2 * m :] - 2 * S[m : N - m + 1] + S[: N - 2 * m + 1]
        d = d[: N - 2 * m]  # safety crop
        var = (d**2).sum() / (2.0 * (N - 2 * m) * m * m)
        ad[i] = np.sqrt(var)
    return ad


def _slope_band(taus: np.ndarray, ad: np.ndarray, target: float) -> np.ndarray:
    """Indices (into taus/ad) where the local log-log slope is target +/- SLOPE_TOL.

    Returns an empty array when fewer than 5 finite points exist.
    """
    valid = np.where(~np.isnan(ad))[0]
    if len(valid) < 5:
        return np.empty(0, dtype=int)
    log_t = np.log10(taus[valid])
    log_s = np.log10(ad[valid])
    slopes = np.gradient(log_s, log_t)
    in_band = np.abs(slopes - target) < SLOPE_TOL
    return valid[in_band]


def fit_white_noise(taus: np.ndarray, ad: np.ndarray) -> float:
    """N (noise density). sigma(tau) = N / sqrt(tau) in the -1/2-slope region.

    Median of the per-point estimate N = sigma(tau) * sqrt(tau) over the
    detected -1/2-slope band; falls back to the sample nearest tau = 1 s
    when no band is found (short or heavily contaminated logs).
    """
    band = _slope_band(taus, ad, -0.5)
    if len(band):
        return float(np.exp(np.median(np.log(ad[band] * np.sqrt(taus[band])))))
    valid = ~np.isnan(ad)
    if not valid.any():
        return float("nan")
    i = int(np.nanargmin(np.abs(np.log10(taus[valid]) - 0.0)))
    idx = np.where(valid)[0][i]
    return float(ad[idx] * np.sqrt(taus[idx]))


def fit_random_walk(taus: np.ndarray, ad: np.ndarray) -> float:
    """K (random walk). sigma(tau) = K * sqrt(tau/3) in the +1/2-slope region.

    Median of the per-point estimate K = sigma(tau) * sqrt(3/tau) over the
    detected +1/2-slope band; falls back to the largest valid tau (assuming
    the random-walk slope holds there) when no band is found.
    """
    band = _slope_band(taus, ad, 0.5)
    if len(band):
        return float(np.exp(np.median(np.log(ad[band] * np.sqrt(3.0 / taus[band])))))
    valid = ~np.isnan(ad)
    if valid.sum() < 5:
        return float("nan")
    tau = taus[valid][-1]
    sigma = ad[valid][-1]
    return float(sigma * np.sqrt(3.0 / tau))


def stitch_segments(
    ts: list[np.ndarray],
    values: list[np.ndarray],
    tol: float = 1e-12,
    probe_len: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reassemble an auto-split continuous recording into one (t, values) pair.

    Each segment is ``(t_i [s], values_i [N_i, C])`` with a clock that restarts
    per segment. Consecutive segments must overlap: the head of segment i+1
    repeats the tail of segment i sample-for-sample. The alignment is found
    by locating the first ``probe_len`` rows of segment i+1 inside segment
    i's tail with max-abs difference <= tol, then the duplicated rows are
    dropped and segment i+1's clock is shifted onto segment i's.

    Raises ValueError when a boundary has no unique match (gap between files,
    wrong ordering, or a constant/ambiguous signal).
    """
    if len(ts) != len(values):
        raise ValueError(
            f"got {len(ts)} timestamp arrays but {len(values)} value arrays"
        )
    if not ts:
        raise ValueError("no segments given")

    out_t = ts[0].astype(np.float64, copy=True)
    out_v = np.asarray(values[0])
    for seg_idx, (t_b, v_b) in enumerate(zip(ts[1:], values[1:]), start=1):
        v_b = np.asarray(v_b)
        if len(t_b) <= probe_len:
            raise ValueError(
                f"segment {seg_idx} has only {len(t_b)} samples (need > {probe_len})"
            )
        probe = v_b[:probe_len]
        # Candidate positions: tail rows whose first channel matches the
        # probe's first sample; then verify the full probe block there.
        # On noise-like data this is O(N) + O(few * probe_len).
        tail_start = max(0, len(out_v) - probe_len - _MAX_OVERLAP_SAMPLES)
        tail = out_v[tail_start:]
        cand = np.where(np.abs(tail[:, 0] - probe[0, 0]) <= tol)[0]
        cand = cand[cand + probe_len <= len(tail)]
        matches = [
            i for i in cand if np.abs(tail[i : i + probe_len] - probe).max() <= tol
        ]
        if not matches:
            raise ValueError(
                f"segment {seg_idx}: no duplicated-sample overlap found in the previous "
                f"segment's tail - files may be non-consecutive or from different runs"
            )
        if len(matches) > 1:
            raise ValueError(
                f"segment {seg_idx}: {len(matches)} candidate alignments - "
                f"signal too repetitive to stitch unambiguously"
            )
        m = tail_start + matches[0]
        n_dup = len(out_t) - m
        # The probe only proves the first probe_len rows; require the
        # duplication to hold through the previous segment's very last row.
        if n_dup > len(v_b) or np.abs(out_v[m:] - v_b[:n_dup]).max() > tol:
            raise ValueError(
                f"segment {seg_idx}: overlap matches at its head but diverges before "
                f"the previous segment ends - possible dropped samples at the boundary"
            )
        offset = out_t[m] - t_b[0]
        out_t = np.concatenate([out_t, t_b[n_dup:] + offset])
        out_v = np.concatenate([out_v, v_b[n_dup:]])
    return out_t, out_v

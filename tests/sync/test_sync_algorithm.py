"""Algorithm invariants and confidence boundaries for audio synchronization."""

import dataclasses

import numpy as np
import pytest
from scipy.ndimage import maximum_filter

import kiwi.sync.audio as audio


def _unchunked_peaks(x, p):
    n = (len(x) - p.nfft) // p.hop
    idx = np.arange(p.nfft)[None, :] + p.hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(p.nfft)
    logm = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
    keep = (maximum_filter(logm, size=p.peak_neighborhood) == logm) & (
        logm > np.median(logm) + p.peak_floor
    )
    return np.nonzero(keep)


@pytest.mark.parametrize("block", [1, 7, 64, 256])
def test_chunk_boundaries_preserve_unchunked_landmarks(block):
    p = dataclasses.replace(audio.SyncParams(), analysis_block_frames=block)
    rng = np.random.default_rng(77)
    x = rng.normal(0, 15, 4 * p.sample_rate)
    for at in (0.074, 0.73, 1.56, 2.08, 3.73):
        start = int(at * p.sample_rate)
        x[start : start + 2500] += rng.normal(0, 3000, 2500)
    expected = _unchunked_peaks(x, p)
    actual = audio._spectral_peaks(x, p)
    for lhs, rhs in zip(expected, actual):
        np.testing.assert_array_equal(lhs, rhs)


@pytest.mark.parametrize("constant", [False, True])
def test_radix_median_is_exact_for_large_duplicate_distributions(constant):
    # Cross the bounded in-memory fast-path threshold. Duplicate central
    # values and all-zero spectra exercise rank selection with tied prefixes.
    rng = np.random.default_rng(88)
    values = rng.integers(0, 100, (2100, 1025)).astype(np.float64)
    if constant:
        values.fill(0)
    assert audio._block_median(values, 127) == np.median(values)


def _drift(low, high, *, slope=50, residual_ms=0):
    return audio.DriftResult(
        slope_ppm=slope,
        ci_low_ppm=low,
        ci_high_ppm=high,
        n_windows=8,
        scatter_ms=0,
        window_offsets=(),
        intercept_s=25,
        coverage_fraction=1.0,
        covers_endpoints=True,
        max_residual_ms=residual_ms,
    )


def test_wide_interval_is_uncertain_not_bounded():
    status, bounded, budget, _ = audio._drift_checks(
        _drift(-1000, 1000), 120, audio.SyncParams()
    )
    assert status == "uncertain drift"
    assert not bounded and not budget
    assert audio._measured_drift_acceptable(
        _drift(-1000, 1000), bounded, budget, audio.SyncParams()
    )


def test_unmeasurable_overlap_lowers_confidence_without_rejecting_the_offset():
    p = audio.SyncParams()
    status, bounded, budget, error = audio._drift_checks(None, 120, p)
    assert status == "unmeasurable on long overlap"
    assert not bounded and not budget and error is None
    assert audio._measured_drift_acceptable(None, bounded, budget, p)
    confidence = audio._confidence(
        {"vote_margin": True, "measured_drift_acceptable": True}, bounded, budget
    )
    assert confidence["level"] == "offset_only"
    status, _, _, error = audio._drift_checks(None, 12, p)
    assert status == "not measured: short overlap" and error is None


def test_healthy_rate_still_exceeds_budget_over_long_recording():
    d = _drift(45, 55)
    assert audio._drift_checks(d, 120, audio.SyncParams())[1:3] == (True, True)
    status, bounded, budget, error = audio._drift_checks(d, 600, audio.SyncParams())
    assert bounded and not budget
    assert status == "exceeds constant-offset timing budget"
    assert error == pytest.approx(0.033)
    assert not audio._measured_drift_acceptable(d, bounded, budget, audio.SyncParams())


def test_rejected_fit_points_still_consume_error_budget():
    d = _drift(-1, 1, slope=0, residual_ms=40)
    assert not audio._drift_checks(d, 120, audio.SyncParams())[2]


def test_missing_tail_cannot_establish_whole_recording_drift():
    p = dataclasses.replace(audio.SyncParams(), sample_rate=1000)
    a = np.random.default_rng(41).normal(0, 3000, 220_000)
    b = a.copy()
    b[140_000:] = 0
    d = audio.estimate_drift(a, b, 0, params=p)
    assert d is not None
    assert d.coverage_fraction < 0.8 and not d.covers_endpoints
    status, bounded, budget, _ = audio._drift_checks(d, 220, p)
    assert status == "insufficient drift coverage"
    assert not bounded and not budget
    assert audio._measured_drift_acceptable(d, bounded, budget, p)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_drift_is_refused_even_without_temporal_support(value):
    d = dataclasses.replace(_drift(-1, 1), slope_ppm=value, coverage_fraction=0.1)
    _, bounded, budget, _ = audio._drift_checks(d, 120, audio.SyncParams())
    assert not audio._measured_drift_acceptable(d, bounded, budget, audio.SyncParams())


@pytest.mark.parametrize("coverage", [0.25, 1.0])
def test_measured_excessive_skew_is_refused_even_with_partial_coverage(coverage):
    d = dataclasses.replace(_drift(350, 450, slope=400), coverage_fraction=coverage)
    _, bounded, budget, _ = audio._drift_checks(d, 120, audio.SyncParams())
    assert not audio._measured_drift_acceptable(d, bounded, budget, audio.SyncParams())


def test_short_match_between_scheduled_sections_can_refine_a_long_recording():
    p = dataclasses.replace(audio.SyncParams(), sample_rate=1000)
    a, b = np.zeros(602_000), np.zeros(600_000)
    pattern = np.random.default_rng(201).normal(0, 3000, 5000)
    a[82_000:87_000] = pattern
    b[80_000:85_000] = pattern
    result = audio._refine_recording(
        a, b, 2.02, p, anchor_times_s=tuple(np.linspace(82.1, 86.9, 20))
    )
    assert result.offset_s == pytest.approx(2.0, abs=1 / p.sample_rate)
    assert not result.at_window_edge


def test_long_refinement_never_ffts_entire_recording(monkeypatch):
    p = dataclasses.replace(audio.SyncParams(), sample_rate=8000)
    rng = np.random.default_rng(9)
    source = rng.normal(0, 3000, 145 * p.sample_rate)
    a = source[: 120 * p.sample_rate]
    b = source[25 * p.sample_rate : 145 * p.sample_rate]
    lengths = []
    original = audio.gcc_phat

    def measured(x, y):
        lengths.append((len(x), len(y)))
        return original(x, y)

    monkeypatch.setattr(audio, "gcc_phat", measured)
    result = audio._refine_recording(a, b, 25.02, p)
    assert result.offset_s == pytest.approx(25, abs=2 / p.sample_rate)
    assert result.global_argmax_s is None
    assert max(max(pair) for pair in lengths) <= p.refine_max_duration_s * p.sample_rate
    assert sum(hi - lo for lo, hi in result.windows_s) == pytest.approx(150.0)
    assert audio._covered_duration(result.windows_s) == pytest.approx(94.98)


def test_silence_does_not_manufacture_zero_strength_onsets():
    assert audio.onset_times(np.zeros(12 * 48000), 8) == []


def test_per_resolution_cache_reuses_features_and_respects_recipe(
    monkeypatch, tmp_path
):
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"recording")
    calls = {"probe": 0, "decode": 0, "peaks": 0}

    def probe(*args, **kwargs):
        calls["probe"] += 1
        return object()

    def decode(*args, **kwargs):
        calls["decode"] += 1
        return np.zeros(1000)

    def peaks(*args, **kwargs):
        calls["peaks"] += 1
        return np.array([], dtype=int), np.array([], dtype=int)

    monkeypatch.setattr(audio, "probe_media", probe)
    monkeypatch.setattr(audio, "load_audio", decode)
    monkeypatch.setattr(audio, "_spectral_peaks", peaks)
    cache = audio.AudioCache()
    p = audio.SyncParams()
    first = cache.prepare(path, p, "ffmpeg", "ffprobe")
    assert cache.prepare(path, p, "ffmpeg", "ffprobe") is first
    assert calls == {"probe": 1, "decode": 1, "peaks": 1}
    cache.prepare(path, dataclasses.replace(p, fan=4), "ffmpeg", "ffprobe")
    assert calls == {"probe": 2, "decode": 2, "peaks": 2}
    path.write_bytes(b"replacement recording")
    assert cache.prepare(path, p, "ffmpeg", "ffprobe") is not first
    assert calls == {"probe": 3, "decode": 3, "peaks": 3}

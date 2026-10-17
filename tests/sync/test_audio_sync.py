"""Unit tests for kiwi.sync.audio.

Pure numpy/scipy/yaml on synthetic signals and throwaway registries - no media
files, no ffmpeg (the two subprocess-touching functions are monkeypatched), no
cv2. It runs in the pip-only CI lane.

Two properties carry most of the weight here:

* the **sign convention**; and
* the A/V table **never defaulting to zero**.

Run: pixi run test
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from kiwi.sync.audio import (
    AV_SYNC_SCHEMA,
    EXTRACTION,
    GATES,
    AudioOffset,
    AVCorrection,
    Clip,
    MediaProbe,
    RawOffset,
    RecordingMode,
    SyncParams,
    av_delay_from_ground_truth,
    estimate_drift,
    gcc_phat,
    load_audio,
    onset_agreement,
    probe_media,
    read_av_delay,
    refine_offset,
    resolve_av_correction,
    vote_offset,
)

SR = 48_000
P = SyncParams()


def _delay_fractional(x: np.ndarray, samples: float) -> np.ndarray:
    """Delay by a possibly non-integer number of samples via an FFT phase ramp."""
    n = len(x)
    f = np.fft.rfft(x)
    k = np.arange(len(f))
    return np.fft.irfft(f * np.exp(-2j * np.pi * k * samples / n), n)


def _periodic_scene(
    seed=0, dur=16.0, period=1.0, true_off=-5.7, uniq_gain_b=0.15, noise=0.02
):
    """Two mics hearing one repeated broadband sound plus a few unique events.

    The repeated sound is what makes real task audio ambiguous: the correlation
    has strong peaks at the true lag AND at true±k*period.
    """
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    pad = int(abs(true_off) * SR) + SR
    proto = rng.standard_normal(1200) * np.exp(-np.arange(1200) / 200.0)
    a_s, b_s = np.zeros(n + pad), np.zeros(n + pad)
    t = period
    while t < dur - 1:
        i = int(t * SR)
        a_s[i : i + len(proto)] += proto
        b_s[i : i + len(proto)] += proto
        t += period
    for k in range(3):
        u = rng.standard_normal(1200) * np.exp(-np.arange(1200) / 200.0)
        i = int((2.3 + k * 4.1) * SR)
        a_s[i : i + len(u)] += 1.5 * u
        b_s[i : i + len(u)] += uniq_gain_b * u
    sh = int(round(true_off * SR))
    a = a_s[:n] + noise * rng.standard_normal(n)
    b = np.roll(b_s, -sh)[:n] + noise * rng.standard_normal(n)
    a = np.convolve(a, [1.0, 0.5, 0.2], "same") * 3000
    b = np.convolve(b, [1.0, -0.3, 0.1], "same") * 3000
    return a, b, true_off


# --------------------------------------------------------------------------- #
# correlation core
# --------------------------------------------------------------------------- #
def test_gcc_phat_recovers_subsample_delay():
    """The parabolic vertex gives sub-sample accuracy."""
    rng = np.random.default_rng(0)
    a = np.convolve(rng.standard_normal(5 * SR), np.ones(4) / 4, "same") * 3000
    d = 137.4  # deliberately fractional
    b = _delay_fractional(a, d)
    b += 0.01 * np.abs(a).mean() * rng.standard_normal(len(b))  # ~20 dB SNR
    got = refine_offset(a, b, -d / SR, params=P).offset_s
    assert abs(got - (-d / SR)) < 0.5 / SR  # within half a sample


def test_refine_sign_convention_positive_means_other_started_later():
    """b[n] = a[n-D] means `other` is a DELAYED copy, i.e. it started EARLIER on
    the shared clock, so the offset is negative. Downstream composes this as
    global(other) = global(ref) + offset; an inversion here is silent."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(3 * SR) * 3000
    D = 4800  # 0.1 s
    b = np.roll(a, D)
    got = refine_offset(a, b, -D / SR, params=P).offset_s
    assert got == pytest.approx(-D / SR, abs=2 / SR)


# --------------------------------------------------------------------------- #
# the two-stage recipe
# --------------------------------------------------------------------------- #
def test_periodic_audio_has_strong_wrong_period_peaks():
    """The synthetic scene has competing peaks a whole period from the truth."""
    a, b, truth = _periodic_scene()
    lags, cc = gcc_phat(a, b)
    acc = np.abs(cc)

    def peak_near(s):
        m = np.abs(lags / SR - s) <= 0.05
        return acc[m].max()

    true_peak = peak_near(truth)
    for k in (-1, 1):
        assert peak_near(truth + k) > 0.2 * true_peak, (
            "no competing period peak - scene too easy"
        )


def test_coarse_stage_decides_which_period_wins():
    """refine_offset only searches ±refine_window_s, so its SEED picks the period.

    Swap the seed and the answer moves by a whole period.
    """
    a, b, truth = _periodic_scene()
    assert refine_offset(a, b, truth, params=P).offset_s == pytest.approx(
        truth, abs=0.02
    )
    for wrong in (truth + 1.0, truth - 1.0):
        got = refine_offset(a, b, wrong, params=P).offset_s
        assert got == pytest.approx(wrong, abs=0.02)
        assert abs(got - truth) > 0.9  # locked onto the wrong period, no complaint


def test_vote_picks_the_true_period_and_refine_polishes_it():
    a, b, truth = _periodic_scene()
    v = vote_offset(a, b, params=P)
    assert abs(v.offset_s - truth) <= P.vote_bin_s, "vote landed on the wrong period"
    assert v.votes >= P.min_votes and v.votes >= P.vote_margin * v.runner_up
    assert refine_offset(a, b, v.offset_s, params=P).offset_s == pytest.approx(
        truth, abs=0.01
    )


def test_uncorrelated_noise_never_yields_a_confident_offset():
    """Two unrelated clips must fail loudly or fail the margin - never a number."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal(6 * SR) * 3000
    b = rng.standard_normal(6 * SR) * 3000
    try:
        v = vote_offset(a, b, params=P)
    except RuntimeError:
        return  # no matching landmarks at all - the correct outcome
    assert not (v.votes >= P.min_votes and v.votes >= P.vote_margin * v.runner_up)


def test_onset_agreement_matches_only_at_the_true_offset():
    rng = np.random.default_rng(3)
    n = 12 * SR
    a = 0.001 * rng.standard_normal(n)
    for t in (1.0, 3.0, 5.0, 7.0, 9.0, 10.5):
        i = int(t * SR)
        a[i : i + 800] += rng.standard_normal(800) * np.exp(-np.arange(800) / 150.0)
    a *= 3000
    off = -1.25
    b = np.roll(a, int(round(-off * SR)))
    good, total = onset_agreement(a, b, off, params=P)
    bad, _ = onset_agreement(a, b, off + 0.5, params=P)
    # 6 events planted; the 1.5 s gap + edge exclusion caps the picks below
    # onset_count, and not every pick survives the wrap-around at the seam.
    assert 0 < total <= P.onset_count
    assert good >= 4
    assert bad <= 1 and good > bad


# --------------------------------------------------------------------------- #
# recording mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mode,expect",
    [
        (RecordingMode(1440, 2560, 50.0, -90), "2560x1440@50"),
        (RecordingMode(1536, 2720, 30000 / 1001, 270), "2720x1536@29.97"),
        (RecordingMode(2560, 1440, 50.0, 0), "2560x1440@50"),
        (RecordingMode(1920, 1920, 120000 / 1001, 0), "1920x1920@119.88"),
    ],
)
def test_mode_key_normalizes_rotation_and_fps(mode, expect):
    """The calibration clips are stored portrait with a display matrix and the
    production clips landscape; both must hit the SAME A/V table entry."""
    assert mode.key == expect


# --------------------------------------------------------------------------- #
# the A/V correction
# --------------------------------------------------------------------------- #
def _probe(mode, path="x.mp4", codec="aac", rate=48000, ch=2):
    return MediaProbe(
        path=Path(path),
        mode=mode,
        video_duration_s=10.0,
        audio_codec=codec,
        audio_sample_rate=rate,
        audio_channels=ch,
    )


def _registry(tmp_path, unit, modes, *, model="Insta360 GO 3", schema=AV_SYNC_SCHEMA):
    d = tmp_path / unit
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text(yaml.safe_dump({"model": model, "nickname": unit}))
    (d / "av-sync.yaml").write_text(
        yaml.safe_dump(
            {"schema": schema, "unit": unit, "extraction": EXTRACTION, "modes": modes}
        )
    )
    return d


X5 = RecordingMode(1920, 1920, 120000 / 1001)
GO = RecordingMode(
    2560, 1440, 50.0
)  # a measured GO3 mode, used by the synthetic fixtures
EGO = RecordingMode(2688, 2688, 30000 / 1001)  # the mode the shipped ego clips are in


def test_same_mode_and_model_cancels_without_reading_any_table(tmp_path):
    """An identical pair needs no calibration at all - the delay is common-mode.
    The registry establishes the model; there is no table to read."""
    unit = tmp_path / "known-unit"
    unit.mkdir()
    (unit / "camera.yaml").write_text("model: Insta360 X5\n")
    a = Clip(tmp_path / "a.insv", unit, "left")
    b = Clip(tmp_path / "b.insv", unit, "right")
    corr = resolve_av_correction(a, _probe(X5), b, _probe(X5))
    assert corr.seconds == 0.0
    assert "identical" in corr.provenance


def test_a_named_unit_without_a_model_is_refused_not_cancelled(tmp_path):
    """Two DIFFERENT units, each with a real table, whose camera.yaml carries a
    serial but no `model:`: the pair is refused, not cancelled as one device."""
    a = _registry(tmp_path, "unit-a", {GO.key: {"av_delay_s": 0.0}})
    b = _registry(tmp_path, "unit-b", {GO.key: {"av_delay_s": 0.059}})
    for d in (a, b):
        (d / "camera.yaml").write_text(
            yaml.safe_dump({"serial": d.name, "av_sync": "av-sync.yaml"})
        )
    with pytest.raises(RuntimeError, match="declares no `model:`"):
        resolve_av_correction(
            Clip(tmp_path / "a", a), _probe(GO), Clip(tmp_path / "b", b), _probe(GO)
        )


def test_matching_container_formats_cannot_establish_camera_identity(tmp_path):
    """Different cameras can record matching formats with different A/V delays."""
    with pytest.raises(RuntimeError, match="no camera unit"):
        resolve_av_correction(
            Clip(tmp_path / "a"), _probe(GO), Clip(tmp_path / "b"), _probe(GO)
        )


def test_correction_applies_the_difference_with_the_documented_sign(tmp_path):
    x5 = _registry(tmp_path, "x5", {X5.key: {"av_delay_s": 0.0}}, model="Insta360 X5")
    go = _registry(tmp_path, "go", {GO.key: {"av_delay_s": 0.059}})
    corr = resolve_av_correction(
        Clip(tmp_path / "a", x5), _probe(X5), Clip(tmp_path / "b", go), _probe(GO)
    )
    assert corr.seconds == pytest.approx(0.059)
    # ...and the calibration round-trips through the one function that owns the sign
    assert av_delay_from_ground_truth(
        raw_audio_offset_s=1.000, video_offset_s=1.059
    ) == pytest.approx(0.059)


def test_unknown_mode_raises_and_names_what_it_knows(tmp_path):
    go = _registry(tmp_path, "go", {GO.key: {"av_delay_s": 0.0}})
    other = RecordingMode(2720, 1536, 30000 / 1001)
    with pytest.raises(RuntimeError) as e:
        read_av_delay(go, other)
    msg = str(e.value)
    assert other.key in msg and GO.key in msg and "av-sync.yaml" in msg


@pytest.mark.parametrize(
    "modes,schema,make",
    [
        ({GO.key: {"av_delay_s": 0.0}}, "wrong_schema_v9", True),
        ({}, AV_SYNC_SCHEMA, True),
        ({"2720x1536@29.97": {"av_delay_s": 0.0}}, AV_SYNC_SCHEMA, True),
        ({GO.key: {"note": "no value"}}, AV_SYNC_SCHEMA, True),
        (None, None, False),  # no unit dir / no av-sync.yaml at all
    ],
    ids=[
        "bad-schema",
        "empty-modes",
        "missing-mode",
        "entry-without-value",
        "missing-file",
    ],
)
def test_av_delay_never_silently_defaults_to_zero(tmp_path, modes, schema, make):
    """Every degenerate table must RAISE."""
    d = (
        _registry(tmp_path, "go", modes, schema=schema)
        if make
        else (tmp_path / "absent")
    )
    with pytest.raises(RuntimeError):
        read_av_delay(d, GO)


def test_differing_modes_without_a_unit_raises(tmp_path):
    """No registry unit + different modes = no way to know the correction."""
    with pytest.raises(RuntimeError, match="no camera unit"):
        resolve_av_correction(
            Clip(tmp_path / "a"), _probe(X5), Clip(tmp_path / "b"), _probe(GO)
        )


@pytest.mark.parametrize("modes", [(X5, GO), (X5, X5)])
def test_a_corrupt_camera_yaml_is_a_runtime_error_here_too(tmp_path, modes):
    """The module boundary: `_run`'s docstring promises exactly SyncGateError
    and RuntimeError leave this module, and `resolve_av_correction` reads a
    unit for its MODEL before it ever reaches `read_av_delay`. The translation
    sits at the load, so an unparsable camera.yaml is a RuntimeError even for a
    same-mode pair that needs no A/V table at all."""
    d = tmp_path / "insta360-X5-001"
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text("model: [unterminated\n")
    with pytest.raises(RuntimeError, match="broken camera registry unit") as e:
        resolve_av_correction(
            Clip(tmp_path / "a", d),
            _probe(modes[0]),
            Clip(tmp_path / "b"),
            _probe(modes[1]),
        )
    assert isinstance(e.value.__cause__, ValueError)
    assert isinstance(e.value.__cause__.__cause__, yaml.YAMLError)
    assert f"{d / 'camera.yaml'}: not valid YAML" in str(e.value)


# --------------------------------------------------------------------------- #
# gating
# --------------------------------------------------------------------------- #
def _raw(gates):
    return RawOffset(
        offset_s=1.0,
        voted_offset_s=1.0,
        votes=100,
        runner_up=2,
        onsets_matched=5,
        onsets_total=8,
        gates=gates,
        diagnostics={"global_argmax_offset_s": 1.0, "argmax_is_false_lock": False},
        ref=_probe(X5, "ref.insv"),
        other=_probe(GO, "other.mp4"),
        params=P,
    )


def test_require_confident_raises_and_names_the_failed_gate():
    bad = AudioOffset(
        1.0,
        _raw({"vote_margin": False, "refine_in_window": True}),
        AVCorrection(0.0, "-"),
    )
    assert not bad.confident
    with pytest.raises(RuntimeError, match="vote_margin"):
        bad.require_confident()


def test_require_confident_passes_through_when_all_gates_pass():
    ok = AudioOffset(1.0, _raw({"vote_margin": True}), AVCorrection(0.0, "-"))
    assert ok.require_confident() is ok
    assert "PASS" in ok.report()


def test_load_audio_raises_on_an_empty_stream(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: type("R", (), {"stdout": b"", "returncode": 0})(),
    )
    with pytest.raises(RuntimeError, match="0 audio samples"):
        load_audio(tmp_path / "x.mp4")


# --- the subprocess boundary ------------------------------------------------ #
# This module promises callers exactly two failure types: SyncGateError when the
# gates refuse a measurable pair, RuntimeError when a pair cannot be measured at
# all. `subprocess` raises neither, so its failures are translated. Both are
# pinned here, at the boundary that makes the promise.


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda p: probe_media(p, ffprobe="kiwi-no-such-tool"), id="probe"),
        pytest.param(lambda p: load_audio(p, ffmpeg="kiwi-no-such-tool"), id="decode"),
    ],
)
def test_a_missing_media_tool_is_a_runtime_error(call, tmp_path):
    clip = tmp_path / "x.insv"
    clip.write_bytes(b"media")
    with pytest.raises(RuntimeError, match="not found on PATH") as e:
        call(clip)
    assert "Install ffmpeg" in str(e.value)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(probe_media, id="probe"),
        pytest.param(load_audio, id="decode"),
    ],
)
def test_an_undecodable_clip_carries_the_tools_own_message(call, monkeypatch, tmp_path):
    import subprocess

    clip = tmp_path / "x.insv"

    def boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            cmd,
            stderr=f"{clip}: Invalid data found when processing input\n".encode(),
        )

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RuntimeError) as e:
        call(clip)
    msg = str(e.value)
    # The tool's stderr is the only real diagnostic, and capture_output keeps it
    # out of CalledProcessError's own message, so it has to be carried across.
    assert "Invalid data found when processing input" in msg
    assert str(clip) in msg
    # ...but only once: ffprobe prefixes the path to its own message too.
    assert msg.count(str(clip)) == 1


def test_a_tool_that_fails_silently_still_names_its_exit_status(monkeypatch, tmp_path):
    import subprocess

    def boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(9, cmd, stderr=b"")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="exit status 9"):
        probe_media(tmp_path / "x.insv")


# --- pair -> shared-clock offsets ------------------------------------------ #
def test_pair_to_global_offsets_matches_the_timecode_convention():
    from kiwi.sync.audio import pair_to_global_offsets

    # other started later -> ref IS the origin
    assert pair_to_global_offsets(0.25) == (0.0, 0.25)
    # other started earlier -> other becomes the origin
    assert pair_to_global_offsets(-0.25) == (0.25, 0.0)
    # simultaneous
    assert pair_to_global_offsets(0.0) == (0.0, 0.0)


def test_pair_to_global_offsets_are_nonnegative_with_one_zero():
    from kiwi.sync.audio import pair_to_global_offsets

    for d in (-3.5, -0.001, 0.0, 0.001, 3.5):
        a, b = pair_to_global_offsets(d)
        assert min(a, b) == 0.0
        assert a >= 0.0 and b >= 0.0
        # the relative offset is preserved
        assert b - a == d


# --------------------------------------------------------------------------- #
# repetition span gate + drift
# --------------------------------------------------------------------------- #
def _burst(rng, n=1200, decay=200.0):
    return rng.standard_normal(n) * np.exp(-np.arange(n) / decay)


def test_vote_support_span_is_wide_when_matches_cover_the_clip():
    """Healthy pairs: winning votes draw support from all along the recording."""
    a, b, _ = _periodic_scene()
    v = vote_offset(a, b, params=P)
    assert v.support_span_s >= P.min_vote_span_s


def test_vote_support_span_flags_single_region_support():
    """Shared content confined to one ~1.4 s patch -> tiny span, the signature of
    a repetition false lock (Six & Leman's 17/1000; Cotton & Ellis's
    single-region hash chains). The offset itself is right here, and the span
    still falls below the gate.
    """
    rng = np.random.default_rng(3)
    n = int(20.0 * SR)
    a = 30.0 * rng.standard_normal(n)
    b = 30.0 * rng.standard_normal(n)
    off = int(2.0 * SR)
    for k in range(5):
        u = 3000.0 * _burst(rng)
        i = int((9.0 + 0.35 * k) * SR)
        a[i : i + len(u)] += u
        b[i - off : i - off + len(u)] += u
    v = vote_offset(a, b, params=P)
    assert abs(v.offset_s - 2.0) < 0.1
    assert v.support_span_s < P.min_vote_span_s


def test_estimate_drift_recovers_an_imposed_rate():
    """A 200 ppm resample of one track reads back as ~200 ppm with a CI that
    excludes zero; the undrifted control reads ~0 and stays insignificant."""
    rng = np.random.default_rng(7)
    dur = 60.0
    n = int(dur * SR)
    a = 30.0 * rng.standard_normal(n)
    for k in range(40):
        u = 3000.0 * _burst(rng)
        i = int((0.7 + 1.45 * k) * SR)
        a[i : i + len(u)] += u
    eps = 200e-6
    b = np.interp(np.arange(n) * (1.0 + eps), np.arange(n), a)
    b = b + 5.0 * rng.standard_normal(n)

    d = estimate_drift(a, b, 0.0, params=P, window_s=10.0, hop_s=5.0)
    assert d is not None and d.n_windows >= P.drift_min_windows
    assert d.significant
    assert abs(abs(d.slope_ppm) - 200.0) < 60.0

    b0 = a + 5.0 * rng.standard_normal(n)
    d0 = estimate_drift(a, b0, 0.0, params=P, window_s=10.0, hop_s=5.0)
    assert d0 is not None
    assert abs(d0.slope_ppm) < 20.0
    assert not d0.significant


def test_gate_wiring_includes_span_and_drift(monkeypatch):
    """estimate_raw_offset carries the span and drift gates; a short clip's
    unmeasurable drift is a PASS (never a guess), and the span gate mirrors its
    diagnostic."""
    import kiwi.sync.audio as mod

    a, b, _ = _periodic_scene()
    monkeypatch.setattr(mod, "load_audio", lambda p, **k: a if "a" in str(p) else b)
    monkeypatch.setattr(mod, "probe_media", lambda p, **k: _probe(X5, str(p)))
    raw = mod.estimate_raw_offset("a.wav", "b.wav")
    assert set(raw.gates) == set(mod.GATES)
    assert "onset_agreement" not in raw.gates
    assert 0 < raw.onsets_total <= P.onset_count
    assert raw.drift is None
    assert raw.gates["measured_drift_acceptable"] is True
    assert raw.diagnostics["drift_bounded"] is False
    assert raw.diagnostics["confidence"]["drift_supported"] is False
    assert raw.gates["vote_support_span"] == (
        raw.diagnostics["vote_support_span_s"] >= P.min_vote_span_s
    )


def test_onset_agreement_is_reported_but_never_gates(monkeypatch):
    """Zero matched onsets on an otherwise healthy pair is still confident:
    a head and a wrist microphone hear different loudest events, so the count
    is recorded for the report and decides nothing."""
    assert "onset_agreement" not in GATES
    raw = dataclasses.replace(
        _raw({name: True for name in GATES}), onsets_matched=0, onsets_total=8
    )
    assert raw.confident and "onset_agreement" not in raw.failed_gates

    a, b, _ = _periodic_scene()
    mod = _stub_clips(monkeypatch, {"ref.insv": a, "other.insv": b})
    monkeypatch.setattr(mod, "onset_agreement", lambda *args, **kwargs: (0, 8))
    raw = mod.estimate_raw_offset("ref.insv", "other.insv")
    assert (raw.onsets_matched, raw.onsets_total) == (0, 8)
    assert raw.confident, raw.failed_gates
    assert raw.diagnostics["confidence"]["level"] != "rejected"
    assert (
        "0/8 matched (advisory, not a gate)"
        in AudioOffset(raw.offset_s, raw, AVCorrection(0.0, "-")).report()
    )


# --- level gates: per clip, on the worse one ------------------------------- #
# Level gates judge each clip on its own: a dead microphone on one camera must
# fail even if the other heard something, and a track pinned at full scale must
# trip the clip gate.


def _stub_clips(monkeypatch, clips: dict[str, np.ndarray]):
    """Serve `load_audio`/`probe_media` from memory, keyed on the EXACT path
    text - a substring match is the trap test_clock.py warns about."""
    import kiwi.sync.audio as mod

    def probe(p, **k):
        # The durations gate compares against the array, not a constant.
        return dataclasses.replace(
            _probe(X5, str(p)), video_duration_s=len(clips[str(p)]) / SR
        )

    monkeypatch.setattr(mod, "load_audio", lambda p, **k: clips[str(p)])
    monkeypatch.setattr(mod, "probe_media", probe)
    return mod


LEVEL_GATES = {"audio_non_silent", "audio_unclipped"}


def test_level_gates_pass_on_a_healthy_pair(monkeypatch):
    a, b, _ = _periodic_scene()
    mod = _stub_clips(monkeypatch, {"ref.insv": a, "other.insv": b})
    raw = mod.estimate_raw_offset("ref.insv", "other.insv")
    assert not LEVEL_GATES & set(raw.failed_gates)
    assert raw.diagnostics["clipped_fraction"] == 0.0


# Every degradation below is applied to EITHER side in turn.
BOTH_SIDES = pytest.mark.parametrize("bad", ["ref.insv", "other.insv"])


def _pair():
    a, b, _ = _periodic_scene()
    return {"ref.insv": a, "other.insv": b}


@BOTH_SIDES
def test_a_silent_clip_fails_audio_non_silent(monkeypatch, bad):
    """One microphone at 1% of full scale while the other is healthy: judged
    on the worse clip, this is silence, and the pair is not confident."""
    clips = _pair()
    clips[bad] = clips[bad] * (0.01 * 32768 / np.abs(clips[bad]).max())
    mod = _stub_clips(monkeypatch, clips)
    raw = mod.estimate_raw_offset("ref.insv", "other.insv")
    assert "audio_non_silent" in raw.failed_gates
    assert "audio_unclipped" not in raw.failed_gates
    assert not raw.confident
    assert raw.diagnostics["min_peak_fraction"] == pytest.approx(0.01)


@BOTH_SIDES
def test_a_dead_microphone_names_the_mic_not_the_pairing(monkeypatch, bad):
    """Exact digital zeros - the case the gate cannot report on its own.
    Silence yields no landmarks, so `vote_offset` raises before any gate dict
    is returned; the error names the dead mic, not the pairing."""
    clips = _pair()
    clips[bad] = np.zeros_like(clips[bad])
    mod = _stub_clips(monkeypatch, clips)
    with pytest.raises(RuntimeError, match="audio_non_silent") as e:
        mod.estimate_raw_offset("ref.insv", "other.insv")
    assert bad in str(e.value)
    assert "do not overlap" not in str(e.value)


@BOTH_SIDES
def test_a_hard_clipped_clip_fails_audio_unclipped(monkeypatch, bad):
    """The last 2% of one clip pinned at +-32767 (an overloaded mic): 20x the
    threshold."""
    clips = _pair()
    pinned = clips[bad].copy()
    n = int(0.02 * len(pinned))
    t = np.arange(n) / SR
    pinned[-n:] = np.where(np.sin(2 * np.pi * 100.0 * t) >= 0, 32767.0, -32767.0)
    clips[bad] = pinned
    mod = _stub_clips(monkeypatch, clips)
    raw = mod.estimate_raw_offset("ref.insv", "other.insv")
    assert "audio_unclipped" in raw.failed_gates
    assert "audio_non_silent" not in raw.failed_gates
    assert not raw.confident
    assert raw.diagnostics["clipped_fraction"] == pytest.approx(0.02, rel=1e-3)


@BOTH_SIDES
def test_a_handful_of_full_scale_samples_passes_audio_unclipped(monkeypatch, bad):
    """A few isolated peaks touching full scale are a loud room, not a pinned
    track: the gate is a fraction, not a maximum."""
    clips = _pair()
    clips[bad] = clips[bad].copy()
    clips[bad][[1000, 2000, 3000, 4000, 5000]] = 32767.0
    mod = _stub_clips(monkeypatch, clips)
    raw = mod.estimate_raw_offset("ref.insv", "other.insv")
    assert not LEVEL_GATES & set(raw.failed_gates)
    assert 0 < raw.diagnostics["clipped_fraction"] < P.clip_frac_max


def test_the_level_thresholds_come_from_sync_params(monkeypatch):
    """Both fields must actually reach the gates, so the test sets them away
    from their defaults."""
    clips = _pair()
    clips["other.insv"] = clips["other.insv"].copy()
    clips["other.insv"][[1000, 2000, 3000, 4000, 5000]] = 32767.0
    mod = _stub_clips(monkeypatch, clips)

    strict = mod.estimate_raw_offset(
        "ref.insv", "other.insv", params=SyncParams(silence_peak_frac=0.99)
    )
    assert "audio_non_silent" in strict.failed_gates

    strict = mod.estimate_raw_offset(
        "ref.insv", "other.insv", params=SyncParams(clip_frac_max=0.0)
    )
    assert "audio_unclipped" in strict.failed_gates


# --- the A/V table resolves through the registry pointer ---------------------
#
# A unit's `camera.yaml` names its current A/V table explicitly, and
# `registry.ARTIFACT_NAMES` registers `av_sync` so this module resolves through
# `CameraUnit.artifact` instead of hardcoding the filename.


def test_av_delay_follows_the_camera_yaml_pointer(tmp_path):
    """A dated file named by the pointer wins over the unversioned name."""
    d = tmp_path / "insta360-GO3-001"
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text(
        yaml.safe_dump({"model": "Insta360 GO 3", "av_sync": "av-sync-2026-07-25.yaml"})
    )
    (d / "av-sync-2026-07-25.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": AV_SYNC_SCHEMA,
                "unit": "insta360-GO3-001",
                "extraction": EXTRACTION,
                "modes": {GO.key: {"av_delay_s": 0.059}},
            }
        )
    )
    # A stale unversioned file must NOT win.
    (d / "av-sync.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": AV_SYNC_SCHEMA,
                "unit": "insta360-GO3-001",
                "extraction": EXTRACTION,
                "modes": {GO.key: {"av_delay_s": -99.0}},
            }
        )
    )
    assert read_av_delay(d, GO).seconds == pytest.approx(0.059)


def test_av_delay_dangling_pointer_is_loud(tmp_path):
    """A pointer naming a missing file is a broken registry, not a fallback -
    and a RuntimeError, not the registry's FileNotFoundError: the sync stage
    catches only what this module promises."""
    d = tmp_path / "insta360-GO3-001"
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text(
        yaml.safe_dump({"model": "Insta360 GO 3", "av_sync": "av-sync-2026-01-01.yaml"})
    )
    (d / "av-sync.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": AV_SYNC_SCHEMA,
                "extraction": EXTRACTION,
                "modes": {GO.key: {"av_delay_s": 0.0}},
            }
        )
    )
    with pytest.raises(RuntimeError, match="av-sync-2026-01-01.yaml") as e:
        read_av_delay(d, GO)
    assert isinstance(e.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("corrupt", ["camera.yaml", "av-sync.yaml"])
def test_av_delay_corrupt_yaml_is_a_runtime_error(tmp_path, corrupt):
    """A table (or camera.yaml) that will not parse: neither yaml.YAMLError nor
    the registry's ValueError is RuntimeError, so both are translated here."""
    d = tmp_path / "insta360-GO3-001"
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text(yaml.safe_dump({"model": "Insta360 GO 3"}))
    (d / "av-sync.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": AV_SYNC_SCHEMA,
                "extraction": EXTRACTION,
                "modes": {GO.key: {"av_delay_s": 0.0}},
            }
        )
    )
    (d / corrupt).write_text("modes: {unterminated: [\n")
    with pytest.raises(RuntimeError) as e:
        read_av_delay(d, GO)
    root = e.value.__cause__
    while root.__cause__ is not None:
        root = root.__cause__
    assert isinstance(root, yaml.YAMLError)
    assert str(d / corrupt) in str(e.value)


def test_av_delay_missing_table_names_the_calib_route(tmp_path):
    """The 'never measured' error must point at kiwi calib."""
    d = tmp_path / "insta360-GO3-001"
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text(yaml.safe_dump({"model": "Insta360 GO 3"}))
    with pytest.raises(RuntimeError, match="kiwi calib"):
        read_av_delay(d, GO)


# --------------------------------------------------------------------------- #
# the shipped registry data: the real config/ tree resolves
# --------------------------------------------------------------------------- #
# The tmp_path tests above prove the mechanism; these prove the DATA - the
# tables committed under config/cameras/ carry the measured constants, reach
# the reader through the camera.yaml pointer, and refuse what was never
# measured. A retuned value fails here before it ships.

CONFIG_CAMERAS = Path(__file__).resolve().parents[2] / "config" / "cameras"


def test_shipped_go3_table_serves_the_ego_mode():
    """The ego clips the published dataset ships are 2688x2688@29.97.

    Capture mode: VideoSuperNormal, 2688x2688 cropped from a 3040x3040 sensor
    read, selected output 2560x1440, stabilisation and dewarp off. 2560x1440
    entries share the output resolution, not the capture mode.
    """
    got = read_av_delay(CONFIG_CAMERAS / "insta360-GO3-001", EGO)
    assert got.seconds == 0.0142
    assert got.unit == "insta360-GO3-001"
    assert got.source.name == "av-sync.yaml"


@pytest.mark.parametrize(
    "mode, seconds",
    [
        (RecordingMode(2720, 1536, 30000 / 1001), 0.0625),
        (RecordingMode(2560, 1440, 30000 / 1001), 0.0256),
        (RecordingMode(2560, 1440, 50.0), -0.0047),
    ],
)
def test_shipped_go3_table_carries_the_other_measured_modes(mode, seconds):
    assert read_av_delay(CONFIG_CAMERAS / "insta360-GO3-001", mode).seconds == seconds


def test_shipped_mixed_x5_go3_pair_resolves_through_the_real_tables():
    """The Config 2 headline pair: X5 wrist (reference mode) vs GO3 ego.

    Uses the mode the shipped ego clips are in.
    """
    ref = Clip("x5.insv", CONFIG_CAMERAS / "insta360-X5-001", "left")
    other = Clip("ego.mp4", CONFIG_CAMERAS / "insta360-GO3-001", "ego")
    corr = resolve_av_correction(
        ref, _probe(X5, path="x5.insv"), other, _probe(EGO, path="ego.mp4")
    )
    # other_delay - ref_delay = +0.0142 - 0.0 (the X5 reference IS the zero).
    assert corr.seconds == 0.0142
    assert corr.ref is not None and corr.ref.seconds == 0.0
    assert corr.other is not None and corr.other.seconds == 0.0142


@pytest.mark.parametrize(
    "unit",
    ["insta360-X5-001", "insta360-X5-002", "insta360-X5-003", "insta360-X5-004"],
)
def test_shipped_x5_tables_refuse_a_non_reference_mode(unit):
    """The X5 tables list ONLY the defined-zero mode: any other mode was never
    measured. This is the operating constraint on mixed sessions - the X5 arms
    must record 1920x1920@119.88."""
    assert read_av_delay(CONFIG_CAMERAS / unit, X5).seconds == 0.0
    with pytest.raises(RuntimeError, match="Known modes"):
        read_av_delay(CONFIG_CAMERAS / unit, RecordingMode(1920, 1920, 30000 / 1001))

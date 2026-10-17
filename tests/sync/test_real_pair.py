"""The recipe against REAL recorded audio, not a synthetic scene.

test_audio_sync.py proves the recipe's properties on signals we constructed;
this proves it still lands the right answer on two cameras that actually
recorded a room, against an *optical* ground truth measured independently of
audio. The excerpt comes from a rolling-QR sync capture: both cameras filmed
one monitor showing a rolling QR clock, and fitting each camera to that clock
gives the true cross-camera offset to within a millisecond.

The excerpts are low-passed at 150 Hz before being committed, so they cannot
carry intelligible speech - the lowest vowel formant sits near 250 Hz. The
filter is zero-phase and identical on both clips, so the lag between them is
unchanged. `test_fixture_is_band_limited` enforces it.

No ffmpeg: `load_audio` and `probe_media` are the module's only two subprocess
boundaries, and both are module globals, so the fixture is pre-decoded samples.
See fixtures/README.md for exactly how they were cut.

Run: pixi run test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kiwi.sync.audio as mod
from kiwi.sync.audio import MediaProbe, RecordingMode, SyncParams

FIXTURES = Path(__file__).parent / "fixtures"

SR = 48_000
# Both clips are Insta360 X5, 1920x1920 at 119.88 - the same recording mode, so
# the A/V correction cancels to exactly 0.0 and no registry table is consulted.
MODE = RecordingMode(1920, 1920, 120000 / 1001)

# What the estimator returns on the two WHOLE clips (55.7 s / 60.1 s).
FULL_CLIP_OFFSET_S = -4.3624

# The optical truth: both cameras fitted to one rolling-QR clock, giving
# -4360.4964 ms with a standard error of 0.5684 ms.
QR_GROUND_TRUTH_S = -4.3604964
QR_STANDARD_ERROR_S = 0.0005684

# The x5b excerpt is cut this many samples later than the x5a one, so the two
# 12 s windows overlap almost entirely instead of losing 4.36 s of shared
# content to the offset. See fixtures/README.md.
CUT_SHIFT_SAMPLES = 161395
EXPECTED_OFFSET_S = FULL_CLIP_OFFSET_S + CUT_SHIFT_SAMPLES / SR  # ~= -1.000004


def _audio(name: str) -> np.ndarray:
    """Load a fixture exactly as `load_audio` would hand it over: int16-scaled
    float64 mono at 48 kHz. Normalising to +-1.0 here would trip the
    `audio_non_silent` gate, which reads absolute sample values."""
    with np.load(FIXTURES / f"sync_{name}.npz") as z:
        return z["audio"].astype(np.float64)


def _probe(arr: np.ndarray) -> MediaProbe:
    # video_duration_s must track the array, not a constant: the
    # `audio_video_durations_agree` gate compares them within 150 ms.
    return MediaProbe(
        path=Path("x.insv"),
        mode=MODE,
        video_duration_s=len(arr) / SR,
        audio_codec="aac",
        audio_sample_rate=SR,
        audio_channels=1,
    )


@pytest.fixture
def real_pair(monkeypatch):
    """Drive the estimator from the fixtures, with no ffmpeg anywhere."""
    arrays = {"x5a": _audio("x5a"), "x5b": _audio("x5b")}
    probes = {k: _probe(v) for k, v in arrays.items()}
    monkeypatch.setattr(mod, "load_audio", lambda p, **k: arrays[str(p)])
    monkeypatch.setattr(mod, "probe_media", lambda p, **k: probes[str(p)])
    return arrays


def test_recovers_the_known_offset(real_pair):
    raw = mod.estimate_raw_offset("x5a", "x5b")
    assert raw.offset_s == pytest.approx(EXPECTED_OFFSET_S, abs=0.005)


def test_every_gate_passes(real_pair):
    raw = mod.estimate_raw_offset("x5a", "x5b")
    assert set(raw.gates) == set(mod.GATES)
    assert raw.confident, f"failed: {raw.failed_gates}\n{raw.report()}"
    # Advisory, not a gate, but two X5 arms hear the same loudest events, so
    # the recorded count still agrees on a healthy same-model pair.
    assert raw.onsets_matched >= 2


def test_the_gates_pass_with_margin_not_by_a_hair(real_pair):
    """The excerpt clears both tight gates with 2x headroom."""
    p = SyncParams()
    raw = mod.estimate_raw_offset("x5a", "x5b")
    assert raw.votes >= 2.0 * p.vote_margin * raw.runner_up
    assert raw.diagnostics["vote_support_span_s"] >= 2.0 * p.min_vote_span_s


def test_corrected_for_the_cut_it_reproduces_the_full_clip_answer(real_pair):
    """The excerpt is anchored, not free-floating: undo the documented cut
    shift and the whole-clip offset comes back."""
    raw = mod.estimate_raw_offset("x5a", "x5b")
    recovered = raw.offset_s - CUT_SHIFT_SAMPLES / SR
    assert recovered == pytest.approx(FULL_CLIP_OFFSET_S, abs=0.005)


def test_it_agrees_with_the_optical_ground_truth(real_pair):
    """The claim that actually matters: audio agrees with photons.

    Both cameras filmed one rolling-QR clock, which pins the true offset to
    -4.3604964 s with a 0.57 ms standard error. The estimator must recover it
    from sound alone to within 5 ms.
    """
    raw = mod.estimate_raw_offset("x5a", "x5b")
    recovered = raw.offset_s - CUT_SHIFT_SAMPLES / SR
    assert recovered == pytest.approx(QR_GROUND_TRUTH_S, abs=0.005)
    # Sanity: the optical truth is far more precise than the tolerance above,
    # so a regression cannot hide behind a loose ground truth.
    assert QR_STANDARD_ERROR_S < 0.001


def test_the_coarse_vote_alone_finds_the_right_period(real_pair):
    """On real audio the coarse vote stage alone must pick the true period."""
    a, b = real_pair["x5a"], real_pair["x5b"]
    v = mod.vote_offset(a, b, params=SyncParams())
    assert abs(v.offset_s - EXPECTED_OFFSET_S) <= SyncParams().vote_bin_s


def test_same_mode_pair_needs_no_av_table(real_pair, tmp_path):
    """Both clips are the same camera model in the same mode, so the A/V delay
    is common-mode and cancels exactly - the video offset equals the audio one
    and no A/V table is needed; both models are explicitly established."""
    unit = tmp_path / "x5"
    unit.mkdir()
    (unit / "camera.yaml").write_text("model: Insta360 X5\n")
    res = mod.estimate_offset(mod.Clip(Path("x5a"), unit), mod.Clip(Path("x5b"), unit))
    assert res.correction.seconds == 0.0
    assert res.offset_s == res.raw.offset_s


def test_fixture_is_band_limited(real_pair):
    """The privacy guarantee.

    The samples are low-passed at 150 Hz so no formant information survives
    (the lowest vowel formant is around 250 Hz), which makes intelligible
    speech impossible by construction. If the fixture is ever regenerated
    without the low-pass, this fails.
    """
    for name, arr in real_pair.items():
        power = np.abs(np.fft.rfft(arr * np.hanning(len(arr)))) ** 2
        freq = np.fft.rfftfreq(len(arr), 1 / SR)
        speech_band = power[(freq >= 300) & (freq <= 3400)].sum() / power.sum()
        above = power[freq > 3400].sum() / power.sum()
        assert speech_band < 1e-4, (
            f"{name}: {speech_band:.2%} of energy in the speech band"
        )
        assert above < 1e-4, f"{name}: {above:.2%} of energy above 3.4 kHz"


def test_the_pair_still_defeats_a_raw_argmax(real_pair):
    """Filtering must not have made the problem easy: a raw global GCC-PHAT
    argmax must still false-lock on this fixture.
    """
    raw = mod.estimate_raw_offset("x5a", "x5b")
    assert raw.diagnostics["argmax_is_false_lock"] is True

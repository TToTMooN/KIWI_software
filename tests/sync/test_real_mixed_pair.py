"""The mixed-model path (X5 -> GO 3) against real audio and an optical truth.

test_real_pair.py covers a same-model pair, where the A/V correction cancels
to exactly 0.0 and the registry is never consulted. This file covers the
per-mode ``av_delay_s`` table with the third camera from the same capture: the
Insta360 GO 3 ego camera, whose 2560x1440@29.97 constant (+25.6 ms, measured
on a separate capture) must move this capture's audio estimate onto its
independently measured video timeline.

The GO 3's optical truth comes from the OCR text-clock fallback: the clock's
text line, a Theil-Sen fit, standard error 2.4 ms.

These fixture tests seed :func:`refine_offset` with the documented cut
geometry and pin the correlation refinement on real cross-device audio, and
the shipped table applied through the real registry, end to end against the
optical truth. The coarse vote on cross-device recordings is not covered here.

Run: pixi run test
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import kiwi.sync.audio as mod
from kiwi.sync.audio import Clip, MediaProbe, RecordingMode, SyncParams

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_CAMERAS = Path(__file__).resolve().parents[2] / "config" / "cameras"

SR = 48_000

# Modes exactly as ffprobe reports the source clips: the X5s landscape-square,
# the GO 3 stored portrait 1440x2560 with a -90 display matrix. Keying the
# table through the rotated form exercises the display normalization that
# production ego clips (stored landscape) and calibration clips (stored
# portrait) must agree on.
X5_MODE = RecordingMode(1920, 1920, 120000 / 1001)
GO3_MODE = RecordingMode(1440, 2560, 30000 / 1001, rotation_deg=-90)

# Units matched by embedded serial (IAHYA2511CWJEA / IAHYA25116W5XY), never by
# directory name - the capture paired X5-001 with X5-003.
X5A_UNIT = CONFIG_CAMERAS / "insta360-X5-001"
X5B_UNIT = CONFIG_CAMERAS / "insta360-X5-003"
GO3_UNIT = CONFIG_CAMERAS / "insta360-GO3-001"

# Whole-clip raw audio offsets measured on the source recordings. Anchors for
# the cut geometry only - the accuracy claims below are against the optical
# truth, not against these.
RAW_X5A_GO3_S = -5.8691
RAW_X5B_GO3_S = -1.5064

# The optical truth: each camera fitted to the filmed clock, then
# differenced. OCR text-clock fallback, standard error 2.4 ms.
QR_X5A_GO3_S = -5.8436
QR_X5B_GO3_S = -1.4831
TEXT_CLOCK_SE_S = 0.0024
# The x5 pair's QR-decode truth (see test_real_pair.py), for the triangle test.
QR_X5A_X5B_S = -4.3604964

# Cut geometry (fixtures/README.md): all three excerpts are 12 s windows
# starting 36.0 s into x5a's file, shifted per clip to keep the shared content.
GO3_CUT_SHIFT_SAMPLES = 233_717  # go3 cut vs x5a's; expected offset ~ -1.0 s
X5B_CUT_SHIFT_SAMPLES = 161_395  # x5b cut vs x5a's (matches test_real_pair)

TOL_S = 0.010


def _audio(name: str) -> np.ndarray:
    """int16-scaled float64 mono at 48 kHz, as `load_audio` would hand over."""
    with np.load(FIXTURES / f"sync_{name}.npz") as z:
        return z["audio"].astype(np.float64)


def _probe(arr: np.ndarray, mode: RecordingMode) -> MediaProbe:
    return MediaProbe(
        path=Path("x"),
        mode=mode,
        video_duration_s=len(arr) / SR,
        audio_codec="aac",
        audio_sample_rate=SR,
        audio_channels=1,
    )


@pytest.fixture(scope="module")
def pairs():
    """Both mixed pairs: refine from the documented coarse, correct via the
    real registry. No subprocess boundary is touched anywhere in this path."""
    go3 = _audio("go3")
    go3_probe = _probe(go3, GO3_MODE)
    go3_clip = Clip(Path("go3"), GO3_UNIT)
    out = {}
    for key, x5_name, unit, raw_whole_s, shift in (
        ("a", "x5a", X5A_UNIT, RAW_X5A_GO3_S, GO3_CUT_SHIFT_SAMPLES),
        (
            "b",
            "x5b",
            X5B_UNIT,
            RAW_X5B_GO3_S,
            GO3_CUT_SHIFT_SAMPLES - X5B_CUT_SHIFT_SAMPLES,
        ),
    ):
        x5 = _audio(x5_name)
        expected_s = raw_whole_s + shift / SR
        refined = mod.refine_offset(x5, go3, expected_s, params=SyncParams())
        corr = mod.resolve_av_correction(
            Clip(Path(x5_name), unit), _probe(x5, X5_MODE), go3_clip, go3_probe
        )
        audio_offset_s = refined.offset_s - shift / SR
        out[key] = SimpleNamespace(
            refined=refined,
            corr=corr,
            audio_offset_s=audio_offset_s,
            video_offset_s=audio_offset_s + corr.seconds,
        )
    return out


def test_x5a_go3_recovers_the_optical_truth(pairs):
    """The claim this file exists for: real cross-device audio, corrected by
    the shipped constant (measured on a separate capture), lands on this
    capture's independently measured video timeline."""
    assert pairs["a"].video_offset_s == pytest.approx(QR_X5A_GO3_S, abs=TOL_S)
    # The truth is far more precise than the tolerance, so a regression
    # cannot hide behind a loose ground truth.
    assert TEXT_CLOCK_SE_S < TOL_S / 4


def test_x5b_go3_recovers_the_optical_truth(pairs):
    """The same claim through the other X5 body (X5-003), whose truth chain
    and audio content are independent of x5a's."""
    assert pairs["b"].video_offset_s == pytest.approx(QR_X5B_GO3_S, abs=TOL_S)


def test_the_shipped_correction_is_load_bearing(pairs):
    """The raw audio answer alone is more than 15 ms off the video timeline -
    the table is what closes it. If the correction ever silently degrades to
    zero, both halves of this test fail, not just the accuracy one."""
    for p, truth in ((pairs["a"], QR_X5A_GO3_S), (pairs["b"], QR_X5B_GO3_S)):
        assert abs(p.audio_offset_s - truth) > 0.015
        assert abs(p.video_offset_s - truth) < TOL_S


def test_correction_resolves_through_the_real_registry(pairs):
    """The constant reaches the estimate from config/cameras/, not from a
    test stub."""
    for p in pairs.values():
        assert p.corr.seconds == pytest.approx(0.0256)
        assert p.corr.ref is not None and p.corr.ref.seconds == 0.0
        assert p.corr.other is not None
        assert p.corr.other.unit == "insta360-GO3-001"
        assert p.corr.other.mode_key == "2560x1440@29.97"


def test_triangle_closes_on_the_x5_pair_optical_truth(pairs):
    """(x5a->go3) - (x5b->go3) must equal x5a->x5b. The GO 3 correction is
    common to both legs and cancels in the difference, so this closes against
    the x5 pair's own sub-ms QR truth."""
    closure = pairs["a"].video_offset_s - pairs["b"].video_offset_s
    assert closure == pytest.approx(QR_X5A_X5B_S, abs=TOL_S)


def test_refine_is_stable_across_the_vote_bin():
    """Refine's contract is polishing a coarse that is right to within one
    50 ms vote bin. Seeded anywhere in +-40 ms, it must land on the same
    answer - on this real pair, not just on synthetic signals."""
    x5a, go3 = _audio("x5a"), _audio("go3")
    expected_s = RAW_X5A_GO3_S + GO3_CUT_SHIFT_SAMPLES / SR
    offsets = {
        mod.refine_offset(x5a, go3, expected_s + seed, params=SyncParams()).offset_s
        for seed in (-0.04, -0.02, 0.0, 0.02, 0.04)
    }
    assert len(offsets) == 1, f"refine wandered with the seed: {sorted(offsets)}"


def test_go3_fixture_is_band_limited():
    """The same privacy guarantee test_real_pair enforces for the x5 pair:
    low-passed at 150 Hz, so no formant survives and intelligible speech is
    impossible by construction."""
    arr = _audio("go3")
    power = np.abs(np.fft.rfft(arr * np.hanning(len(arr)))) ** 2
    freq = np.fft.rfftfreq(len(arr), 1 / SR)
    speech_band = power[(freq >= 300) & (freq <= 3400)].sum() / power.sum()
    above = power[freq > 3400].sum() / power.sum()
    assert speech_band < 1e-4, f"{speech_band:.2%} of energy in the speech band"
    assert above < 1e-4, f"{above:.2%} of energy above 3.4 kHz"

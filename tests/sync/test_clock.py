"""Unit tests for kiwi.sync.clock (the offset algebra + multi-stream resolve).

The estimator itself is covered by test_audio_sync.py; here it is replaced by a
table of known pairwise offsets, so these tests pin the *algebra*: which stream
owns the origin, the sign composition, and the loop check that stops a false
lock from shipping. No audio, no ffmpeg.

Run: pixi run test
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kiwi.core.clock import GlobalClock
from kiwi.sync import clock as clock_mod
from kiwi.sync.audio import AVCorrection
from kiwi.sync.clock import (
    LOOP_TOL_S,
    SyncGateError,
    resolve_clock,
    to_last_start_origin,
)


# --- to_last_start_origin ---------------------------------------------------


def test_origin_is_the_last_stream_to_start():
    # Bigger input value = started later = wins the origin and holds 0.0.
    placed, origin = to_last_start_origin({"left": 0.0, "right": 2.5})
    assert origin == "right"
    assert placed == {"left": -2.5, "right": 0.0}


def test_an_extra_camera_can_own_the_origin():
    # The reason phase 1 cannot pick the origin: the ego stream started last.
    placed, origin = to_last_start_origin({"left": 0.0, "right": 1.2, "ego": 3.4})
    assert origin == "ego"
    assert placed["ego"] == 0.0
    assert placed["left"] == pytest.approx(-3.4)
    assert placed["right"] == pytest.approx(-2.2)


def test_output_always_satisfies_the_globalclock_contract():
    placed, origin = to_last_start_origin({"a": 0.0, "b": 7.25, "c": 3.0})
    GlobalClock(origin=origin, offsets_s=placed).validate()  # must not raise


def test_ties_land_at_exactly_zero_and_break_lexicographically():
    placed, origin = to_last_start_origin({"right": 1.0, "left": 1.0})
    assert origin == "left"  # deterministic across re-runs
    assert placed == {"left": 0.0, "right": 0.0}


def test_empty_offsets_rejected():
    with pytest.raises(ValueError, match="empty offsets"):
        to_last_start_origin({})


# --- resolve_clock: the estimator is stubbed out ----------------------------


class _FakeOffset:
    """Stands in for AudioOffset: only what resolve_clock touches."""

    def __init__(self, offset_s, confident=True, failed=()):
        self.offset_s = offset_s
        self.confident = confident
        self.failed_gates = list(failed)
        # resolve_clock serializes these into SyncEdge records.
        self.correction = AVCorrection(seconds=0.0, provenance="(stub: same mode)")
        mode = SimpleNamespace(key="1920x1920@119.88")
        self.raw = SimpleNamespace(
            ref=SimpleNamespace(mode=mode), other=SimpleNamespace(mode=mode)
        )

    def report(self):
        return f"(stub report for {self.offset_s:+.4f}s)"


def _stub_estimator(monkeypatch, table, *, unconfident=()):
    """Patch estimate_offset with a lookup keyed on (ref_label, other_label).

    Keying on the label - not the path - avoids a substring-matching trap,
    where a path containing "a" would silently return the wrong clip.
    """
    seen = []

    def fake(ref, other, *, params=None, _cache=None):
        key = (ref.label, other.label)
        seen.append(key)
        if key in unconfident:
            return _FakeOffset(table[key], confident=False, failed=["vote_margin"])
        return _FakeOffset(table[key])

    monkeypatch.setattr(clock_mod, "estimate_offset", fake)
    return seen


def _clip(label):
    return clock_mod.Clip(Path(f"{label}.insv"), None, label)


def _arms():
    return {"left": _clip("left"), "right": _clip("right")}


def test_arm_pair_only(monkeypatch):
    # right started 2.0 s after left -> left is earlier -> right owns the origin.
    _stub_estimator(monkeypatch, {("left", "right"): 2.0})
    est = resolve_clock(_arms())
    assert est.clock.origin == "right"
    assert est.clock.offsets_s == {"left": -2.0, "right": 0.0}
    assert est.method == "audio_gcc_phat"
    assert est.loop_residuals == {}


def test_sign_convention_composes_the_other_way(monkeypatch):
    # Negative offset: `right` started EARLIER, so `left` owns the origin.
    _stub_estimator(monkeypatch, {("left", "right"): -2.0})
    est = resolve_clock(_arms())
    assert est.clock.origin == "left"
    assert est.clock.offsets_s == {"left": 0.0, "right": -2.0}


def test_extra_stream_is_placed_and_loop_checked(monkeypatch):
    # left at 0, right at +2 (pair frame). ego is +3 from left, so +1 from right;
    # both routes agree, and ego started last so it takes the origin.
    seen = _stub_estimator(
        monkeypatch,
        {
            ("left", "right"): 2.0,
            ("left", "ego"): 3.0,
            ("right", "ego"): 1.0,
        },
    )
    est = resolve_clock(_arms(), {"ego": _clip("ego")})
    assert est.clock.origin == "ego"
    assert est.clock.offsets_s["ego"] == 0.0
    assert est.clock.offsets_s["left"] == pytest.approx(-3.0)
    assert est.clock.offsets_s["right"] == pytest.approx(-1.0)
    # The loop check really ran: ego was estimated against BOTH arms.
    assert ("left", "ego") in seen and ("right", "ego") in seen
    assert est.loop_residuals["ego"] == pytest.approx(0.0, abs=1e-9)


def test_extra_hangs_off_the_earliest_arm_not_the_origin(monkeypatch):
    """The reference for phase 1b is the phase-1 zero, not the clock origin.

    Here `right` ends up owning the origin, but `ego` must still be measured
    against `left` (the earlier arm), because the origin is not yet knowable.
    """
    seen = _stub_estimator(
        monkeypatch,
        {("left", "right"): 5.0, ("left", "ego"): 1.0, ("right", "ego"): -4.0},
    )
    resolve_clock(_arms(), {"ego": _clip("ego")})
    assert seen[1] == ("left", "ego"), "extra was not placed against the earliest arm"


def test_loop_disagreement_halts(monkeypatch):
    # Via left, ego sits at +3.0; via right it sits at 2.0 + 0.5 = +2.5.
    # 500 ms apart is far outside an all-audio triangle: one edge false-locked.
    _stub_estimator(
        monkeypatch,
        {("left", "right"): 2.0, ("left", "ego"): 3.0, ("right", "ego"): 0.5},
    )
    with pytest.raises(SyncGateError, match="disagree by 500 ms"):
        resolve_clock(_arms(), {"ego": _clip("ego")})


def test_loop_residual_inside_tolerance_passes(monkeypatch):
    resid = LOOP_TOL_S * 0.5
    _stub_estimator(
        monkeypatch,
        {
            ("left", "right"): 2.0,
            ("left", "ego"): 3.0,
            ("right", "ego"): 1.0 + resid,
        },
    )
    est = resolve_clock(_arms(), {"ego": _clip("ego")})
    assert est.loop_residuals["ego"] == pytest.approx(resid)


def test_failed_gates_on_the_arm_edge_halt(monkeypatch):
    _stub_estimator(
        monkeypatch, {("left", "right"): 2.0}, unconfident={("left", "right")}
    )
    with pytest.raises(SyncGateError) as e:
        resolve_clock(_arms())
    assert e.value.failed_gates == ("vote_margin",)
    assert "gates FAILED" in str(e.value)


def test_failed_gates_on_a_loop_edge_also_halt(monkeypatch):
    """An estimate that failed its own gates can neither vouch for nor veto the
    primary placement, so an unusable cross-check halts rather than being
    quietly skipped."""
    _stub_estimator(
        monkeypatch,
        {("left", "right"): 2.0, ("left", "ego"): 3.0, ("right", "ego"): 1.0},
        unconfident={("right", "ego")},
    )
    with pytest.raises(SyncGateError, match="loop check"):
        resolve_clock(_arms(), {"ego": _clip("ego")})


def test_single_reference_stream_says_it_could_not_cross_check(monkeypatch):
    _stub_estimator(monkeypatch, {("left", "ego"): 1.0})
    est = resolve_clock({"left": _clip("left")}, {"ego": _clip("ego")})
    assert est.clock.origin == "ego"
    assert est.loop_residuals == {}
    assert any("no loop check possible" in r for r in est.reports)


def test_unmeasurable_pair_becomes_a_gate_error_with_a_pairing_hint(monkeypatch):
    def boom(ref, other, *, params=None, _cache=None):
        raise RuntimeError("clips do not overlap")

    monkeypatch.setattr(clock_mod, "estimate_offset", boom)
    with pytest.raises(SyncGateError, match="mis-sorted"):
        resolve_clock(_arms())


def test_rejects_a_stream_listed_twice():
    with pytest.raises(ValueError, match="both arm and extra"):
        resolve_clock(_arms(), {"left": _clip("left")})


def test_rejects_no_reference_streams():
    with pytest.raises(ValueError, match="no reference streams"):
        resolve_clock({})


def test_rejects_three_reference_streams():
    arms = {**_arms(), "third": _clip("third")}
    with pytest.raises(ValueError, match="expected 1 or 2"):
        resolve_clock(arms)


def test_summary_line_is_one_line(monkeypatch):
    _stub_estimator(monkeypatch, {("left", "right"): 2.0})
    s = resolve_clock(_arms()).summary()
    assert "\n" not in s
    assert "origin=right" in s and "left=-2.0000" in s


# --- the structured edge record ----------------------------------------------


def test_every_measured_edge_is_recorded_structured(monkeypatch):
    """One SyncEdge per _estimate call - the arm pair, the extra's placement,
    and the loop check - in measurement order, carrying the A/V correction."""
    _stub_estimator(
        monkeypatch,
        {("left", "right"): 2.0, ("left", "ego"): 1.0, ("right", "ego"): -1.0},
    )
    est = resolve_clock(_arms(), {"ego": _clip("ego")})
    kinds = [(e.kind, e.ref, e.other) for e in est.edges]
    assert kinds == [
        ("arms", "left", "right"),
        ("extra", "left", "ego"),
        ("loop_check", "right", "ego"),
    ]
    for e in est.edges:
        assert e.av_correction_s == 0.0
        assert "stub" in e.av_provenance
        assert e.modes == ("1920x1920@119.88", "1920x1920@119.88")

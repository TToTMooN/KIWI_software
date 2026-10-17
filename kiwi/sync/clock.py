"""The global clock: offset algebra, and multi-stream resolution.

:func:`to_last_start_origin` is the offset algebra proper - it is the ONLY place
the global origin is decided. :func:`resolve_clock` drives the estimator over every
stream of an episode and hands back a validated
:class:`kiwi.core.clock.GlobalClock`.

This module knows nothing about episode layout: it takes labelled clips and
returns labelled offsets. Finding the clips and persisting the result belong to
:mod:`kiwi.episode`.

An extra stream is placed against one reference arm, then cross-checked via
every other available arm. A loop residual above :data:`LOOP_TOL_S` refuses
the placement. With only one reference arm, no loop check is possible and the
result reports that limitation. Loop closure checks pairwise consistency;
device-specific A/V biases cancel around the loop, so it cannot validate them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from kiwi.core.clock import GlobalClock
from kiwi.sync.audio import (
    AudioCache,
    AudioOffset,
    Clip,
    SyncParams,
    estimate_offset,
    pair_to_global_offsets,
)

__all__ = [
    "LOOP_TOL_S",
    "METHOD",
    "ClockEstimate",
    "SyncGateError",
    "resolve_clock",
    "to_last_start_origin",
]

# Maximum disagreement between an extra stream's direct placement and its
# placement via another reference arm.
LOOP_TOL_S = 0.05

# Provenance recorded with every clock this module produces.
METHOD = "audio_gcc_phat"


class SyncGateError(RuntimeError):
    """An estimate was refused: gates failed, or a loop check did not close.

    Carries the estimator's full diagnostic report so a caller that can afford
    the space (the ``kiwi sync`` verb) can show it, while one that cannot (a
    stage summary line) can fall back to :attr:`failed_gates`.
    """

    def __init__(
        self,
        message: str,
        *,
        report: str = "",
        failed_gates: tuple[str, ...] = (),
        evidence: dict | None = None,
    ):
        super().__init__(message)
        self.report = report
        self.failed_gates = failed_gates
        self.evidence = evidence or {}


@dataclasses.dataclass(frozen=True)
class SyncEdge:
    """One measured audio edge, machine-readable.

    The structured twin of the prose in :attr:`ClockEstimate.reports`: what the
    run record stores so an aggregator never parses the report text. The A/V
    correction and its provenance are the part a consumer cannot re-derive from
    the clock alone - they say whether an ``av_sync`` table was read, and which
    resolution rule fired.
    """

    kind: str  # "arms" | "extra" | "loop_check"
    ref: str
    other: str
    offset_s: float  # VIDEO-timeline offset of this edge (positive = other later)
    av_correction_s: float
    av_provenance: str
    modes: tuple[str, str]  # (ref, other) recording-mode keys
    raw: dict = dataclasses.field(default_factory=dict)


def _edge(kind: str, ref: str, other: str, res: "AudioOffset") -> SyncEdge:
    return SyncEdge(
        kind=kind,
        ref=ref,
        other=other,
        offset_s=res.offset_s,
        av_correction_s=res.correction.seconds,
        av_provenance=res.correction.provenance,
        modes=(res.raw.ref.mode.key, res.raw.other.mode.key),
        raw=dataclasses.asdict(res.raw)
        if dataclasses.is_dataclass(res.raw)
        else vars(res.raw),
    )


@dataclasses.dataclass(frozen=True)
class ClockEstimate:
    """A resolved clock plus the evidence behind it."""

    clock: GlobalClock
    method: str
    reports: tuple[str, ...]  # one per estimated edge, in the order measured
    loop_residuals: dict[str, float]  # extra stream label -> worst loop residual (s)
    edges: tuple[SyncEdge, ...] = ()  # structured twin of reports, same order

    def summary(self) -> str:
        """One line, for a stage result or a progress log."""
        offs = "  ".join(
            f"{k}={v:+.4f}" for k, v in sorted(self.clock.offsets_s.items())
        )
        worst = (
            f"; loop <= {max(self.loop_residuals.values()) * 1e3:.1f} ms"
            if self.loop_residuals
            else ""
        )
        limited = [
            f"{edge.ref}->{edge.other}"
            for edge in self.edges
            if edge.raw.get("diagnostics", {}).get("confidence", {}).get("level")
            == "offset_only"
        ]
        confidence = (
            f"; confidence=offset_only ({', '.join(limited)}; drift unverified)"
            if limited
            else ""
        )
        return f"origin={self.clock.origin}  {offs}{worst}{confidence}"

    def match_summaries(self) -> tuple[str, ...]:
        """Per-edge audio duration and match strength for pipeline logs."""
        lines = []
        for edge in self.edges:
            raw = edge.raw
            d = raw.get("diagnostics", {})
            if "matched_audio_coverage_s" not in d:
                continue
            lines.append(
                f"{edge.ref}->{edge.other}: overlap {d['overlap_s']:.1f} s; "
                f"matched ~{d['matched_audio_coverage_s']:.1f} s "
                f"({d['matched_audio_bin_s']:.0f} s bins), "
                f"span {d['vote_support_span_s']:.1f} s; "
                f"refinement {d['refinement_audio_coverage_s']:.1f} s; "
                f"votes {raw['votes']} vs {raw['runner_up']} "
                f"({d['vote_margin']:.1f}x); "
                f"landmarks {d['landmark_matched_fraction']:.1%}; "
                f"onsets {raw['onsets_matched']}/{raw['onsets_total']} (advisory); "
                f"confidence={d['confidence']['level']}"
            )
        return tuple(lines)


def to_last_start_origin(offsets: Mapping[str, float]) -> tuple[dict[str, float], str]:
    """Re-origin reference-relative frame-0 offsets onto the global clock.

    Input: each stream's frame-0 time in ANY common frame (:func:`resolve_clock`
    uses "seconds after the earliest arm"). Output: the same placements shifted
    so the LAST stream to start sits at exactly 0.0 - the global origin - making
    every offset <= 0. Streams tied with the origin also land at exactly 0.0;
    ``origin`` breaks the tie lexicographically so a re-run is deterministic.

    This is the ONLY place the global origin is decided, and it must run after EVERY
    video stream is placed: an extra camera can start last and own the origin.
    """
    if not offsets:
        raise ValueError("to_last_start_origin: empty offsets")
    m = max(offsets.values())
    origin = min(k for k, v in offsets.items() if v == m)
    return {k: (0.0 if v == m else float(v) - m) for k, v in offsets.items()}, origin


def _estimate(
    ref: Clip, other: Clip, what: str, params: SyncParams, cache: AudioCache
) -> AudioOffset:
    """``estimate_offset`` with both failure modes turned into SyncGateError.

    The estimator raises ``RuntimeError`` for anything it cannot even measure (no
    shared landmarks, no audio stream, a missing per-mode A/V entry); those must
    halt as cleanly as a gate failure, not as a traceback. The zero-landmark case
    gets a pairing hint.
    """
    try:
        res = estimate_offset(ref, other, params=params, _cache=cache)
    except RuntimeError as e:
        hint = ""
        if "do not overlap" in str(e):
            hint = (
                "\nIf these clips were captured simultaneously, the pair is probably "
                "mis-sorted (clips from different takes in one episode dir). Verify "
                "the pairing before re-running."
            )
        raise SyncGateError(f"audio sync ({what}) failed outright: {e}{hint}") from e
    if not res.confident:
        raise SyncGateError(
            f"audio sync ({what}) gates FAILED: {', '.join(res.failed_gates)}. "
            f"Run `kiwi sync pair` on this pair for the full report.",
            report=res.report(),
            failed_gates=tuple(res.failed_gates),
            evidence={
                "raw": dataclasses.asdict(res.raw)
                if dataclasses.is_dataclass(res.raw)
                else vars(res.raw),
                "offset_s": res.offset_s,
            },
        )
    return res


def resolve_clock(
    arms: Mapping[str, Clip],
    extras: Mapping[str, Clip] | None = None,
    *,
    params: SyncParams = SyncParams(),
) -> ClockEstimate:
    """Place every stream on one global clock, from audio alone.

    ``arms`` are the reference streams that are estimated directly against each
    other (the two wrist cameras); ``extras`` are additional streams (e.g. the
    ego camera) placed against them. Three phases, in this order:

    1a. the arm pair, collapsed to two frame-0 offsets in the PAIR frame;
    1b. each extra placed against the REFERENCE arm - the phase-1 zero, NOT the
        clock origin, which is unknowable until every stream is placed - then
        cross-checked via each other arm within :data:`LOOP_TOL_S`;
    2.  :func:`to_last_start_origin` settles the origin once, at the end.

    Raises :class:`SyncGateError` if any edge fails its gates or any loop check
    does not close.
    """
    extras = dict(extras or {})
    if not arms:
        raise ValueError("resolve_clock: no reference streams given")
    if overlap := set(arms) & set(extras):
        raise ValueError(
            f"resolve_clock: {sorted(overlap)} listed as both arm and extra"
        )
    reports: list[str] = []
    edges: list[SyncEdge] = []
    cache = AudioCache()

    def measure(ref, other, what, parameters):
        try:
            return _estimate(ref, other, what, parameters, cache)
        except SyncGateError as exc:
            exc.evidence = {
                "edges": [dataclasses.asdict(edge) for edge in edges],
                "failed_edge": {"pair": what, **exc.evidence},
                "params": dataclasses.asdict(parameters),
            }
            raise

    # --- phase 1a: the arm pair ------------------------------------------
    if len(arms) == 1:
        offsets = {next(iter(arms)): 0.0}
    elif len(arms) == 2:
        (a_label, a_clip), (b_label, b_clip) = sorted(arms.items())
        res = measure(a_clip, b_clip, f"{a_label} vs {b_label}", params)
        reports.append(f"{a_label} vs {b_label} (audio)\n{res.report()}")
        edges.append(_edge("arms", a_label, b_label, res))
        a_off, b_off = pair_to_global_offsets(res.offset_s)
        offsets = {a_label: a_off, b_label: b_off}
    else:
        raise ValueError(
            f"resolve_clock: {len(arms)} reference streams given, expected 1 or 2. "
            f"Pass the extra streams as `extras` so each one is loop-checked."
        )

    # --- phase 1b: extras hang off the reference arm ----------------------
    # The reference is the EARLIEST arm (the phase-1 zero). The final uniform
    # re-origin cannot change which stream that is: it is about differences.
    ref_label = min(sorted(offsets), key=offsets.__getitem__)
    others = {k: v for k, v in arms.items() if k != ref_label}
    loop_residuals: dict[str, float] = {}
    for label, clip in sorted(extras.items()):
        res = measure(arms[ref_label], clip, f"{label} vs {ref_label}", params)
        reports.append(f"{label} vs {ref_label} (audio)\n{res.report()}")
        edges.append(_edge("extra", ref_label, label, res))
        offsets[label] = offsets[ref_label] + res.offset_s

        if not others:
            reports.append(
                f"{label}: no loop check possible - only one reference stream. "
                f"This placement rests on a single audio edge."
            )
            continue
        for other_label, other_clip in sorted(others.items()):
            alt = measure(
                other_clip, clip, f"{label} vs {other_label} (loop check)", params
            )
            edges.append(_edge("loop_check", other_label, label, alt))
            via_other = offsets[other_label] + alt.offset_s
            resid = abs(via_other - offsets[label])
            loop_residuals[label] = max(resid, loop_residuals.get(label, 0.0))
            reports.append(
                f"{label}: loop check via {other_label}: {via_other:+.4f}s vs "
                f"{offsets[label]:+.4f}s -> {resid * 1e3:.1f} ms "
                f"({'ok' if resid <= LOOP_TOL_S else 'FAILED'}; "
                f"tol {LOOP_TOL_S * 1e3:.0f} ms)"
            )
            if resid > LOOP_TOL_S:
                raise SyncGateError(
                    f"{label}: audio placements via {ref_label} and {other_label} "
                    f"disagree by {resid * 1e3:.0f} ms (> {LOOP_TOL_S * 1e3:.0f} ms). "
                    f"That is far more than an all-audio triangle explains, so at "
                    f"least one estimate is a periodic false lock.",
                    report="\n".join(reports),
                    evidence={
                        "edges": [dataclasses.asdict(e) for e in edges],
                        "loop_residuals_s": loop_residuals,
                        "loop_tolerance_s": LOOP_TOL_S,
                    },
                )

    # --- phase 2: settle the global origin, once, at the end --------------
    placed, origin = to_last_start_origin(offsets)
    return ClockEstimate(
        clock=GlobalClock(origin=origin, offsets_s=placed).validate(),
        method=METHOD,
        reports=tuple(reports),
        loop_residuals=loop_residuals,
        edges=tuple(edges),
    )

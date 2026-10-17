"""Cross-camera time sync from what the cameras heard (component: imports ``kiwi.core`` only).

Every stream of an episode - the two Insta360 X5 arm cameras and any extra
camera (e.g. the ego GO3) - is placed on one clock by cross-correlating what
the microphones heard. Audio needs no prior synchronization and no knowledge of
whether the cameras were timecode-jammed.

Owns:

* the two-stage recipe in :mod:`kiwi.sync.audio` - Shazam-style spectral-landmark
  fingerprint **voting** (prior-free), then GCC-PHAT refinement within +-0.3 s
  and a parabolic sub-sample vertex
* the **gates** that decide whether an estimate may be trusted
  (:data:`kiwi.sync.audio.GATES`), and the rule that a failure halts rather
  than guesses
* per-recording-mode **A/V corrections** (audio time is not video time)
* the offset algebra: pairwise -> pair frame -> the global origin, resolved
  into a :class:`kiwi.core.clock.GlobalClock` (:mod:`kiwi.sync.clock`)

**Sign convention**, stated on every offset-bearing signature: a positive offset
means ``other`` started later. An instant at ``ref`` clip time ``t`` sits at
``other`` clip time ``t - offset_s``, so on a shared clock
``global_off(other) = global_off(ref) + offset_s``.

**Audio is not video.** A camera stamps its audio and video tracks with a
constant relative delay that is a property of its exact recording mode. So
:func:`kiwi.sync.audio.resolve_av_correction` has exactly two rules and no
third: identified identical model *and* recording mode cancels to exactly ``0.0``;
anything else needs both clips' measured ``av_sync_v1`` delays or it raises.
There is no default and no zero fallback.

Fingerprint voting selects a content-supported offset before correlation
refines it. :class:`kiwi.sync.audio.SyncParams` defines the shared recipe
defaults and confidence thresholds.

Carve-outs:

* SMPTE ``tmcd`` (:mod:`kiwi.sync.tmcd`) is **verification only**: jam status
  is not observable from a recording. ``kiwi sync survey --tmcd`` reports
  audio-vs-timecode skew as a diagnostic cross-check.
* Measuring a camera's A/V delay (differencing a raw audio offset against
  a rolling-QR ground truth) is ``kiwi calib qr av-delay`` -
  :mod:`kiwi.calib` owns the ``av_sync_v1`` artifact. The one function that
  carries the calibration sign, :func:`kiwi.sync.audio.av_delay_from_ground_truth`,
  lives here.
* Writing the clock into ``episode.yaml`` belongs to :mod:`kiwi.episode` - this
  package hands back a validated clock and never touches a manifest.
"""

from kiwi.sync.audio import (
    GATES,
    AudioOffset,
    AVCorrection,
    AVDelay,
    Clip,
    RawOffset,
    RecordingMode,
    SyncParams,
    estimate_offset,
    estimate_raw_offset,
    pair_to_global_offsets,
)
from kiwi.sync.clock import (
    LOOP_TOL_S,
    ClockEstimate,
    SyncEdge,
    SyncGateError,
    resolve_clock,
    to_last_start_origin,
)

__all__ = [
    "GATES",
    "LOOP_TOL_S",
    "AVCorrection",
    "AVDelay",
    "AudioOffset",
    "ClockEstimate",
    "SyncEdge",
    "Clip",
    "RawOffset",
    "RecordingMode",
    "SyncGateError",
    "SyncParams",
    "estimate_offset",
    "estimate_raw_offset",
    "pair_to_global_offsets",
    "resolve_clock",
    "to_last_start_origin",
]

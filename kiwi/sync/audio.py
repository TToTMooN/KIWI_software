"""Cross-camera time sync from the recorded audio track.

Every stream of an episode - the two Insta360 X5 arm cameras and any extra camera
(e.g. the ego GO3) - is placed on the shared clock by cross-correlating what the
microphones heard. Audio needs no prior synchronization and no knowledge of
whether the cameras were timecode-jammed (jam status is not observable from a
recording). SMPTE timecode (:mod:`kiwi.sync.tmcd`) is an out-of-pipeline
cross-check in verification tooling.

The two-stage algorithm is identified by :data:`SYNC_RECIPE`:

1. *Coarse, prior-free*: Shazam-style spectral-peak landmark hashes ``(f1, f2,
   dt)``; matched hashes vote offsets into a 50 ms histogram and the mode wins.
   Needs no prior time sync.
2. *Fine*: GCC-PHAT peak within ±0.3 s of the voted offset, plus a parabolic
   sub-sample vertex. Long clips use bounded windows supported by the winning
   landmarks.

Content-specific landmark voting selects the coarse offset before correlation
refines it. Onset agreement is an independent envelope check, reported but
advisory; sliding-window drift measurements assess the constant-offset timing
budget and report the available evidence.

**Sign convention** (stated on every offset-bearing signature): a positive offset
means ``other`` started later. An instant at ``ref`` clip time ``t`` sits at
``other`` clip time ``t - offset_s``. So on a shared clock,
``global_off(other) = global_off(ref) + offset_s`` - identical to the convention
the episode's ``sync:`` block uses (see :mod:`kiwi.core.clock`).

**Audio is not video.** A camera stamps its audio and video tracks with a
constant relative delay that depends on the camera model and exact recording
mode, including resolution. :func:`resolve_av_correction` requires both camera
registry identities. The correction cancels when the declared model and
recording mode match; otherwise both units need measured per-mode delays in
the active A/V tables selected by their registry ``av_sync:`` pointers (with
``av-sync.yaml`` as the unversioned fallback). Missing calibration raises
instead of guessing a video offset. The exported clock remains a constant
offset; drift measurements do not remap video timestamps.

Free of ``cv2`` / ``tyro`` / ``matplotlib``; the CLI front end is the
``kiwi sync`` verb.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from scipy.ndimage import maximum_filter

from kiwi.core.camera.registry import ARTIFACT_NAMES, ARTIFACT_SCHEMAS, CameraUnit

# Public estimators, diagnostics, clock algebra, and A/V calibration constants.
__all__ = [
    "AudioCache",
    "AVDelay",
    "AVCorrection",
    "AV_SYNC_FILENAME",
    "AV_SYNC_SCHEMA",
    "AudioOffset",
    "Clip",
    "DriftResult",
    "EXTRACTION",
    "GATES",
    "MediaProbe",
    "RawOffset",
    "RecordingMode",
    "RefineResult",
    "SAMPLE_RATE",
    "SYNC_RECIPE",
    "SyncParams",
    "VoteResult",
    "av_delay_from_ground_truth",
    "estimate_drift",
    "estimate_offset",
    "estimate_raw_offset",
    "gcc_phat",
    "load_audio",
    "onset_agreement",
    "onset_times",
    "pair_to_global_offsets",
    "probe_media",
    "read_av_delay",
    "refine_offset",
    "resolve_av_correction",
    "vote_offset",
]

SAMPLE_RATE = 48_000
SYNC_RECIPE = "audio_landmarks_gcc_phat"
# The schema tag and filename come from the registry's artifact tables, which
# the writer (kiwi.calib, which measures the delay) also reads; neither
# component may import the other.
AV_SYNC_SCHEMA = ARTIFACT_SCHEMAS["av_sync"]
AV_SYNC_FILENAME = ARTIFACT_NAMES["av_sync"]
# The exact decode the A/V constants were measured under, recorded into the
# YAML. A constant measured with one recipe is not valid for another.
EXTRACTION = "ffmpeg -map 0:a:0 -ac 1 -ar 48000 -f s16le -"


@dataclasses.dataclass(frozen=True)
class SyncParams:
    """Parameters recorded in the synchronization input identity.

    These defaults define the recipe; a change to one must come with a fresh
    ``kiwi sync survey`` over real pairs. The recipe uses spectral matching,
    refines landmark-supported windows, and records offset confidence
    separately from measured drift support.

    The onset agreement (``onset_count``, ``onset_tol_s``) is advisory: it is
    computed and reported but does not gate acceptance.
    """

    sample_rate: int = SAMPLE_RATE
    # --- stage 2 (GCC-PHAT refine) ---
    refine_window_s: float = 0.3
    # --- gates ---
    min_votes: int = 15
    vote_margin: float = 1.5
    duration_tol_s: float = 0.15
    # --- advisory onset agreement (reported, not gating) ---
    onset_count: int = 8
    onset_tol_s: float = 0.150
    # --- level gates ---
    # Each clip is judged on its own, against the WORSE one: one dead
    # microphone or one pinned track must trip even when the other clip is
    # healthy.
    silence_peak_frac: float = 0.05  # min over both clips of peak / full scale
    clip_frac_max: float = 1e-3  # share of samples pinned at full scale
    # --- stage 1 (landmark fingerprint vote) ---
    nfft: int = 2048  # ~43 ms window
    hop: int = 512  # ~10.7 ms hop
    fan: int = 8  # landmark pairs per anchor peak
    dt_min: int = 3  # pair separation, frames
    dt_max: int = 60
    df_max: int = 96  # pair separation, freq bins
    vote_bin_s: float = 0.05
    peak_neighborhood: tuple[int, int] = (15, 31)
    peak_floor: float = 1.5
    max_hash_pairs: int = 64  # degenerate near-constant hash guard
    # --- repetition + drift gates ---
    min_vote_span_s: float = 3.0  # winning votes' anchors must span at least this
    drift_window_s: float = 20.0  # sliding-window refine length
    drift_hop_s: float = 10.0
    drift_min_duration_s: float = 90.0  # shorter clips: drift not measured
    drift_min_windows: int = 5
    max_drift_ppm: float = 150.0  # measured CI outside this range is excessive
    # Drift evidence thresholds: missing support lowers confidence, while
    # measured excessive skew or a well-supported budget failure refuses.
    max_drift_error_s: float = 0.020
    min_overlap_s: float = 3.0
    # Diagnostics by default.
    min_landmark_match_fraction: float = 0.0
    min_support_coverage: float = 0.0
    drift_min_coverage: float = 0.8
    analysis_block_frames: int = 256
    refine_max_duration_s: float = 30.0
    refine_windows: int = 5


# --------------------------------------------------------------------------- #
# recording mode + media probe
# --------------------------------------------------------------------------- #
def _fps_label(fps: float) -> str:
    """``30000/1001`` -> ``"29.97"``, ``50/1`` -> ``"50"`` (stable mode-key text)."""
    return f"{fps:.2f}".rstrip("0").rstrip(".")


@dataclasses.dataclass(frozen=True)
class RecordingMode:
    """A clip's *displayed* geometry + frame rate - the A/V table's key.

    Normalized by rotation: the GO 3 stores some modes portrait with a -90
    display matrix and others landscape, and the same physical mode hits the
    same table entry either way. The key is computed from coded dims (as
    ``ffprobe`` reports them) plus explicit rotation math.
    """

    width: int  # coded
    height: int  # coded
    fps: float
    rotation_deg: int = 0

    @property
    def displayed(self) -> tuple[int, int]:
        return (
            (self.height, self.width)
            if self.rotation_deg % 180
            else (self.width, self.height)
        )

    @property
    def key(self) -> str:
        w, h = self.displayed
        return f"{w}x{h}@{_fps_label(self.fps)}"

    def __str__(self) -> str:  # noqa: D105
        return self.key


@dataclasses.dataclass(frozen=True)
class MediaProbe:
    """What ffprobe says about one clip, for gates and the A/V lookup."""

    path: Path
    mode: RecordingMode
    video_duration_s: float
    audio_codec: str
    audio_sample_rate: int
    audio_channels: int


def _parse_fraction(text: str | None) -> float:
    if not text:
        return 0.0
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(text)


def _run(cmd: list[str], *, what: str, path: Path) -> bytes:
    """Run a media tool, translating both failure modes into ``RuntimeError``.

    This module raises exactly two things across the component boundary:
    ``SyncGateError`` when the gates refuse a measurable pair, and
    ``RuntimeError`` when a pair cannot be measured at all. A missing binary
    or an undecodable file raises ``RuntimeError``, carrying the last line of
    the tool's own stderr.
    """
    try:
        return subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{cmd[0]} not found on PATH, so {what} is impossible. "
            f"Install ffmpeg - it ships ffprobe too - and re-run."
        ) from e
    except subprocess.CalledProcessError as e:
        lines = (e.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        why = lines[-1].strip() if lines else f"exit status {e.returncode}"
        why = why.removeprefix(f"{path}: ")  # ffprobe prefixes its own message
        raise RuntimeError(f"{what} failed for {path}: {why}") from e


def probe_media(path: str | Path, *, ffprobe: str = "ffprobe") -> MediaProbe:
    """Probe ``path`` for the geometry/rate/audio facts this module needs.

    Container-agnostic: an ``.insv`` (audio is stream 2, tmcd is stream 3) and a
    plain ``.mp4`` (audio is stream 1, no tmcd) are handled identically.

    Raises ``RuntimeError`` when there is no decodable audio stream - an offset
    cannot be estimated at all in that case.
    """
    path = Path(path)
    out = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        what="probing media",
        path=path,
    )
    doc = json.loads(out.decode("utf-8", "replace") or "{}")
    streams = doc.get("streams") or []

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise RuntimeError(
            f"no audio stream in {path}: audio sync is impossible for this clip. "
            f"Place it on the clock by another means, or hand-write its offset "
            f"into the episode's sync block."
        )

    rotation = 0
    for sd in video.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if not rotation:
        rotation = int(round(float((video.get("tags") or {}).get("rotate", 0) or 0)))

    duration = float(
        video.get("duration") or (doc.get("format") or {}).get("duration") or 0.0
    )
    return MediaProbe(
        path=path,
        mode=RecordingMode(
            width=int(video["width"]),
            height=int(video["height"]),
            fps=_parse_fraction(video.get("r_frame_rate")),
            rotation_deg=rotation % 360,
        ),
        video_duration_s=duration,
        audio_codec=str(audio.get("codec_name", "")),
        audio_sample_rate=int(audio.get("sample_rate") or 0),
        audio_channels=int(audio.get("channels") or 0),
    )


def load_audio(
    path: str | Path, *, sample_rate: int = SAMPLE_RATE, ffmpeg: str = "ffmpeg"
) -> np.ndarray:
    """Mono float64 PCM at ``sample_rate`` from the clip's first audio stream.

    This exact decode is part of the A/V calibration (see :data:`EXTRACTION`) -
    changing the channel mix, rate or sample format invalidates the measured
    per-mode constants.
    """
    path = Path(path)
    raw = _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-",
        ],  # fmt: skip
        what="decoding audio",
        path=path,
    )
    if not raw:
        raise RuntimeError(
            f"decoded 0 audio samples from {path} (empty or unreadable stream)"
        )
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64)


# --------------------------------------------------------------------------- #
# stage 1: prior-free landmark fingerprint vote
# --------------------------------------------------------------------------- #
def _block_median(values: np.ndarray, rows: int) -> float:
    """Exact median of nonnegative finite float64 values using bounded buffers.

    Positive IEEE-754 floats sort in the same order as their unsigned bit
    patterns. Eight radix passes select the middle ranks without copying or
    partitioning the full spectrogram. The two ranks usually share a prefix.
    """
    if values.nbytes <= 16 * 1024 * 1024:
        return float(np.median(values))
    ranks = [(0, (values.size - 1) // 2), (0, values.size // 2)]
    for shift in range(56, -1, -8):
        counts = {prefix: np.zeros(256, dtype=np.int64) for prefix, _ in ranks}
        for start in range(0, len(values), rows):
            bits = np.asarray(values[start : start + rows]).view(np.uint64).ravel()
            for prefix, hist in counts.items():
                selected = bits if shift == 56 else bits[bits >> (shift + 8) == prefix]
                digits = ((selected >> shift) & 255).astype(np.int64)
                hist += np.bincount(digits, minlength=256)
        updated = []
        for prefix, rank in ranks:
            cumulative = np.cumsum(counts[prefix])
            digit = int(np.searchsorted(cumulative, rank, side="right"))
            before = int(cumulative[digit - 1]) if digit else 0
            updated.append(((prefix << 8) | digit, rank - before))
        ranks = updated
    middle = np.array([prefix for prefix, _ in ranks], dtype=np.uint64).view(np.float64)
    return float(np.mean(middle))


def _spectral_peaks(x: np.ndarray, p: SyncParams) -> tuple[np.ndarray, np.ndarray]:
    """Global-floor STFT spectral peaks, computed in bounded blocks.

    A temporary memory map holds the spectrogram; RAM buffers and FFT work
    arrays are bounded by analysis_block_frames. Halo rows preserve the exact
    maximum-filter result across block boundaries.
    """
    n = (len(x) - p.nfft) // p.hop
    if n <= 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if p.analysis_block_frames < 1:
        raise ValueError("analysis_block_frames must be positive")
    block = p.analysis_block_frames
    frames = np.lib.stride_tricks.sliding_window_view(x, p.nfft)[:: p.hop][:n]
    window = np.hanning(p.nfft)
    with tempfile.TemporaryFile() as scratch:
        logm = np.memmap(
            scratch, dtype=np.float64, mode="w+", shape=(n, p.nfft // 2 + 1)
        )
        for start in range(0, n, block):
            stop = min(start + block, n)
            logm[start:stop] = np.log1p(
                np.abs(np.fft.rfft(frames[start:stop] * window, axis=1))
            )
        floor = _block_median(logm, block) + p.peak_floor
        ts, fs = [], []
        halo = p.peak_neighborhood[0] // 2
        for start in range(0, n, block):
            stop = min(start + block, n)
            lo, hi = max(0, start - halo), min(n, stop + halo)
            maxima = maximum_filter(logm[lo:hi], size=p.peak_neighborhood)
            center = logm[start:stop]
            keep = (maxima[start - lo : stop - lo] == center) & (center > floor)
            t, f = np.nonzero(keep)
            ts.append(t + start)
            fs.append(f)
        del logm
    return np.concatenate(ts), np.concatenate(fs)


def _landmarks(t: np.ndarray, f: np.ndarray, p: SyncParams) -> dict:
    """hash ``(f1, f2, dt)`` -> list of anchor frame indices."""
    order = np.argsort(t)
    t, f = t[order], f[order]
    table: dict = {}
    for i in range(len(t)):
        n_paired = 0
        for j in range(i + 1, len(t)):
            dt = t[j] - t[i]
            if dt < p.dt_min:
                continue
            if dt > p.dt_max:
                break
            if abs(int(f[j]) - int(f[i])) > p.df_max:
                continue
            table.setdefault((int(f[i]), int(f[j]), int(dt)), []).append(int(t[i]))
            n_paired += 1
            if n_paired >= p.fan:
                break
    return table


@dataclasses.dataclass(frozen=True)
class VoteResult:
    """Coarse offset from landmark-hash voting, with the histogram for plots."""

    offset_s: float
    votes: int
    runner_up: int
    hist_centers: np.ndarray
    hist_counts: np.ndarray
    # Anchor-time span (s) of the votes in the winning bin: how much of the clip
    # actually supports this offset.
    support_span_s: float = 0.0
    overlap_s: float = 0.0
    matched_fraction: float = 0.0
    support_coverage: float = 0.0
    anchor_times_s: tuple[float, ...] = ()  # unique winning anchors on ref clock
    supported_duration_s: float = 0.0  # occupied 1 s bins, clipped to overlap

    @property
    def margin(self) -> float:
        return self.votes / max(self.runner_up, 1)


def vote_offset(
    a: np.ndarray,
    b: np.ndarray,
    *,
    params: SyncParams = SyncParams(),
    _prepared_landmarks: tuple[dict, dict] | None = None,
) -> VoteResult:
    """Prior-free coarse offset of ``b`` relative to ``a`` (positive = b later)."""
    p = params
    if _prepared_landmarks is None:
        la = _landmarks(*_spectral_peaks(a, p), p)
        lb = _landmarks(*_spectral_peaks(b, p), p)
    else:
        la, lb = _prepared_landmarks
    frame_s = p.hop / p.sample_rate
    votes: list[float] = []
    anchors: list[float] = []  # each vote's anchor time in `a`, for the span gate
    for h, tb in lb.items():
        ta = la.get(h)
        if (
            not ta or len(ta) * len(tb) > p.max_hash_pairs
        ):  # degenerate, near-constant hash
            continue
        for t1 in ta:
            for t2 in tb:
                votes.append((t1 - t2) * frame_s)
                anchors.append(t1 * frame_s)
    if not votes:
        raise RuntimeError("no matching landmarks at all - clips likely do not overlap")
    v = np.asarray(votes)
    anchor_t = np.asarray(anchors)
    lo = v.min()
    bins = np.floor((v - lo) / p.vote_bin_s).astype(int)
    counts = np.bincount(bins)
    merged = counts.copy()
    merged[:-1] += counts[1:]  # events straddling a bin edge
    best = int(np.argmax(merged))
    best_votes = int(merged[best])
    sel = (bins == best) | (bins == best + 1)
    offset = float(np.median(v[sel]))
    span = float(anchor_t[sel].max() - anchor_t[sel].min())
    aside = merged.copy()
    aside[max(0, best - 2) : best + 3] = 0
    runner = int(aside.max()) if aside.size else 0
    centers = lo + (np.arange(len(counts)) + 0.5) * p.vote_bin_s
    overlap_start = max(0.0, offset)
    overlap_end = min(len(a) / p.sample_rate, len(b) / p.sample_rate + offset)
    overlap_s = max(0.0, overlap_end - overlap_start)
    matched_a = anchor_t[sel]
    matched_b = anchor_t[sel] - v[sel]

    def fraction(table, matched, start, stop):
        eligible = {
            t for times in table.values() for t in times if start <= t * frame_s < stop
        }
        supported = {int(round(t / frame_s)) for t in matched if start <= t < stop}
        return len(supported & eligible) / max(1, len(eligible))

    matched_fraction = min(
        fraction(la, matched_a, overlap_start, overlap_end),
        fraction(lb, matched_b, overlap_start - offset, overlap_end - offset),
    )
    # Count occupied one-second intervals, not only the earliest/latest vote.
    inside = matched_a[(matched_a >= overlap_start) & (matched_a < overlap_end)]
    occupied_bins = np.unique(np.floor(inside - overlap_start))
    occupied = occupied_bins.size
    coverage = occupied / max(1, int(np.ceil(overlap_s)))
    supported_duration_s = float(np.minimum(1.0, overlap_s - occupied_bins).sum())
    return VoteResult(
        offset,
        best_votes,
        runner,
        centers,
        counts,
        span,
        overlap_s,
        matched_fraction,
        coverage,
        tuple(float(t) for t in np.unique(inside)),
        supported_duration_s,
    )


# --------------------------------------------------------------------------- #
# stage 2: GCC-PHAT refine
# --------------------------------------------------------------------------- #
def gcc_phat(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-lag GCC-PHAT of ``a`` (ref) vs ``b`` (other).

    Returns ``(lag_samples, cc)``. Positive lag = ``other`` started later:
    ``a[t]`` aligns with ``b[t - lag/sample_rate]``.
    """
    n = int(2 ** np.ceil(np.log2(len(a) + len(b))))
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    cross = A * np.conj(B)
    cc = np.fft.irfft(cross / (np.abs(cross) + 1e-12), n)
    cc = np.concatenate((cc[-(len(b) - 1) :], cc[: len(a)]))
    lags = np.arange(-(len(b) - 1), len(a))
    return lags, cc


@dataclasses.dataclass(frozen=True)
class RefineResult:
    offset_s: float
    at_window_edge: bool
    global_argmax_s: float | None  # absent when long clips use bounded windows
    peak_value: float
    windows_s: tuple[tuple[float, float], ...] = ()  # contributing ref-audio intervals


def refine_offset(
    a: np.ndarray, b: np.ndarray, coarse_s: float, *, params: SyncParams = SyncParams()
) -> RefineResult:
    """Sub-sample offset: GCC-PHAT peak within ±``refine_window_s`` of ``coarse_s``."""
    p = params
    lags, cc = gcc_phat(a, b)
    acc = np.abs(cc)
    sel = np.abs(lags / p.sample_rate - coarse_s) <= p.refine_window_s
    if not sel.any():
        raise RuntimeError(
            f"refinement window around {coarse_s:+.3f}s contains no lags"
        )
    wl, wa = lags[sel], acc[sel]
    i = int(np.argmax(wa))
    y0, y1, y2 = wa[max(i - 1, 0)], wa[i], wa[min(i + 1, len(wa) - 1)]
    den = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    offset = float((wl[i] + frac) / p.sample_rate)
    g = int(np.argmax(acc))
    return RefineResult(
        offset_s=offset,
        at_window_edge=not (0 < i < len(wa) - 1),
        global_argmax_s=float(lags[g] / p.sample_rate),
        peak_value=float(y1),
        windows_s=((0.0, len(a) / p.sample_rate),),
    )


def _aligned_audio(a, b, offset_s, sample_rate):
    """Shared content, integer removed offset, and start time on the ref clock."""
    shift = int(round(offset_s * sample_rate))
    a0, b0 = max(shift, 0), max(-shift, 0)
    count = max(0, min(len(a) - a0, len(b) - b0))
    return a[a0 : a0 + count], b[b0 : b0 + count], shift / sample_rate, a0 / sample_rate


def _refine_recording(a, b, coarse_s, p, *, anchor_times_s=None):
    """Refine bounded sections supported by the winning landmark pattern."""
    if max(len(a), len(b)) <= p.refine_max_duration_s * p.sample_rate:
        return refine_offset(a, b, coarse_s, params=p)
    aa, bb, shift, ref_start = _aligned_audio(a, b, coarse_s, p.sample_rate)
    win = min(len(aa), int(p.refine_max_duration_s * p.sample_rate))
    if win < 2 or p.refine_windows < 1:
        raise RuntimeError("no shared audio available for refinement")
    starts = np.unique(np.linspace(0, len(aa) - win, p.refine_windows).astype(int))
    if anchor_times_s is not None:
        anchors = (np.asarray(anchor_times_s) - ref_start) * p.sample_rate
        anchors = anchors[(anchors >= 0) & (anchors < len(aa))]
        if not len(anchors):
            raise RuntimeError("no shared landmark support available for refinement")
        starts = np.asarray(
            [s for s in starts if np.any((anchors >= s) & (anchors < s + win))],
            dtype=int,
        )
        if not len(starts):
            # A short match can lie between the bounded sections of a long
            # recording. Put a section on the actual evidence in that case.
            centers = np.quantile(anchors, np.linspace(0, 1, p.refine_windows))
            starts = np.unique(np.clip(centers - win / 2, 0, len(aa) - win).astype(int))
    results = [
        refine_offset(aa[s : s + win], bb[s : s + win], coarse_s - shift, params=p)
        for s in starts
    ]
    valid = [r for r in results if not r.at_window_edge]
    if len(valid) * 2 <= len(results):
        raise RuntimeError(
            "fewer than a majority of supported windows could be refined"
        )
    return RefineResult(
        offset_s=shift + float(np.median([r.offset_s for r in valid])),
        at_window_edge=False,
        global_argmax_s=None,
        peak_value=float(np.median([r.peak_value for r in valid])),
        windows_s=tuple(
            (ref_start + s / p.sample_rate, ref_start + (s + win) / p.sample_rate)
            for s, result in zip(starts, results)
            if not result.at_window_edge
        ),
    )


def _covered_duration(intervals):
    """Duration of interval union; overlapping analysis windows count once."""
    covered, end = 0.0, -math.inf
    for lo, hi in sorted(intervals):
        covered += max(0.0, hi - max(lo, end))
        end = max(end, hi)
    return float(covered)


# --------------------------------------------------------------------------- #
# drift: relative media-clock rate between the two recordings
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class DriftResult:
    """Relative clock rate of ``other`` vs ``ref``, from sliding-window refines.

    ``slope_ppm`` > 0 means the offset grows over time (``other``'s media clock
    runs slow relative to ``ref``'s, so ``other`` keeps starting 'later').
    """

    slope_ppm: float
    ci_low_ppm: float
    ci_high_ppm: float
    n_windows: int
    scatter_ms: float  # MAD of the fit residuals
    window_offsets: tuple[tuple[float, float], ...]  # (t_center_s, offset_s)
    intercept_s: float = 0.0
    max_residual_ms: float = 0.0
    coverage_fraction: float = 0.0
    covers_endpoints: bool = False

    @property
    def significant(self) -> bool:
        """95% CI excludes zero."""
        return self.ci_low_ppm > 0 or self.ci_high_ppm < 0

    @property
    def accumulated_ms(self) -> float:
        """Offset movement over the measured span, in ms."""
        if len(self.window_offsets) < 2:
            return 0.0
        t0, t1 = self.window_offsets[0][0], self.window_offsets[-1][0]
        return self.slope_ppm * (t1 - t0) * 1e-3


def estimate_drift(
    a: np.ndarray,
    b: np.ndarray,
    coarse_s: float,
    *,
    params: SyncParams = SyncParams(),
    window_s: float | None = None,
    hop_s: float | None = None,
    anchor_times_s: tuple[float, ...] | None = None,
) -> DriftResult | None:
    """Relative clock-rate estimate from independently refined sliding windows.

    Remove the coarse integer lag before selecting windows, so each pair
    contains the same events even when recordings start far apart. A Theil-Sen
    fit of absolute offsets against ref-clock window-center time yields the
    relative rate. Constant A/V stamping delays cancel out of the slope.

    Returns ``None`` when fewer than ``params.drift_min_windows`` windows
    survive the quality filters. Missing measurements lower confidence without
    vetoing an offset supported by a short shared pattern.
    """
    p = params
    win = int((window_s if window_s is not None else p.drift_window_s) * p.sample_rate)
    hop = int((hop_s if hop_s is not None else p.drift_hop_s) * p.sample_rate)
    if min(win, hop) <= 0:
        raise ValueError("drift window and hop must be positive")
    a, b, shift, ref_start = _aligned_audio(a, b, coarse_s, p.sample_rate)
    residual_coarse = coarse_s - shift
    n = len(a)
    anchors = (
        None
        if anchor_times_s is None
        else (np.asarray(anchor_times_s) - ref_start) * p.sample_rate
    )
    pts: list[tuple[float, float]] = []
    s = 0
    while s + win <= n:
        if anchors is not None and not np.any((anchors >= s) & (anchors < s + win)):
            s += hop
            continue
        try:
            r = refine_offset(a[s : s + win], b[s : s + win], residual_coarse, params=p)
            # a window locked to a periodic side peak or with no shared content
            # is a mislock, not evidence about the rate
            if not r.at_window_edge and abs(r.offset_s - residual_coarse) < 0.2:
                pts.append(
                    (ref_start + (s + win / 2) / p.sample_rate, shift + r.offset_s)
                )
        except RuntimeError:
            pass
        s += hop
    if len(pts) < p.drift_min_windows:
        return None
    t = np.asarray([q[0] for q in pts])
    o = np.asarray([q[1] for q in pts])
    all_t, all_o = t.copy(), o.copy()
    ts = stats.theilslopes(o, t, 0.95)
    resid = o - (ts.slope * t + ts.intercept)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    keep = np.abs(resid - np.median(resid)) <= max(4 * mad, 1e-4)
    if p.drift_min_windows <= int(keep.sum()) < len(o):
        t, o = t[keep], o[keep]
        pts = [q for q, k in zip(pts, keep) if k]
        ts = stats.theilslopes(o, t, 0.95)
        resid = o - (ts.slope * t + ts.intercept)
        mad = float(np.median(np.abs(resid - np.median(resid))))
    # Union of quality-accepted observations, before robust fit trimming.
    # Trimmed observations still constrain max_residual_ms.
    intervals = sorted(
        (center - win / (2 * p.sample_rate), center + win / (2 * p.sample_rate))
        for center in all_t
    )
    covered, end = 0.0, -np.inf
    for lo, hi in intervals:
        covered += max(0.0, hi - max(lo, end))
        end = max(end, hi)
    last_center = ref_start + (((n - win) // hop) * hop + win / 2) / p.sample_rate
    endpoints = (
        abs(all_t[0] - (ref_start + win / (2 * p.sample_rate))) < 1 / p.sample_rate
        and abs(all_t[-1] - last_center) < 1 / p.sample_rate
    )
    return DriftResult(
        slope_ppm=float(ts.slope * 1e6),
        ci_low_ppm=float(ts.low_slope * 1e6),
        ci_high_ppm=float(ts.high_slope * 1e6),
        n_windows=len(pts),
        scatter_ms=mad * 1e3,
        window_offsets=tuple(pts),
        intercept_s=float(ts.intercept),
        max_residual_ms=float(
            np.max(np.abs(all_o - (ts.slope * all_t + ts.intercept))) * 1e3
        ),
        coverage_fraction=min(1.0, covered / (n / p.sample_rate)),
        covers_endpoints=bool(endpoints),
    )


# --------------------------------------------------------------------------- #
# independent check: onset agreement (advisory, reported but not gating)
# --------------------------------------------------------------------------- #
def onset_times(
    x: np.ndarray,
    k: int,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop: int = 240,
    gap_s: float = 1.5,
) -> list[tuple[float, float]]:
    """Times + strengths of the ``k`` strongest RMS-derivative onsets, ``gap_s`` apart."""
    n = len(x) // hop
    if n < 2:
        return []
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    d = np.diff(rms, prepend=rms[0])
    d[d < 0] = 0
    picked: list[tuple[float, float]] = []
    for i in np.argsort(d)[::-1]:
        if d[i] <= 0:
            break
        t = i * hop / sample_rate
        if 0.5 < t < len(x) / sample_rate - 0.5 and all(
            abs(t - q) > gap_s for q, _ in picked
        ):
            picked.append((float(t), float(d[i])))
        if len(picked) == k:
            break
    return sorted(picked)


def onset_agreement(
    a: np.ndarray, b: np.ndarray, offset_s: float, *, params: SyncParams = SyncParams()
) -> tuple[int, int]:
    """``(matched, total)`` strong onsets that line up under ``offset_s``.

    An independent check on the correlation: it uses the signal envelope, not the
    spectrum. It is only ever a check, never an estimator, and it is
    advisory: :func:`estimate_raw_offset` records the counts without gating
    on them (see :class:`SyncParams`).
    """
    p = params
    a, b, shift, _ = _aligned_audio(a, b, offset_s, p.sample_rate)
    residual = offset_s - shift
    a_on = onset_times(a, p.onset_count, sample_rate=p.sample_rate)
    b_on = onset_times(b, p.onset_count, sample_rate=p.sample_rate)
    if not a_on or not b_on:
        return 0, min(len(a_on), len(b_on))

    def targets(x, references):
        # Keep the reference events well separated, but do not let a louder
        # nearby sound suppress their counterparts in the other microphone.
        # Candidates must still be local maxima at least as strong as the
        # weakest independently selected target reference onset.
        hop = 240
        n = len(x) // hop
        rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
        d = np.diff(rms, prepend=rms[0])
        floor = min(strength for _, strength in references)
        idx = (
            np.flatnonzero((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:]) & (d[1:-1] >= floor))
            + 1
        )
        times = idx * hop / p.sample_rate
        return times[(times > 0.5) & (times < len(x) / p.sample_rate - 0.5)]

    a_targets, b_targets = targets(a, a_on), targets(b, b_on) + residual

    def matched(references, candidates):
        return sum(
            bool(np.any(np.abs(candidates - t) <= p.onset_tol_s)) for t, _ in references
        )

    ab = matched(a_on, b_targets)
    ba = matched([(t + residual, strength) for t, strength in b_on], a_targets)
    return min(ab, ba), min(len(a_on), len(b_on))


# --------------------------------------------------------------------------- #
# the audio-vs-video (A/V) delay table
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class AVDelay:
    """One measured ``t_audio = t_video + seconds`` constant for one mode."""

    seconds: float
    mode_key: str
    unit: str
    measured_at: str
    source: Path


@dataclasses.dataclass(frozen=True)
class AVCorrection:
    """What to ADD to a raw audio offset to get the video offset."""

    seconds: float
    provenance: str
    ref: AVDelay | None = None
    other: AVDelay | None = None


def av_delay_from_ground_truth(
    *, raw_audio_offset_s: float, video_offset_s: float, ref_delay_s: float = 0.0
) -> float:
    """The ``av_delay_s`` implied for ``other`` by a ground-truth video offset.

    THE one place the calibration sign lives. From
    ``offset_video = offset_audio + (d_other - d_ref)``::

        d_other = video_offset - raw_audio_offset + d_ref
    """
    return float(video_offset_s - raw_audio_offset_s + ref_delay_s)


def _load_unit(camera_dir: Path) -> CameraUnit:
    """``CameraUnit.load``, with a broken unit translated at the load.

    The registry raises ``OSError`` for an unreadable ``camera.yaml`` and
    ``ValueError`` for one that is not a valid YAML mapping; both become
    ``RuntimeError``, one of the two types this module raises across its
    boundary (see :func:`_run`).
    """
    try:
        return CameraUnit.load(camera_dir)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"broken camera registry unit {camera_dir}: {e}") from e


def _unit_model(camera_dir: Path | None) -> str | None:
    """The unit's declared ``model:``, or None when no unit is named at all.

    A named unit must declare a nonempty model. Container geometry and codecs
    cannot identify a camera model or establish that its A/V delay cancels.
    """
    if camera_dir is None:
        return None
    model = _load_unit(camera_dir).meta.get("model")
    if model is None:
        raise RuntimeError(
            f"{camera_dir}/camera.yaml declares no `model:`, so this unit "
            f"cannot be told apart from another one that records the same "
            f"container format. Add the `model:` line to the unit."
        )
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError(
            f"{camera_dir}/camera.yaml `model:` must be a nonempty string, "
            f"got {model!r}"
        )
    return model


def read_av_delay(camera_dir: Path, mode: RecordingMode) -> AVDelay:
    """The measured delay for ``mode`` from the registry-selected ``av_sync`` artifact.

    The registry uses ``av-sync.yaml`` as the unversioned fallback.

    Raises ``RuntimeError`` - never returns a default - when the file, schema or
    mode entry is absent. The measured delay is specific to the camera's
    recording mode, including resolution and frame rate.
    """
    # Resolved through the registry, never by joining the filename: the active
    # artifact is whatever `camera.yaml`'s `av_sync:` pointer names (dated,
    # append-only), with the unversioned name as the fallback for a unit that
    # carries no pointer.
    try:
        path = _load_unit(camera_dir).artifact("av_sync")
    except OSError as e:
        # A pointer naming a missing file is a broken registry unit, raised as
        # RuntimeError like _run's missing binary.
        raise RuntimeError(f"broken camera registry unit {camera_dir}: {e}") from e
    hint = (
        "Measure it with `kiwi calib qr av-delay` (film the rolling QR clock "
        "with both cameras, then difference the QR delta against the raw "
        f"audio offset). Or copy a measured {AV_SYNC_FILENAME} from an "
        "existing unit, or record both clips in the same mode on the same "
        "camera model so the delay cancels exactly."
    )
    if path is None:
        raise RuntimeError(
            f"no {AV_SYNC_FILENAME} in {camera_dir}: this device's audio-vs-video delay "
            f"has never been measured, so an audio offset cannot be converted to a "
            f"video offset.\n{hint}"
        )
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:  # unreadable or corrupt table
        raise RuntimeError(f"{path}: {e}") from e
    if not isinstance(doc, dict):
        raise RuntimeError(f"{path}: A/V delay table must be a mapping")
    if doc.get("schema") != AV_SYNC_SCHEMA:
        raise RuntimeError(
            f"{path}: schema is {doc.get('schema')!r}, expected {AV_SYNC_SCHEMA!r}"
        )
    # A table must declare the decoding recipe its delays were measured under;
    # constants measured under a different recipe are never applied.
    extraction = doc.get("extraction")
    if extraction != EXTRACTION:
        raise RuntimeError(
            f"{path}: A/V delays use extraction {extraction!r}, "
            f"expected {EXTRACTION!r}; remeasure with the current recipe"
        )
    modes = doc.get("modes") or {}
    if not isinstance(modes, dict):
        raise RuntimeError(f"{path}: `modes` must be a mapping")
    entry = modes.get(mode.key)
    if entry is not None and not isinstance(entry, dict):
        raise RuntimeError(f"{path}: mode {mode.key} must be a mapping")
    if entry is None or "av_delay_s" not in entry:
        raise RuntimeError(
            f"no measured A/V delay for mode {mode.key} in {path} "
            f"(schema {AV_SYNC_SCHEMA}, measured {doc.get('measured_at')}). "
            f"Known modes: {', '.join(sorted(map(str, modes))) or '(none)'}.\n"
            f"This constant depends on the EXACT recording mode, resolution included, "
            f"and must be re-measured after any mode change.\n{hint}"
        )
    value = entry["av_delay_s"]
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError(
            f"{path}: mode {mode.key} `av_delay_s` must be a finite number, "
            f"got {value!r}"
        ) from None
    if isinstance(value, bool) or not math.isfinite(seconds):
        raise RuntimeError(
            f"{path}: mode {mode.key} `av_delay_s` must be a finite number, "
            f"got {value!r}"
        )
    return AVDelay(
        seconds=seconds,
        mode_key=mode.key,
        unit=str(doc.get("unit", camera_dir.name)),
        measured_at=str(doc.get("measured_at", "unknown")),
        source=path,
    )


@dataclasses.dataclass(frozen=True)
class Clip:
    """One clip to sync: the media plus the registry unit that recorded it."""

    path: Path
    camera_dir: Path | None = None  # registry identity required for video offsets
    label: str = ""

    def __post_init__(self) -> None:  # noqa: D105
        object.__setattr__(self, "path", Path(self.path))
        if self.camera_dir is not None:
            object.__setattr__(self, "camera_dir", Path(self.camera_dir))

    @property
    def name(self) -> str:
        return self.label or self.path.name


def resolve_av_correction(
    ref: Clip, ref_probe: MediaProbe, other: Clip, other_probe: MediaProbe
) -> AVCorrection:
    """Seconds to ADD to a raw audio offset so it becomes a video offset.

    Two rules, and no third:

    1. **Same model and same recording mode** -> exactly ``0.0``. Only the
       *difference* of two per-clip A/V delays is observable, so an identical
       pair cancels. Both clips must name registry units declaring that same
       model.
    2. **Anything else** -> both clips' measured delays, or ``RuntimeError``.
    """
    ref_model, other_model = _unit_model(ref.camera_dir), _unit_model(other.camera_dir)
    same_mode = ref_probe.mode.key == other_probe.mode.key
    for clip, model in ((ref, ref_model), (other, other_model)):
        if model is None:
            raise RuntimeError(
                f"{clip.name}: no camera unit was given, so its camera model "
                "and audio-vs-video delay cannot be established from matching "
                "containers. Name the camera unit "
                "(`kiwi sync pair --ref-camera/--other-camera`); "
                "use estimate_raw_offset for audio-only calibration."
            )
    if same_mode and ref_model == other_model:
        return AVCorrection(
            seconds=0.0,
            provenance=(
                f"identical recording mode ({ref_model} {ref_probe.mode.key}) - "
                f"the A/V delay is common-mode and cancels exactly"
            ),
        )
    rd = read_av_delay(ref.camera_dir, ref_probe.mode)
    od = read_av_delay(other.camera_dir, other_probe.mode)
    seconds = od.seconds - rd.seconds
    if not math.isfinite(seconds):
        raise RuntimeError("A/V delay difference must be finite")
    return AVCorrection(
        seconds=seconds,
        provenance=(
            f"{od.unit} {od.mode_key} ({od.seconds * 1e3:+.1f} ms, measured {od.measured_at}) "
            f"minus {rd.unit} {rd.mode_key} ({rd.seconds * 1e3:+.1f} ms)"
        ),
        ref=rd,
        other=od,
    )


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
GATES = (
    "audio_streams_present",
    "audio_video_durations_agree",
    "audio_non_silent",
    "audio_unclipped",
    "vote_margin",
    "vote_support_span",
    "vote_overlap_support",
    "refine_in_window",
    "measured_drift_acceptable",
)


@dataclasses.dataclass(frozen=True)
class RawOffset:
    """An offset on the AUDIO timeline, before any A/V correction.

    Calibration-only. This type has no field that can be mistaken for a video
    offset: the number a caller may write into ``sync:`` comes from
    :class:`AudioOffset`.
    """

    offset_s: float
    voted_offset_s: float
    votes: int
    runner_up: int
    onsets_matched: int
    onsets_total: int
    gates: dict[str, bool]
    diagnostics: dict
    ref: MediaProbe
    other: MediaProbe
    params: SyncParams
    drift: "DriftResult | None" = None  # too short or too few supported windows

    @property
    def confident(self) -> bool:
        return all(self.gates.values())

    @property
    def failed_gates(self) -> list[str]:
        return [k for k, v in self.gates.items() if not v]


@dataclasses.dataclass(frozen=True)
class AudioOffset:
    """An offset on the VIDEO timeline - the number downstream may use."""

    offset_s: float
    raw: RawOffset
    correction: AVCorrection

    @property
    def confident(self) -> bool:
        return not self.failed_gates

    @property
    def failed_gates(self) -> list[str]:
        failed = self.raw.failed_gates
        values = (self.offset_s, self.raw.offset_s, self.correction.seconds)
        try:
            finite = all(
                not isinstance(value, bool) and math.isfinite(value) for value in values
            )
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            failed.append("finite_video_offset")
        return failed

    def report(self) -> str:
        r = self.raw
        lines = [
            f"audio sync  {r.ref.path.name} -> {r.other.path.name}",
            f"  modes       : {r.ref.mode.key}  vs  {r.other.mode.key}",
            f"  voted       : {r.voted_offset_s:+.4f} s   "
            f"(votes {r.votes} vs runner-up {r.runner_up}, margin "
            f"{r.votes / max(r.runner_up, 1):.2f}x, "
            f"span {r.diagnostics.get('vote_support_span_s', 0.0):.1f} s)",
            f"  refined     : {r.offset_s:+.4f} s",
            f"  A/V corr    : {self.correction.seconds * 1e3:+.1f} ms  "
            f"[{self.correction.provenance}]",
            f"  VIDEO offset: {self.offset_s:+.4f} s   (positive = other started later)",
            f"  onsets      : {r.onsets_matched}/{r.onsets_total} matched "
            "(advisory, not a gate)",
        ]
        if r.drift is not None:
            d = r.drift
            lines.append(
                f"  drift       : {d.slope_ppm:+.1f} ppm "
                f"[{d.ci_low_ppm:+.1f}, {d.ci_high_ppm:+.1f}] over {d.n_windows} windows, "
                f"scatter {d.scatter_ms:.2f} ms; coverage {d.coverage_fraction:.0%}"
            )
            lines.append(
                "  drift state : " + r.diagnostics.get("drift_status", "not assessed")
            )
        else:
            lines.append(
                "  drift       : " + r.diagnostics.get("drift_status", "not measured")
            )
        confidence = r.diagnostics.get("confidence")
        if confidence is not None:
            lines.append(f"  confidence  : {confidence['level']} (evidence level)")
        lines.append(
            f"  overlap     : {r.diagnostics.get('overlap_s', 0.0):.1f} s; "
            f"landmark match {r.diagnostics.get('landmark_matched_fraction', 0.0):.1%}, "
            f"time coverage {r.diagnostics.get('vote_support_coverage', 0.0):.1%}"
        )
        if "matched_audio_coverage_s" in r.diagnostics:
            d = r.diagnostics
            lines.append(
                f"  audio used  : ref {d['ref_audio_duration_s']:.1f} s, "
                f"other {d['other_audio_duration_s']:.1f} s decoded; "
                f"{d['refinement_audio_coverage_s']:.1f} s in refinement windows"
            )
            lines.append(
                f"  match length: ~{d['matched_audio_coverage_s']:.1f} s "
                f"({d['matched_audio_bin_s']:.0f} s bins); "
                f"{d['vote_support_span_s']:.1f} s first-to-last match span"
            )
        budget = r.diagnostics.get("drift_error_bound_s")
        if budget is not None:
            lines.append(
                f"  drift budget: {budget * 1e3:.2f} ms / {r.params.max_drift_error_s * 1e3:.2f} ms"
            )
        arg = r.diagnostics.get("global_argmax_offset_s")
        if r.diagnostics.get("argmax_is_false_lock"):
            lines.append(
                f"  note        : raw correlation max sits at {arg:+.3f} s - "
                f"a periodic false lock the vote ignored"
            )
        lines.append("  gates:")
        lines += [
            f"    {'PASS' if ok else 'FAIL'}  {name}" for name, ok in r.gates.items()
        ]
        if "finite_video_offset" in self.failed_gates:
            lines.append("    FAIL  finite_video_offset")
        return "\n".join(lines)

    def require_confident(self) -> "AudioOffset":
        """Return self, or raise with the full report if any gate failed.

        The guard for anyone persisting a single pair's offset (the episode
        path goes through :mod:`kiwi.sync.clock`, which checks ``confident``
        itself). Never catch this and fall back to a default.
        """
        if not self.confident:
            if "finite_video_offset" in self.failed_gates:
                raise RuntimeError(
                    "audio sync not confident (failed: "
                    f"{', '.join(self.failed_gates)}): "
                    f"video={self.offset_s!r}, raw={self.raw.offset_s!r}, "
                    f"A/V correction={self.correction.seconds!r}"
                )
            raise RuntimeError(
                f"audio sync not confident (failed: {', '.join(self.failed_gates)})\n"
                + self.report()
            )
        return self


def _drift_checks(drift, overlap_s, p):
    """Assess drift evidence independently of whether the offset is accepted.

    Bounded drift needs a contained rate interval, broad temporal support,
    and a duration budget. Missing or limited evidence is not a measured
    defect and does not by itself invalidate a supported audio offset.
    """
    if drift is None:
        short = overlap_s < p.drift_min_duration_s
        return (
            "not measured: short overlap" if short else "unmeasurable on long overlap",
            False,
            False,
            None,
        )
    finite = all(
        np.isfinite(v)
        for v in (
            drift.slope_ppm,
            drift.ci_low_ppm,
            drift.ci_high_ppm,
            drift.max_residual_ms,
        )
    )
    supported = (
        drift.covers_endpoints and drift.coverage_fraction >= p.drift_min_coverage
    )
    bounded = (
        finite
        and supported
        and -p.max_drift_ppm <= drift.ci_low_ppm <= drift.ci_high_ppm <= p.max_drift_ppm
    )
    error = (
        max(abs(drift.ci_low_ppm), abs(drift.ci_high_ppm)) * 1e-6 * overlap_s
        + 2 * drift.max_residual_ms * 1e-3
    )
    budget_ok = finite and supported and error <= p.max_drift_error_s
    excessive = (
        drift.ci_low_ppm > p.max_drift_ppm or drift.ci_high_ppm < -p.max_drift_ppm
    )
    status = (
        "bounded"
        if bounded
        else ("excessive drift" if excessive else "uncertain drift")
    )
    if not supported:
        status = "insufficient drift coverage"
    if bounded and not budget_ok:
        status = "exceeds constant-offset timing budget"
    return status, bool(bounded), bool(budget_ok), float(error) if finite else None


def _measured_drift_acceptable(drift, bounded, budget_ok, p):
    """Reject measured defects, not the absence of whole-overlap evidence."""
    if drift is None:
        return True
    values = (
        drift.slope_ppm,
        drift.ci_low_ppm,
        drift.ci_high_ppm,
        drift.max_residual_ms,
    )
    if not all(np.isfinite(value) for value in values):
        return False
    excessive = (
        drift.ci_low_ppm > p.max_drift_ppm or drift.ci_high_ppm < -p.max_drift_ppm
    )
    return bool(not excessive and not (bounded and not budget_ok))


def _confidence(gates, drift_bounded, drift_budget_ok):
    """Evidence categories, not an uncalibrated probability of correctness."""
    offset_supported = all(
        value for name, value in gates.items() if name != "measured_drift_acceptable"
    )
    drift_supported = drift_bounded and drift_budget_ok
    accepted = all(gates.values())
    return {
        "schema": "sync_confidence_v1",
        "level": (
            "rejected"
            if not accepted
            else "offset_and_drift"
            if drift_supported
            else "offset_only"
        ),
        "offset_supported": bool(offset_supported),
        "drift_supported": bool(drift_supported),
        "calibrated_probability": None,
    }


class AudioCache:
    """Per-resolution decoded audio and landmark reuse, never a persistent cache.

    A clock resolver owns one instance and releases all samples on completion.
    Recording metadata and the complete recipe key prevent accidental reuse
    after an input or extraction setting changes during that resolution.
    Saved-clock reuse is human-controlled; these checks only reuse work
    within one requested estimation.
    """

    def __init__(self):
        self._prepared = {}

    def prepare(self, path, params, ffmpeg, ffprobe):
        try:
            stat = path.stat()
        except OSError as exc:
            raise RuntimeError(f"cannot read audio input {path}: {exc}") from exc
        key = (
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            params,
            ffmpeg,
            ffprobe,
        )
        if key not in self._prepared:
            probe = probe_media(path, ffprobe=ffprobe)
            samples = load_audio(path, sample_rate=params.sample_rate, ffmpeg=ffmpeg)
            landmarks = _landmarks(*_spectral_peaks(samples, params), params)
            self._prepared[key] = (probe, samples, landmarks)
        return self._prepared[key]


def estimate_raw_offset(
    ref: Clip | str | Path,
    other: Clip | str | Path,
    *,
    params: SyncParams = SyncParams(),
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    _cache: AudioCache | None = None,
) -> RawOffset:
    """Offset of ``other`` relative to ``ref`` on the AUDIO timeline.

    For calibration (differencing against a QR/video ground truth) only - it
    applies no A/V correction, so it is *not* a video offset. Use
    :func:`estimate_offset` for anything that reaches a dataset.
    """
    ref = ref if isinstance(ref, Clip) else Clip(Path(ref))
    other = other if isinstance(other, Clip) else Clip(Path(other))
    p = params

    prepared = None
    if _cache is None:
        rp = probe_media(ref.path, ffprobe=ffprobe)
        op = probe_media(other.path, ffprobe=ffprobe)
        a = load_audio(ref.path, sample_rate=p.sample_rate, ffmpeg=ffmpeg)
        b = load_audio(other.path, sample_rate=p.sample_rate, ffmpeg=ffmpeg)
    else:
        rp, a, la = _cache.prepare(ref.path, p, ffmpeg, ffprobe)
        op, b, lb = _cache.prepare(other.path, p, ffmpeg, ffprobe)
        prepared = la, lb

    gates: dict[str, bool] = {
        "audio_streams_present": True
    }  # probe_media raises otherwise
    gates["audio_video_durations_agree"] = all(
        abs(len(x) / p.sample_rate - pr.video_duration_s) < p.duration_tol_s
        for x, pr in ((a, rp), (b, op))
    )
    # Level gates on the WORSE clip: one dead microphone or one pinned track
    # must trip even when the other clip is healthy. Samples are int16-scaled
    # (see load_audio), so full scale is 32767, not 1.0.
    peaks = (np.abs(a).max() / 32768, np.abs(b).max() / 32768)
    peak_frac = min(peaks)
    clipped_frac = max(np.mean(np.abs(a) >= 32767), np.mean(np.abs(b) >= 32767))
    gates["audio_non_silent"] = bool(peak_frac > p.silence_peak_frac)
    gates["audio_unclipped"] = bool(clipped_frac <= p.clip_frac_max)

    try:
        vote = vote_offset(a, b, params=p, _prepared_landmarks=prepared)
    except RuntimeError as e:
        # Exact digital zeros (and DC) produce no landmarks, so for a dead
        # microphone the vote raises before the level gate is reported. Name
        # the dead clip.
        if gates["audio_non_silent"]:
            raise
        quiet = (ref if peaks[0] <= peaks[1] else other).name
        raise RuntimeError(
            f"audio_non_silent failed: {quiet} peaks at {peak_frac:.5f} of full "
            f"scale (threshold {p.silence_peak_frac}) and carries no landmarks "
            f"at all - check that camera's microphone, not the pairing"
        ) from e

    gates["vote_margin"] = bool(
        vote.votes >= p.min_votes and vote.votes >= p.vote_margin * vote.runner_up
    )
    gates["vote_support_span"] = bool(vote.support_span_s >= p.min_vote_span_s)
    gates["vote_overlap_support"] = bool(
        vote.overlap_s >= p.min_overlap_s
        and vote.matched_fraction >= p.min_landmark_match_fraction
        and vote.support_coverage >= p.min_support_coverage
    )

    ref_res = _refine_recording(
        a, b, vote.offset_s, p, anchor_times_s=vote.anchor_times_s
    )
    gates["refine_in_window"] = bool(
        not ref_res.at_window_edge
        and abs(ref_res.offset_s - vote.offset_s) < p.refine_window_s * 0.98
    )

    # Recorded, never gating (see SyncParams): the counts stay in the result
    # so the record shows how well the loudest events agreed.
    matched, total = onset_agreement(a, b, ref_res.offset_s, params=p)

    overlap_s = max(
        0.0,
        min(len(a) / p.sample_rate, len(b) / p.sample_rate + ref_res.offset_s)
        - max(0.0, ref_res.offset_s),
    )
    drift = None
    if overlap_s >= p.drift_min_duration_s:
        drift = estimate_drift(
            a, b, ref_res.offset_s, params=p, anchor_times_s=vote.anchor_times_s
        )
    drift_status, drift_bounded, drift_budget_ok, drift_error = _drift_checks(
        drift, overlap_s, p
    )
    gates["measured_drift_acceptable"] = _measured_drift_acceptable(
        drift, drift_bounded, drift_budget_ok, p
    )

    return RawOffset(
        offset_s=ref_res.offset_s,
        voted_offset_s=vote.offset_s,
        votes=vote.votes,
        runner_up=vote.runner_up,
        onsets_matched=matched,
        onsets_total=total,
        gates={k: gates[k] for k in GATES},
        diagnostics={
            "global_argmax_offset_s": ref_res.global_argmax_s,
            # Recorded, never gating.
            "argmax_is_false_lock": bool(
                ref_res.global_argmax_s is not None
                and abs(ref_res.global_argmax_s - ref_res.offset_s) > p.refine_window_s
            ),
            "refinement_method": "whole_clip"
            if ref_res.global_argmax_s is not None
            else "overlap_windows",
            "overlap_s": overlap_s,
            "ref_audio_duration_s": len(a) / p.sample_rate,
            "other_audio_duration_s": len(b) / p.sample_rate,
            "matched_audio_coverage_s": vote.supported_duration_s,
            "matched_audio_bin_s": 1.0,
            "refinement_windows_s": ref_res.windows_s,
            "refinement_audio_coverage_s": _covered_duration(ref_res.windows_s),
            "refinement_audio_processed_s": float(
                sum(hi - lo for lo, hi in ref_res.windows_s)
            ),
            "landmark_matched_fraction": vote.matched_fraction,
            "vote_support_coverage": vote.support_coverage,
            "drift_status": drift_status,
            "drift_bounded": drift_bounded,
            "drift_error_budget_met": drift_budget_ok,
            "drift_error_bound_s": drift_error,
            "confidence": _confidence(gates, drift_bounded, drift_budget_ok),
            # The min over both clips; `kiwi sync pair --json` writes it
            # verbatim.
            "min_peak_fraction": float(peak_frac),
            "clipped_fraction": float(clipped_frac),
            "vote_margin": vote.margin,
            "vote_support_span_s": vote.support_span_s,
            "drift_ppm": None if drift is None else drift.slope_ppm,
        },
        ref=rp,
        other=op,
        params=p,
        drift=drift,
    )


def estimate_offset(
    ref: Clip | str | Path,
    other: Clip | str | Path,
    *,
    params: SyncParams = SyncParams(),
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    _cache: AudioCache | None = None,
) -> AudioOffset:
    """Offset of ``other`` relative to ``ref`` on the VIDEO timeline.

    Positive = ``other`` started later, so on a shared clock
    ``global_off(other) = global_off(ref) + offset_s``.

    Call :meth:`AudioOffset.require_confident` before persisting the result.

    Accuracy floor: the microphones sit up to ~1.5 m apart, so the estimate
    carries a few ms of acoustic time-of-flight on top of the A/V constant's own
    spread. Do not gate tighter than about one video frame.
    """
    ref = ref if isinstance(ref, Clip) else Clip(Path(ref))
    other = other if isinstance(other, Clip) else Clip(Path(other))
    raw = estimate_raw_offset(
        ref, other, params=params, ffmpeg=ffmpeg, ffprobe=ffprobe, _cache=_cache
    )
    corr = resolve_av_correction(ref, raw.ref, other, raw.other)
    return AudioOffset(offset_s=raw.offset_s + corr.seconds, raw=raw, correction=corr)


def pair_to_global_offsets(delta_s: float) -> tuple[float, float]:
    """Turn a relative pair offset into two frame-0 offsets in the PAIR frame.

    ``delta_s`` is :attr:`AudioOffset.offset_s` for (ref, other): positive means
    ``other`` started later. The returned ``(ref_off, other_off)`` place both
    clips relative to the EARLIER frame 0 (both >= 0, exactly one 0.0) - the
    phase-1 frame. This is NOT the final global clock:
    the global origin (the LAST video stream to start) is settled afterwards
    by :func:`kiwi.sync.clock.to_last_start_origin`, once every stream has
    been placed.
    """
    return (0.0, delta_s) if delta_s >= 0 else (-delta_s, 0.0)

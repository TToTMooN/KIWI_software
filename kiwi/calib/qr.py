"""The rolling-QR clock: render, decode, robust-fit, difference two cameras.

One monitor shows a rolling QR of the host's ``time.monotonic()`` clock
(payload ``KIWI:<seconds>``). Each camera clip filmed at it is fitted
independently to the on-screen clock,

    monotonic = video_t + offset          (one fit per camera),

and because both cameras watch the *same photons*, the monitor's display
latency and refresh phase are COMMON to both and cancel when the two offsets
are differenced, leaving the cross-camera stream offset

    delta = offset_right - offset_left    (positive = right started later)

This is the independent ground truth for the audio sync - and differencing
it against a pair's RAW audio offset is how a camera's per-recording-mode
A/V delay is measured (:mod:`kiwi.calib.avsync`).

Decode rules:

* **Flat decoder only.** Do not use ``detectAndDecodeCurved``.
* **Fisheye clips must be rectified first** (``undistort=``): the flat
  decoder is homography-based.
* **Frame times come from container PTS**, never an assumed index grid.
  One primed read settles POS_MSEC.
* **Hold each code ~50 ms when generating** (see the CLI): a frame decodes
  only if no code transition falls inside its exposure. A held code's
  staleness is common to every camera watching the screen and cancels in
  the difference.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "PAYLOAD_PREFIX",
    "ScanResult",
    "cross_camera_delta",
    "decode_payload",
    "fit_clip",
    "fit_time_map",
    "located_fraction",
    "render_clock_frame",
    "scan_video",
]

PAYLOAD_PREFIX = "KIWI:"


def fit_time_map(
    t_video: np.ndarray, t_mono: np.ndarray, fit_drift: bool
) -> dict[str, float]:
    """Robust-fit ``t_mono ~ slope * t_video + intercept`` from decoded pairs.

    Median offset with the slope fixed at 1 by default; the Theil-Sen slope
    (median of pairwise slopes over well-separated pairs) replaces it under
    ``fit_drift``, for a filmed span minutes long. The fitted drift is
    reported as a diagnostic either way.
    """
    t_video = np.asarray(t_video, dtype=np.float64)
    t_mono = np.asarray(t_mono, dtype=np.float64)
    span = float(t_video.max() - t_video.min()) if len(t_video) > 1 else 0.0

    # Theil-Sen slope: close-in-time pairs have huge slope noise, so only use
    # pairs separated by at least a quarter of the scanned span.
    slope = 1.0
    if span > 0:
        i, j = np.triu_indices(len(t_video), k=1)
        dt = t_video[j] - t_video[i]
        keep = dt > span / 4
        if keep.any():
            slope = float(np.median((t_mono[j] - t_mono[i])[keep] / dt[keep]))

    used_slope = slope if fit_drift else 1.0
    intercept = float(np.median(t_mono - used_slope * t_video))
    resid = t_mono - (used_slope * t_video + intercept)
    t_mid = float(t_video.min() + span / 2)
    return {
        "slope": used_slope,
        "intercept": intercept,
        # The model evaluated mid-span (equals the intercept at slope 1).
        "offset_s": intercept + (used_slope - 1.0) * t_mid,
        "drift_ppm": (slope - 1.0) * 1e6,
        "resid_median_ms": float(np.median(np.abs(resid))) * 1e3,
        "resid_p95_ms": float(np.percentile(np.abs(resid), 95)) * 1e3,
        "resid_max_ms": float(np.abs(resid).max()) * 1e3,
        "n_outliers_100ms": int(np.sum(np.abs(resid) > 0.1)),
    }


def render_clock_frame(
    encoder,  # cv2.QRCodeEncoder
    payload: str,
    qr_size: int,
    margin: int,
    show_timestamp: bool,
) -> np.ndarray:
    """One single-channel QR-clock frame: QR on top, payload text below.

    The text band is sized against a generous 8-integer-digit sample so a
    real (equal-or-shorter) payload never overflows the QR width, keeping
    the canvas - and thus the filmed framing - fixed as the digits roll.
    """
    qr = encoder.encode(payload)
    qr = cv2.resize(qr, (qr_size, qr_size), interpolation=cv2.INTER_NEAREST)
    if margin > 0:
        qr = cv2.copyMakeBorder(
            qr, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255
        )
    full = qr_size + 2 * margin
    if not show_timestamp:
        return qr

    font = cv2.FONT_HERSHEY_SIMPLEX
    sample = f"{PAYLOAD_PREFIX}00000000.000000"
    (w1, _), _ = cv2.getTextSize(sample, font, 1.0, 2)
    scale = 0.9 * full / w1
    thick = max(2, round(scale * 2))
    (_, sh), sb = cv2.getTextSize(sample, font, scale, thick)

    canvas = np.full((full + sh + sb + 2 * margin, full), 255, dtype=np.uint8)
    canvas[:full, :full] = qr
    (tw, th), _ = cv2.getTextSize(payload, font, scale, thick)
    cv2.putText(
        canvas,
        payload,
        ((full - tw) // 2, full + margin + th),
        font,
        scale,
        0,
        thick,
        cv2.LINE_AA,
    )
    return canvas


def clock_frames(
    encoder,  # cv2.QRCodeEncoder
    qr_size: int,
    margin: int,
    show_timestamp: bool,
    hold_s: float,
    now,  # () -> float, time.monotonic
):
    """Yield ``(payload, frame)`` for the rolling clock, resampling on a grid.

    THE payload-timing rule, shared by every display backend: the payload is
    stamped when the QR is BUILT, not when it is shown. A held code's
    staleness is then uniform on ``[0, hold]``, common to every camera
    watching the screen, and cancels in the cross-camera difference exactly
    as display latency does. A backend calls this as fast as it can redraw;
    the grid, not the caller, decides when the label advances.
    """
    payload, frame, next_sample = "", None, 0.0
    while True:
        t = now()
        if frame is None or t >= next_sample:
            payload = f"{PAYLOAD_PREFIX}{t:.6f}"
            frame = render_clock_frame(
                encoder, payload, qr_size, margin, show_timestamp
            )
            next_sample = t + hold_s
        yield payload, frame


def pgm_bytes(frame) -> bytes:
    """Single-channel frame as binary PGM, which Tk's PhotoImage reads natively.

    The Tk backend's equivalent of ``cv2.cvtColor(..., GRAY2BGR)``: a format
    shim, no resampling, so both backends display the same pixels.
    """
    height, width = frame.shape
    return b"P5\n%d %d\n255\n" % (width, height) + frame.tobytes()


def decode_payload(
    detector,  # cv2.QRCodeDetector
    frame: np.ndarray,
) -> float | None:
    """Decode one frame's clock value, or None on any kind of miss.

    Flat ``detectAndDecode`` only; detector-internal cv2 errors, alien QR
    content, and blended decodes that kept the prefix all count as a plain
    miss.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    try:
        payload, _, _ = detector.detectAndDecode(gray)
    except cv2.error:
        return None
    if not payload or not payload.startswith(PAYLOAD_PREFIX):
        return None
    try:
        return float(payload[len(PAYLOAD_PREFIX) :])
    except ValueError:
        return None  # garbled digits behind an intact prefix


@dataclasses.dataclass(frozen=True)
class ScanResult:
    """One clip's decoded clock samples."""

    t_video: np.ndarray  # (N,) container-PTS seconds
    t_mono: np.ndarray  # (N,) decoded monotonic seconds
    n_scanned: int
    fps: float


def scan_video(
    video: Path,
    *,
    start: float = 0.0,
    duration: float | None = None,
    stride: int = 1,
    max_decodes: int = 300,
    undistort: Path | None = None,
    undistort_fov: float = 130.0,
    undistort_lens: str = "front",
) -> ScanResult:
    """Scan a clip for the QR clock; raises ``RuntimeError`` on unopenable input.

    Frame times come from the container PTS; rectification, when requested,
    is spatial only - the frame's PTS is untouched.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError(f"{video} reports no FPS; cannot map frames to time")
    start_idx = int(round(start * fps))
    if start_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    # Prime the decoder before trusting POS_MSEC: on the first read after an
    # open it reports the newest fed *packet's* PTS, which leads the delivered
    # frame by the B-pyramid depth. One discarded read settles it.
    cap.read()

    detector = cv2.QRCodeDetector()
    rectify = None  # built lazily: the source size is only known from a frame
    t_video: list[float] = []
    t_mono: list[float] = []
    scanned = 0
    n = 0  # frames consumed since priming, for stride bookkeeping only
    t = start
    while len(t_video) < max_decodes:
        if n % stride:
            if not cap.grab():
                break
            n += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        # Backends without PTS report 0; fall back to the nominal grid.
        t = pos_ms / 1000.0 if pos_ms > 0 else (start_idx + n) / fps
        if duration is not None and t - start > duration:
            break
        scanned += 1
        if undistort is not None:
            if rectify is None:
                from kiwi.core.rectify import load_rectifier_auto

                h, w = frame.shape[:2]
                rectify = load_rectifier_auto(
                    undistort,
                    (w, h),
                    lens=undistort_lens,
                    out_size=(w, h),
                    fov_deg=undistort_fov,
                )
            frame = rectify.apply(frame)
        mono = decode_payload(detector, frame)
        if mono is not None:
            t_video.append(t)
            t_mono.append(mono)
    cap.release()
    return ScanResult(
        t_video=np.asarray(t_video),
        t_mono=np.asarray(t_mono),
        n_scanned=scanned,
        fps=float(fps),
    )


def fit_clip(scan: ScanResult, fit_drift: bool = False) -> dict:
    """One clip's fit, with the standard error the differencing step needs.

    Raises ``RuntimeError`` below 5 decodes - too few to fit; the caller
    diagnoses (framing vs unreadable) via :func:`located_fraction`. The
    median has ~1.253x the standard error of the mean, so the offset SE is
    ``1.2533 * resid_median / sqrt(n)``.
    """
    if len(scan.t_video) < 5:
        raise RuntimeError(
            f"only {len(scan.t_video)} QR decode(s) in {scan.n_scanned} scanned frames"
        )
    fit = fit_time_map(scan.t_video, scan.t_mono, fit_drift)
    fit["offset_se_ms"] = 1.2533 * fit["resid_median_ms"] / np.sqrt(len(scan.t_video))
    fit["n_decoded"] = len(scan.t_video)
    fit["n_scanned"] = scan.n_scanned
    fit["fps"] = scan.fps
    return fit


def cross_camera_delta(fit_left: dict, fit_right: dict) -> tuple[float, float]:
    """``(delta_s, se_ms)``: right minus left, positive = right started later.

    ``offset = monotonic - video_t``, so a later start means a larger
    offset - the sign matches the audio and episode.yaml convention.
    """
    delta = fit_right["offset_s"] - fit_left["offset_s"]
    se = float(np.hypot(fit_left["offset_se_ms"], fit_right["offset_se_ms"]))
    return float(delta), se


def located_fraction(video: Path, n_samples: int = 12) -> float:
    """Fraction of sampled frames where the detector at least LOCATES a QR.

    Decode failures are ambiguous alone; this splits them into "not in
    frame" (framing/distance/glare) vs "in frame but unreadable" (exposure
    spanning code transitions, or a fisheye bending the module grid) -
    which point at opposite fixes.
    """
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    det, found = cv2.QRCodeDetector(), 0
    for k in range(n_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 0.5) / n_samples))
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            _, pts, _ = det.detectAndDecode(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        except cv2.error:
            pts = None
        found += pts is not None
    cap.release()
    return found / n_samples

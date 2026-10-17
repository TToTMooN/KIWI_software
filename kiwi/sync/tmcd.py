"""SMPTE ``tmcd`` timecode reading for Insta360 ``.insv`` clips.

Insta360 app-recorded clips carry a 4th ``tmcd`` (timecode) data stream that the
app writes from the phone's wall clock at jam-sync time. The label is non-drop
``HH:MM:SS:FF`` on a 29.97 base **even for 119.88 fps HFR clips** - four video
frames then share one timecode label.

Converting a label to elapsed seconds must therefore use the *tmcd* stream's own
base rate (read from the ``d:0`` data stream), NOT the video stream's
``r_frame_rate``.

**Verification only.** Jam status is not observable from a recording: an X5
writes a tmcd track whether jammed or not. :mod:`kiwi.sync.audio` estimates the
offsets used by :mod:`kiwi.sync.clock`; this module supports
``kiwi sync survey --tmcd`` by reporting audio-vs-timecode skew as an independent
cross-check. Timecode does not produce the pipeline clock.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Non-drop SMPTE runs the label counter at 30 frames per labeled second while the
# real media rate is 30000/1001 = 29.97..., so each labeled second lasts 1.001 s.
NDF_BASE_FPS = 30000.0 / 1001.0


def _ffprobe(args: list[str], path: Path, ffprobe: str) -> str:
    out = subprocess.run(
        [ffprobe, "-v", "error", *args, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def read_tmcd(path: Path, ffprobe: str = "ffprobe") -> tuple[str, float]:
    """Return ``(timecode, base_fps)`` from the clip's tmcd/data stream.

    ``timecode`` is the SMPTE ``HH:MM:SS:FF`` label string. ``base_fps`` is the
    tmcd *label* base rate (~29.97 for non-drop), read from the ``d:0`` data
    stream so it is correct regardless of the video stream's frame rate. Raises
    ``RuntimeError`` if the clip carries no timecode tag.
    """
    tc = _ffprobe(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream_tags=timecode",
            "-of",
            "default=nw=1:nk=1",
        ],
        path,
        ffprobe,
    )
    if not tc:
        raise RuntimeError(f"no SMPTE timecode (tmcd) in {Path(path).name}")
    base = _ffprobe(
        [
            "-select_streams",
            "d:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=nw=1:nk=1",
        ],
        path,
        ffprobe,
    )
    num, _, den = base.partition("/")
    try:
        base_fps = float(num) / float(den)
        if base_fps <= 0:
            raise ValueError
    except (ValueError, ZeroDivisionError):
        base_fps = NDF_BASE_FPS  # e.g. "0/0" or an empty/absent d:0 rate
    return tc.strip(), base_fps


def tc_to_seconds(tc: str, base_fps: float) -> float:
    """Non-drop SMPTE ``HH:MM:SS:FF`` -> elapsed seconds at the track's true rate.

    Count label frames first, then divide by the real rate. A non-drop label
    second holds ``round(base_fps)`` frames but lasts ``round(base_fps)/base_fps``
    real seconds (1.001 s at 29.97).
    """
    hh, mm, ss, ff = (int(p) for p in re.split("[:;]", tc.strip()))
    labels_per_sec = round(base_fps)
    return ((hh * 3600 + mm * 60 + ss) * labels_per_sec + ff) / base_fps


def read_tmcd_seconds(path: Path, ffprobe: str = "ffprobe") -> float:
    """Elapsed seconds of the clip's start timecode (``read_tmcd`` + conversion)."""
    tc, base_fps = read_tmcd(path, ffprobe)
    return tc_to_seconds(tc, base_fps)

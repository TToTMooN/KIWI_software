"""The ``av_sync_v1`` writer - a camera's per-mode audio-vs-video delay table.

``t_audio = t_video + av_delay_s``, one constant per **exact** recording mode
(resolution included). Only the DIFFERENCE between two clips' delays is
observable, so the scale needs a declared zero - the reference camera/mode
named in the document.

The measurement itself is a composition this module does not perform: a
rolling-QR video ground truth (:mod:`kiwi.calib.qr`) differenced against
the pair's RAW audio offset, converted by
``kiwi.sync.audio.av_delay_from_ground_truth`` - the single place the
calibration sign lives. The ``kiwi calib qr av-delay`` verb composes the
two components; this module only owns the document.

Multiple runs of the same mode accumulate: each run's derived value is
appended to ``derived_ms`` provenance and ``av_delay_s`` is their mean.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import yaml

from kiwi.core.camera.registry import ARTIFACT_SCHEMAS

__all__ = [
    "AV_DELAY_PLAUSIBLE_MAX_S",
    "AV_SYNC_SCHEMA",
    "ExtractionMismatch",
    "add_measurement",
    "carry_forward",
    "load_av_sync",
    "new_av_sync_doc",
    "write_av_sync",
]

AV_SYNC_SCHEMA = ARTIFACT_SCHEMAS["av_sync"]

# Upper bound on |av_delay_s|: a larger magnitude (a swapped ref/other
# against the qr sync left/right, or a periodic false lock of the audio
# estimator) is refused rather than averaged into av_delay_s.
AV_DELAY_PLAUSIBLE_MAX_S = 0.25

_HEADER = """\
# Audio-vs-video stamping delay, per recording mode.
#
#   t_audio = t_video + av_delay_s
#
# Only the DIFFERENCE between two clips' delays is observable, so the scale
# needs a zero: the `reference:` camera/mode below is it, by definition.
#
# A mode absent from `modes:` is UNMEASURED and MUST fail rather than default
# to zero. This constant depends on the exact recording mode, RESOLUTION
# INCLUDED. Re-measure after ANY mode change.
#
# Keys are DISPLAY-normalized (rotation applied) - kiwi.sync's RecordingMode
# normalizes portrait/landscape storage of the same physical mode to one key.
"""


def new_av_sync_doc(
    *,
    unit: str,
    reference: str,
    extraction: str,
    model: str | None = None,
) -> dict:
    """A fresh ``av_sync_v1`` document with no modes yet.

    ``extraction`` records the exact audio-decode recipe the constants are
    measured under (``kiwi.sync.audio.EXTRACTION``); a constant measured
    with one recipe is not valid for another. ``reference`` names the
    camera/mode that defines the scale's zero.
    """
    return {
        "schema": AV_SYNC_SCHEMA,
        "unit": unit,
        **({"model": model} if model else {}),
        "reference": reference,
        "method": (
            "rolling-QR video ground truth (kiwi calib qr sync -> delta_qr_ms) "
            "minus the raw audio estimate (kiwi.sync.audio.estimate_raw_offset); "
            "converted by kiwi.sync.audio.av_delay_from_ground_truth, the single "
            "place the sign lives."
        ),
        "extraction": extraction,
        "measured_at": str(datetime.date.today()),
        "modes": {},
    }


def add_measurement(doc: dict, *, mode_key: str, delay_s: float, run: str) -> dict:
    """Fold one measured run into a mode entry; returns its updated entry.

    Appends the run label and its derived value to the provenance lists and
    recomputes ``av_delay_s`` as the mean of every derived value.
    """
    if doc.get("schema") != AV_SYNC_SCHEMA:
        raise ValueError(f"not an {AV_SYNC_SCHEMA} document")
    modes = doc.setdefault("modes", {})
    entry = modes.setdefault(mode_key, {"runs": [], "derived_ms": []})
    entry.setdefault("runs", []).append(run)
    entry.setdefault("derived_ms", []).append(round(delay_s * 1e3, 1))
    entry["av_delay_s"] = round(
        sum(entry["derived_ms"]) / len(entry["derived_ms"]) / 1e3, 4
    )
    doc["measured_at"] = str(datetime.date.today())
    return entry


def load_av_sync(path: Path) -> dict:
    """Read an ``av_sync_v1`` document back for updating; schema-checked."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    if doc.get("schema") != AV_SYNC_SCHEMA:
        raise ValueError(
            f"{path}: schema {doc.get('schema')!r}, expected {AV_SYNC_SCHEMA!r}"
        )
    return doc


class ExtractionMismatch(ValueError):
    """A published table was measured under a different extraction recipe.

    Its constants are not valid under the current one, so they cannot be
    carried into a new document - the modes need re-measuring.
    """

    def __init__(self, path: Path, published: str | None, current: str) -> None:
        self.path = Path(path)
        self.published = published
        self.current = current
        super().__init__(
            f"{path} was measured under a different extraction recipe: "
            f"published {published!r}, current {current!r}"
        )


def carry_forward(path: Path, *, extraction: str) -> dict:
    """The published document to keep measuring into.

    A new dated table STARTS from the published one, so modes accumulate
    across documents, not just across runs of one.

    Raises :class:`ExtractionMismatch` when the published constants were
    measured under a different audio-decode recipe: those modes are
    unmeasured under the new recipe and must fail rather than default (the
    rule in the file header).
    """
    doc = load_av_sync(path)
    published = doc.get("extraction")
    if published != extraction:
        raise ExtractionMismatch(path, published, extraction)
    return doc


def write_av_sync(path: Path, doc: dict) -> Path:
    """Write the document with its contract header - the single writer."""
    if doc.get("schema") != AV_SYNC_SCHEMA:
        raise ValueError(f"not an {AV_SYNC_SCHEMA} document")
    path = Path(path)
    path.write_text(_HEADER + yaml.safe_dump(doc, sort_keys=False))
    return path

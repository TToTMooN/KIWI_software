"""The ``kiwi_calib.json`` run manifest - run facts recorded at staging time.

KIWI's staging layer records here the run facts it knows at write time:
image sizes, per-camera lens identity (the staged topics are
``/cam0/image_raw`` for BOTH lenses - bagcreater derives topics from folder
names, so the topic does not say which lens; the manifest and the run-dir
name carry it), frame counts for the in-container bag-message assertion, and
the clock constants that make a re-run reproducible.

With the manifest in place the bag itself is deletable after a run -
nothing reads it back on the host.
"""

from __future__ import annotations

import json
from pathlib import Path

from kiwi.core.provenance import provenance_block

__all__ = [
    "MANIFEST_NAME",
    "RUN_SCHEMA",
    "image_size",
    "read_manifest",
    "write_manifest",
]

RUN_SCHEMA = "kiwi_calib_run_v1"
MANIFEST_NAME = "kiwi_calib.json"


def write_manifest(
    work_dir: Path,
    *,
    kind: str,  # "cameras" | "imucam"
    cams: list[dict],  # per-camera: id, lens, topic, image_size [W, H], n_frames, ...
    imu: dict | None = None,  # topic, csv, n_samples, units
    clock: dict | None = None,  # fps, period_ns, offset_ns
    bag: dict | None = None,  # name, expected_messages
    unit: str | None = None,
    extras: dict | None = None,
) -> Path:
    """Write the run manifest into ``work_dir``; returns its path."""
    doc = {
        "schema": RUN_SCHEMA,
        "kind": kind,
        **({"unit": unit} if unit else {}),
        "cams": cams,
        **({"imu": imu} if imu else {}),
        **({"clock": clock} if clock else {}),
        **({"bag": bag} if bag else {}),
        "provenance": provenance_block(),
        **(extras or {}),
    }
    out = Path(work_dir) / MANIFEST_NAME
    out.write_text(json.dumps(doc, indent=2) + "\n")
    return out


def read_manifest(work_dir: Path) -> dict:
    """Read and validate the run manifest of ``work_dir``.

    Raises ``FileNotFoundError`` when the run dir has none (a hand-built dir -
    :func:`kiwi.calib.intrinsics.staged_image_size` then probes a staged PNG,
    never the bag) and ``ValueError`` on a wrong schema.
    """
    p = Path(work_dir) / MANIFEST_NAME
    if not p.is_file():
        raise FileNotFoundError(
            f"{work_dir}: no {MANIFEST_NAME} (not a kiwi calib run dir?)"
        )
    doc = json.loads(p.read_text())
    if doc.get("schema") != RUN_SCHEMA:
        raise ValueError(f"{p}: schema {doc.get('schema')!r}, expected {RUN_SCHEMA!r}")
    return doc


def image_size(work_dir_or_doc: Path | dict, cam_id: str = "cam0") -> tuple[int, int]:
    """``(W, H)`` of one staged camera stream, from the manifest.

    Raises ``KeyError`` when the manifest has no such camera.
    """
    doc = (
        work_dir_or_doc
        if isinstance(work_dir_or_doc, dict)
        else read_manifest(work_dir_or_doc)
    )
    for cam in doc.get("cams", []):
        if cam.get("id") == cam_id:
            w, h = cam["image_size"]
            return int(w), int(h)
    raise KeyError(f"manifest has no camera {cam_id!r}")

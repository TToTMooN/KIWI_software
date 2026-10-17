"""Readers for register-stage publications and their recorded file digests.

The register stage writes one marker document per arm and an aggregate
that binds those marker documents by digest. By default, the readers
verify the registration trajectory, detections, state record and state
trajectory against their recorded digests. Missing or changed artifacts
invalidate the publication.
"""

from __future__ import annotations

import json
from pathlib import Path

from kiwi.core import layout
from kiwi.core.config_hash import hash_path

__all__ = ["read_register_inputs", "read_registration_bundle"]

_MARKER_FIELDS = {
    "schema",
    "method",
    "register_tum_bytes_sha256",
    "detections_bytes_sha256",
    "state_inputs_bytes_sha256",
}


def read_register_inputs(stream_dir: Path, *, verify: bool = True) -> dict:
    """One arm's ``register.inputs.json``, verified against its artifacts.

    ``verify=True`` checks the digests of ``register.tum``,
    ``detections.json`` and ``scene/state.inputs.json``, then verifies the
    state record's trajectory digest. A missing file or a digest mismatch
    refuses the document - the marker describes those exact artifact bytes.
    """
    stream_dir = Path(stream_dir)
    path = stream_dir / layout.REGISTER_INPUTS
    doc = json.loads(path.read_text(encoding="utf-8"))
    missing = _MARKER_FIELDS - set(doc)
    if missing:
        raise ValueError(f"{path}: missing required field(s) {sorted(missing)}")
    if doc["schema"] != "register_inputs":
        raise ValueError(
            f"{path}: schema is {doc['schema']!r}, expected 'register_inputs'"
        )
    if doc["method"] != "marker_anchor":
        raise ValueError(
            f"{path}: method {doc['method']!r} is not a method this reader "
            f"knows; a new method arrives with its own reader support"
        )
    state_name = f"{layout.SCENE_DIR}/{layout.STATE_INPUTS}"
    if verify:
        for field, name in (
            ("register_tum_bytes_sha256", layout.REGISTER_TUM),
            ("detections_bytes_sha256", layout.DETECTIONS),
            ("state_inputs_bytes_sha256", state_name),
        ):
            artifact = stream_dir / name
            if not artifact.is_file():
                raise ValueError(f"{path}: referenced {name} is missing")
            actual = hash_path(artifact)
            if actual != doc[field]:
                raise ValueError(
                    f"{path}: {name} digest mismatch (recorded "
                    f"{doc[field][:12]}.., actual {actual[:12]}..) - the "
                    f"artifact changed after publication; no valid "
                    f"registration exists for this arm"
                )
        from kiwi.core.digest import verify_digest

        state_path = stream_dir / state_name
        state = json.loads(state_path.read_text(encoding="utf-8"))
        outputs = state.get("outputs") if isinstance(state, dict) else None
        trajectory = outputs.get("trajectory") if isinstance(outputs, dict) else None
        if not isinstance(trajectory, dict) or not isinstance(
            trajectory.get("digest"), str
        ):
            raise ValueError(f"{state_path}: missing trajectory digest")
        name = trajectory.get("path")
        if (
            not isinstance(name, str)
            or (state_path.parent / name).resolve()
            != (state_path.parent / layout.STATE_TUM).resolve()
        ):
            raise ValueError(
                f"{state_path}: trajectory path must name {layout.STATE_TUM}"
            )
        verify_digest(trajectory["digest"], state_path.parent / name)
    return doc


def read_registration_bundle(episode_dir: Path, *, verify: bool = True) -> dict:
    """Read the aggregate and verify every listed arm's marker digest.

    The arm mapping must be nonempty. ``verify=True`` also verifies each
    marker's referenced artifacts through :func:`read_register_inputs`.
    """
    episode_dir = Path(episode_dir)
    path = episode_dir / layout.REGISTRATION_BUNDLE
    doc = json.loads(path.read_text(encoding="utf-8"))
    for field in ("schema", "method", "arms"):
        if field not in doc:
            raise ValueError(f"{path}: missing required field {field!r}")
    if doc["schema"] != "registration_bundle":
        raise ValueError(
            f"{path}: schema is {doc['schema']!r}, expected 'registration_bundle'"
        )
    if doc["method"] != "marker_anchor":
        raise ValueError(f"{path}: unknown method {doc['method']!r}")
    if not isinstance(doc["arms"], dict) or not doc["arms"]:
        raise ValueError(f"{path}: arms must be a non-empty mapping")
    for arm, entry in doc["arms"].items():
        recorded = entry.get("register_inputs_bytes_sha256")
        if not recorded:
            raise ValueError(f"{path}: arm {arm!r} records no marker digest")
        marker = episode_dir / arm / layout.REGISTER_INPUTS
        if not marker.is_file():
            raise ValueError(f"{path}: arm {arm!r} has no {layout.REGISTER_INPUTS}")
        actual = hash_path(marker)
        if actual != recorded:
            raise ValueError(
                f"{path}: arm {arm!r} marker digest mismatch - a later "
                f"per-arm rewrite invalidated this bundle; re-run the "
                f"register stage"
            )
        if verify:
            read_register_inputs(episode_dir / arm)
    return doc

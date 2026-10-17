"""Content hashes for the calibration + config files a pipeline run consumed.

:mod:`kiwi.core.provenance` answers "which code produced this artifact" with a git
commit. This module pins the calibration files a run read (intrinsics, cam<->IMU
extrinsics, the tool design's ``session:`` recipe and the gripper unit's
``build:`` block - its tag, its calibrated ``fixed_pose``, its mount
overrides - the robot description URDF the export's camera->tip constant is
composed from, and the mapping profile) by content.

Two normalizations:

* **YAML is parsed, not read.** The hash covers the *parsed* document: comment
  and layout edits leave it unchanged, and it moves only when a value moves.
* **.npz is hashed by array, not by file.** An npz is a zip, and zip embeds a
  per-entry mtime, so regenerating byte-identical arrays yields a different file
  hash. Hashing dtype + shape + buffer per key is stable across regeneration.

Nothing in here raises. Stamping is observational: a missing or unreadable input
must be *recorded* as such, never turned into a failed pipeline run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from kiwi.core.opencv_yaml import strip_opencv_directive

__all__ = [
    "hash_dir",
    "hash_npz",
    "hash_path",
    "hash_yaml",
    "stamp_input",
    "stamp_inputs",
]

# Field separator, so ("ab", "c") and ("a", "bc") cannot hash alike.
_SEP = b"\x00"

# Documentation, in the same class as a YAML comment.
_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt"})


def _canonical(doc: object) -> str:
    """Parsed YAML -> one canonical string, with key order and layout removed.

    ``default=str`` covers the types PyYAML emits that json does not carry on its
    own - ``datetime.date`` from an unquoted ``date:`` field, most often.
    """
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str)


def hash_yaml(path: str | Path) -> str:
    """sha256 of a YAML file's *parsed content* - comment- and layout-insensitive.

    Accepts OpenCV's dialect as well as standard YAML. The %YAML:1.0
    directive is a format marker, not content; dropping it loses no values.
    """
    text = strip_opencv_directive(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(_canonical(yaml.safe_load(text)).encode("utf-8")).hexdigest()


def hash_npz(path: str | Path) -> str:
    """sha256 of an ``.npz``'s arrays - stable across an identical regeneration.

    numpy is imported lazily.
    """
    import numpy as np

    h = hashlib.sha256()
    # Own the handle even if np.load raises before returning an NpzFile.
    with Path(path).open("rb") as source, np.load(source, allow_pickle=False) as z:
        for key in sorted(z.files):
            arr = np.ascontiguousarray(z[key])
            for field in (
                key.encode(),
                str(arr.dtype).encode(),
                str(arr.shape).encode(),
            ):
                h.update(field)
                h.update(_SEP)
            h.update(arr.tobytes())
            h.update(_SEP)
    return h.hexdigest()


def hash_dir(path: str | Path) -> str:
    """sha256 over a directory tree: each file's relative path plus its own hash.

    Recurses through :func:`hash_path`, so a directory of YAML inherits
    :func:`hash_yaml`'s comment-insensitivity. Files are walked in sorted order
    so the result does not depend on filesystem iteration order, and
    documentation files are skipped, like comments.
    """
    path = Path(path)
    h = hashlib.sha256()
    files = (
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() not in _DOC_SUFFIXES
    )
    for f in sorted(files):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(_SEP)
        h.update(hash_path(f).encode())
        h.update(_SEP)
    return h.hexdigest()


def hash_path(path: str | Path) -> str:
    """Content hash of a file or directory, dispatched on kind then suffix."""
    path = Path(path)
    if path.is_dir():
        return hash_dir(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return hash_yaml(path)
    if suffix == ".npz":
        return hash_npz(path)
    # Streamed in chunks: same digest as ``read_bytes()``, bounded memory.
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path, root: Path | None) -> str:
    """``path`` relative to ``root`` when it is underneath it, else unchanged."""
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def stamp_input(path: str | Path, *, root: str | Path | None = None) -> dict:
    """One provenance entry: ``{path, target?, sha256 | missing | error}``.

    ``root`` (the repo root) shortens ``path`` to a repo-relative string. A
    symlinked input also records its ``target`` (the dated target name, e.g.
    ``extrinsic-imu.yaml``).

    A missing input is recorded, not raised.
    """
    path = Path(path)
    root = Path(root) if root is not None else None
    entry: dict = {"path": _rel(path, root)}
    if path.is_symlink():
        entry["target"] = str(path.readlink())
    if not path.exists():
        entry["missing"] = True
        return entry
    try:
        entry["sha256"] = hash_path(path)
    except Exception as exc:  # provenance must never fail the run it describes
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def stamp_inputs(
    paths: dict[str, str | Path | None], *, root: str | Path | None = None
) -> dict:
    """Stamp a ``{label: path}`` mapping, dropping ``None`` paths.

    ``None`` means "this run does not use that input at all" (an arm with no
    ``gripper_unit``, a stream with no camera registry entry), which is a
    different claim from an input that should exist and does not - so it is
    omitted rather than recorded as missing.
    """
    return {
        label: stamp_input(path, root=root)
        for label, path in paths.items()
        if path is not None
    }

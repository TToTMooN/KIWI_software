"""Small, observational generation records shared by the pipeline writers.

These helpers do not decide whether an existing result may be reused. A
writer calls them when it actually generates a result; an input that cannot
be recovered is recorded as missing, never attributed to the current code.
Stage-specific fields stay in each stage's own record.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile

import yaml
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from kiwi.core.atomic import atomic_write_text
from kiwi.core.digest import (
    digest_bytes,
    digest_canonical_json,
    digest_canonical_yaml,
    digest_npz,
)
from kiwi.core.paths import repo_root
from kiwi.core.provenance import provenance_block


def json_value(value):
    """Encode already-computed arrays and missing values as ordinary JSON."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, SimpleNamespace):
        return json_value(vars(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_record(path: str | Path, payload: dict) -> Path:
    """Atomically write JSON; uncomputed/nonfinite measurements become null."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_text(
        path,
        json.dumps(json_value(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


@lru_cache(maxsize=256)
def _file_digest(location: str, selected: str, stat_key: tuple) -> str:
    """Cache only while the file's size, mtime, ctime and inode stay unchanged."""
    path = Path(location)
    try:
        if selected in ("json", "canonical_json_sha256"):
            return digest_canonical_json(json.loads(path.read_text()))
        if selected in ("yaml", "yml", "canonical_yaml_sha256"):
            return digest_canonical_yaml(path)
        if selected in ("npz", "npz_semantic_sha256"):
            return digest_npz(path)
    except (ValueError, TypeError, yaml.YAMLError, zipfile.BadZipFile):
        # A malformed or foreign file can still be identified by its bytes.
        # This fallback changes the named algorithm, never fakes its result.
        return digest_bytes(path)
    return digest_bytes(path)


def file_ref(path, *, base=None, record=None, digest=None, kind=None) -> dict:
    """Identify an actual file and optionally its own generation record.

    Paths are relative to ``base`` (normally the containing record's
    directory), including external inputs through ``..``. An existing
    typed digest may be reused.
    A missing file is recorded explicitly (``status: missing``); this is not
    a reuse gate.
    """
    if path is None:
        return {"path": None, "digest": None, "status": "unknown"}
    path = Path(path).absolute()
    location = os.path.relpath(path, Path(base).absolute()) if base else str(path)
    ref = {"path": location}
    if record is not None:
        ref["record"] = (
            record if isinstance(record, dict) else file_ref(record, base=base)
        )
    if digest is None:
        if not path.is_file():
            return {**ref, "digest": None, "status": "missing"}
        selected = kind or path.suffix.lower().lstrip(".")
        stat = path.stat()
        digest = _file_digest(
            str(path),
            selected,
            (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino),
        )
    ref["digest"] = digest
    return ref


def producer_info(algorithm: str, *, dependencies=()) -> dict:
    """Record generation-time code and the actual available dependency versions."""
    info = {"algorithm": algorithm, **provenance_block()}
    versions = {}
    aliases = {"opencv-python-headless": "cv2", "opencv-python": "cv2"}
    for name in dependencies:
        module = sys.modules.get(aliases.get(name, name))
        loaded = getattr(module, "__version__", None)
        try:
            versions[name] = loaded or importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = loaded
            if name in ("ffmpeg", "ffprobe") and (binary := shutil.which(name)):
                try:
                    result = subprocess.run(
                        [binary, "-version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=True,
                    )
                    versions[name] = {
                        "version": result.stdout.splitlines()[0],
                        "binary": file_ref(binary, kind="bytes"),
                    }
                except (OSError, subprocess.SubprocessError, IndexError):
                    pass
    info["dependencies"] = versions
    # A dirty checkout also records its tracked Python changes and untracked
    # Python modules; a clean checkout records no snapshot.
    root = repo_root()
    if root is not None and info["kiwi_git"].get("dirty"):
        try:
            patch = subprocess.run(
                ["git", "-C", str(root), "diff", "--binary", "HEAD", "--", "kiwi"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout
            untracked = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "--",
                    "kiwi",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.splitlines()
            info["source_changes"] = {
                "patch": patch,
                "untracked_python": {
                    name: (root / name).read_text()
                    for name in untracked
                    if name.endswith(".py")
                },
            }
        except (OSError, subprocess.SubprocessError, UnicodeError):
            info["source_changes"] = None
    return info

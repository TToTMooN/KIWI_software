"""Atomic small-file writes: unique temp in the same directory, then rename.

A reader sees the previous complete file or the new complete file, never a
mix: ``os.replace`` on the same filesystem gives exactly that, and the
temporary name is UNIQUE per write (``tempfile.mkstemp``), so two writers
racing on one path cannot stomp each other's temp file - the loser's rename
lands second.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["atomic_write_bytes", "atomic_write_text"]


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))

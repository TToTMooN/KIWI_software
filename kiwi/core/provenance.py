"""Provenance stamping: which code + toolchain produced a derived artifact.

Every derived artifact in the pipeline - LeRobot exports, run records, Kalibr
run manifests, generation records - records the git commit of this repo that
produced it. The commit stamp is a validity key for a *backed-up* derived
artifact (``commit == HEAD && not dirty`` -> trust; else diff the relevant
paths and regenerate if they changed).

The git commit pins *our* code; ``pixi.lock`` pins the external toolchain
(ffmpeg / opencv / ...), which the commit alone does not. Both together are what
make a derived artifact reproducible.

Import-light (stdlib only), so any package can stamp.

Repo-root discovery lives in :mod:`kiwi.core.paths` (the one root-finder);
the stamp key is ``kiwi_git``.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from kiwi.core.paths import repo_root

__all__ = ["git_stamp", "provenance_block"]


def _git(root: Path, *args: str) -> str | None:
    """Run ``git -C root <args>``; return stripped stdout, or None on any failure."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_stamp(start: str | Path | None = None) -> dict:
    """``{commit, dirty, branch}`` for the repo containing ``start``.

    ``commit`` is the full HEAD sha (``None`` when git is unavailable or ``start``
    is not in a work tree). ``dirty`` is True when the work tree has uncommitted
    changes - a dirty stamp means the artifact is *not* reproducible from the
    commit alone, so it must never be trusted as a cache entry.
    """
    root = repo_root(start)
    if root is None:
        return {"commit": None, "dirty": None, "branch": None}
    status = _git(root, "status", "--porcelain")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "dirty": (bool(status) if status is not None else None),
        "branch": (branch or None),
    }


def _pixi_lock_sha(root: Path) -> str | None:
    lock = root / "pixi.lock"
    if not lock.is_file():
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:12]


def provenance_block(start: str | Path | None = None) -> dict:
    """The stamp embedded in a derived artifact's provenance: git + toolchain.

    ``{"kiwi_git": {commit, dirty, branch}, "pixi_lock_sha": <hex|None>}`` - the
    commit pins our code, the lock hash pins ffmpeg/opencv/etc. Drop into an
    artifact's provenance dict with ``{**provenance_block(), ...}``.
    """
    block: dict = {"kiwi_git": git_stamp(start)}
    root = repo_root(start)
    block["pixi_lock_sha"] = _pixi_lock_sha(root) if root is not None else None
    return block

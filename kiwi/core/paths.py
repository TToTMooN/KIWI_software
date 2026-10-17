"""Where the deployment tree lives: repo-root discovery + the config root.

The single config-root seam. There is one repo-root finder, and every module
that reads deployment config (cameras, tools, markers and mapping
profiles) routes through :func:`config_root`, so a deployment can relocate its
config tree with one environment variable.

Resolution order for the config root:

1. ``<episode>/config``, when an episode is supplied and has a local copy.
2. ``$KIWI_CONFIG``, when set - an installed (non-checkout) deployment
   points this at its config tree.
3. ``<repo_root>/config`` - the layout of a git checkout, where
   ``config/`` ships with the repo.

:func:`descriptions_root` is the same seam for the robot-description tree
(``descriptions/`` - URDF / MJCF / meshes for export and replay), with
``$KIWI_DESCRIPTIONS`` playing the role of ``$KIWI_CONFIG``. The wheel
ships code only, so an installed deployment that exports or replays must
point the variable at a checkout's ``descriptions/`` tree.

Stdlib-only (like :mod:`kiwi.core.provenance`).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "CONFIG_ENV",
    "DESCRIPTIONS_ENV",
    "config_root",
    "descriptions_root",
    "repo_root",
]

# Environment variable overriding the config-tree location.
CONFIG_ENV = "KIWI_CONFIG"

# Environment variable overriding the robot-description-tree location.
DESCRIPTIONS_ENV = "KIWI_DESCRIPTIONS"


def _is_work_tree(p: Path) -> bool:
    """Whether ``p`` holds a git marker git itself would accept.

    A work tree's ``.git`` is either a directory with a ``HEAD`` or, for linked
    worktrees and submodules, a file pointing at one. A bare ``.git`` directory
    without ``HEAD`` (a stray ``mkdir``) is not a repository.
    """
    dotgit = p / ".git"
    return dotgit.is_file() or (dotgit / "HEAD").is_file()


def repo_root(start: str | Path | None = None) -> Path | None:
    """Nearest ancestor of ``start`` that is a git work tree.

    With an explicit ``start``, this is the generic "which repo contains this
    path" lookup (what provenance stamping wants). Without one it means *the
    KIWI checkout this module runs from*: the found root must actually contain
    this file at ``kiwi/core/paths.py``, so a wheel installed into a venv that
    happens to sit under some unrelated git repo still returns ``None`` (and
    :func:`config_root` then fails loudly) instead of silently adopting that
    repo's root.
    """
    if start is not None:
        anchor = Path(start).resolve()
        checkout_only = False
    else:
        anchor = Path(__file__).resolve()
        checkout_only = True
    for p in (anchor, *anchor.parents):
        if _is_work_tree(p):
            if checkout_only and (p / "kiwi" / "core" / "paths.py").resolve() != anchor:
                return None
            return p
    return None


def config_root(episode_dir: str | Path | None = None) -> Path:
    """Episode-local ``config/`` when present, else the deployment config tree.

    With no episode snapshot, resolve ``$KIWI_CONFIG``, then ``<repo>/config``.
    This lookup is read-only; pipeline entry points create the snapshot.

    Raises ``RuntimeError`` when neither resolves - an installed package with
    no checkout must set ``$KIWI_CONFIG`` explicitly.
    """
    if episode_dir is not None:
        local = Path(episode_dir).resolve() / "config"
        if local.exists() or local.is_symlink():
            if not local.is_dir():
                raise ValueError(f"episode config is not a directory: {local}")
            return local
    env = os.environ.get(CONFIG_ENV)
    if env:
        return Path(env).expanduser()
    root = repo_root()
    if root is not None:
        return root / "config"
    raise RuntimeError(
        "cannot locate the config tree: not running from a checkout and "
        f"${CONFIG_ENV} is not set"
    )


def descriptions_root() -> Path:
    """The robot-description tree: ``$KIWI_DESCRIPTIONS``, else ``<repo>/descriptions``.

    Raises ``RuntimeError`` when neither resolves, mirroring
    :func:`config_root` - an installed package with no checkout must set
    ``$KIWI_DESCRIPTIONS`` explicitly.
    """
    env = os.environ.get(DESCRIPTIONS_ENV)
    if env:
        return Path(env).expanduser()
    root = repo_root()
    if root is not None:
        return root / "descriptions"
    raise RuntimeError(
        "cannot locate the descriptions tree: not running from a checkout and "
        f"${DESCRIPTIONS_ENV} is not set"
    )

"""Keep a real copy of the deployment configuration beside an episode.

Snapshot before resolving inputs, including on the first run of an existing
episode. Later runs use that copy without refreshing it from the deployment.
Read-only manifest loading and export never create or overwrite a snapshot.
"""

from __future__ import annotations

import errno
import shutil
import tempfile
from pathlib import Path

from kiwi.core.paths import config_root


def snapshot_config(episode_dir: Path) -> Path:
    """Copy the complete config tree atomically, materializing file symlinks.

    The tree includes camera registry pointers and their dated calibration
    files and gripper designs/builds. Keeping its relative
    layout lets those pointers resolve without the original checkout.
    """
    episode_dir = Path(episode_dir).resolve()
    destination = episode_dir / "config"
    if destination.is_symlink():
        raise ValueError(
            f"episode config must be a local copy, not a symlink: {destination}"
        )
    if destination.exists():
        return config_root(episode_dir)
    source = config_root().resolve()
    if not source.is_dir():
        raise ValueError(f"cannot snapshot missing config directory: {source}")
    if destination.is_relative_to(source):
        raise ValueError(f"config source contains the episode: {source}")
    try:
        with tempfile.TemporaryDirectory(prefix=".config-", dir=episode_dir) as tmp:
            staged = Path(tmp) / "config"
            shutil.copytree(source, staged, symlinks=False)
            try:
                staged.rename(destination)
            except OSError as e:
                # Another worker can prepare the same episode during a scene
                # build. Its complete snapshot wins; never merge two trees.
                if e.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                if destination.is_symlink() or not destination.is_dir():
                    raise
                return config_root(episode_dir)
    except OSError as e:
        raise ValueError(f"cannot copy config into {destination}: {e}") from e
    return destination

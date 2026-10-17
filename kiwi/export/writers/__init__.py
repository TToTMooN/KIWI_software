"""Dataset-format writers, keyed by ``--format``.

Add a format by implementing the
:class:`~kiwi.export.writers.base.DatasetWriter` protocol and registering it
in :data:`WRITERS`. LeRobot formats keep their ``codebase_version`` strings as
keys (``"v2.1"``, ``"v3.0"``); non-LeRobot targets use their own names.
"""

from __future__ import annotations

from .base import DatasetInfo, DatasetWriter
from .lerobot_v21 import LeRobotV21Writer
from .lerobot_v30 import LeRobotV30Writer
from .umi_zarr import UmiZarrWriter

WRITERS: dict[str, type] = {
    LeRobotV21Writer.format: LeRobotV21Writer,
    LeRobotV30Writer.format: LeRobotV30Writer,
    UmiZarrWriter.format: UmiZarrWriter,
}


def get_writer(format: str) -> DatasetWriter:
    """Instantiate the writer for ``format`` (e.g. ``"v3.0"``)."""
    try:
        return WRITERS[format]()
    except KeyError:
        raise ValueError(
            f"unknown export format {format!r}; have {sorted(WRITERS)}"
        ) from None


__all__ = ["WRITERS", "DatasetInfo", "DatasetWriter", "get_writer"]

"""The export funnel: what was considered, what shipped, what dropped, and why.

``meta/export_provenance.json``'s ``skipped`` list records export-time
curation cuts; episodes that failed upstream never reach the exporter. This
module rolls up what the export CAN see: for every source episode handed to
it, the verdict its ``run_status.json`` recorded - which stage the run
reached and whether it halted.

The counts a dataset card wants ("N considered, M exported, dropped: ...")
fall out of the same pass. Episodes never handed to the export are not
covered.

Composer boundary: ``kiwi.export`` never imports ``kiwi.episode``, so the
``run_status.json`` name and shape are read here as the on-disk contract
(schema ``run_status_v1``), exactly as ``write._write_provenance`` already
stamps that file by name.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["build_funnel"]

# The on-disk name kiwi.episode.run_record writes (schema run_status_v1).
_RUN_STATUS = "run_status.json"


def _source_entry(episode_dir: Path) -> dict:
    """One considered source's roll-up, read from its ``run_status.json``.

    Purely observational, like all provenance: a missing or unreadable record
    is *recorded* as such, never raised.
    """
    path = Path(episode_dir) / _RUN_STATUS
    if not path.is_file():
        return {"run_status": "missing"}
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {"run_status": f"unreadable: {type(exc).__name__}: {exc}"}
    entry: dict = {
        "reached": record.get("reached"),
        "halted": record.get("halted"),
        "stages": {
            s["stage"]: s.get("status")
            for s in record.get("stages", [])
            if isinstance(s, dict) and "stage" in s
        },
    }
    return entry


def build_funnel(sources, jobs, skipped: list[dict]) -> dict:
    """Assemble the ``funnel`` block for ``meta/export_provenance.json``.

    ``sources`` are every :class:`~kiwi.export.source.EpisodeSource` handed to
    the export (the "considered" population), ``jobs`` the per-output-episode
    expansion that survived curation, ``skipped`` the recorded cuts
    (passed through verbatim - they already carry ``source`` + ``reason``).
    """
    # Keyed by episode DIR name - the same key curation and ExportJob.source
    # use - not EpisodeSource.name, which a CLI override can restyle.
    exported_names = {job.source for job in jobs}
    per_source = {
        src.episode_dir.name: _source_entry(src.episode_dir)
        for src in sorted(sources, key=lambda s: s.episode_dir.name)
    }
    for name, entry in per_source.items():
        entry["exported"] = name in exported_names
    return {
        "considered": len(per_source),
        "exported_sources": len(exported_names),
        "output_episodes": len(jobs),
        "dropped": list(skipped),
        "sources": per_source,
    }

"""Load a ``curation_v1`` file and expand episodes into export jobs.

A curation file is the durable form of a downstream replayability verdict:
per **source episode** (the episode dir name, e.g. ``ep1``) a list of
``keep_windows`` on the episode global clock - the same time base as
``episode.yaml``'s ``task_window``. Re-exporting through it turns each kept
window into its own output episode, so curation always re-materializes from
source artifacts instead of slicing a previous export.

The file is machine-generated and checked into git next to the dataset
(``data/<dataset>/curation/<name>.json``). This module owns
parsing/validation and the pure job-expansion step; the exporter
(:mod:`kiwi.export.write`) owns the I/O.

The ``curation_v1`` field set is a stable format: the filter that writes these
files lives outside this repository, so the key set and its spellings do not
change here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .source import EpisodeSource

__all__ = ["SCHEMA", "Curation", "CurationEpisode", "ExportJob", "expand_episodes"]

SCHEMA = "curation_v1"


@dataclasses.dataclass(frozen=True)
class CurationEpisode:
    """One source episode's verdict: which global-clock windows to keep."""

    source: str
    keep_windows: tuple[tuple[float, float], ...]
    kept_frac: float | None = None


@dataclasses.dataclass(frozen=True)
class Curation:
    """A parsed curation file plus the identity needed for export provenance."""

    path: Path
    generated: str | None
    dataset_name: str | None
    filter_config: str | None
    filter_config_sha1: str | None
    postprocess: dict
    episodes: dict[str, CurationEpisode]  # keyed by source name
    # The immutable source pin (the filter records these when it filters a
    # Hub revision; None for an unpinned local-path filter). Surfaced into the
    # curated dataset's export provenance so it chains back to the exact
    # source dataset it descends from without opening the curation file.
    source_repo_id: str | None = None  # HF dataset repo the filter read
    source_revision: str | None = None  # resolved HF revision (immutable)
    source_commit: str | None = None  # commit that produced the source dataset

    @classmethod
    def load(cls, path: str | Path) -> "Curation":
        path = Path(path)
        raw = json.loads(path.read_text())
        if raw.get("schema") != SCHEMA:
            raise ValueError(
                f"{path}: schema is {raw.get('schema')!r}, expected {SCHEMA!r}"
            )
        ds = raw.get("dataset") or {}
        episodes: dict[str, CurationEpisode] = {}
        for e in raw.get("episodes", []):
            source = str(e["source"])
            if source in episodes:
                raise ValueError(
                    f"{path}: duplicate curation entry for source {source!r}"
                )
            windows: list[tuple[float, float]] = []
            for w in e.get("keep_windows", []):
                ok = (
                    isinstance(w, (list, tuple))
                    and len(w) == 2
                    and all(
                        isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in w
                    )
                    and float(w[1]) > float(w[0])
                )
                if not ok:
                    raise ValueError(
                        f"{path}: {source}: keep_windows entries must be "
                        f"[start_s, end_s] with end > start, got {w!r}"
                    )
                windows.append((float(w[0]), float(w[1])))
            kf = e.get("kept_frac")
            episodes[source] = CurationEpisode(
                source=source,
                keep_windows=tuple(windows),
                kept_frac=float(kf) if kf is not None else None,
            )
        if not episodes:
            raise ValueError(f"{path}: curation file lists no episodes")
        flt = raw.get("filter") or {}
        return cls(
            path=path,
            generated=raw.get("generated"),
            dataset_name=ds.get("name"),
            filter_config=flt.get("config"),
            filter_config_sha1=flt.get("config_sha1"),
            postprocess=dict(raw.get("postprocess") or {}),
            episodes=episodes,
            source_repo_id=ds.get("repo_id"),
            source_revision=ds.get("revision"),
            source_commit=ds.get("source_commit"),
        )

    def provenance(self) -> dict:
        """Identity block recorded in the curated dataset's export provenance."""
        return {
            "file": self.path.name,
            "generated": self.generated,
            "source_dataset": self.dataset_name,
            "source_repo_id": self.source_repo_id,
            "source_revision": self.source_revision,
            "source_commit": self.source_commit,
            "filter_config": self.filter_config,
            "filter_config_sha1": self.filter_config_sha1,
            "postprocess": self.postprocess,
        }


@dataclasses.dataclass(frozen=True)
class ExportJob:
    """One output episode: a resolved source spec + the window to export."""

    episode: EpisodeSource
    window: tuple[float, float] | None  # None = spec/full-overlap window
    span_index: int | None = None  # position within the source's keep_windows

    @property
    def source(self) -> str:
        """The source name curation keys on: the episode DIR name."""
        return self.episode.episode_dir.name


def expand_episodes(
    episodes: list[EpisodeSource],
    curation: Curation | None,
    window: tuple[float, float] | None,
) -> tuple[list[ExportJob], list[dict]]:
    """Expand resolved episode specs into per-output-episode jobs.

    Without curation this is one job per spec carrying the (optional) global
    ``window``. With curation, each spec maps to one job per kept window;
    specs the curation file doesn't list, and specs whose verdict kept
    nothing, are skipped and reported (never silently dropped). ``window`` and
    curation are mutually exclusive - per-span windows already encode the
    trim.
    """
    if curation is None:
        return [ExportJob(ep, window) for ep in episodes], []
    if window is not None:
        raise ValueError("pass either an explicit window or a curation file, not both")

    jobs: list[ExportJob] = []
    skipped: list[dict] = []
    for ep in episodes:
        name = ep.episode_dir.name
        entry = curation.episodes.get(name)
        if entry is None:
            skipped.append({"source": name, "reason": "not in curation file"})
            continue
        if not entry.keep_windows:
            skipped.append({"source": name, "reason": "curated out (no kept spans)"})
            continue
        for k, w in enumerate(entry.keep_windows):
            jobs.append(ExportJob(ep, w, span_index=k))
    return jobs, skipped

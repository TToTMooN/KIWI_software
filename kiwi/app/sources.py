"""Resolve episode manifests into the plain inputs visualization consumes."""

import math
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class VizResolved:
    sources: object  # kiwi.viz.sources.EpisodeSources
    arms: dict


def preview_offset(stream_dir: Path) -> float | None:
    """Read a finite preview offset from offset_s.txt, separate from pipeline sync."""
    try:
        offset = float((stream_dir / "offset_s.txt").read_text().strip())
    except (OSError, ValueError):
        return None
    return offset if math.isfinite(offset) else None


def preview_streams(sources):
    """Include extra cameras with an explicit, visibly provisional alignment."""
    offsets = sources.clock.offsets_s if sources.clock else {}
    streams, provisional, missing = [], {}, []
    for stream in sources.streams:
        if stream.label in offsets:
            streams.append(replace(stream, offset_s=float(offsets[stream.label])))
            continue
        offset = preview_offset(stream.stream_dir) if stream.kind == "camera" else None
        if offset is None:
            missing.append(stream.label)
        else:
            streams.append(replace(stream, offset_s=offset))
            provisional[stream.label] = offset
    return replace(sources, streams=tuple(streams)), provisional, missing


def viz_sources(episode: Path) -> VizResolved:
    from kiwi.core import layout
    from kiwi.episode.manifest import EpisodeConfig
    from kiwi.viz.sources import EpisodeSources, StreamSource

    cfg = EpisodeConfig.load(episode / "episode.yaml")
    clock = cfg.global_clock()
    offsets = clock.offsets_s if clock is not None else {}
    streams = tuple(
        StreamSource(
            label=spec.label,
            kind=kind,
            recording=spec.recording,
            stream_dir=layout.stream_dir(cfg.root, spec.label),
            offset_s=float(offsets.get(spec.label, 0.0)),
        )
        for kind, specs in (("arm", cfg.arms), ("camera", cfg.cameras))
        for spec in specs.values()
    )
    return VizResolved(
        sources=EpisodeSources(
            name=cfg.name,
            root=cfg.root,
            streams=streams,
            clock=clock,
            task_window=cfg.task_window,
            tool=cfg.default_tool,
            description=str(cfg.meta.get("description") or ""),
            date=str(cfg.meta.get("date") or ""),
        ),
        arms={a.label: (a.estimator, dict(a.session or {})) for a in cfg.arms.values()},
    )

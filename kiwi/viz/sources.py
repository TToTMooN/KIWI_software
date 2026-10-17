"""Resolved episode inputs for viz - the hand-off seam below the manifest.

``kiwi.viz`` may not import ``kiwi.episode`` (composer -> composer is a
forbidden DAG edge), so the CLI walks ``episode.yaml`` and hands viz an
:class:`EpisodeSources`: plain paths, floats, and the episode's
``GlobalClock``. Everything viz needs beyond that is DERIVED here from
naming conventions the producing stages own:

* a stream's playable video is the recording itself for plain containers,
  or the convert stage's ``lens0.mp4`` in the stream's directory for
  ``.insv`` (derived, never demuxed - viz is read-only over artifacts);
* the state stage writes ``<episode>/<label>/scene/state.tum``
  (:mod:`kiwi.core.layout` spells the tree; this module applies it);
* per-frame clip time maps onto the global clock as ``offset + i / fps``
  (the ``eef_state_v1`` contract makes this the consumer's job).

Constructing these types requires no manifest.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from kiwi.core import layout
from kiwi.core.clock import GlobalClock

__all__ = [
    "EpisodeSources",
    "StreamSource",
    "clip_to_global",
]


def clip_to_global(frame_index: float, fps: float, offset_s: float) -> float:
    """A clip-clock frame index as seconds on the episode global clock."""
    if not fps > 0:
        raise ValueError(f"fps must be positive to place frames, got {fps!r}")
    return offset_s + frame_index / fps


@dataclasses.dataclass(frozen=True)
class StreamSource:
    """One video stream of the episode, as the manifest declared it.

    ``kind`` is ``"arm"`` (contributes trajectory + tool state) or
    ``"camera"`` (an extra stream such as the ego view: video only).
    ``stream_dir`` is where the stream's derived artifacts live
    (``<episode>/<label>/``, per :mod:`kiwi.core.layout`); the CLI resolves
    it. ``offset_s`` is the stream's offset on the global clock. Recorded
    sync offsets are <= 0; an unsynchronized stream defaults to 0.0. A
    provisional GUI preview offset may have either sign.
    """

    label: str
    kind: str
    recording: Path
    stream_dir: Path
    offset_s: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("arm", "camera"):
            raise ValueError(f"kind must be 'arm' or 'camera', got {self.kind!r}")

    @property
    def video(self) -> Path:
        """The stream's directly playable video, derived, possibly absent.

        ``.insv`` recordings play through the convert stage's lens-0 demux
        in the stream's own directory; anything else (the ego GO3 mp4, an
        already-demuxed clip) plays as-is. Existence is the caller's
        question - a missing file means the producing stage has not run,
        which the player reports rather than fixes.
        """
        rec = Path(self.recording)
        if rec.suffix.lower() == ".insv":
            return Path(self.stream_dir) / layout.LENS0
        return rec

    @property
    def proxy(self) -> Path:
        """Where this stream's H.264 rendition belongs, built or not.

        Always inside the stream directory, never beside the source.
        """
        return Path(self.stream_dir) / layout.PROXY


@dataclasses.dataclass(frozen=True)
class EpisodeSources:
    """Everything the CLI resolved from one episode for viz to consume."""

    name: str
    root: Path
    streams: tuple[StreamSource, ...]
    clock: GlobalClock | None = None
    task_window: tuple[float | None, float | None] | None = None
    tool: str | None = None
    description: str = ""
    date: str = ""

    def __post_init__(self) -> None:
        labels = [s.label for s in self.streams]
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate stream labels: {labels}")

    @property
    def arms(self) -> tuple[StreamSource, ...]:
        return tuple(s for s in self.streams if s.kind == "arm")

    def tum_path(self, label: str) -> Path:
        """Where the state stage writes ``label``'s trajectory."""
        return layout.scene_dir(Path(self.root), label) / layout.STATE_TUM

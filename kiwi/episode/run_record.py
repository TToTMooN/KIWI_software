"""``run_status.json``: what a ``kiwi run`` did, written as it happens.

Two properties:

* **Written after every stage, atomically** (temp file plus rename). A run that
  crashes or is interrupted leaves a truthful record of what had completed, not
  an empty file or a half-written one.
* **The manifest is stamped, and re-stamped at the end.** ``episode.yaml`` is
  rewritten mid-run - the sync stage writes the clock into it.

Stage-level input stamps cover synchronization tables and optional marker-scan
intrinsics. The producing stages also publish the
full generation records: exact raw/calibration/map inputs, settings,
producer identity and numerical quality evidence.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from kiwi.core.config_hash import stamp_input
from kiwi.core.generation import json_value
from kiwi.core.provenance import provenance_block

__all__ = ["RECORD_NAME", "RECORD_SCHEMA", "RunRecord", "stage_inputs"]

RECORD_NAME = "run_status.json"
RECORD_SCHEMA = "run_status_v1"


def stage_inputs(stage: str, episode_dir: Path) -> list[Path]:
    """Calibration inputs stamped in run_status.json.

    Sync names A/V tables; register names the front-lens intrinsics. Full
    producer/input/output attribution is
    in each stage's generation record. An unreadable manifest attributes
    nothing here and is reported by the stage itself.
    """
    if stage not in ("sync", "register"):
        return []
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig

    manifest = episode_dir / MANIFEST_NAME
    if not manifest.is_file():
        return []
    try:
        cfg = EpisodeConfig.load(manifest, check_exists=False)
    except (ValueError, OSError):
        return []
    from kiwi.core.camera.registry import CameraUnit

    if stage == "sync":
        paths: set[Path] = set()
        dirs = [a.camera_dir for a in cfg.arms.values()] + [
            c.camera_dir for c in cfg.cameras.values() if c.camera_dir is not None
        ]
        for d in dirs:
            try:
                p = CameraUnit.load(d).artifact("av_sync")
            except (FileNotFoundError, KeyError, ValueError, OSError):
                continue
            if p is not None:
                paths.add(p)
        return sorted(paths)

    paths: set[Path] = set()
    for arm in cfg.arms.values():
        try:
            p = CameraUnit.load(arm.camera_dir).artifact("intrinsics")
        except (FileNotFoundError, KeyError, ValueError, OSError):
            continue
        if p is not None:
            paths.add(p)
    return sorted(paths)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


@dataclass
class RunRecord:
    """Accumulates one run's outcome and persists it after every stage."""

    episode_dir: Path
    manifest: Path
    requested: tuple[str, ...] = ()
    stages: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        root = self.episode_dir.resolve()
        self.meta = {
            "schema": RECORD_SCHEMA,
            "episode": self.episode_dir.name,
            "started": _now(),
            "requested": list(self.requested),
            "manifest": stamp_input(self.manifest, root=root),
            **provenance_block(),
        }

    @property
    def path(self) -> Path:
        return self.episode_dir / RECORD_NAME

    def start(self, stage: str) -> None:
        # Entered before its inputs are resolved: resolving them loads the
        # manifest, and a load that raises leaves this entry for `close` to
        # mark "error" rather than no entry for the stage at all.
        entry = {"stage": stage, "status": "running", "started": _now(), "inputs": []}
        self.stages.append(entry)
        root = self.episode_dir.resolve()
        entry["inputs"] = [
            stamp_input(p, root=root) for p in stage_inputs(stage, self.episode_dir)
        ]
        self._write()

    def finish(self, result, seconds: float) -> None:
        rec = self.stages[-1]
        rec["status"] = "ok" if result.ok else "failed"
        rec["seconds"] = round(seconds, 3)
        rec["ended"] = _now()
        rec["summary"] = result.summary
        rec["artifacts"] = [str(a) for a in result.artifacts]
        # Machine-readable stage supplement (StageResult.data), recorded only
        # when non-empty.
        if result.data:
            rec["data"] = result.data
        self._write()

    def close(self, *, reached: str | None, halted: bool) -> None:
        # A stage still marked "running" here raised or was interrupted.
        # Record that outcome instead of leaving it looking in-flight.
        if self.stages and self.stages[-1]["status"] == "running":
            self.stages[-1]["status"] = "error"
            self.stages[-1]["ended"] = _now()
        self.meta["ended"] = _now()
        self.meta["reached"] = reached
        self.meta["halted"] = halted
        # Re-stamp: the sync stage rewrites episode.yaml mid-run, so the opening
        # stamp describes a file this run has already replaced.
        self.meta["manifest_end"] = stamp_input(
            self.manifest, root=self.episode_dir.resolve()
        )
        self._write()

    def _write(self) -> None:
        """Atomic replace, so an interrupted run never leaves a torn record."""
        doc = {**self.meta, "stages": self.stages}
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(json_value(doc), indent=2, sort_keys=False, allow_nan=False)
                + "\n"
            )
            os.replace(tmp, self.path)
        except OSError:
            # A failed record write never fails the run - same rule stamp_input
            # follows.
            tmp.unlink(missing_ok=True)

"""``pipeline_run_status.json``: what a profile-governed run did, exactly.

``run_status.json`` (schema ``run_status_v1``) records one entry per
*started* stage, in start order. This document records one outcome per stage
of the requested chain:

* ``completed`` - the stage ran and reported ok;
* ``failed`` - the stage ran and did not complete: it reported not-ok,
  raised, or was interrupted;
* ``not_requested`` - the governing profile skips this stage, and the run
  did not ask for it by name;
* ``not_executed`` - required, but the run ended before reaching it.

The governing profile is recorded by name AND content digest. Written
atomically after every stage, like ``run_status.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from kiwi.core.atomic import atomic_write_text
from kiwi.core.digest import digest_canonical_json

__all__ = ["STATUS_NAME", "OUTCOMES", "PipelineRunStatus"]

STATUS_NAME = "pipeline_run_status.json"
OUTCOMES = ("completed", "failed", "not_requested", "not_executed")


@dataclass
class PipelineRunStatus:
    """Accumulates one profile-governed run and persists it as it goes."""

    episode_dir: Path
    profile_name: str
    profile_digest: str
    chain: tuple[str, ...]
    not_requested: frozenset[str] = frozenset()
    outcomes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def start(
        cls, episode_dir: Path, profile, asked: frozenset[str] = frozenset()
    ) -> "PipelineRunStatus":
        """Open the record. ``asked`` names the stages the run requested
        directly, which the profile's skips never cover."""
        digest = digest_canonical_json(
            {
                "name": profile.name,
                "trajectory_mode": profile.trajectory_mode,
                "dispositions": dict(profile.dispositions),
                "state_profile": profile.state_profile,
            }
        )
        return cls(
            episode_dir=Path(episode_dir),
            profile_name=profile.name,
            profile_digest=digest,
            chain=(),
            not_requested=frozenset(
                s
                for s, d in profile.dispositions.items()
                if d != "required" and s not in asked
            ),
        )

    def begin_chain(self, chain: tuple[str, ...]) -> None:
        self.chain = tuple(chain)
        self._write()

    def record(self, stage: str, outcome: str) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}; one of {OUTCOMES}")
        self.outcomes[stage] = outcome
        self._write()

    def finish(self) -> None:
        """Mark every still-unrecorded stage: a profile skip stays
        ``not_requested`` (it was never going to run), everything else
        becomes ``not_executed`` (the run ended before reaching it)."""
        for stage in self.chain:
            self.outcomes.setdefault(
                stage,
                "not_requested" if stage in self.not_requested else "not_executed",
            )
        self._write()

    def _write(self) -> None:
        doc = {
            "schema": "pipeline_run_status",
            "profile": {"name": self.profile_name, "digest": self.profile_digest},
            "stages": [
                {
                    "stage": s,
                    "outcome": self.outcomes.get(
                        s,
                        "not_requested" if s in self.not_requested else "not_executed",
                    ),
                }
                for s in self.chain
            ],
        }
        try:
            atomic_write_text(
                self.episode_dir / STATUS_NAME,
                json.dumps(doc, indent=2) + "\n",
            )
        except OSError:
            pass  # a record must never fail the run it describes

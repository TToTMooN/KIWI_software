"""Which stages a run requires, including explicit optional consumers.

The default global-state profile runs sync, convert, state and eef.
State localizes the lens videos against the published scene maps and fuses
those observations with raw IMU.
Prepare runs sync, convert and eef before scene maps are available.
Head and ground are opt-in consumers of scene state. Marker registration is
an on-demand evaluation stage after state, absent from every builtin chain.
Profiles state every disposition and validate stage ordering, including
every upstream product a stage entry refuses to run without.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kiwi.episode.stages import STAGES

__all__ = [
    "DEFAULT_PROFILE",
    "DISPOSITIONS",
    "GLOBAL_STATE_GROUND_PROFILE",
    "GLOBAL_STATE_HEAD_GROUND_PROFILE",
    "GLOBAL_STATE_HEAD_PROFILE",
    "PROFILES",
    "TRAJECTORY_MODES",
    "PipelineProfile",
    "PREPARE_PROFILE",
    "GLOBAL_STATE_PROFILE",
]

DISPOSITIONS = ("required", "not_requested")
TRAJECTORY_MODES = ("alignment_only", "global_state")

# The stages each stage needs first, validated here so a profile cannot
# request a stage while skipping one it needs. Most edges are refusals the
# stage entries raise themselves ("run the <X> stage first"): every consumer
# of convert's lens videos, every consumer of the state products (register,
# head, ground), and the sync edges of register and head, which need
# the episode global clock. The sync edges of convert, state, eef and ground
# are ordering policy instead. convert and eef never read the clock, and
# without one state publishes on the recording's own IMU clock, which ground
# then reads.
_DEPENDS: dict[str, tuple[str, ...]] = {
    "convert": ("sync",),
    "register": ("sync", "convert", "state"),
    "state": ("sync", "convert"),
    "head": ("sync", "convert", "state"),
    "ground": ("sync", "convert", "state"),
    "eef": ("sync", "convert"),
}


@dataclass(frozen=True)
class PipelineProfile:
    """One disposition per pipeline stage, plus the trajectory mode.

    ``trajectory_mode`` is the scene-state scheduling switch:
    ``alignment_only`` runs no global state estimation (the ``state``
    stage must be ``not_requested`` and naming a state profile is an
    error); ``global_state`` requires BOTH the ``state`` stage and a named
    state profile.
    The default profile runs sync, convert, state and eef.
    """

    name: str
    dispositions: Mapping[str, str]
    trajectory_mode: str = "alignment_only"
    state_profile: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a pipeline profile needs a non-empty name")
        if self.trajectory_mode not in TRAJECTORY_MODES:
            raise ValueError(
                f"profile {self.name!r}: trajectory_mode "
                f"{self.trajectory_mode!r}, expected one of {TRAJECTORY_MODES}"
            )
        stated = set(self.dispositions)
        if stated != set(STAGES):
            raise ValueError(
                f"profile {self.name!r} must state every stage exactly once: "
                f"missing {sorted(set(STAGES) - stated)}, "
                f"unknown {sorted(stated - set(STAGES))}"
            )
        for stage, disposition in self.dispositions.items():
            if disposition not in DISPOSITIONS:
                raise ValueError(
                    f"profile {self.name!r}: stage {stage!r} has disposition "
                    f"{disposition!r}, expected one of {DISPOSITIONS}"
                )
        for stage, deps in _DEPENDS.items():
            if self.dispositions[stage] != "required":
                continue
            skipped = [d for d in deps if self.dispositions[d] != "required"]
            if skipped:
                raise ValueError(
                    f"profile {self.name!r}: stage {stage!r} is required but "
                    f"needs {skipped} first, which the profile skips"
                )
        if self.trajectory_mode == "global_state":
            if self.dispositions["state"] != "required":
                raise ValueError(
                    f"profile {self.name!r}: global_state mode requires the state stage"
                )
            if not self.state_profile:
                raise ValueError(
                    f"profile {self.name!r}: global_state mode requires a "
                    f"named state profile"
                )
        else:
            if self.dispositions["state"] == "required":
                raise ValueError(
                    f"profile {self.name!r}: alignment_only mode must not "
                    f"require the state stage"
                )
            if self.state_profile:
                raise ValueError(
                    f"profile {self.name!r}: alignment_only mode names a "
                    f"state profile it would never use - remove it or switch "
                    f"the mode"
                )

    def requires(self, stage: str) -> bool:
        return self.dispositions[stage] == "required"


PREPARE_PROFILE = PipelineProfile(
    name="prepare",
    dispositions={
        "sync": "required",
        "convert": "required",
        # The table marker is a ground-truth instrument, not a capture input:
        # run it on demand with `--from-stage register --to-stage register`.
        "register": "not_requested",
        "state": "not_requested",
        "head": "not_requested",
        "eef": "required",
        "ground": "not_requested",
    },
)


#: The default profile publishes the global visual-inertial batch estimate.
GLOBAL_STATE_PROFILE = PipelineProfile(
    name="global-state",
    trajectory_mode="global_state",
    state_profile="scene-fusion",
    dispositions={
        "sync": "required",
        "convert": "required",
        "register": "not_requested",
        "state": "required",
        "head": "not_requested",
        "eef": "required",
        "ground": "not_requested",
    },
)

#: Explicit opt-in for the passive head-marker estimator. ``global-state``
#: does not request the head stage, so adding head-marker config alone never
#: makes an already selected pipeline do more work.
GLOBAL_STATE_HEAD_PROFILE = PipelineProfile(
    name="global-state-head",
    trajectory_mode="global_state",
    state_profile="scene-fusion",
    dispositions={
        "sync": "required",
        "convert": "required",
        "register": "not_requested",
        "state": "required",
        "head": "required",
        "eef": "required",
        "ground": "not_requested",
    },
)

#: The scene-mapping profile plus the floor: once both arms are in the one
#: scene frame, the ``ground`` stage estimates its height. Selected by name
#: (``kiwi run --profile global-state-ground``); needs the ``ground`` pixi env.
GLOBAL_STATE_GROUND_PROFILE = PipelineProfile(
    name="global-state-ground",
    trajectory_mode="global_state",
    state_profile="scene-fusion",
    dispositions={
        "sync": "required",
        "convert": "required",
        "register": "not_requested",
        "state": "required",
        "head": "not_requested",
        "eef": "required",
        "ground": "required",
    },
)

#: Head marker and floor together, both after the fusion
#: (``kiwi run --profile global-state-head-ground``).
GLOBAL_STATE_HEAD_GROUND_PROFILE = PipelineProfile(
    name="global-state-head-ground",
    trajectory_mode="global_state",
    state_profile="scene-fusion",
    dispositions={
        "sync": "required",
        "convert": "required",
        "register": "not_requested",
        "state": "required",
        "head": "required",
        "eef": "required",
        "ground": "required",
    },
)

DEFAULT_PROFILE = GLOBAL_STATE_PROFILE


#: Builtin pipeline profiles, by name. A site-specific variant gets a NEW
#: name here.
PROFILES: dict[str, PipelineProfile] = {
    PREPARE_PROFILE.name: PREPARE_PROFILE,
    GLOBAL_STATE_PROFILE.name: GLOBAL_STATE_PROFILE,
    GLOBAL_STATE_HEAD_PROFILE.name: GLOBAL_STATE_HEAD_PROFILE,
    GLOBAL_STATE_GROUND_PROFILE.name: GLOBAL_STATE_GROUND_PROFILE,
    GLOBAL_STATE_HEAD_GROUND_PROFILE.name: GLOBAL_STATE_HEAD_GROUND_PROFILE,
}

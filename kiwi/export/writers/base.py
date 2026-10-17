"""Writer contract + shared helpers for emitting a dataset from episodes.

A :class:`DatasetWriter` is opened once with a :class:`DatasetInfo` (the
dataset's schema + identity), fed one
:class:`~kiwi.core.observation_episode.ObservationEpisode` at a time, and
closed once (which flushes ``meta/``). Per-version layout (one file per
episode for v2.1, aggregated shards for v3) lives in the concrete writers;
the ``action`` synthesis and chunk math are shared here.

The ``delta`` action policy differences the position and gripper columns
(world frame, so ``p_{t+1} = p_t + dp``), while the quaternion blocks (located
by the state column names) carry the body-frame relative rotation
``q_t^-1 * q_{t+1}``, so ``q_{t+1} = q_t * dq``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.core.observation_episode import FeatureColumn, ObservationEpisode

from ..schema import STATE_KEY

__all__ = [
    "ACTION_POLICIES",
    "DatasetInfo",
    "DatasetWriter",
    "chunk_index",
    "quat_spans",
    "synthesize_action",
]

ACTION_POLICIES = ("next_state", "copy", "delta")

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


@dataclasses.dataclass
class DatasetInfo:
    """A dataset's identity + schema, fixed for the whole write.

    ``features`` is a representative episode's feature dict (its ``data`` is
    ignored - only key/dtype/shape/names matter). ``action_policy`` selects
    how each episode's ``action`` column is synthesized from
    ``observation.state``.
    """

    repo_id: str
    robot_type: str
    fps: float
    features: dict[str, FeatureColumn]
    action_policy: str = "next_state"
    chunks_size: int = 1000
    codec: str = "hevc"
    include_action: bool = True

    def __post_init__(self) -> None:
        if self.action_policy not in ACTION_POLICIES:
            raise ValueError(
                f"action_policy {self.action_policy!r} not in {ACTION_POLICIES}"
            )
        state = self.features.get(STATE_KEY)
        if (
            self.include_action
            and self.action_policy == "delta"
            and state is not None
            and not state.names
        ):
            # Without names the quaternion blocks cannot be located.
            raise ValueError(
                f"action_policy 'delta' needs named {STATE_KEY} columns "
                f"to locate the quaternion blocks"
            )


@runtime_checkable
class DatasetWriter(Protocol):
    format: str

    def open(self, root: Path, info: DatasetInfo) -> None: ...

    def add_episode(
        self, episode: ObservationEpisode, episode_index: int, task_index: int
    ) -> None: ...

    def close(self) -> None: ...


def quat_spans(names: list[str] | None) -> tuple[int, ...]:
    """Start indices of 4-wide xyzw quaternion blocks, located by column names.

    Recognizes the schema's ``<label>_ee_qx`` .. ``<label>_ee_qw`` runs
    (:func:`kiwi.export.schema.arm_state_names`); anything else is not a
    quaternion and differences component-wise.
    """
    if not names:
        return ()
    spans: list[int] = []
    i = 0
    while i + 3 < len(names):
        if (
            names[i].endswith("_qx")
            and names[i + 1].endswith("_qy")
            and names[i + 2].endswith("_qz")
            and names[i + 3].endswith("_qw")
        ):
            spans.append(i)
            i += 4
        else:
            i += 1
    return tuple(spans)


def synthesize_action(
    state: np.ndarray, policy: str, *, quat_spans: tuple[int, ...] = ()
) -> np.ndarray:
    """Derive an ``action`` column from the per-frame ``observation.state``.

    ``next_state`` (default): ``action[t] = state[t+1]`` with the last frame
    repeated - the standard target for pose-trajectory demonstrations.
    ``copy``: ``action[t] = state[t]``. ``delta``: first difference with a
    trailing zero row, except the ``quat_spans`` blocks, which carry the
    body-frame relative rotation ``q_t^-1 * q_{t+1}`` (xyzw; trailing row =
    identity) - composing ``q_t * delta_t`` recovers ``q_{t+1}`` exactly.
    """
    state = np.asarray(state, dtype=np.float32)
    if len(state) == 0:
        return state.copy()
    if policy == "copy":
        return state.copy()
    if policy == "next_state":
        return np.vstack([state[1:], state[-1:]])
    if policy == "delta":
        d = np.vstack(
            [np.diff(state, axis=0), np.zeros((1, state.shape[1]), dtype=np.float32)]
        )
        for s in quat_spans:
            r = Rotation.from_quat(state[:, s : s + 4].astype(np.float64))
            if len(state) > 1:
                rel = (r[:-1].inv() * r[1:]).as_quat().astype(np.float32)
                d[:-1, s : s + 4] = rel
            d[-1, s : s + 4] = _IDENTITY_QUAT
        return d
    raise ValueError(f"unknown action policy {policy!r}")


def chunk_index(episode_index: int, chunks_size: int) -> int:
    """LeRobot ``episode_chunk`` = ``episode_index // chunks_size``."""
    return episode_index // chunks_size

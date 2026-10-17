"""``replay_state.npz`` - the portable EE-state seam to ``kiwi.replay``.

Dumps a built episode's ``observation.state`` (plus the component names, the
shared timestamps, fps, and the frame labels) to a plain numpy ``.npz`` - no
video. This is the single source of truth for the EE geometry: downstream
replay tools read this npz instead of re-deriving EE poses from the raw
``.tum`` + extrinsics.

The key set is a stable format that replay tools outside this repository read,
as does ``kiwi.replay``: ``state`` (N, D) float32, ``names`` (D,),
``timestamps`` (N,) s, ``fps``, ``frame`` (world frame), ``ee_frame`` (body
frame), ``task``, plus ``tcp_measured`` (bool) and ``tcp`` (JSON per-arm
provenance). Units: position m, quaternion xyzw, gripper [0, 1].

``world_scene`` (JSON, like ``tcp``) is the key that makes ``frame`` usable
as an identity: under the scene world every scene declares the same constant
frame name, so replay - which anchors a whole episode set on one shared
origin - needs the scene name and map-set digest to tell two scenes apart.
The reader treats a dump that lacks the key as "unknown scene" rather than
refusing it.

``ee_frame`` is always ``"tcp"``: the tool tip is the only body frame the
exporter produces. The key is part of the contract: replay verifies it and
refuses a dump that names another body.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from kiwi.core.observation_episode import ObservationEpisode

from .schema import GROUND_PLANE_KEY, HEAD_MARKER_POSE_KEY, STATE_KEY

__all__ = ["ReplayStateResult", "write_replay_state"]


@dataclasses.dataclass(frozen=True)
class ReplayStateResult:
    """What was written, plus anything the caller should warn about."""

    path: Path
    frames: int
    dims: int
    tcp_measured: bool
    warning: str | None = None


def write_replay_state(
    episode: ObservationEpisode,
    out: Path,
    *,
    assert_tcp_xyz: tuple[float, float, float] | None = None,
    assert_tol: float = 0.01,
) -> ReplayStateResult:
    """Write the episode's state to ``out`` under the stable key set.

    ``assert_tcp_xyz`` fails unless every arm's stored TCP translation in the
    camera frame matches the expected position within ``assert_tol`` meters -
    the measure-and-assert guard against a drifted or placeholder transform.
    An UNMEASURED ``T_camera_tcp`` never fails silently either: it comes back
    as a ``warning`` for the caller to surface.
    """
    if HEAD_MARKER_POSE_KEY in episode.features:
        raise ValueError(
            "replay state does not support head markers; use LeRobot export"
        )
    if GROUND_PLANE_KEY in episode.features:
        raise ValueError(
            "replay state does not support the ground plane; use LeRobot export"
        )
    out = Path(out)
    col = episode.features[STATE_KEY]

    tcp = episode.meta.get("tcp") or {}
    tcp_measured = bool(tcp) and all((v or {}).get("measured") for v in tcp.values())
    warning: str | None = None
    if not tcp_measured:
        warning = (
            "exporting at the tool TCP with an UNMEASURED T_camera_tcp "
            "(a gripper unit's mount override declares `measured: false`). "
            "The trajectory is only as correct as that placeholder - "
            "measure the tip and set `measured: true` before trusting "
            "replay geometry."
        )
    if assert_tcp_xyz is not None:
        want = np.asarray(assert_tcp_xyz, dtype=np.float64)
        for label, v in tcp.items():
            have = np.asarray(
                (v or {}).get("t_camera_tcp_xyz", [np.nan] * 3), dtype=np.float64
            )
            d = float(np.linalg.norm(have - want))
            if not (d <= assert_tol):
                raise ValueError(
                    f"assert_tcp_xyz failed for arm {label!r}: stored tip "
                    f"{have.tolist()} is {d * 1000:.1f} mm from expected "
                    f"{want.tolist()} (tol {assert_tol * 1000:.1f} mm)"
                )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        state=np.asarray(col.data, dtype=np.float32),
        names=np.asarray(col.names),
        timestamps=np.asarray(episode.timestamps, dtype=np.float64),
        fps=np.float64(episode.fps),
        frame=episode.frame,
        world_scene=json.dumps(episode.meta.get("world_scene")),
        ee_frame="tcp",
        task=episode.task,
        tcp_measured=tcp_measured,
        tcp=json.dumps(tcp),
    )
    return ReplayStateResult(
        path=out,
        frames=episode.num_frames,
        dims=int(col.shape[0]),
        tcp_measured=tcp_measured,
        warning=warning,
    )

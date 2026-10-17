"""``replay_solve.npz`` - the solved-trajectory artifact of ``kiwi replay solve``.

A whole-episode derived artifact (episode root, see ``kiwi.core.layout``):
per arm the solved joint trajectory and the commanded/achieved TCP pose logs,
plus the solve metrics, the L0 continuity summary, and the full parameter set
the run used - self-describing and reproducible.

Key set (schema ``replay_solve_v1``): ``schema``; per side ``solved_<side>``
(N, dof) float64 rad, ``target_<side>`` / ``actual_<side>`` (N, 6)
[x, y, z (m), roll, pitch, yaw (deg)]; ``fps`` (the solved stream's rate),
``decim`` (total capture-clock decimation behind it); ``status``; metrics
scalars ``frames``, ``worst_pos_m``, ``mean_pos_m``, ``worst_rot_rad``,
``mean_rot_rad``, ``qp_failures``, ``self_frames``, ``cross_frames``,
``grip_frames``; continuity ``continuity_min`` (NaN when degenerate),
``continuity_min_frame``, ``continuity_frame_scores`` (float32, per step);
``params`` (JSON: every rig/solver knob).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kiwi.replay.solve import SIDES, SolveResult

__all__ = ["SCHEMA", "write_solve_npz"]

SCHEMA = "replay_solve_v1"


def write_solve_npz(
    out: Path,
    result: dict,
    *,
    status: str,
    continuity: dict,
    fps: float,
    decim: int,
    params: dict,
) -> Path:
    """Write one episode's solve to ``out``.

    ``result`` is the ``EpisodeSet.results`` entry
    (``{"solved", "target", "actual", "metrics"}``); ``continuity`` the
    episode's ``continuity_score`` dict; ``params`` the run's full parameter
    set (JSON-serializable values only).
    """
    out = Path(out)
    m: SolveResult = result["metrics"]
    cont_min = continuity.get("min")
    arrays: dict = {}
    for s in SIDES:
        arrays[f"solved_{s}"] = np.asarray(result["solved"][s], dtype=np.float64)
        arrays[f"target_{s}"] = np.asarray(result["target"][s], dtype=np.float64)
        arrays[f"actual_{s}"] = np.asarray(result["actual"][s], dtype=np.float64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        schema=SCHEMA,
        status=status,
        fps=np.float64(fps),
        decim=np.int64(decim),
        frames=np.int64(m.n),
        worst_pos_m=np.float64(m.worst_pos),
        mean_pos_m=np.float64(m.mean_pos),
        worst_rot_rad=np.float64(m.worst_rot),
        mean_rot_rad=np.float64(m.mean_rot),
        qp_failures=np.int64(m.qp_failures),
        self_frames=np.int64(m.self_f),
        cross_frames=np.int64(m.cross_f),
        grip_frames=np.int64(m.grip_f),
        continuity_min=np.float64(np.nan if cont_min is None else cont_min),
        continuity_min_frame=np.int64(continuity.get("min_frame", -1)),
        continuity_frame_scores=np.asarray(
            continuity.get("frame_scores", np.empty(0, np.float32)), dtype=np.float32
        ),
        params=json.dumps(params, sort_keys=True),
        **arrays,
    )
    return out

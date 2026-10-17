"""``state_latent``: the estimator's IMU-body pose beside its pose output.

The published ``state.tum`` is the front camera per native frame. The graph
solves for the IMU-BODY pose at every node; this artifact keeps it, together
with which nodes had a tracked front or rear localization.

The artifact is one ``.npz`` per arm beside the trajectory, with the
frame and body of every array declared in its own metadata so a reader
never has to infer them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["LATENT_SCHEMA", "read_state_latent", "write_state_latent"]

LATENT_SCHEMA = "state_latent"


def write_state_latent(
    path: Path,
    *,
    t,
    body,
    front,
    front_candidate_present,
    rear_candidate_present,
    frame: str,
    metadata: dict | None = None,
) -> Path:
    """Write one arm's latent state beside its trajectory.

    Positions are metres in ``frame``; rotations are Hamilton xyzw
    quaternions of that body IN that frame. The ``*_candidate_present``
    masks mark the nodes where a tracked localization was available.
    """
    path = Path(path)
    meta = {
        **(metadata or {}),
        "schema": LATENT_SCHEMA,
        "frame": frame,
        "latent_body": "imu_body_centre",
        "output_body": "front_camera_optical_centre",
        "position_units": "metres",
        "rotation_convention": "hamilton_xyzw",
    }
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            metadata=np.asarray(json.dumps(meta, sort_keys=True)),
            t=np.asarray(t, dtype=float),
            body_position=body[:, :3, 3],
            body_quaternion_xyzw=Rotation.from_matrix(body[:, :3, :3]).as_quat(),
            front_position=front[:, :3, 3],
            front_quaternion_xyzw=Rotation.from_matrix(front[:, :3, :3]).as_quat(),
            front_candidate_present=np.asarray(front_candidate_present, dtype=bool),
            rear_candidate_present=np.asarray(rear_candidate_present, dtype=bool),
        )
    tmp.replace(path)
    return path


def read_state_latent(path: Path) -> dict:
    """Load the latent state, with its declared metadata parsed."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "metadata"}
        meta = json.loads(str(z["metadata"]))
    if meta.get("schema") != LATENT_SCHEMA:
        raise ValueError(
            f"{path}: schema is {meta.get('schema')!r}, expected {LATENT_SCHEMA!r}"
        )
    n = len(out["t"])
    for key, array in out.items():
        if len(array) != n:
            raise ValueError(
                f"{path}: {key} has {len(array)} rows, expected {n} - the "
                f"latent state is one row per graph node"
            )
    out["metadata"] = meta
    return out

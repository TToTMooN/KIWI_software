"""Reading a trained Gaussian scene asset and the arm sources it overlays."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from kiwi.core.digest import digest_bytes

SCHEMA = "kiwi_real2sim_v1"


@dataclasses.dataclass(frozen=True)
class ArmSource:
    episode: Path
    label: str
    camera_dir: Path
    sync_offset_s: float
    scene_manifest: Path


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def file_identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "digest": digest_bytes(path)}


def validate_transform(T: np.ndarray) -> None:
    if (
        T.shape != (4, 4)
        or not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7)
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("camera extrinsic must be a finite rigid SE(3) transform")


def read_asset(asset: Path) -> dict:
    """Resolve the paths needed to render a completed asset."""
    root = Path(asset).resolve()
    record = json.loads((root / "run.json").read_text())
    if record.get("status") != "completed":
        raise ValueError(f"{root}: training has not completed")
    return {
        **record,
        "root": root,
        **{
            key: root / record[key]
            for key in ("dataset", "checkpoint", "ply", "trajectories")
            if record.get(key)
        },
    }

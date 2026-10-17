"""Sample the optional marker trajectory without crossing missing segments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from kiwi.core.digest import verify_digest
from kiwi.core.observation_episode import FeatureColumn
from kiwi.core.trajectory import read_tum_header

from .schema import (
    HEAD_MARKER_POSE_KEY,
    HEAD_MARKER_POSE_NAMES,
    HEAD_MARKER_SEGMENT_ID_KEY,
    HEAD_MARKER_VALID_KEY,
)
from .source import HeadSource


def sample_head(
    source: HeadSource | None,
    grid: np.ndarray,
    *,
    world_scene: dict,
    transform: np.ndarray | None,
) -> tuple[dict[str, FeatureColumn], dict]:
    """Keep unsupported times NaN/false/-1; interpolate only within one segment."""
    pose = np.full((len(grid), 7), np.nan, dtype=np.float32)
    valid = np.zeros((len(grid), 1), dtype=bool)
    segment_ids = np.full((len(grid), 1), -1, dtype=np.int64)
    metadata = {
        "body": "head_marker",
        "world_scene": world_scene,
        "sampling": "linear position and quaternion Slerp within each segment; no extrapolation",
        "validity": "trajectory support, not an accuracy threshold",
        "missing": "pose=NaN, valid=false, segment_id=-1",
        "status": "unavailable",
    }
    if source is not None and Path(source.trajectory).is_file():
        path, report_path = Path(source.trajectory), Path(source.report)
        report = json.loads(report_path.read_text())
        header = read_tum_header(path)
        if (
            header is None
            or header.body != "head_marker"
            or header.frame != world_scene["frame"]
        ):
            raise ValueError(
                f"{path}: head marker body/world frame disagrees with hands"
            )
        if not header.clock or not header.clock.startswith("episode global clock"):
            raise ValueError(
                f"{path}: head timestamps must use the episode global clock"
            )
        source_world = report.get("world_scene") or {}
        if any(
            source_world.get(key) != world_scene.get(key)
            for key in ("frame", "scene", "map_set_digest")
        ):
            raise ValueError(
                f"{report_path}: head world_scene disagrees with hands or is unknown; "
                "rerun head estimation against the exported state"
            )
        placement = source_world.get("scene_frame_digest")
        if (
            placement
            and world_scene.get("scene_frame_digest")
            and placement != world_scene["scene_frame_digest"]
        ):
            raise ValueError(
                f"{report_path}: head scene_frame_digest disagrees with hands"
            )
        metadata["source_world_scene"] = source_world
        metadata["placement_verification"] = (
            "matched"
            if placement and world_scene.get("scene_frame_digest")
            else "unknown"
        )
        output = report.get("output") or {}
        if not output.get("digest"):
            raise ValueError(f"{report_path}: head report records no output digest")
        verify_digest(output["digest"], path)
        rows = np.loadtxt(path, comments="#", ndmin=2)
        if rows.shape[1] != 8 or not np.isfinite(rows).all():
            raise ValueError(f"{path}: expected finite 8-column TUM rows")
        t, xyz, quat = rows[:, 0], rows[:, 1:4], rows[:, 4:8]
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"{path}: head timestamps must strictly increase")
        diagnostics = Path(source.diagnostics)
        if not diagnostics.is_file():
            raise ValueError(f"{diagnostics}: head diagnostics are missing")
        diagnostics_ref = (report.get("quality_outputs") or {}).get("diagnostics") or {}
        if not diagnostics_ref.get("digest"):
            raise ValueError(
                f"{report_path}: head report records no diagnostics digest"
            )
        verify_digest(diagnostics_ref["digest"], diagnostics)
        with np.load(diagnostics, allow_pickle=False) as data:
            diag_t = np.asarray(data["t"])
            ids = np.asarray(data["segment_id"])
        if (
            diag_t.shape != t.shape
            or ids.shape != t.shape
            or not np.allclose(diag_t, t, atol=1e-8, rtol=0)
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
        ):
            raise ValueError(f"{diagnostics}: t/segment_id must match head TUM rows")
        metadata["segment_source"] = str(diagnostics)
        for segment_id in np.unique(ids):
            ix = np.flatnonzero(ids == segment_id)
            if len(ix) > 1 and np.any(np.diff(ix) != 1):
                raise ValueError(f"{path}: a head segment must occupy contiguous rows")
            ts = t[ix]
            take = (grid >= ts[0]) & (grid <= ts[-1])
            if len(ix) == 1:
                take = np.isclose(grid, ts[0], atol=1e-9, rtol=0)
                p = np.broadcast_to(xyz[ix[0]], (int(take.sum()), 3)).copy()
                r = (
                    Rotation.from_quat(
                        np.broadcast_to(quat[ix[0]], (int(take.sum()), 4))
                    )
                    if take.any()
                    else None
                )
            else:
                p = np.column_stack(
                    [np.interp(grid[take], ts, xyz[ix, k]) for k in range(3)]
                )
                r = (
                    Slerp(ts, Rotation.from_quat(quat[ix]))(grid[take])
                    if take.any()
                    else None
                )
            if not take.any():
                continue
            if transform is not None:
                p = p @ transform[:3, :3].T + transform[:3, 3]
                r = Rotation.from_matrix(transform[:3, :3]) * r
            pose[take] = np.column_stack([p, r.as_quat()])
            valid[take, 0] = True
            segment_ids[take, 0] = segment_id
        metadata.update(
            status="available", trajectory=str(path), report=str(report_path)
        )
    elif source is not None and Path(source.report).is_file():
        report = json.loads(Path(source.report).read_text())
        metadata["source_status"] = report.get("status")
        metadata["source_reason"] = report.get("reason")
        metadata["report"] = str(source.report)
    metadata["valid_frames"] = int(valid.sum())
    metadata["statistics_status"] = "computed" if valid.any() else "not_computed"
    features = {
        HEAD_MARKER_POSE_KEY: FeatureColumn(
            HEAD_MARKER_POSE_KEY, "float32", (7,), list(HEAD_MARKER_POSE_NAMES), pose
        ),
        HEAD_MARKER_VALID_KEY: FeatureColumn(
            HEAD_MARKER_VALID_KEY, "bool", (1,), ["valid"], valid
        ),
        HEAD_MARKER_SEGMENT_ID_KEY: FeatureColumn(
            HEAD_MARKER_SEGMENT_ID_KEY, "int64", (1,), ["segment_id"], segment_ids
        ),
    }
    return features, metadata

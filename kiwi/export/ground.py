"""Place the optional floor plane and camera heights beside the hands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kiwi.core.digest import verify_digest
from kiwi.core.observation_episode import FeatureColumn

from .schema import GROUND_PLANE_KEY, GROUND_PLANE_NAMES, camera_height_key
from .source import GroundSource

COMPUTED_STATUS = "computed"  # the ground stage's status for a GREEN/YELLOW level


def transform_plane(plane: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    """The plane ``(n, d)`` (``n . p + d = 0``) after ``p' = s R p + t``.

    ``n' = R n`` and ``d' = s d - n' . t``; a rigid ``transform`` has ``s = 1``.
    The export's :func:`kiwi.export.build.world_transform` is rigid; the
    scale term serves a sim(3) caller.
    """
    plane = np.asarray(plane, dtype=np.float64)
    if transform is None:
        return plane
    transform = np.asarray(transform, dtype=np.float64)
    linear = transform[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    normal = (linear / scale) @ plane[:3]
    return np.append(normal, scale * plane[3] - normal @ transform[:3, 3])


def sample_ground(
    source: GroundSource | None,
    grid: np.ndarray,
    camera_positions: dict[str, np.ndarray],
    *,
    world_scene: dict,
    transform: np.ndarray | None,
) -> tuple[dict[str, FeatureColumn], dict]:
    """The floor plane and each arm's height above it on the export grid, or NaN.

    ``camera_positions`` are the arms' camera optical centres already in the
    exported world (the same trajectory and transform the hands use), so the
    height is the signed distance of the camera from the plane - invariant to
    the rigid re-seating both sides received.
    """
    n = len(grid)
    plane = np.full((n, 4), np.nan, dtype=np.float32)
    heights = {
        label: np.full((n, 1), np.nan, dtype=np.float32) for label in camera_positions
    }
    metadata = {
        "plane": "n . p + d = 0, unit normal n = (nx, ny, nz) pointing up, in the exported world",
        "camera_height": "signed distance of the arm's camera optical centre from the plane [m]",
        "world_scene": world_scene,
        "missing": "NaN when the episode has no computed ground report",
        "status": "absent",
    }
    report_path = Path(source.report) if source is not None else None
    if report_path is not None and report_path.is_file():
        report = json.loads(report_path.read_text())
        metadata.update(
            report=str(report_path),
            status=report.get("status"),
            verdict=report.get("verdict"),
            reasons=report.get("reasons"),
        )
        if report.get("status") == COMPUTED_STATUS:
            if report.get("frame") != world_scene["frame"]:
                raise ValueError(
                    f"{report_path}: ground frame {report.get('frame')!r} disagrees "
                    f"with the hands' {world_scene['frame']!r}"
                )
            source_world = report.get("world_scene") or {}
            if any(
                source_world.get(key) != world_scene.get(key)
                for key in ("frame", "scene", "map_set_digest")
            ):
                raise ValueError(
                    f"{report_path}: ground world_scene disagrees with hands or is "
                    "unknown; rerun the ground stage against the exported state"
                )
            placement = source_world.get("scene_frame_digest")
            if (
                placement
                and world_scene.get("scene_frame_digest")
                and placement != world_scene["scene_frame_digest"]
            ):
                raise ValueError(
                    f"{report_path}: ground scene_frame_digest disagrees with hands"
                )
            metadata["source_world_scene"] = source_world
            metadata["placement_verification"] = (
                "matched"
                if placement and world_scene.get("scene_frame_digest")
                else "unknown"
            )
            floor_z = report.get("floor_z_m")
            if floor_z is None or not np.isfinite(float(floor_z)):
                raise ValueError(f"{report_path}: computed report without floor_z_m")
            observations = (report.get("quality_outputs") or {}).get(
                "observations"
            ) or {}
            if (
                source.observations is not None
                and Path(source.observations).is_file()
                and observations.get("digest")
            ):
                verify_digest(observations["digest"], source.observations)
                metadata["observations"] = str(source.observations)
            exported = transform_plane(
                np.array([0.0, 0.0, 1.0, -float(floor_z)]), transform
            )
            plane[:] = exported.astype(np.float32)
            for label, positions in camera_positions.items():
                positions = np.asarray(positions, dtype=np.float64).reshape(n, 3)
                heights[label][:, 0] = positions @ exported[:3] + exported[3]
            metadata.update(
                floor_z_m=float(floor_z),
                sigma_m=report.get("sigma_m"),
                plane_scene_frame=[0.0, 0.0, 1.0, -float(floor_z)],
            )
    metadata["valid_frames"] = int(np.isfinite(plane[:, 3]).sum())
    metadata["statistics_status"] = (
        "computed" if metadata["valid_frames"] else "not_computed"
    )
    features = {
        GROUND_PLANE_KEY: FeatureColumn(
            GROUND_PLANE_KEY, "float32", (4,), list(GROUND_PLANE_NAMES), plane
        )
    }
    for label, column in heights.items():
        key = camera_height_key(label)
        features[key] = FeatureColumn(
            key, "float32", (1,), [f"{label}_camera_height_m"], column
        )
    return features, metadata

"""Per-frame board corner extraction.

``build_extractor`` returns a closure that detects the AprilGrid in one RGB
frame and returns matched ``(object_points, image_points)`` for it. It is
used by the Kalibr calibration workflow (the ``kiwi calib intrinsics run``
verb) purely as a *frame-count* pre-filter: a frame is admitted to the
calibration bag when at least ``min_corners_per_view`` corners are
detected. Kalibr itself does the authoritative corner detection and
calibration inside its docker image, so the per-tag corner *ordering* here
is irrelevant to the calibration result (it would only matter if these
correspondences were fed to a solver).

See ``docs/hardware.md`` for the calibration workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from kiwi.core.markers import apriltag_grid_object_points, detect_aruco_markers

ExtractFn = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray] | None"]


def _extract_apriltag_grid(
    img_rgb: np.ndarray,
    dict_id: int,
    allowed_ids: set[int],
    grid_obj_pts: dict[int, np.ndarray],
    min_corners: int,
    border_bits: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    corners, ids = detect_aruco_markers(
        img_rgb, marker_dict=dict_id, allowed_ids=allowed_ids, border_bits=border_bits
    )
    if ids is None:
        return None
    obj_list: list[np.ndarray] = []
    img_list: list[np.ndarray] = []
    for c, tid in zip(corners, ids.flatten()):
        tid = int(tid)
        if tid not in grid_obj_pts:
            continue
        obj_list.append(grid_obj_pts[tid])  # (4, 3) float32
        img_list.append(c.reshape(4, 2))  # (4, 2) float32
    if not obj_list or sum(len(p) for p in obj_list) < min_corners:
        return None
    obj = np.vstack(obj_list).astype(np.float64).reshape(-1, 1, 3)
    img = np.vstack(img_list).astype(np.float64).reshape(-1, 1, 2)
    return obj, img


def build_extractor(
    marker_config: Path,
    min_corners_per_view: int,
    *,
    name: str | None = None,
) -> ExtractFn:
    """Build the per-frame corner-extraction closure.

    Returns a callable ``extract(rgb) -> (obj_pts, img_pts) | None`` for the
    AprilGrid described by ``marker_config``.

    ``name`` picks the entry from a multi-target file. It is resolved through
    :func:`kiwi.calib.kalibr.yamls.resolve_grid`, the same rule that picks the
    grid Kalibr solves against; an ambiguous file is a refusal.
    """
    from kiwi.calib.kalibr.yamls import resolve_grid

    name, grid_cfg = resolve_grid(marker_config, name)
    dict_id = grid_cfg.cv2_dictionary
    allowed_ids = set(grid_cfg.tag_ids)
    grid_obj_pts = apriltag_grid_object_points(grid_cfg)
    print(
        f"Target: AprilGrid '{name}' ({grid_cfg.tag_cols}x{grid_cfg.tag_rows}, "
        f"{grid_cfg.dictionary}, border_bits={grid_cfg.border_bits})"
    )

    def extract(rgb: np.ndarray):
        return _extract_apriltag_grid(
            rgb,
            dict_id,
            allowed_ids,
            grid_obj_pts,
            min_corners_per_view,
            border_bits=grid_cfg.border_bits,
        )

    return extract

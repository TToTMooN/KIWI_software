"""Project metric overlays into the exact native-render camera image.

The resulting RGB frame contains both layers. It remains self-consistent while
an interactive browser camera advances ahead of server-side rendering.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def project_segments(points, camera_pose, width, height, fov, near=0.01, far=100.0):
    """Return visible segments as Nx2x2 pixel-centre coordinates.

    ``camera_pose`` is OpenCV camera-to-world: +X right, +Y down, +Z forward.
    The projection matches NativeRenderer's rays at (u + .5, v + .5).
    Segments are clipped to all six viewing-frustum planes before division,
    including ones that cross the near plane. Invalid segments are omitted.
    """
    if width < 1 or height < 1 or not (0 < fov < math.pi):
        raise ValueError("Invalid image dimensions or vertical field of view")
    if not (0 < near < far < math.inf):
        raise ValueError("Expected 0 < near < far < infinity")
    pose = np.asarray(camera_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite 4x4 camera-to-world pose")
    segments = np.asarray(points, dtype=np.float64)
    if segments.size == 0:
        return np.empty((0, 2, 2), dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3):
        raise ValueError("Expected world segments with shape Nx2x3")
    segments = segments[np.isfinite(segments).all(axis=(1, 2))]
    camera = (segments - pose[:3, 3]) @ pose[:3, :3]
    start, delta = camera[:, 0], camera[:, 1] - camera[:, 0]
    focal = 0.5 * height / math.tan(fov / 2)
    cx, cy = width / 2 - 0.5, height / 2 - 0.5
    # Every plane is represented by dot(normal, point) + offset >= 0.
    planes = (
        ((0.0, 0.0, 1.0), -near),
        ((0.0, 0.0, -1.0), far),
        ((focal, 0.0, cx), 0.0),
        ((-focal, 0.0, width - 1 - cx), 0.0),
        ((0.0, focal, cy), 0.0),
        ((0.0, -focal, height - 1 - cy), 0.0),
    )
    lower, upper = np.zeros(len(camera)), np.ones(len(camera))
    valid = np.ones(len(camera), dtype=bool)
    for normal, offset in planes:
        value = start @ normal + offset
        slope = delta @ normal
        parallel = np.abs(slope) < 1e-14
        valid &= ~(parallel & (value < 0))
        crossing = np.zeros_like(value)
        np.divide(-value, slope, out=crossing, where=~parallel)
        lower = np.maximum(lower, np.where(slope > 1e-14, crossing, 0.0))
        upper = np.minimum(upper, np.where(slope < -1e-14, crossing, 1.0))
    valid &= lower <= upper
    clipped = np.stack(
        (start + lower[:, None] * delta, start + upper[:, None] * delta), axis=1
    )[valid]
    if not len(clipped):
        return np.empty((0, 2, 2), dtype=np.float64)
    uv = clipped[:, :, :2] / clipped[:, :, 2, None] * focal + (cx, cy)
    uv = uv[np.isfinite(uv).all(axis=(1, 2))]
    # Roundoff at the frustum planes must never produce huge drawing coordinates.
    return np.clip(uv, (0.0, 0.0), (width - 1.0, height - 1.0))


def composite_overlay(rgb, camera_pose, fov, groups):
    """Return a new RGB uint8 image with projected, anti-aliased metric lines.

    Each group has ``points`` (Nx2x3 world segments), ``color`` (RGB), and
    ``thickness`` (image pixels, default 2). No scene-depth occlusion is applied.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an HxWx3 uint8 RGB image")
    output = image.copy()
    height, width = output.shape[:2]
    for group in groups:
        color = tuple(int(c) for c in group["color"])
        if len(color) != 3 or not all(0 <= c <= 255 for c in color):
            raise ValueError("Expected three RGB channel values in [0, 255]")
        thickness_value = float(group.get("thickness", 2.0))
        if not math.isfinite(thickness_value) or thickness_value <= 0:
            raise ValueError("Line thickness must be positive and finite")
        thickness = max(1, round(thickness_value))
        segments = project_segments(group["points"], camera_pose, width, height, fov)
        for segment in np.rint(segments * 16).astype(np.int32):
            # OpenCV writes tuple components directly; no BGR conversion is
            # needed because both this array and the supplied tuple are RGB.
            cv2.line(
                output,
                tuple(segment[0]),
                tuple(segment[1]),
                color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
                shift=4,
            )
    return output


def frustum_segments(pose, fov=math.radians(65), aspect=1.0, scale=0.08):
    """Return Viser-compatible camera-frustum wireframe in world coordinates.

    Matches CameraFrustumVariants.tsx: volume-normalized pyramid, four rectangle
    edges, four rays to the optical centre, and one upward direction indicator.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite 4x4 camera-to-world pose")
    if not (
        0 < fov < math.pi
        and math.isfinite(aspect)
        and aspect > 0
        and math.isfinite(scale)
        and scale > 0
    ):
        raise ValueError("Invalid frustum geometry")
    y = math.tan(fov / 2)
    dimensions = np.array([y * aspect, y, 1.0])
    dimensions *= scale / np.cbrt(np.prod(dimensions) / 3.0)
    vertices = (
        np.array(
            [
                [[-1, -1, 1], [1, -1, 1]],
                [[1, -1, 1], [1, 1, 1]],
                [[1, 1, 1], [-1, 1, 1]],
                [[-1, 1, 1], [-1, -1, 1]],
                [[-1, -1, 1], [0, 0, 0]],
                [[0, 0, 0], [1, -1, 1]],
                [[-1, 1, 1], [0, 0, 0]],
                [[0, 0, 0], [1, 1, 1]],
                [[0, -1.2, 1], [0, -0.9, 1]],
            ],
            dtype=np.float64,
        )
        * dimensions
    )
    return vertices @ pose[:3, :3].T + pose[:3, 3]


def trajectory_overlay(
    tracks,
    timestamp,
    *,
    show_paths=True,
    show_frusta=True,
    full_trajectory=False,
    trail_window_s=20.0,
    pose_body="camera",
):
    """Camera/TCP paths and actual front/rear poses; gap samples have no frusta."""
    groups = []
    colors = {"left": (45, 150, 255), "right": (50, 225, 120)}
    for arm, track in tracks.items():
        if show_paths:
            groups.append(
                dict(
                    points=track.segments(
                        timestamp, trail_window_s, full_trajectory, pose_body
                    ),
                    color=colors[arm],
                    thickness=2.5,
                )
            )
        if show_frusta:
            front = track.at(timestamp)
            if front is not None:
                for pose in (front, front @ track.rear_from_front):
                    groups.append(
                        dict(
                            points=frustum_segments(pose),
                            color=colors[arm],
                            thickness=2.0,
                        )
                    )
    return groups

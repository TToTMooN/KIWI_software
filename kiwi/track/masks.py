"""Keypoint exclusion masks for the front-map builds: U-Net segmentation plus a
vignette ring.

A mask is a ``<t_ns>.png`` named after its frame: 255 marks pixels the engine must
not extract keypoints from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kiwi.track.frames import MAP_IMAGE_SIZE

__all__ = ["MASKING_MODES", "RING_RADIUS_FRACTION", "ring_mask", "write_front_masks"]

MASKING_MODES = ("learned",)
RING_RADIUS_FRACTION = 466.0 / 960.0


def ring_mask(
    cx: float, cy: float, radius_px: float, shape: tuple[int, int]
) -> np.ndarray:
    """0/255 uint8 mask with 255 outside the illuminated fisheye circle."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) > radius_px**2).astype(np.uint8) * 255


def write_front_masks(
    masker,
    frames_dir: Path,
    t_ns_subset: np.ndarray,
    mask_dir: Path,
    kb4_params: np.ndarray,
) -> Path:
    """Write the ``max(unet, ring)`` mask of every listed frame, the ring at the principal point."""
    import cv2

    mask_dir.mkdir(parents=True, exist_ok=True)
    size = (MAP_IMAGE_SIZE, MAP_IMAGE_SIZE)
    cx, cy = float(kb4_params[2]), float(kb4_params[3])
    ring = ring_mask(cx, cy, RING_RADIUS_FRACTION * MAP_IMAGE_SIZE, size)
    for t in (int(t) for t in np.asarray(t_ns_subset).ravel()):
        img = cv2.imread(str(frames_dir / f"{t}.jpg"))
        cv2.imwrite(str(mask_dir / f"{t}.png"), np.maximum(masker.segment(img), ring))
    return mask_dir

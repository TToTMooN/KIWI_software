"""Circular-dot blob detection, polarity-dispatched.

One detector for all three dot polarities; ``polarity`` selects the
segmentation channel and threshold direction:

- ``"dark"``: dark blobs on a light background. Grayscale + THRESH_BINARY_INV.
- ``"light"``: bright blobs on a dark background. Grayscale + THRESH_BINARY.
- ``"red"``: red blobs on a dark/neutral background - what every shipped
  Chopsticks tool uses. Segments on the Lab ``a*`` channel (red-green axis,
  neutral=128); pixels with ``a* > threshold`` are kept (150 is mildly red;
  raise toward 170+ for saturated reds, lower toward 140 if the dot is dim).

Callers pass ``polarity``; the per-polarity threshold defaults live in
:data:`kiwi.eef.config.DEFAULT_THRESHOLD_BY_POLARITY`. The input is always a
rectified or plane-warped frame, so lens correction is the rectifier's job,
never the detector's.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["detect_dots"]


def detect_dots(
    frame: np.ndarray,
    polarity: str,
    *,
    threshold: int,
    min_area: int,
    max_area: int,
    min_circularity: float,
) -> tuple[list[dict], np.ndarray]:
    """Detect circular dots of one polarity in a BGR frame.

    Segmentation channel by ``polarity`` (above), then Gaussian blur,
    threshold, open/close morphology, external contours, and an area +
    circularity filter. Centers come from ``cv2.fitEllipse`` (or plain
    moments below 5 contour points), so an elliptical projection still
    yields its geometric center.

    ``min_area`` / ``max_area`` are in pixels² of the frame given - the
    plane-warped grid in the estimator, so tune them at that resolution.

    Returns ``(detections, mask)``: one dict per dot (``center`` (2,) float,
    ``area``, ``circularity``, ``axes``, ``ellipse_angle``, ``contour``,
    ``polarity``) plus the post-morphology binary mask for diagnosis.
    """
    if polarity not in ("dark", "light", "red"):
        raise ValueError(
            f"polarity must be 'dark', 'light', or 'red', got {polarity!r}"
        )

    if polarity == "red":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        chan = lab[:, :, 1]  # a*: neutral=128, red>128, green<128
    else:
        chan = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    chan_blur = cv2.GaussianBlur(chan, (5, 5), 0)

    thresh_type = cv2.THRESH_BINARY_INV if polarity == "dark" else cv2.THRESH_BINARY
    _, mask = cv2.threshold(chan_blur, threshold, 255, thresh_type)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 1e-6:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        if len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            center = np.array(ellipse[0], dtype=float)
            axes = ellipse[1]
            ellipse_angle = ellipse[2]
        else:
            M = cv2.moments(c)
            if abs(M["m00"]) < 1e-9:
                continue
            center = np.array(
                [M["m10"] / M["m00"], M["m01"] / M["m00"]],
                dtype=float,
            )
            axes = None
            ellipse_angle = None

        detections.append(
            {
                "center": center,
                "area": area,
                "circularity": circularity,
                "axes": axes,
                "ellipse_angle": ellipse_angle,
                "contour": c,
                "polarity": polarity,
            }
        )

    return detections, mask

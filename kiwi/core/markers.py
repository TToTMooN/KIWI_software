"""ArUco / AprilTag detection and YAML config loading.

All detectors run through OpenCV's ``cv2.aruco`` module. AprilTag families
(``DICT_APRILTAG_36h11`` etc.) are predefined ArUco dictionaries in OpenCV,
so ``detect_aruco_markers`` works for them. This is what we use for calib.io
AprilGrid targets.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import yaml

ARUCO_DICT_MAP: dict[str, int] = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


def resolve_aruco_dict(name: str) -> int:
    """Look up the OpenCV enum value for an ArUco / AprilTag dictionary name."""
    if name not in ARUCO_DICT_MAP:
        raise ValueError(
            f"Unknown ArUco dictionary '{name}'. Valid options: {sorted(ARUCO_DICT_MAP)}"
        )
    return ARUCO_DICT_MAP[name]


@dataclasses.dataclass(frozen=True)
class MarkerConfig:
    id: int | None  # None when the YAML omits it (a per-setup id supplied elsewhere)
    size: float
    dictionary: str
    label: str = ""

    @property
    def cv2_dictionary(self) -> int:
        if self.dictionary not in ARUCO_DICT_MAP:
            raise ValueError(
                f"Unknown ArUco dictionary '{self.dictionary}'. "
                f"Valid options: {sorted(ARUCO_DICT_MAP)}"
            )
        return ARUCO_DICT_MAP[self.dictionary]


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def detect_aruco_markers(
    img: np.ndarray,
    marker_dict: int = cv2.aruco.DICT_4X4_50,
    allowed_ids: set[int] | None = None,
    upscale: int = 1,
    border_bits: int = 1,
) -> tuple[tuple | None, np.ndarray | None]:
    """Detect ArUco markers in an RGB image.

    ``border_bits`` is the width, in tag cells, of the marker's black border.
    **It must match how the tag was printed or nothing decodes**, because the
    detector divides the quad it finds into ``markerSize + 2*border_bits`` cells
    and samples the data bits off that grid; get it wrong and every bit lands
    between cells. Two conventions are in use here:

    - ``1`` (default) - the OpenCV/ArUco convention. Standalone ArUco markers:
      the anchor marker, the gripper/tool tags (``DICT_4X4_50``).
    - ``2`` - the **Kalibr AprilGrid** convention. Kalibr's target generator
      draws a 2-cell border (``createTargetPDF.py``: ``bitSquareSize =
      metricSize / (sqrtBits + borderBits*2)`` with ``borderBits=2``), and its
      detector defaults to ``blackTagBorder=2`` to match. Pass
      ``AprilGridConfig.border_bits`` rather than hardcoding it.

    Using 1 on a Kalibr AprilGrid does not fail cleanly: the quads that are
    found lock onto the inner 8 of the 10 printed cells, which undersizes the
    tag against ``apriltag_grid_object_points`` (corners at the full
    ``tag_size``) and biases any PnP built on it.

    With ``upscale > 1`` the image is bicubically resized by that factor
    before detection, ``CORNER_REFINE_NONE`` is used, and corner coordinates
    are divided back into native pixels before being returned.

    Returns ``(corners, ids)`` where ``corners`` is a tuple of ``(1, 4, 2)`` float32
    arrays and ``ids`` is an ``(N, 1)`` int32 array; both ``None`` if no markers
    pass the filter.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict)
    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = int(border_bits)
    if upscale > 1:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        h, w = img.shape[:2]
        det_img = cv2.resize(
            img,
            (w * upscale, h * upscale),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det_img = img
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(det_img)
    if ids is None:
        return None, None

    if allowed_ids is not None:
        mask = np.isin(ids.flatten(), list(allowed_ids))
        if not np.any(mask):
            return None, None
        corners = tuple(c for c, m in zip(corners, mask) if m)
        ids = ids[mask].reshape(-1, 1)

    if upscale > 1:
        corners = tuple((c.astype(np.float32) / upscale) for c in corners)

    return corners, ids


def marker_object_points(marker_size: float) -> np.ndarray:
    """3D corners of a planar square marker centered at the origin.

    Order: top-left, top-right, bottom-right, bottom-left (OpenCV convention).
    """
    half = marker_size / 2.0
    return np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# AprilGrid (calib.io / Kalibr-style)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AprilGridConfig:
    """calib.io / Kalibr-style AprilGrid: a rigid grid of AprilTags."""

    tag_cols: int
    """Number of tags in the x dimension (calib.io 'Columns')."""

    tag_rows: int
    """Number of tags in the y dimension (calib.io 'Rows')."""

    tag_size: float
    """Side length of one tag in metres (calib.io 'Tag Size')."""

    tag_spacing: float
    """Gap between adjacent tags as a fraction of ``tag_size`` (Kalibr ratio).
    Example: 15 mm gap on a 50 mm tag = 0.30."""

    dictionary: str = "DICT_APRILTAG_36h11"
    """OpenCV aruco dictionary name. calib.io's default for AprilGrid is 36h11."""

    border_bits: int = 2
    """Width of the tag's black border in cells. Kalibr AprilGrids are drawn
    with **2** (``createTargetPDF.py`` uses ``borderBits=2``, and Kalibr's own
    detector defaults to ``blackTagBorder=2``), unlike standalone OpenCV ArUco
    markers which use 1. It is a property of how the board was PRINTED.

    Consequently ``tag_size`` spans ``6 + 2*border_bits = 10`` cells - the full
    printed black square - which is what ``apriltag_grid_object_points`` places
    its corners on. Detecting the same board at ``border_bits=1`` returns quads
    that lock onto the inner 8 of 10 cells, silently biasing any PnP."""

    start_id: int = 0
    """First tag ID used by the grid (calib.io 'Start Id'). The grid occupies
    IDs ``[start_id, start_id + tag_cols * tag_rows)``."""

    label: str = ""

    @property
    def cv2_dictionary(self) -> int:
        return resolve_aruco_dict(self.dictionary)

    @property
    def tag_ids(self) -> range:
        """All tag IDs belonging to this grid."""
        return range(self.start_id, self.start_id + self.tag_cols * self.tag_rows)

    @property
    def stride(self) -> float:
        """Centre-to-centre distance between adjacent tags (m)."""
        return self.tag_size * (1.0 + self.tag_spacing)


def load_apriltag_grid_configs(path: Path) -> dict[str, AprilGridConfig]:
    """Load named AprilGrid presets from the ``apriltag_grid:`` section of a YAML file."""
    raw = _load_yaml(path)
    grids = raw.get("apriltag_grid", {})
    return {name: AprilGridConfig(**fields) for name, fields in grids.items()}


def apriltag_grid_object_points(config: AprilGridConfig) -> dict[int, np.ndarray]:
    """3D positions of each tag's 4 corners in the board frame (Kalibr convention).

    Follows Kalibr's ``GridCalibrationTargetAprilgrid::createGridPoints``
    (``aslam_cv/aslam_cameras_april/src/GridCalibrationTargetAprilgrid.cpp``):

      - **Origin** at the bottom-left corner of tag ``start_id`` (the
        (col=0, row=0) tag).
      - **x axis** along columns (rightward).
      - **y axis** along rows pointing **up** (so ``row=0`` is the bottom
        row of the board).
      - z = 0 (board lies in the xy plane).

    Tag IDs increment row-major from the bottom-left tag::

        tag_id = start_id + row * tag_cols + col   # row 0 is the bottom row

    Per-tag corner order is OpenCV ArUco's ``[TL, TR, BR, BL]`` (matches
    ``cv2.aruco.ArucoDetector.detectMarkers`` output), which is the reverse
    of Kalibr's internal ``[BL, BR, TR, TL]`` pIdx storage - same four
    points, swapped order, so ``solvePnP(obj_pts, image_corners)`` works
    directly without reordering the detected corners.

    Returns ``{tag_id: (4, 3) float32 ndarray}`` for every tag in the grid.
    """
    stride = config.stride
    s = config.tag_size
    points: dict[int, np.ndarray] = {}
    for r in range(config.tag_rows):
        for c in range(config.tag_cols):
            tag_id = config.start_id + r * config.tag_cols + c
            x0 = c * stride  # left edge of this tag
            y0 = r * stride  # bottom edge of this tag (y-up)
            points[tag_id] = np.array(
                [
                    [x0, y0 + s, 0.0],  # TL - top edge has the higher y
                    [x0 + s, y0 + s, 0.0],  # TR
                    [x0 + s, y0, 0.0],  # BR
                    [x0, y0, 0.0],  # BL
                ],
                dtype=np.float32,
            )
    return points

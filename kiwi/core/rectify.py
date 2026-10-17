"""Fisheye -> single-pinhole rectification per Insta360 lens.

Build a per-lens remap LUT once from the calibration file, apply it to
every frame of the corresponding per-lens video. Each rectified frame
behaves like a standard pinhole camera with the synthetic intrinsics
``K_pinhole`` returned here, so downstream ``cv2.solvePnP`` works
directly (with zero distortion).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Per-lens rectifier
# ---------------------------------------------------------------------------


def _pinhole_K(out_width: int, out_height: int, fov_deg: float) -> np.ndarray:
    """Synthetic pinhole intrinsics that span ``fov_deg`` horizontally."""
    fov_rad = math.radians(fov_deg)
    fx = (out_width / 2.0) / math.tan(fov_rad / 2.0)
    return np.array(
        [[fx, 0.0, out_width / 2.0], [0.0, fx, out_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _build_double_sphere_map(
    ds_params: np.ndarray,
    K_pinhole: np.ndarray,
    out_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the float remap LUT for a Double-Sphere -> pinhole rectification.

    `ds_params` is ``[xi, alpha, fx, fy, cx, cy]`` in kalibr's convention.
    For each output pinhole pixel, we unproject it to a 3D ray via
    ``K_pinhole``, then project that ray onto the source fisheye image via
    the DS forward model (Usenko et al. 2018, eq. 6). Pixels outside the
    DS projection's valid region (rays pointing too far behind the camera)
    are marked with a sentinel coordinate so ``cv2.remap`` fills them with
    the border color.
    """
    xi, alpha, fx, fy, cx, cy = (float(x) for x in ds_params)
    W, H = int(out_size[0]), int(out_size[1])

    # Pinhole pixel grid (each row = a v value, each col = a u value).
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    pixels = np.stack(
        [u_grid.ravel(), v_grid.ravel(), np.ones(W * H)], axis=1
    )  # (N, 3)

    # Unproject pinhole pixels into 3D rays (K^{-1} * [u, v, 1]).
    rays = pixels @ np.linalg.inv(K_pinhole).T  # (N, 3)
    X = rays[:, 0]
    Y = rays[:, 1]
    Z = rays[:, 2]

    # DS validity (Usenko 2018, eq. 8). A ray is "in front of" the DS
    # camera iff Z > -w2 * ||(X, Y, Z)||. Outside this region the forward
    # projection is undefined, so the pinhole pixel gets no source pixel.
    w1 = (1.0 - alpha) / alpha if alpha > 0.5 else alpha / (1.0 - alpha)
    w2 = (w1 + xi) / math.sqrt(2.0 * w1 * xi + xi * xi + 1.0)
    d1 = np.sqrt(X * X + Y * Y + Z * Z)
    valid = Z > -w2 * d1

    # Forward projection (eq. 6 of the paper):
    #   d2     = ||(X, Y, xi*d1 + Z)||
    #   m_xy   = (X, Y) / (alpha*d2 + (1 - alpha)*(xi*d1 + Z))
    #   pixel  = K * [m_xy; 1]
    zb = xi * d1 + Z
    d2 = np.sqrt(X * X + Y * Y + zb * zb)
    denom = alpha * d2 + (1.0 - alpha) * zb
    safe_denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)
    x_norm = X / safe_denom
    y_norm = Y / safe_denom
    u_src = fx * x_norm + cx
    v_src = fy * y_norm + cy

    invalid = ~valid | (denom <= 1e-12)
    # Negative sentinels make BORDER_CONSTANT fill these pixels with the
    # remap border color (default black).
    u_src[invalid] = -1.0
    v_src[invalid] = -1.0

    return u_src.reshape(H, W).astype(np.float32), v_src.reshape(H, W).astype(
        np.float32
    )


@dataclasses.dataclass
class Rectifier:
    """Precomputed fisheye -> pinhole remap for one lens.

    The map is built with ``cv2.fisheye.initUndistortRectifyMap`` for an
    equidistant (``K``/``D``) calibration, or by projecting each pinhole ray
    through the Double-Sphere model for a kalibr ``ds`` calibration. Either
    way the result is one LUT, so the downstream code path is identical: one
    ``cv2.remap`` per frame.
    """

    K_pinhole: np.ndarray
    map1: np.ndarray
    map2: np.ndarray
    out_size: tuple[int, int]  # (W, H)

    @classmethod
    def build(
        cls,
        K_fisheye: np.ndarray,
        D_fisheye: np.ndarray,
        out_size: tuple[int, int],
        fov_deg: float,
    ) -> "Rectifier":
        K_p = _pinhole_K(out_size[0], out_size[1], fov_deg)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K_fisheye.astype(np.float64),
            D_fisheye.astype(np.float64).reshape(4, 1),
            np.eye(3, dtype=np.float64),
            K_p,
            out_size,
            cv2.CV_16SC2,
        )
        return cls(K_pinhole=K_p, map1=map1, map2=map2, out_size=out_size)

    @classmethod
    def build_double_sphere(
        cls,
        ds_params: np.ndarray,
        out_size: tuple[int, int],
        fov_deg: float,
    ) -> "Rectifier":
        """Rectify a Double-Sphere fisheye (Usenko 2018) to a virtual pinhole.

        Drop-in for ``build`` when the source calibration used kalibr's
        ``ds-none`` model instead of ``cv2.fisheye``'s equidistant. The
        output ``K_pinhole`` and downstream ``cv2.solvePnP`` path are
        unchanged; only the source-image sampling math differs.

        ``ds_params`` is the 6-element ``[xi, alpha, fx, fy, cx, cy]``
        vector kalibr emits in ``calib-camchain.yaml``.
        """
        K_p = _pinhole_K(out_size[0], out_size[1], fov_deg)
        map_x, map_y = _build_double_sphere_map(ds_params, K_p, out_size)
        fixed_xy, fixed_extra = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        return cls(
            K_pinhole=K_p,
            map1=fixed_xy,
            map2=fixed_extra,
            out_size=out_size,
        )

    def apply(self, lens_frame: np.ndarray) -> np.ndarray:
        return cv2.remap(
            lens_frame,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


# ---------------------------------------------------------------------------
# Intrinsics bundle (the registry intrinsics.npz layout)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IntrinsicsBundle:
    """Per-lens intrinsics loaded from a registry ``intrinsics.npz``.

    Either lens may be missing if it wasn't calibrated yet - fields are
    ``None`` in that case. ``image_size`` is always present once any lens
    has been calibrated.
    """

    K_front: np.ndarray | None
    D_front: np.ndarray | None
    K_back: np.ndarray | None
    D_back: np.ndarray | None
    image_size: tuple[int, int]

    # Double-Sphere (Usenko 2018) intrinsics - used when the lens was
    # calibrated via kalibr `--model ds-none` instead of cv2.fisheye.
    # Stored as the 6-element `[xi, alpha, fx, fy, cx, cy]` vector that
    # kalibr writes into calib-camchain.yaml.
    ds_front: np.ndarray | None = None
    ds_back: np.ndarray | None = None

    @classmethod
    def load(cls, path: Path) -> "IntrinsicsBundle":
        d = np.load(str(path))
        size = tuple(int(v) for v in d["image_size"])
        if len(size) != 2:
            raise ValueError(f"image_size in {path} has unexpected shape {size}")
        return cls(
            K_front=d["K_front"] if "K_front" in d.files else None,
            D_front=d["D_front"] if "D_front" in d.files else None,
            K_back=d["K_back"] if "K_back" in d.files else None,
            D_back=d["D_back"] if "D_back" in d.files else None,
            image_size=size,  # type: ignore[arg-type]
            ds_front=d["ds_front"] if "ds_front" in d.files else None,
            ds_back=d["ds_back"] if "ds_back" in d.files else None,
        )

    def rectifier_for(
        self,
        lens: str,
        out_size: tuple[int, int],
        fov_deg: float,
    ) -> Rectifier:
        # Double Sphere when present, else equidistant fisheye.
        ds: np.ndarray | None = getattr(self, f"ds_{lens}")
        if ds is not None:
            return Rectifier.build_double_sphere(ds, out_size, fov_deg)

        K = getattr(self, f"K_{lens}")
        D = getattr(self, f"D_{lens}")
        if K is None or D is None:
            raise RuntimeError(
                f"Intrinsics for '{lens}' lens not present in calibration file"
            )
        return Rectifier.build(K, D, out_size, fov_deg)


def load_rectifier_auto(
    npz_path: Path,
    src_size: tuple[int, int],
    *,
    lens: str = "front",
    out_size: tuple[int, int] | None = None,
    fov_deg: float = 110.0,
) -> Rectifier:
    """Build a Rectifier for ``lens`` from a registry intrinsics npz.

    The npz carries ``image_size`` plus per-lens ``K_{front,back}`` /
    ``D_{front,back}`` (equidistant fisheye) or ``ds_{front,back}``
    (Double-Sphere), as written by the calibration tools. It is routed
    through ``IntrinsicsBundle.load(...).rectifier_for(lens, out_size,
    fov_deg)``; ``out_size`` defaults to ``src_size`` when omitted.
    """
    bundle = IntrinsicsBundle.load(npz_path)
    return bundle.rectifier_for(lens, out_size or src_size, fov_deg)

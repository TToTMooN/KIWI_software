"""Distill a Kalibr run into the registry ``intrinsics.npz``.

Kalibr's canonical output is ``calib-camchain.yaml``, written only when the
global bundle adjustment converges. When it does not, the salvage path
parses the per-batch "Calibration results" blocks out of ``kalibr.log`` and
picks the most-viewed fit. The parser covers the project's default model: a
``ds-none`` block prints ``distortion: [] +- []``, and its projection is 6
values where ``pinhole-equi`` prints 4.

npz schema (consumed by :class:`kiwi.core.rectify.IntrinsicsBundle`):

  ds-none (Double-Sphere, the default):
    ``ds_front`` / ``ds_back``   (6,) [xi, alpha, fx, fy, cx, cy] per lens
  pinhole-equi (equidistant fisheye):
    ``K_front``/``D_front``, ``K_back``/``D_back``  (cv2.fisheye)
  ``image_size``                 (2,) int32 (W, H), both models

The two lenses face opposite directions, so each is its own recording and
its own run dir; :func:`save_intrinsics_npz` merges whichever sides are
given, and ``base`` carries an untouched lens forward from the currently
published bundle (the registry keeps both lenses in one npz).

Image sizes come from the run manifest (:mod:`kiwi.calib.kalibr.manifest`),
falling back to probing a staged PNG for a hand-built run dir - never from
the bag.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import yaml

from kiwi.calib.kalibr import layout, manifest

__all__ = [
    "Batch",
    "DoubleSphereResult",
    "FisheyeResult",
    "best_batch",
    "lens_result",
    "parse_kalibr_log",
    "read_camchain",
    "result_from_batch",
    "save_intrinsics_npz",
    "staged_image_size",
]

# Kalibr writes per-batch "Calibration results" blocks. pinhole-equi:
#       distortion: [ 0.0459 -0.0051  0.0041 -0.0010] +- [...]
#       projection: [354.89 354.77 664.98 661.97] +- [...]
#       reprojection error: [-0.000000, -0.000000] +- [0.282028, 0.299070]
# Double-Sphere (ds-none): distortion is EMPTY and projection is 6 values
# [xi alpha fu fv cu cv]:
#       distortion: [] +- []
_RE_VIEWS = re.compile(
    r"Processed\s+\d+\s+of\s+\d+\s+views\s+with\s+(\d+)\s+views used"
)
# `*` (not `+`) so an empty DS distortion `[]` matches.
_RE_DIST = re.compile(r"distortion:\s+\[([^\]]*)\]\s+\+-")
_RE_PROJ = re.compile(r"projection:\s+\[([^\]]+)\]\s+\+-")
_RE_REPRO_STD = re.compile(r"reprojection error:\s+\[[^\]]+\]\s+\+-\s+\[([^\]]+)\]")


@dataclasses.dataclass(frozen=True)
class Batch:
    views_used: int
    # Variable length by model: pinhole-equi is 4 projection (fx, fy, cx, cy)
    # + 4 distortion (k1..k4); Double-Sphere is 6 projection + 0 distortion.
    projection: tuple[float, ...]
    distortion: tuple[float, ...]
    repro_err_std: tuple[float, float]  # std of per-view residuals, (u, v)


def parse_kalibr_log(log_path: Path) -> list[Batch]:
    """Parse every per-batch 'Calibration results' block from a kalibr log.

    Returns one Batch per block in file order; a block only commits once all
    four fields (views, projection, distortion, reprojection error) are
    present, so partial blocks (a crash mid-print) are skipped silently.

    The file is segmented on the 'views used' lines and each field is
    searched within its block, NOT line by line: numpy line-wraps the
    6-element DS projection array in Kalibr logs
    (``projection: [ ... five values\\n sixth] +- [...]``), and the
    ``[^\\]]`` classes span newlines. For a dual-camera block, the first
    (cam0) parameter set is taken.
    """
    text = Path(log_path).read_text()
    # split with a capturing group keeps the views counts at odd indices:
    # [preamble, views_1, body_1, views_2, body_2, ...]
    parts = _RE_VIEWS.split(text)
    batches: list[Batch] = []
    for views_s, body in zip(parts[1::2], parts[2::2]):
        mp = _RE_PROJ.search(body)
        md = _RE_DIST.search(body)
        mr = _RE_REPRO_STD.search(body)
        if not (mp and md and mr):
            continue  # partial block (e.g. a crash mid-print)
        batches.append(
            Batch(
                views_used=int(views_s),
                projection=tuple(float(x) for x in mp.group(1).split()),
                distortion=tuple(float(x) for x in md.group(1).split()),
                # Kalibr formats this list comma-separated, unlike the others.
                repro_err_std=tuple(  # type: ignore[arg-type]
                    float(x) for x in mr.group(1).replace(",", " ").split()
                ),
            )
        )
    return batches


def best_batch(batches: list[Batch]) -> Batch:
    """The batch fit on the most views.

    Ties (across restart attempts) break to the lowest mean residual.
    """
    if not batches:
        raise ValueError(
            "kalibr.log contained no parseable 'Calibration results' blocks"
        )
    return max(batches, key=lambda b: (b.views_used, -sum(b.repro_err_std) / 2.0))


@dataclasses.dataclass(frozen=True)
class FisheyeResult:
    """An equidistant-fisheye (kalibr ``pinhole-equi``) fit."""

    K: np.ndarray  # 3x3
    D: np.ndarray  # (4,)
    size: tuple[int, int]  # (W, H)


@dataclasses.dataclass(frozen=True)
class DoubleSphereResult:
    """A Double-Sphere (kalibr ``ds-none``) fit."""

    ds: np.ndarray  # (6,) [xi, alpha, fx, fy, cx, cy]
    size: tuple[int, int]  # (W, H)


def batch_to_K_D(b: Batch) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = b.projection
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, np.array(b.distortion, dtype=np.float64)


def result_from_batch(
    b: Batch, size: tuple[int, int]
) -> FisheyeResult | DoubleSphereResult:
    """A lens result from a salvaged per-batch fit, dispatched on model arity.

    6 projection values = Double-Sphere ([xi, alpha, fx, fy, cx, cy], no
    distortion; the log's order already matches ``core.rectify``'s ``ds_*``
    key, no reordering); 4 = equidistant fisheye + 4 distortion coeffs.
    """
    if len(b.projection) == 6:
        return DoubleSphereResult(
            ds=np.array(b.projection, dtype=np.float64), size=size
        )
    if len(b.projection) == 4:
        K, D = batch_to_K_D(b)
        return FisheyeResult(K=K, D=D, size=size)
    raise ValueError(
        f"salvage: unexpected projection length {len(b.projection)} "
        "(expected 4 for pinhole-equi or 6 for ds-none)"
    )


def read_camchain(
    run_dir: Path, cam: str = "cam0"
) -> FisheyeResult | DoubleSphereResult | None:
    """Intrinsics from kalibr's canonical ``calib-camchain.yaml``, or None.

    ``None`` means the file does not exist (kalibr never converged on this
    run) and the caller should fall back to the log salvage. A malformed or
    unsupported camchain raises ``ValueError``, including a ``pinhole``
    entry with a non-``equidistant`` distortion model.
    """
    camchain = Path(run_dir) / "calib-camchain.yaml"
    if not camchain.exists():
        return None
    cfg = yaml.safe_load(camchain.read_text())
    entry = cfg.get(cam)
    if entry is None:
        raise ValueError(f"{camchain}: missing {cam} entry")
    model = entry.get("camera_model")
    intrinsics = list(entry["intrinsics"])
    resolution = tuple(int(x) for x in entry["resolution"])

    if model == "ds":
        if len(intrinsics) != 6:
            raise ValueError(
                f"{camchain}: ds camera_model expects 6 intrinsics, got {len(intrinsics)}"
            )
        return DoubleSphereResult(
            ds=np.array(intrinsics, dtype=np.float64), size=resolution
        )
    if model == "pinhole":
        if len(intrinsics) != 4:
            raise ValueError(
                f"{camchain}: pinhole camera_model expects 4 intrinsics, "
                f"got {len(intrinsics)}"
            )
        dmodel = entry.get("distortion_model")
        if dmodel != "equidistant":
            raise ValueError(
                f"{camchain}: the pinhole branch handles only the equidistant "
                f"distortion model (cv2.fisheye), not {dmodel!r}"
            )
        fx, fy, cx, cy = intrinsics
        dist = list(entry.get("distortion_coeffs") or [])
        if len(dist) != 4:
            raise ValueError(
                f"{camchain}: pinhole-equi expects 4 distortion coeffs, got {len(dist)}"
            )
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return FisheyeResult(K=K, D=np.array(dist, dtype=np.float64), size=resolution)
    raise ValueError(f"{camchain}: unsupported camera_model {model!r}")


def staged_image_size(run_dir: Path, cam: str = "cam0") -> tuple[int, int]:
    """``(W, H)`` for the salvage path: the manifest, else a staged PNG.

    A hand-built run dir may carry frames but no manifest. Never the bag.
    """
    run_dir = Path(run_dir)
    try:
        return manifest.image_size(run_dir, cam)
    except (FileNotFoundError, KeyError):
        pass
    frames = sorted((run_dir / layout.FRAMES_DIR / cam).glob(f"*{layout.IMAGE_SUFFIX}"))
    if frames:
        import cv2  # deferred: the parsing half of this module stays cv2-free

        img = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape[:2]
            return (w, h)
    raise ValueError(
        f"{run_dir}: cannot determine image size for {cam} - no run-manifest entry "
        f"and no staged frames under {layout.FRAMES_DIR}/{cam}/"
    )


def lens_result(
    run_dir: Path, cam: str = "cam0"
) -> tuple[FisheyeResult | DoubleSphereResult, list[Batch] | None]:
    """The best calibration result for one lens run: ``(result, salvage)``.

    Prefers the canonical camchain (``salvage`` is then None); falls back to
    the per-batch log parse when the global solve diverged, in which case
    ``salvage`` carries the parsed batches for the caller to display.
    """
    result = read_camchain(run_dir, cam)
    if result is not None:
        return result, None
    log = Path(run_dir) / "kalibr.log"
    if not log.exists():
        raise ValueError(f"{run_dir}: has neither calib-camchain.yaml nor {log.name}")
    batches = parse_kalibr_log(log)
    best = best_batch(batches)
    return result_from_batch(best, staged_image_size(run_dir, cam)), batches


def save_intrinsics_npz(
    output: Path,
    results: dict[str, FisheyeResult | DoubleSphereResult],
    base: Path | None = None,
) -> list[str]:
    """Assemble the per-unit npz from per-lens results; returns the keys written.

    ``results`` maps side ("front"/"back") to that lens's fit. ``base`` seeds
    the output with the currently published bundle so the lens NOT re-shot
    carries forward - without it the untouched lens is silently dropped.
    """
    out: dict[str, np.ndarray] = {}
    if base is not None:
        with np.load(str(base)) as z:
            out.update({k: z[k] for k in z.files})
    sizes: list[tuple[int, int]] = []
    for side, r in results.items():
        if side not in ("front", "back"):
            raise ValueError(f"unknown lens side {side!r}")
        if isinstance(r, DoubleSphereResult):
            out[f"ds_{side}"] = r.ds
            # A DS re-fit supersedes any stale fisheye keys for this side.
            out.pop(f"K_{side}", None)
            out.pop(f"D_{side}", None)
        else:
            out[f"K_{side}"] = r.K
            out[f"D_{side}"] = r.D
            out.pop(f"ds_{side}", None)
        sizes.append(r.size)
    if not sizes and "image_size" not in out:
        raise ValueError("no lens results and no base bundle - nothing to write")
    if len(sizes) == 2 and sizes[0] != sizes[1]:
        raise ValueError(
            f"front and back image sizes differ: {sizes[0]} vs {sizes[1]} - "
            "the two lenses share a sensor, so the runs disagree about the "
            "crop; refusing to write a mixed bundle"
        )
    # The same rule across a re-shoot. A lens the base supplies and this call
    # does not re-fit is carried forward at the BASE's crop, so overwriting
    # image_size under it assembles the mixed bundle above one run at a time.
    # Only carried-forward lenses matter: re-fitting both at a new crop is a
    # legitimate full re-shoot, and nothing survives to disagree.
    carried = sorted(
        side
        for side in ("front", "back")
        if side not in results and (f"ds_{side}" in out or f"K_{side}" in out)
    )
    if sizes and carried and "image_size" in out:
        base_size = tuple(int(v) for v in out["image_size"])
        if base_size != sizes[0]:
            raise ValueError(
                f"re-fit image size {sizes[0]} differs from the base bundle's "
                f"{base_size} ({base}); the {', '.join(carried)} lens carried "
                f"forward was measured at the base crop, so this would write a "
                f"mixed bundle. Re-fit both lenses at the new crop instead"
            )
    if sizes:
        out["image_size"] = np.array(sizes[0], dtype=np.int32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output), **out)
    return sorted(out)

"""Scan a lens video for one ArUco marker and persist every detection.

The scan is separate from the anchor solve:

* **Re-anchoring reads the cache.** The scan rectifies, detects and runs
  solvePnP over ~120 s of 1920x1920 video; the solve reads the cached
  detections and does not re-decode the clip.
* **``max_marker_dist`` stays a solve-time filter.** EVERY detection is
  written, distance included. Retuning the PnP-flip filter therefore needs
  no re-scan.
* **Front lens only.** :func:`kiwi.register.anchor.solve_anchor` composes
  ``T_cam_marker`` with the FRONT camera's track pose and has no leg for
  ``T_camF_camB``. The state stage publishes the front optical camera pose,
  so the anchor consumes front detections without an additional
  camera-to-camera transform. The scan takes no lens argument and refuses a
  ``lens1.mp4`` by name; the file records ``lens`` and the readers refuse
  anything but ``front``.

The file is the contract, not this module: schema ``marker_detections_v1``,
a ``meta`` block plus one row per detected frame. ``meta`` carries the
rectifier parameters (``lens``, ``fov_deg``, ``out_size``) of the virtual
pinhole frame ``T_cam_marker`` is expressed in; :func:`require_compatible`
checks them against a solve.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from kiwi.core import layout

__all__ = [
    "FRONT",
    "LENS_INDEX",
    "SCHEMA",
    "MarkerObservation",
    "load_detections",
    "refuse_rear_clip",
    "require_compatible",
    "scan_video_for_marker",
    "write_debug_image",
    "write_detections",
]

SCHEMA = "marker_detections_v1"

#: The ``convert`` demux mapping: ``<stem>_lens0.mp4`` (front), ``_lens1.mp4``
#: (rear). The rectifier and the IntrinsicsBundle both spell the rear lens
#: "back". It is file-format vocabulary - it names the stream a file's
#: ``lens_index`` refers to - not a read path: both lens refusals below
#: compare the ``lens`` STRING in ``meta``, never this index.
LENS_INDEX = {"front": 0, "back": 1}

#: The only lens an anchor can be solved from - see the module docstring.
FRONT = "front"


@dataclasses.dataclass(frozen=True)
class MarkerObservation:
    """One frame in which the marker was found."""

    frame: int
    t_video: float  # frame / fps; the stream's sync offset maps it to the global clock
    T_cam_marker: np.ndarray  # 4x4, in the RECTIFIED pinhole frame (see meta)
    corners: np.ndarray  # 4x2 image points, same frame
    reproj: float  # RMS reprojection error (px); the anchor weight is 1/reproj^2

    @property
    def distance_m(self) -> float:
        """||t|| of the marker in the camera frame - the ``max_marker_dist`` knob."""
        return float(np.linalg.norm(self.T_cam_marker[:3, 3]))


def write_detections(
    path: Path, *, meta: dict, observations: list[MarkerObservation]
) -> Path:
    """Write ``marker_detections_v1``. Rows are ordered by frame."""
    doc = {
        "schema": SCHEMA,
        "meta": meta,
        "detections": [
            {
                "frame": int(o.frame),
                "t_video": float(o.t_video),
                "reproj": float(o.reproj),
                "distance_m": o.distance_m,
                "T_cam_marker": [float(v) for v in np.asarray(o.T_cam_marker).ravel()],
                "corners": [float(v) for v in np.asarray(o.corners).ravel()],
            }
            for o in sorted(observations, key=lambda o: o.frame)
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


def refuse_rear_clip(video: Path) -> None:
    """Raise if ``video`` is a demuxed REAR-lens clip, by name.

    The scan takes no ``lens`` argument. Called at both boundaries an
    operator can reach: the CLI verb (before it goes looking for intrinsics)
    and the scan itself.
    """
    name = Path(video).name
    if name == layout.LENS1 or name.endswith(f"_{layout.LENS1}"):
        raise ValueError(
            f"{video} is the REAR lens ({layout.LENS1} / *_{layout.LENS1}); "
            f"scan {layout.LENS0} instead. The anchor is solved in the FRONT "
            f"camera's frame; the track comes from the front stream and there "
            f"is no T_camF_camB leg."
        )


def load_detections(path: Path) -> tuple[dict, list[MarkerObservation]]:
    """Read ``marker_detections_v1`` -> ``(meta, observations)``."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(
            f"{path}: expected schema {SCHEMA!r}, got {doc.get('schema')!r}"
        )
    obs = [
        MarkerObservation(
            frame=int(r["frame"]),
            t_video=float(r["t_video"]),
            T_cam_marker=np.asarray(r["T_cam_marker"], dtype=float).reshape(4, 4),
            corners=np.asarray(r["corners"], dtype=float).reshape(4, 2),
            reproj=float(r["reproj"]),
        )
        for r in doc.get("detections", [])
    ]
    meta = doc.get("meta", {})
    got_lens = meta.get("lens")
    # An EXPLICIT rear lens is refused here. An ABSENT lens is read as front.
    # The reuse path does not share the tolerance - the stage passes
    # lens="front" to require_compatible, which calls an unrecorded lens a
    # cache miss and re-scans - so absence is only accepted where an operator
    # named the file themselves (`kiwi register anchor`).
    if got_lens is not None and str(got_lens) != FRONT:
        raise ValueError(
            f"{path}: scanned on the {got_lens} lens. The anchor solve has no "
            f"T_camF_camB leg. Re-scan the front lens mp4."
        )
    return meta, obs


def _unrecorded(what: str) -> str:
    """Message for a cache that does not say what it was scanned with."""
    return (
        f"detections do not record {what}, so this solve cannot tell whether "
        f"they were produced for it; re-run the scan (which records it) rather "
        f"than trust an unverifiable cache"
    )


def require_compatible(
    meta: dict,
    *,
    marker_id: int,
    fov_deg: float,
    out_size: int,
    marker_size: float | None = None,
    marker_dict: str | None = None,
    lens: str | None = None,
    scan_seconds: float | None = None,
) -> None:
    """Raise unless cached detections were produced for this solve.

    ``T_cam_marker`` lives in the virtual-pinhole frame defined by
    ``(fov_deg, out_size)`` and by the LENS, so detections rectified
    differently, or scanned on the other lens, are refused. ``marker_size``
    is checked too when given: the PnP translation is LINEAR in the
    configured size.

    ``marker_dict`` and ``scan_seconds`` are checked when given.
    ``scan_seconds`` is one-sided - a cache scanned LONGER than asked
    already covers the requested window, so only a shorter one is refused.
    What is compared is the window the cache ASKED for, not the
    ``frames_scanned`` it managed: a clip that ended first is re-scanned
    too.

    The three checks the caller opts into (dictionary, lens, window) also
    refuse a cache that does not RECORD the value: unverifiable is a cache
    miss, not a pass, and the re-scan it forces writes the key. The other
    four keys (marker id and size, ``fov_deg``, ``out_size``) stay tolerant
    of absence so hand-written files load.
    """
    got_id = meta.get("marker", {}).get("id")
    if got_id is not None and int(got_id) != int(marker_id):
        raise ValueError(f"detections are for marker {got_id}, solve wants {marker_id}")
    got_size = meta.get("marker", {}).get("size")
    if (
        marker_size is not None
        and got_size is not None
        and float(got_size) != float(marker_size)
    ):
        raise ValueError(
            f"detections were scanned with marker size {got_size} m but this solve "
            f"uses {marker_size} m; PnP translation scales linearly with the size, "
            f"so re-run the scan"
        )
    if marker_dict is not None:
        got_dict = meta.get("marker", {}).get("dictionary")
        if got_dict is None:
            raise ValueError(_unrecorded("which ArUco dictionary they used"))
        if str(got_dict) != str(marker_dict):
            raise ValueError(
                f"detections were scanned with dictionary {got_dict} but this solve "
                f"uses {marker_dict}; id {marker_id} names a different physical tag "
                f"in another family, so re-run the scan"
            )
    if lens is not None:
        got_lens = meta.get("lens")
        if got_lens is None:
            raise ValueError(_unrecorded("which lens they were scanned on"))
        if str(got_lens) != str(lens):
            raise ValueError(
                f"detections were scanned on the {got_lens} lens but this solve "
                f"uses {lens}; T_cam_marker is expressed in THAT lens's pinhole "
                f"frame, and the anchor composes it with the FRONT camera's "
                f"track pose, so re-run the scan on the front lens"
            )
    if scan_seconds is not None:
        got_scan = meta.get("scan_seconds")
        if got_scan is None:
            raise ValueError(_unrecorded("the window they scanned"))
        if float(got_scan) < float(scan_seconds):
            raise ValueError(
                f"detections cover only the first {got_scan} s but this solve asks "
                f"for {scan_seconds} s; a marker first visible later would be missed, "
                f"so re-run the scan"
            )
    for key, want in (("fov_deg", float(fov_deg)), ("out_size", int(out_size))):
        got = meta.get(key)
        if got is not None and float(got) != float(want):
            raise ValueError(
                f"detections were rectified with {key}={got} but this solve uses "
                f"{want}; T_cam_marker is expressed in that pinhole frame, so "
                f"re-run the scan"
            )


def scan_video_for_marker(
    video: Path,
    *,
    intrinsics: Path,
    marker_id: int,
    marker_size: float,
    marker_dict: str = "DICT_4X4_50",
    fov_deg: float = 140.0,
    out_size: int = 1920,
    scan_seconds: float = 120.0,
) -> tuple[dict, list[MarkerObservation]]:
    """Detect ``marker_id`` in the first ``scan_seconds`` of ``video``.

    Returns ``(meta, observations)`` ready for :func:`write_detections`. No
    distance filtering happens here - every detection is kept, so the
    consumer can retune ``max_marker_dist`` without re-scanning.

    Detection runs on a DS->pinhole rectified view. The rectifier applies an
    identity rotation about the optical centre
    (``kiwi.core.rectify.Rectifier.build_double_sphere``), so the pinhole
    frame IS the camera frame every published extrinsic refers to - swap in a
    rotated view and every ``T_cam_marker`` silently changes frame.

    The rear clip is refused (:func:`refuse_rear_clip`): this is the boundary
    every caller (the CLI verb, the stage, a library user) passes through.
    """
    refuse_rear_clip(video)
    # Imported here so the pure schema helpers above stay importable without cv2.
    from kiwi.core.markers import MarkerConfig
    from kiwi.core.rectify import load_rectifier_auto
    from kiwi.core.video import VideoInfo, iter_frames

    from kiwi.register.pnp import detect_per_marker_pnp

    if not 0.001 <= float(marker_size) <= 1.0:
        # Size sets the metric scale of every T_cam_marker (the same range
        # check kiwi.core.tool_config applies to the gripper tag size).
        raise ValueError(
            f"marker_size {marker_size} is out of range - it is the tag's "
            f"black-square edge in METRES (expected 0.001-1.0; 149 mm is "
            f"0.149, not 149)"
        )

    fps = VideoInfo.read(video).fps
    n_scan = int(round(scan_seconds * fps))
    rect = load_rectifier_auto(
        intrinsics,
        src_size=(out_size, out_size),
        lens=FRONT,
        out_size=(out_size, out_size),
        fov_deg=fov_deg,
    )
    K = rect.K_pinhole
    D = np.zeros((1, 5), dtype=np.float64)  # the rectified frame is distortion-free
    marker_cfgs = {
        marker_id: MarkerConfig(
            id=marker_id, size=marker_size, dictionary=marker_dict, label="world"
        )
    }

    observations: list[MarkerObservation] = []
    n_frames = 0
    for idx, bgr in iter_frames(video):
        if idx >= n_scan:
            break
        n_frames += 1
        obs, _, _ = detect_per_marker_pnp(rect.apply(bgr), marker_cfgs, K, D)
        if marker_id not in obs:
            continue
        T_C_M, img_pts, reproj = obs[marker_id]
        observations.append(
            MarkerObservation(
                frame=idx,
                t_video=idx / fps,
                T_cam_marker=np.asarray(T_C_M, dtype=float),
                corners=np.asarray(img_pts, dtype=float).reshape(4, 2),
                reproj=float(reproj),
            )
        )

    from kiwi.core.generation import file_ref, producer_info

    meta = {
        "producer": producer_info(
            "aruco_marker_scan", dependencies=("opencv-python-headless", "numpy")
        ),
        "inputs": {
            "video": file_ref(video, record=video.parent / "convert.inputs.json"),
            "intrinsics": file_ref(intrinsics),
        },
        "video": str(video),
        "lens": FRONT,
        "lens_index": LENS_INDEX[FRONT],
        "fps": float(fps),
        "scan_seconds": float(scan_seconds),
        "frames_scanned": n_frames,
        "n_detections": len(observations),
        "marker": {
            "id": int(marker_id),
            "size": float(marker_size),
            "dictionary": marker_dict,
        },
        "intrinsics": str(intrinsics),
        "fov_deg": float(fov_deg),
        "out_size": int(out_size),
    }
    return meta, observations


def write_debug_image(
    video: Path,
    intrinsics: Path,
    meta: dict,
    observations: list[MarkerObservation],
    out: Path,
) -> Path | None:
    """The best (lowest-reproj) detection frame with the marker axes drawn.

    A human sanity check on the anchor - one decode of the winning frame.
    Returns ``None`` when that frame cannot be re-read (a re-encoded or
    truncated video); never raises for a missing frame.
    """
    import cv2

    from kiwi.core.rectify import load_rectifier_auto
    from kiwi.core.video import iter_frames

    best = min(observations, key=lambda o: o.reproj)
    out_size = int(meta.get("out_size", 1920))
    rect = load_rectifier_auto(
        intrinsics,
        src_size=(out_size, out_size),
        lens=str(meta.get("lens", "front")),
        out_size=(out_size, out_size),
        fov_deg=float(meta.get("fov_deg", 140.0)),
    )
    pin = None
    for idx, bgr in iter_frames(video):
        if idx == best.frame:
            pin = rect.apply(bgr)
            break
        if idx > best.frame:
            break
    if pin is None:
        return None
    K = rect.K_pinhole
    D = np.zeros((1, 5), dtype=np.float64)
    T = best.T_cam_marker
    axis_len = 1.5 * float(meta.get("marker", {}).get("size", 0.05))
    cv2.drawFrameAxes(pin, K, D, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], axis_len, 3)
    for c in best.corners.astype(int):
        cv2.circle(pin, tuple(c), 6, (0, 255, 0), -1)
    cv2.putText(
        pin,
        f"frame {best.frame}  reproj={best.reproj:.2f}px",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), pin)
    return out

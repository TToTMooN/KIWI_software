"""Resolve which camera-registry unit a recording came from, by serial.

The camera registry (``config/cameras/<unit>/``) keys each physical camera by
the ``serial:`` field in its ``camera.yaml``; a recording carries the same
serial in its ``.insv`` telemetry. Matching the two pairs a clip with its
calibration.

Artifacts: ``camera.yaml`` names the active file of each calibration artifact
with a pointer key (``intrinsics: intrinsics.npz``).
Resolution lives in :meth:`CameraUnit.artifact` - the one place
"unit dir -> named artifact" is spelled out - with a fallback to the
conventional unversioned filename (``intrinsics.npz``) for a hand-assembled
unit dir that carries no pointers. Publication defaults (the dated
filename, and the refuse-to-clobber-published guard) live in
:func:`default_artifact_out`.

Mode-keyed pointers: a unit that records in several video modes carries one
calibration PER MODE, because the modes are different sensor crops with
different optical mappings (the GO3's square vs 16:9 is not a rescale). Such
a unit's pointer value is a mapping keyed by ``"<W>x<H>"``::

    intrinsics:
      2688x2688: intrinsics-2688x2688.npz
      2720x1536: intrinsics-2720x1536.npz

and resolves ONLY through :meth:`CameraUnit.artifact_for_size` with the
recording's frame size - a plain :meth:`CameraUnit.artifact` call refuses
the mapping rather than guessing a mode. Single-mode units (the X5s) keep
the scalar pointer, for which both methods agree.

Callers pass the registry directory explicitly (``<repo>/config/cameras``);
this module has no repo-layout knowledge of its own.
"""

from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path
from typing import NamedTuple

import yaml

__all__ = [
    "ARTIFACT_NAMES",
    "ARTIFACT_SCHEMAS",
    "CameraMatch",
    "CameraUnit",
    "default_artifact_out",
    "find_camera_by_serial",
    "read_insv_serial",
    "resolve_camera_for_video",
]

# Artifact key -> conventional unversioned filename (the pointer fallback).
# One entry per artifact the registry contract defines. A new artifact kind
# registers its key HERE and resolves through CameraUnit - never by
# hardcoding config/cameras/<unit>/ filenames in the package that reads it.
ARTIFACT_NAMES: dict[str, str] = {
    "intrinsics": "intrinsics.npz",  # DoubleSphere per lens; GO3: pinhole-equi
    "extrinsic_imu": "extrinsic-imu.yaml",  # Kalibr cam-IMU chain (front)
    "extrinsic_imu_back": "extrinsic-imu-back.yaml",  # rear-lens chain
    "av_sync": "av-sync.yaml",  # per-recording-mode A/V delay (kiwi.sync)
    "imu_noise": "imu-noise.yaml",  # Allan noise densities (kiwi.calib)
    "intrinsics_kb4": "intrinsics-kb4.json",  # KB4 fit of the DS model (kiwi.track)
    "intrinsics_kb4_back": "intrinsics-kb4-back.json",  # rear-lens KB4 fit
}

# Artifact key -> the schema tag its YAML/JSON body must carry, for the
# artifacts that carry one (npz bundles and Kalibr camchains are implicit
# formats and have no entry). This is the ONE place a schema string is
# spelled: the component that writes the artifact and the component that
# reads it both take it from here.
ARTIFACT_SCHEMAS: dict[str, str] = {
    "av_sync": "av_sync_v1",
    "imu_noise": "imu_noise_v1",
    # One canonical KB4 document for both lenses; it records lens + unit
    # and the reader (kiwi.core.camera.kb4.read_kb4) gates on the expected
    # lens, so a fit attached to the wrong pointer fails loudly.
    "intrinsics_kb4": "kb4_intrinsics",
    "intrinsics_kb4_back": "kb4_intrinsics",
}


class CameraMatch(NamedTuple):
    """A recording matched to a camera-registry unit."""

    unit: str  # registry dir name, e.g. "insta360-X5-001"
    serial: str  # serial probed from the recording
    intrinsics: (
        Path | None
    )  # active intrinsics file (dated when pointed), None if absent


@dataclasses.dataclass(frozen=True)
class CameraUnit:
    """One registry unit: its directory plus the parsed ``camera.yaml``."""

    path: Path  # config/cameras/<unit>/
    meta: dict  # parsed camera.yaml ({} when the file is absent/empty)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def serial(self) -> str:
        return str(self.meta.get("serial") or self.meta.get("serial_number") or "")

    @classmethod
    def load(cls, unit_dir: Path) -> "CameraUnit":
        cam_yaml = Path(unit_dir) / "camera.yaml"
        # A broken camera.yaml is a ValueError naming the file - the type
        # every caller turns into a refusal. The parser names no file itself.
        try:
            meta = yaml.safe_load(cam_yaml.read_text()) if cam_yaml.is_file() else None
        except yaml.YAMLError as e:
            raise ValueError(f"{cam_yaml}: not valid YAML: {e}") from e
        if meta is not None and not isinstance(meta, dict):
            raise ValueError(
                f"{cam_yaml}: must be a YAML mapping, got {type(meta).__name__}"
            )
        return cls(path=Path(unit_dir), meta=meta or {})

    def artifact(self, key: str) -> Path | None:
        """The active file for ``key`` (see :data:`ARTIFACT_NAMES`), or None.

        A ``camera.yaml`` pointer wins, and a pointer naming a missing file
        is a broken registry - that raises ``FileNotFoundError`` rather than
        silently falling back. Without a pointer, the conventional
        unversioned filename is used when present. An unknown ``key`` is a
        programming error and raises ``KeyError``. A mode-keyed pointer
        mapping raises ``ValueError``: no recording size, no way to pick a
        mode - resolve through :meth:`artifact_for_size` instead.
        """
        fallback_name = ARTIFACT_NAMES[key]
        pointer = self.meta.get(key)
        if isinstance(pointer, dict):
            raise ValueError(
                f"{self.path / 'camera.yaml'}: {key} is mode-keyed "
                f"({', '.join(sorted(map(str, pointer)))}) - resolve with "
                "artifact_for_size(key, (W, H)) using the recording's frame "
                "size; a plain artifact() call cannot pick a mode"
            )
        if pointer:
            p = self.path / str(pointer)
            if not p.is_file():
                raise FileNotFoundError(
                    f"{self.path / 'camera.yaml'}: {key}: {str(pointer)!r} does not exist"
                )
            return p
        fallback = self.path / fallback_name
        return fallback if fallback.is_file() else None

    def artifact_for_size(self, key: str, size: tuple[int, int]) -> Path | None:
        """The active file for ``key`` matching a recording's frame size.

        A mode-keyed mapping resolves by the ``"<W>x<H>"`` key; a size with
        no entry raises ``ValueError`` naming the modes that exist. A scalar
        pointer resolves exactly as :meth:`artifact` - single-mode units
        keep their consumer-side ``image_size`` guards as the size check.
        """
        pointer = self.meta.get(key)
        if not isinstance(pointer, dict):
            return self.artifact(key)
        mode = f"{int(size[0])}x{int(size[1])}"
        entry = pointer.get(mode)
        if entry is None:
            raise ValueError(
                f"{self.path / 'camera.yaml'}: {key} has no {mode!r} entry "
                f"(has: {', '.join(sorted(map(str, pointer)))}) - the recording's "
                "video mode was never calibrated; run `kiwi calib intrinsics` "
                "on a recording in this mode first"
            )
        p = self.path / str(entry)
        if not p.is_file():
            raise FileNotFoundError(
                f"{self.path / 'camera.yaml'}: {key}[{mode}]: {str(entry)!r} "
                "does not exist"
            )
        return p


def default_artifact_out(unit_dir: Path, key: str, mode: str | None = None) -> Path:
    """The dated default path a calib verb publishes ``key`` to, dir ensured.

    ``<stem>-<today><ext>`` from the artifact's conventional filename
    (:data:`ARTIFACT_NAMES`), e.g. ``imu-noise-2026-08-28.yaml``; ``mode``
    (a ``"<W>x<H>"`` string, for units whose pointer is mode-keyed) lands
    between them - ``intrinsics-2688x2688-2026-09-12.npz`` - so two modes
    published the same day cannot collide. Guard: when ``camera.yaml``'s
    pointer (or any entry of a mode-keyed mapping) already names exactly
    this file, the default would rewrite the PUBLISHED artifact in place,
    so this raises :class:`FileExistsError` instead. Overwriting today's
    un-pointed candidate is allowed.
    """
    conventional = Path(ARTIFACT_NAMES[key])
    unit_dir = Path(unit_dir)
    unit_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{conventional.stem}-{mode}" if mode else conventional.stem
    out = unit_dir / f"{stem}-{datetime.date.today()}{conventional.suffix}"
    pointer = CameraUnit.load(unit_dir).meta.get(key)
    published = (
        {unit_dir / str(v) for v in pointer.values()}
        if isinstance(pointer, dict)
        else {unit_dir / str(pointer)}
        if pointer
        else set()
    )
    if out in published:
        raise FileExistsError(
            f"{unit_dir / 'camera.yaml'}: {key}: the default output {out.name!r} "
            "is the published artifact (the pointer names it) - refusing to "
            "rewrite it in place; pass an explicit output path, or move the "
            "pointer only after the day's measuring is done"
        )
    return out


def read_insv_serial(insv: Path) -> str | None:
    """Read the camera serial from an ``.insv``'s telemetry metadata.

    Returns ``None`` when the file has no readable telemetry (non-.insv
    input, truncated file) - callers treat that as "cannot auto-resolve",
    not an error.
    """
    import telemetry_parser  # deferred: keep this module importable without it

    try:
        tel = telemetry_parser.Parser(str(insv)).telemetry()
        serial = tel[0]["Default"]["Metadata"].get("serial_number")
    except Exception:
        return None
    return str(serial) if serial else None


def find_camera_by_serial(cameras_dir: Path, serial: str) -> Path | None:
    """Find the registry unit dir whose ``camera.yaml`` matches ``serial``.

    Substring match in either direction, since one side may carry a prefix
    the other omits; both sides must be non-empty.
    """
    if not serial:
        return None
    for cam_yaml in sorted(cameras_dir.glob("*/camera.yaml")):
        s = CameraUnit.load(cam_yaml.parent).serial
        if s and (s in serial or serial in s):
            return cam_yaml.parent
    return None


def resolve_camera_for_video(video: Path, cameras_dir: Path) -> CameraMatch | None:
    """Match a recording to its camera-registry unit via the probed serial.

    Only ``.insv`` files carry a readable serial (a demuxed per-lens
    ``.mp4`` does not); returns ``None`` for other inputs, unknown serials,
    or an unmatched registry.
    """
    if video.suffix.lower() != ".insv":
        return None
    serial = read_insv_serial(video)
    if serial is None:
        return None
    unit_dir = find_camera_by_serial(cameras_dir, serial)
    if unit_dir is None:
        return None
    unit = CameraUnit.load(unit_dir)
    try:
        intrinsics = unit.artifact("intrinsics")
    except ValueError:
        # Mode-keyed unit: there is no single active file to report, and
        # this serial-level match has no frame size to pick one with.
        intrinsics = None
    return CameraMatch(unit.name, serial, intrinsics)

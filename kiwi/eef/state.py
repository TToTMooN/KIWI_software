"""The ``eef_state_v1`` artifact: per-clip tool state as NaN-parallel arrays.

This is the ONE place the schema is spelled. The eef stage writes it as
``eef.npz`` in the arm's stream directory (:mod:`kiwi.core.layout`; a
standalone `kiwi eef run` on a bare clip writes ``<stem>_eef.npz`` beside
it instead); export and visualization read it back through :func:`read_eef_state`.
The representation is the
``imu_raw_v1`` pattern: a compressed npz with the schema string inside, and
the in-memory contract IS the on-disk contract. ``value[i]`` is NaN when
frame ``i`` has no surviving raw estimate after the estimator's post-passes;
validity is ``isfinite(value)``.

The generic keys (everything outside the estimator namespace) are
tool-agnostic: a consumer that reads ``value`` / ``output`` /
``unit`` and normalizes with the tool config's closed/open endpoints never
branches on which estimator produced the file. Everything tool-specific
(fit modes, per-side tilts, rescue audits) lives under ``<estimator>/...``
keys that generic consumers never read.

Every artifact contains ``value_completed`` and generic ``completion/...``
audit arrays; ``output`` is this complete series, and a file without it is
refused by :func:`read_eef_state`. Completion interpolates every interior
missing run and holds the nearest raw anchor at the edges. It never changes a
finite raw value. A clip with no finite raw anchor cannot produce completed
data. Gap classes are duration labels, not calibrated confidence. The raw
channel may already contain a one-jaw fallback estimate or a long-gap rescue,
identified in the estimator namespace.

Index IS the frame: every array has one entry per source video frame, so
there is no frame column and nothing to keep contiguous. Frame ``i`` sits at
``t = i / fps`` on the clip's own clock; mapping onto the episode global
clock (adding the sync offset) is the consumer's job.

``pose_mode`` and ``pose_source`` record which plane-pose mode ran. Whole-series
calibration results such as the one-jaw symmetry-axis delta live in the
estimator namespace.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

__all__ = [
    "EEF_STATE_SCHEMA",
    "EEF_STATE_SUFFIX",
    "OVERLAY_SUFFIX",
    "POSE_MODES",
    "TRACE_SUFFIX",
    "EefState",
    "read_eef_state",
    "write_eef_state",
]

EEF_STATE_SCHEMA = "eef_state_v1"

# The standalone convention: `kiwi eef run` on a bare clip outside any
# episode writes ``<stem><suffix>`` beside it. Inside an episode the names
# are the fixed ones in ``kiwi.core.layout``. The overlay mp4 is a human
# debug artifact and is never listed as a stage artifact.
EEF_STATE_SUFFIX = "_eef.npz"
TRACE_SUFFIX = "_eef.png"
OVERLAY_SUFFIX = "_eef.mp4"

POSE_MODES = ("fixed_pose", "marker_pnp")

_SCALAR_STR = ("tool", "estimator", "unit", "source", "pose_mode", "pose_source")
_GENERIC_ARRAYS = (
    "value",
    "rejected_value",
    "has_pose",
    "n_detected",
    "n_kept",
    "error",
)


@dataclasses.dataclass(frozen=True)
class EefState:
    """One clip's tool-state series, exactly as stored on disk.

    Per-frame arrays are all length ``n_frames`` and indexed by frame:

    - ``value`` (f8): the scalar tool state in ``unit``; NaN = dropout.
    - ``value_completed`` (f8): the consumer output, finite on every frame.
    - ``rejected_value`` (f8): the fit a gate refused, NaN = none. Diagnosis
      only, never for use.
    - ``has_pose`` (bool): a plane pose existed. Unconditionally True in
      fixed-pose mode, so check ``pose_mode`` before reading meaning into it.
    - ``n_detected`` / ``n_kept`` (i4): dot candidates before / after the
      spatial filters.
    - ``error`` (str): the gate/fit failure reason, ``""`` = none. Never
      derive validity from it - a post-pass can revoke a value on a frame
      whose error is empty, and set one where a rejected fit still carries
      diagnostics.

    ``detail`` holds the estimator's own arrays (stored namespaced as
    ``<estimator>/<key>``), plus 0-d entries for whole-series results.
    Generic consumers must never need it. ``completion`` is a separate,
    tool-independent namespace containing per-frame fill methods, durations,
    anchor indices and distances, plus scalar policy and summary values.
    """

    tool: str
    estimator: str
    unit: str
    fps: float
    source: str
    pose_mode: str
    pose_source: str
    value: np.ndarray
    rejected_value: np.ndarray
    has_pose: np.ndarray
    n_detected: np.ndarray
    n_kept: np.ndarray
    error: np.ndarray
    value_completed: np.ndarray
    completion: dict[str, np.ndarray]
    detail: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pose_mode not in POSE_MODES:
            raise ValueError(
                f"pose_mode must be one of {POSE_MODES}, got {self.pose_mode!r}"
            )
        if not self.fps > 0:
            raise ValueError(f"fps must be positive, got {self.fps!r}")
        n = int(np.asarray(self.value).shape[0]) if np.ndim(self.value) else -1
        for name in _GENERIC_ARRAYS:
            arr = np.asarray(getattr(self, name))
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(
                    f"{name} must be a length-{n} 1-d array (one entry per "
                    f"source frame), got shape {arr.shape}"
                )
        for key, arr in self.detail.items():
            a = np.asarray(arr)
            if a.ndim == 1 and a.shape[0] != n:
                raise ValueError(
                    f"detail[{key!r}] is per-frame shaped but length "
                    f"{a.shape[0]} != {n}"
                )
            if a.ndim > 1:
                raise ValueError(
                    f"detail[{key!r}] must be 0-d or 1-d, got shape {a.shape}"
                )
        completed = np.asarray(self.value_completed)
        if (
            completed.dtype.kind not in "fiu"
            or completed.shape != (n,)
            or n == 0
            or not np.isfinite(completed).all()
        ):
            raise ValueError("value_completed must contain one finite value per frame")
        raw = np.asarray(self.value)
        if (
            raw.dtype.kind not in "fiu"
            or np.isinf(raw).any()
            or not np.isfinite(raw).any()
        ):
            raise ValueError(
                "value_completed requires at least one finite raw anchor and no infinities"
            )
        anchors = np.isfinite(raw)
        if not np.array_equal(completed[anchors], raw[anchors]):
            raise ValueError("value_completed must preserve every finite raw value")
        for key, value in self.completion.items():
            a = np.asarray(value)
            if a.dtype.hasobject or a.ndim > 1 or (a.ndim == 1 and a.shape != (n,)):
                raise ValueError(
                    f"completion[{key!r}] must be a scalar or length-{n} non-object array"
                )
        filled = np.asarray(self.completion.get("filled"))
        if filled.dtype.kind != "b" or not np.array_equal(filled, ~anchors):
            raise ValueError(
                "completion.filled must exactly identify missing raw frames"
            )

    def __len__(self) -> int:
        return int(self.value.size)

    @property
    def valid(self) -> np.ndarray:
        """Frames whose raw value survived every gate and post-pass."""
        return np.isfinite(self.value)

    @property
    def output(self) -> np.ndarray:
        """The consumer series: completed raw estimates, finite on every frame."""
        return self.value_completed

    @property
    def valid_output(self) -> np.ndarray:
        return np.isfinite(self.output)


def write_eef_state(path: Path, state: EefState) -> Path:
    """Write ``state`` to ``path`` atomically (tmp + rename).

    The estimator's ``detail`` arrays are stored under ``<estimator>/<key>``
    so the generic namespace can never collide with a tool's.
    """
    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(EEF_STATE_SCHEMA),
        "fps": np.asarray(float(state.fps)),
    }
    for name in _SCALAR_STR:
        payload[name] = np.asarray(str(getattr(state, name)))
    payload["value"] = np.asarray(state.value, dtype=np.float64)
    payload["rejected_value"] = np.asarray(state.rejected_value, dtype=np.float64)
    payload["has_pose"] = np.asarray(state.has_pose, dtype=bool)
    payload["n_detected"] = np.asarray(state.n_detected, dtype=np.int32)
    payload["n_kept"] = np.asarray(state.n_kept, dtype=np.int32)
    payload["error"] = np.asarray(state.error, dtype=str)
    payload["value_completed"] = np.asarray(state.value_completed, dtype=np.float64)
    for key, arr in state.completion.items():
        payload[f"completion/{key}"] = np.asarray(arr)
    for key, arr in state.detail.items():
        payload[f"{state.estimator}/{key}"] = np.asarray(arr)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    try:
        with open(tmp, "wb") as fp:
            np.savez_compressed(fp, **payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_eef_state(path: Path) -> EefState:
    """Read an ``eef_state_v1`` npz back into :class:`EefState`.

    Fails loudly on a missing file, a foreign schema, or a file without the
    completed output (``value_completed``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no eef state at {path}. The eef stage writes it into the "
            f"arm's stream directory; run `kiwi run` (or `kiwi eef run`) first."
        )
    with np.load(path) as data:
        schema = str(data["schema"][()]) if "schema" in data else None
        if schema != EEF_STATE_SCHEMA:
            raise ValueError(
                f"{path}: schema is {schema!r}, expected {EEF_STATE_SCHEMA!r}"
            )
        if "value_completed" not in data:
            raise ValueError(
                f"{path}: no value_completed channel, so the artifact has no "
                f"consumer output; re-run the eef stage to regenerate it."
            )
        estimator = str(data["estimator"][()])
        prefix = estimator + "/"
        detail = {
            key[len(prefix) :]: data[key]
            for key in data.files
            if key.startswith(prefix)
        }
        return EefState(
            tool=str(data["tool"][()]),
            estimator=estimator,
            unit=str(data["unit"][()]),
            fps=float(data["fps"][()]),
            source=str(data["source"][()]),
            pose_mode=str(data["pose_mode"][()]),
            pose_source=str(data["pose_source"][()]),
            value=data["value"],
            rejected_value=data["rejected_value"],
            has_pose=data["has_pose"],
            n_detected=data["n_detected"],
            n_kept=data["n_kept"],
            error=data["error"],
            detail=detail,
            value_completed=data["value_completed"],
            completion={
                key.removeprefix("completion/"): data[key]
                for key in data.files
                if key.startswith("completion/")
            },
        )

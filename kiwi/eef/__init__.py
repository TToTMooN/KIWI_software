"""End-effector (tool) state estimation, per tool (component: imports ``kiwi.core`` only).

The ``design.estimator`` field in a tool config selects a registered bundle
(``config/gripper/<tool>/tool.yaml``, schema ``tool_v1``). The current bundle is
**chopsticks** (:mod:`kiwi.eef.chopsticks`): marker-plane homography -> dot
detection -> per-jaw line fit (with outlier trim) -> hinge angle, then the
drop-dot retry, the one-jaw fallback and the long-gap rescue. The batch runner
completes the surviving raw series for downstream consumers. The tuning GUI
lives in ``kiwi.viz`` and calls the same estimator.

Contract notes for implementers and consumers:

* **The registry bundle defines the tool-specific operations.** :class:`ToolEstimator`
  carries everything tool-specific: the per-frame estimator factory (its
  product satisfies :class:`kiwi.core.pipeline.GripperEstimator` and exposes
  ``params`` / ``pose_mode`` / ``pose_source`` / ``warnings``), the optional
  marker detector, the whole-series post-passes, the value ``unit``, and the
  ``closed_open`` normalization endpoints. Everything else - the batch
  runner, the stage entry, export - is tool-blind and never
  mentions angles. A new estimator also registers its session-key validator
  in :mod:`kiwi.eef.config`'s ``_SESSION_VALIDATORS`` table and supplies a
  matching tool config; adding a bundle alone is insufficient.
* **Plane pose comes from one of two mutually exclusive modes** picked once
  per run, never mixed and never falling back mid-run: calibrated
  ``fixed_pose`` from the gripper unit's ``build`` block (``has_pose`` is then
  True by construction, not a detection-quality signal) or per-frame marker PnP
  (``--live-pnp``, or when no ``fixed_pose`` resolves). Which mode ran is
  recorded in the artifact (``pose_mode`` / ``pose_source``).
* **The artifact is ``eef_state_v1``** (:mod:`kiwi.eef.state`): a per-clip
  npz of parallel arrays, one entry per source frame. Raw validity is
  ``isfinite(value)``. ``value_completed`` is the consumer output: every
  finite raw anchor unchanged, linear interior fills and nearest edge holds.
  Generic ``completion/`` arrays record methods, durations and anchor indices.
  At least one raw anchor is required.
* **Config resolves in layers**: the ``session:`` recipe merges tool YAML <
  gripper unit < episode arm (by the episode manifest, each layer validated
  against the estimator's key set via
  :func:`kiwi.eef.config.validate_session_block`); CLI flags arrive as
  overrides, through :mod:`kiwi.eef.config`'s single ``resolve_params``.
  Build facts (the tag id/size, the calibrated ``fixed_pose``) come from the
  gripper unit alone - there is no version-wide pose.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

__all__ = [
    "ESTIMATOR_KEYS",
    "EstimatorContext",
    "ToolEstimator",
    "get_estimator",
]


@dataclasses.dataclass(frozen=True)
class EstimatorContext:
    """Everything a tool bundle's factory may need, resolved by the caller.

    The caller (the stage entry via ``ArmSpec``, or a CLI verb via the
    tool/unit YAMLs) owns the layering; the factory only consumes it.
    ``fixed_pose_source`` names the YAML the pose came from, for the
    artifact's ``pose_source`` field.
    """

    session_block: Mapping[str, object]
    K: np.ndarray
    image_wh: tuple[int, int]
    marker_id: int | None = None
    marker_size_m: float | None = None
    marker_dictionary: str | None = None
    fixed_pose: Mapping[str, object] | None = None
    fixed_pose_source: str = ""
    live_pnp: bool = False
    overrides: Mapping[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ToolEstimator:
    """One tool's estimator, postprocessing, and display operations.

    ``make_frame_estimator(ctx)`` returns the per-frame callable (satisfying
    the core ``GripperEstimator`` Protocol, and exposing ``params``,
    ``pose_mode``, ``pose_source``, ``warnings``). ``make_detector(ctx)``
    returns the per-frame marker detector the runner should feed it, or
    ``None`` when the mode needs none. ``postprocess(states, params, fps)``
    runs the whole-series passes and returns ``(detail_arrays, notes)``.
    ``closed_open(session_block)`` returns the normalization endpoints in
    ``unit``. ``draw_overlay(canvas, state)`` (optional) annotates a
    rectified BGR frame in place for the opt-in debug overlay video -
    drawing is tool knowledge too.
    """

    name: str
    unit: str
    make_frame_estimator: Callable[[EstimatorContext], Any]
    make_detector: Callable[[EstimatorContext], Any]
    postprocess: Callable[..., tuple[dict, list[str]]]
    closed_open: Callable[[Mapping], tuple[float, float]]
    draw_overlay: Callable[..., None] | None = None


def _load_chopsticks() -> ToolEstimator:
    from kiwi.eef import chopsticks

    return ToolEstimator(
        name="chopsticks",
        unit="deg",
        make_frame_estimator=chopsticks.make_frame_estimator,
        make_detector=chopsticks.make_detector,
        postprocess=chopsticks.postprocess,
        closed_open=chopsticks.closed_open,
        draw_overlay=chopsticks.draw_overlay,
    )


# Loaded lazily (no cv2 until a bundle is actually used). New estimators also
# register their session-key validator in kiwi.eef.config and provide a
# matching tool config.
_ESTIMATORS: dict[str, Callable[[], ToolEstimator]] = {
    "chopsticks": _load_chopsticks,
}

ESTIMATOR_KEYS = tuple(sorted(_ESTIMATORS))


def get_estimator(key: str, *, source: str = "") -> ToolEstimator:
    """The bundle registered under ``key`` (a tool YAML's ``estimator:``)."""
    try:
        loader = _ESTIMATORS[key]
    except KeyError:
        where = f" (from {source})" if source else ""
        raise ValueError(
            f"unknown estimator {key!r}{where}; registered estimators are "
            f"{list(ESTIMATOR_KEYS)}. A tool_v1 config names its estimator "
            f"with the `design.estimator` key."
        ) from None
    return loader()

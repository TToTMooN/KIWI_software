"""The ``session:`` detection/fit recipe: YAML layer + resolved parameters.

Two shapes live here, one per stage of a knob's life:

* :class:`GripperDotConfig` - the raw YAML layer. Every field is optional:
  a config may legitimately carry only a partial override set (e.g. a gripper
  unit's ``session:`` block layered onto a tool's), and ``None`` means "no
  opinion at this layer".
* :class:`DotParams` - the fully resolved parameter set the estimator runs
  with. Built by :func:`resolve_params`, which folds
  **override > config > builtin** for every knob - the top of the session
  ladder (`config/README.md`). The episode manifest merges
  tool < unit < episode-arm into one config block. Standalone commands merge
  tool < unit and supply CLI ``--set`` / flag overrides. The batch estimator
  and tuner share this resolution.

This module also owns per-estimator SESSION-KEY VALIDATION
(:func:`validate_session_block`): the tool_v1 schema's ``session:`` block is
strict - an unknown key raises, at the layer that carries it - and which keys
exist is estimator knowledge, so the registry lives here, importable from the
manifest parser.

The module does not import cv2.
"""

from __future__ import annotations

import dataclasses
import difflib
import math
from collections.abc import Mapping
from numbers import Real

__all__ = [
    "DEFAULT_THRESHOLD_BY_POLARITY",
    "DotParams",
    "GripperDotConfig",
    "config_from_block",
    "resolve_params",
    "validate_session_block",
]

_VALID_POLARITIES = ("dark", "light", "red")
_VALID_ROI_KEEP = ("+y", "-y", "+x", "-x")

# Per-polarity intensity-threshold builtin when neither an override nor the
# YAML supplies one, shared by the batch estimator and tuner.
DEFAULT_THRESHOLD_BY_POLARITY = {"dark": 80, "light": 175, "red": 150}


def _validate_gap_thresholds(short, medium) -> None:
    """Check gap thresholds without imposing types on unrelated session knobs."""
    for name, value in (("gap_short_max_s", short), ("gap_medium_max_s", medium)):
        finite = False
        if isinstance(value, Real) and not isinstance(value, bool):
            try:
                finite = math.isfinite(value)
            except (OverflowError, ValueError):
                pass
        if not finite:
            raise ValueError(f"gripper.{name} must be a finite number, got {value!r}")
        if value < 0:
            raise ValueError(f"gripper.{name} must be >= 0, got {value!r}")
    if short > medium:
        raise ValueError("gripper.gap_short_max_s must be <= gap_medium_max_s")


@dataclasses.dataclass(frozen=True)
class GripperDotConfig:
    # Every field is optional: this is purely a defaults layer. Callers
    # resolve None through resolve_params' builtin defaults.
    polarity: str | None = None
    min_area: int | None = None
    max_area: int | None = None
    threshold: int | None = None
    min_circularity: float | None = None
    # Rectifier / plane-warp defaults.
    fov_deg: float | None = None
    extent_x: float | None = None
    extent_y: float | None = None
    # Dot-search ROI (marker-anchored plane mm, +X right, +Y up): a SEMICIRCLE
    # of roi_radius_mm about (roi_center_x_mm, roi_center_y_mm), keeping the
    # half-disk on the roi_semicircle_keep side (+y/-y/+x/-x).
    # roi_inner_radius_mm turns the half-disk into a half-ANNULUS (used to
    # exclude the operator's-hand blob cluster inside the innermost real dot).
    # None = no shape filter beyond the in-marker square. clip_to_image drops
    # dots whose image pixel is off-frame.
    roi_radius_mm: float | None = None
    roi_inner_radius_mm: float | None = None
    roi_center_x_mm: float | None = None
    roi_center_y_mm: float | None = None
    roi_semicircle_keep: str | None = None
    clip_to_image: bool | None = None
    # Reject a frame's hinge angle when the two side fits disagree by more
    # than this (|left_angle + right_angle|, deg).
    max_asymmetry_deg: float | None = None
    # The estimator's fallback steps, in the order they apply:
    # fit (with outlier trim) -> drop-dot retry -> one-jaw fallback ->
    # long-gap rescue -> completion.
    #
    # Outlier trim: drop stray dots from each jaw's line via leave-one-out
    # deleted residuals before the PCA fit.
    enable_outlier_trim: bool | None = None
    outlier_trim_mm: float | None = None  # absolute outlier floor (plane mm)
    outlier_trim_mad_k: float | None = None  # MAD multiplier, adaptive floor
    # Drop-dot retry: when a both-side frame fails max_asymmetry_deg, refit
    # once without the single dot whose removal minimises asymmetry. Off by
    # default.
    enable_drop_dot_retry: bool | None = None
    drop_dot_min_side_dots: int | None = None  # >= 3, so a drop leaves two
    # One-jaw fallback: when one jaw is occluded, estimate the open angle
    # from the visible jaw x2. symmetry_axis_offset_deg (delta) is the
    # marker->symmetry-axis offset; None = auto-calibrate from the clip.
    enable_one_jaw_fallback: bool | None = None
    symmetry_axis_offset_deg: float | None = None
    # Reject a one-jaw estimate above this (deg).
    one_jaw_max_open_deg: float | None = None
    # Long-gap rescue: recover frames in rejected runs longer than
    # long_gap_min_s from the jaw whose tilt drifted least from the nearest
    # accepted frame, via the one-jaw x2 formula. long_gap_min_s is a
    # DURATION (seconds), converted to a frame count against each clip's own
    # fps. Shorter runs are never touched. Off by default.
    enable_long_gap_rescue: bool | None = None
    long_gap_min_s: float | None = None
    long_gap_max_jaw_drift_deg: float | None = None  # jaw-trust radius
    # Completion: duration boundaries for the short / medium / long gap
    # classes.
    gap_short_max_s: float | None = None
    gap_medium_max_s: float | None = None
    # Normalization endpoints: how the export maps the tool value onto the
    # [0,1] tool_state channel (closed -> 0, open -> 1). Session keys like
    # every other knob (tool default < unit measured override < episode-arm),
    # but consumed by the EXPORT side - the bundle's ``closed_open`` - not by
    # the estimator: ``resolve_params`` ignores them, and they have no CLI
    # override.
    angle_closed_deg: float | None = None
    angle_open_deg: float | None = None

    def __post_init__(self) -> None:
        if self.polarity is not None and self.polarity not in _VALID_POLARITIES:
            raise ValueError(
                f"gripper.polarity must be one of {_VALID_POLARITIES}, "
                f"got {self.polarity!r}"
            )
        # Validate the ROI keep-side at load time too.
        if (
            self.roi_semicircle_keep is not None
            and self.roi_semicircle_keep not in _VALID_ROI_KEEP
        ):
            raise ValueError(
                f"gripper.roi_semicircle_keep must be one of {_VALID_ROI_KEEP}, "
                f"got {self.roi_semicircle_keep!r}"
            )


def config_from_block(
    block: Mapping[str, object], *, source: str = ""
) -> GripperDotConfig | None:
    """Build a config from a raw ``session:`` mapping, ``None`` if empty.

    An unknown key raises, with a did-you-mean hint when a close match
    exists.

    Entry point both for a block that is already layered - the episode
    manifest's ``ArmSpec.session`` merges tool < unit < episode-arm before eef
    ever sees it - and for one layer alone (``source`` attributes the error
    to the file that carries the typo).
    """
    if not block:
        return None
    where = f"{source}: " if source else ""
    known = {f.name for f in dataclasses.fields(GripperDotConfig)}
    unknown = sorted(set(map(str, block)) - known)
    if unknown:
        hints = []
        for k in unknown:
            close = difflib.get_close_matches(k, known, n=1)
            hints.append(k + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(
            f"{where}unknown session key(s): {', '.join(hints)}. The "
            f"chopsticks session keys are {sorted(known)}."
        )
    return GripperDotConfig(**{k: v for k, v in block.items() if k in known})


# Per-estimator session-key validation. The ``session:`` block's key set is
# estimator knowledge, so a new tool family registers its validator here
# alongside its bundle in ``kiwi.eef``. This table is cv2-free and the
# manifest parser calls it.
_SESSION_VALIDATORS = {
    "chopsticks": config_from_block,
}


def validate_session_block(
    estimator: str, block: Mapping[str, object], source: str
) -> None:
    """Validate ONE layer's ``session:`` block against an estimator's keys.

    Called per layer (tool file, unit file, episode-arm block) so the error
    names the file that carries the bad key, not the merged result. Raises
    for an unknown key or an unregistered estimator.
    """
    try:
        validator = _SESSION_VALIDATORS[estimator]
    except KeyError:
        raise ValueError(
            f"{source}: no session-key validator for estimator {estimator!r}; "
            f"registered estimators are {sorted(_SESSION_VALIDATORS)}"
        ) from None
    validator(block, source=source)


@dataclasses.dataclass(frozen=True)
class DotParams:
    """The fully resolved detection / fit parameter set the estimator runs on.

    Unlike :class:`GripperDotConfig`, every field is concrete - the only
    ``None`` values left are knobs whose absence means something (no shape
    ROI, no inner annulus cut, auto-calibrated symmetry axis). Build through
    :func:`resolve_params`, never by hand.
    """

    polarity: str
    threshold: int
    min_area: int
    max_area: int
    min_circularity: float
    fov_deg: float
    extent_x: float
    extent_y: float
    roi_radius_mm: float | None
    roi_inner_radius_mm: float | None
    roi_center_x_mm: float
    roi_center_y_mm: float
    roi_semicircle_keep: str
    clip_to_image: bool
    max_asymmetry_deg: float
    enable_outlier_trim: bool
    outlier_trim_mm: float
    outlier_trim_mad_k: float
    enable_drop_dot_retry: bool
    drop_dot_min_side_dots: int
    enable_one_jaw_fallback: bool
    symmetry_axis_offset_deg: float | None
    one_jaw_max_open_deg: float
    enable_long_gap_rescue: bool
    long_gap_min_s: float
    long_gap_max_jaw_drift_deg: float
    gap_short_max_s: float = 0.167
    gap_medium_max_s: float = 1.0

    def __post_init__(self) -> None:
        _validate_gap_thresholds(self.gap_short_max_s, self.gap_medium_max_s)
        # Normalize after validation so runtime arrays stay numeric without
        # admitting coercible strings or booleans into these fields.
        object.__setattr__(self, "gap_short_max_s", float(self.gap_short_max_s))
        object.__setattr__(self, "gap_medium_max_s", float(self.gap_medium_max_s))


def resolve_params(
    cfg: GripperDotConfig | None = None,
    overrides: Mapping[str, object] | None = None,
) -> DotParams:
    """Fold ``overrides > cfg > builtin`` into a concrete :class:`DotParams`.

    ``overrides`` is keyed by :class:`DotParams` field names (a CLI flag left
    unset arrives as ``None`` and defers to the config); an unknown override
    key raises.

    The ``long_gap_min_s`` default is five frames at 29.97 fps; the actual
    frame count follows each clip's frame rate.
    """
    ov = dict(overrides or {})
    known = {f.name for f in dataclasses.fields(DotParams)}
    unknown = sorted(set(ov) - known)
    if unknown:
        raise ValueError(
            f"unknown parameter override(s) {unknown}; knobs are {sorted(known)}"
        )

    def pick(name: str, builtin):
        v = ov.get(name)
        if v is not None:
            return v
        c = getattr(cfg, name) if cfg is not None else None
        return c if c is not None else builtin

    polarity = str(pick("polarity", "dark"))
    if polarity not in DEFAULT_THRESHOLD_BY_POLARITY:
        raise ValueError(
            f"polarity must be one of {_VALID_POLARITIES}, got {polarity!r}"
        )
    keep = str(pick("roi_semicircle_keep", "-y"))
    if keep not in _VALID_ROI_KEEP:
        raise ValueError(
            f"roi_semicircle_keep must be one of {_VALID_ROI_KEEP}, got {keep!r}"
        )
    radius = pick("roi_radius_mm", None)
    inner = pick("roi_inner_radius_mm", None)
    if inner is not None:
        if radius is None:
            raise ValueError(
                "roi_inner_radius_mm needs roi_radius_mm: the annulus has no "
                "outer edge without it"
            )
        if float(inner) >= float(radius):
            raise ValueError(
                f"roi_inner_radius_mm ({inner}) must be smaller than "
                f"roi_radius_mm ({radius}); the annulus would be empty"
            )
    delta = pick("symmetry_axis_offset_deg", None)
    extent_x = float(pick("extent_x", 4.0))
    return DotParams(
        polarity=polarity,
        threshold=int(pick("threshold", DEFAULT_THRESHOLD_BY_POLARITY[polarity])),
        min_area=int(pick("min_area", 70)),
        max_area=int(pick("max_area", 700)),
        min_circularity=float(pick("min_circularity", 0.55)),
        fov_deg=float(pick("fov_deg", 110.0)),
        extent_x=extent_x,
        extent_y=float(pick("extent_y", extent_x)),
        roi_radius_mm=None if radius is None else float(radius),
        roi_inner_radius_mm=None if inner is None else float(inner),
        roi_center_x_mm=float(pick("roi_center_x_mm", 0.0)),
        roi_center_y_mm=float(pick("roi_center_y_mm", 0.0)),
        roi_semicircle_keep=keep,
        clip_to_image=bool(pick("clip_to_image", True)),
        max_asymmetry_deg=float(pick("max_asymmetry_deg", 30.0)),
        enable_outlier_trim=bool(pick("enable_outlier_trim", True)),
        outlier_trim_mm=float(pick("outlier_trim_mm", 3.0)),
        outlier_trim_mad_k=float(pick("outlier_trim_mad_k", 3.0)),
        enable_drop_dot_retry=bool(pick("enable_drop_dot_retry", False)),
        drop_dot_min_side_dots=int(pick("drop_dot_min_side_dots", 3)),
        enable_one_jaw_fallback=bool(pick("enable_one_jaw_fallback", True)),
        symmetry_axis_offset_deg=None if delta is None else float(delta),
        one_jaw_max_open_deg=float(pick("one_jaw_max_open_deg", 180.0)),
        enable_long_gap_rescue=bool(pick("enable_long_gap_rescue", False)),
        long_gap_min_s=float(pick("long_gap_min_s", 0.167)),
        long_gap_max_jaw_drift_deg=float(pick("long_gap_max_jaw_drift_deg", 3.0)),
        gap_short_max_s=pick("gap_short_max_s", 0.167),
        gap_medium_max_s=pick("gap_medium_max_s", 1.0),
    )

"""Tests for kiwi.eef.config - the session: recipe layer + resolved params.

Pure yaml/dataclass logic - no cv2, no video. The tool / gripper-unit
LOADERS are core's and tested in tests/core/test_tool_config.py; this file
covers the eef side: the STRICT session-key check, the per-estimator dispatch
the manifest runs on every layer, the merged tool < unit < arm ladder, and
the override > config > builtin resolution.

Run: pixi run test
"""

from __future__ import annotations

import textwrap
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import pytest

from kiwi.core.tool_config import load_gripper_unit, load_tool_config, merge_session
from kiwi.eef.config import (
    DEFAULT_THRESHOLD_BY_POLARITY,
    DotParams,
    GripperDotConfig,
    config_from_block,
    resolve_params,
    validate_session_block,
)

# The tool file's recipe defaults - the bottom YAML rung of the ladder.
TOOL_SESSION = {
    "polarity": "red",
    "min_area": 200,
    "max_area": 400,
    "roi_semicircle_keep": "-y",
}


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text))
    return p


# --- the session block: what a layer may say -------------------------------


def test_an_empty_layer_is_no_opinion_at_all():
    """A layer that states nothing resolves to None, not to an all-None config:
    the caller then falls through to the layer below it (and to the builtins),
    instead of pinning every knob to "unset"."""
    assert config_from_block({}) is None
    assert config_from_block(None) is None
    # ...and a ladder of nothing-saying layers merges to the same nothing.
    assert config_from_block(merge_session(None, {}, None)) is None


def test_config_from_block_reads_the_session_keys():
    cfg = config_from_block(TOOL_SESSION)
    assert cfg is not None
    assert cfg.polarity == "red"
    assert (cfg.min_area, cfg.max_area) == (200, 400)
    assert cfg.roi_semicircle_keep == "-y"
    assert cfg.threshold is None  # unset stays None for the builtin fallback


def test_the_normalization_endpoints_are_session_fields():
    """angle_closed_deg / angle_open_deg are session fields. They are
    EXPORT knowledge, so resolve_params ignores them - a detection knob
    they are not."""
    cfg = config_from_block(
        {"polarity": "red", "angle_closed_deg": 18.0, "angle_open_deg": 108.0}
    )
    assert (cfg.angle_closed_deg, cfg.angle_open_deg) == (18.0, 108.0)
    params = resolve_params(cfg)
    assert params.polarity == "red"
    assert not hasattr(params, "angle_closed_deg")
    assert not hasattr(params, "angle_open_deg")


def test_bad_polarity_rejected_at_load():
    with pytest.raises(ValueError, match="polarity must be one of"):
        GripperDotConfig(polarity="blue")


def test_bad_roi_keep_rejected_at_load():
    """An invalid roi_semicircle_keep fails at config load, not deep in the
    frame loop after outputs have already been written."""
    with pytest.raises(ValueError, match="roi_semicircle_keep must be one of"):
        config_from_block({"polarity": "red", "roi_semicircle_keep": "up"})


# --- strict keys -----------------------------------------------------------


def test_unknown_session_key_raises_with_a_did_you_mean():
    """An unknown key raises and names the key it meant."""
    with pytest.raises(ValueError, match=r"min_aera \(did you mean 'min_area'\?\)"):
        config_from_block({"polarity": "red", "min_aera": 100})


def test_unknown_session_key_error_lists_the_estimator_key_set():
    """No near miss to suggest: print the whole key set, so the error is the
    documentation for what a session: block may say."""
    with pytest.raises(
        ValueError, match=r"unknown session key\(s\).*chopsticks session keys are"
    ):
        config_from_block({"totally_made_up": 1})


def test_the_error_names_the_layer_that_carries_the_typo():
    """Each layer is validated at its SOURCE, so the message points at the file
    to edit instead of at the merged result, which belongs to no file."""
    with pytest.raises(
        ValueError,
        match=r"units/chopstick-m5\.yaml: unknown session key",
    ):
        config_from_block(
            {"min_aera": 100},
            source="config/gripper/chopstick/units/chopstick-m5.yaml",
        )


def test_long_gap_duration_key_loads():
    cfg = config_from_block({"polarity": "red", "long_gap_min_s": 0.167})
    assert cfg.long_gap_min_s == 0.167


# --- validate_session_block: one layer, one estimator ----------------------


def test_validate_session_block_dispatches_to_the_estimators_key_set():
    assert validate_session_block("chopsticks", TOOL_SESSION, "tool.yaml") is None
    with pytest.raises(ValueError, match=r"tool\.yaml: unknown session key"):
        validate_session_block("chopsticks", {"polarty": "red"}, "tool.yaml")


def test_validate_session_block_refuses_an_unregistered_estimator():
    """Even for an EMPTY block: a tool whose estimator has no key registry
    cannot have its recipe checked at all."""
    with pytest.raises(
        ValueError,
        match=r"no session-key validator for estimator 'parallel'.*chopsticks",
    ):
        validate_session_block("parallel", {}, "config/gripper/parallel-v1/tool.yaml")


# --- the ladder: tool < unit < episode-arm ---------------------------------


def test_the_session_ladder_merges_later_layers_over_earlier_ones():
    merged = merge_session(
        TOOL_SESSION,
        {"roi_semicircle_keep": "+y", "roi_center_y_mm": -5.0},  # the unit
        {"min_area": 111},  # the episode arm
    )
    cfg = config_from_block(merged)
    assert cfg.roi_semicircle_keep == "+y"  # unit wins over tool
    assert cfg.roi_center_y_mm == -5.0  # unit-only key applied
    assert cfg.min_area == 111  # arm wins over both
    assert cfg.max_area == 400  # non-overridden tool value kept


def test_the_ladder_reads_off_real_tool_and_unit_files(tmp_path):
    """The config layers end to end: design facts and recipe defaults from the tool
    file, this build's overrides from the unit file, merged into one block the
    estimator resolves."""
    tool = _write(
        tmp_path / "chopsticks-test.yaml",
        """
        schema: tool_v1
        design:
          estimator: chopsticks
          marker: {dictionary: DICT_4X4_50}
        session:
          polarity: red
          min_area: 200
          angle_closed_deg: 0.0
          angle_open_deg: 80.0
        """,
    )
    unit = _write(
        tmp_path / "unit-m5.yaml",
        """
        schema: chopstick_unit_v1
        tool: chopsticks-test
        build:
          marker: {id: 4, size: 0.014}
        session:
          angle_open_deg: 76.5
        """,
    )
    tc = load_tool_config(tool)
    gu = load_gripper_unit(unit)
    for block, source in ((tc.session, str(tool)), (gu.session, str(unit))):
        validate_session_block(tc.estimator, block, source)

    cfg = config_from_block(merge_session(tc.session, gu.session))
    assert cfg.polarity == "red" and cfg.min_area == 200  # tool defaults
    assert cfg.angle_closed_deg == 0.0  # tool endpoint kept
    assert cfg.angle_open_deg == 76.5  # this build's measured hinge wins


def test_a_typo_in_one_layer_is_caught_at_that_layer(tmp_path):
    """The merge is dumb by design (later wins); validation happens per layer
    at its source, which is what makes the message name the unit file."""
    unit = _write(
        tmp_path / "unit-m6.yaml",
        """
        schema: chopstick_unit_v1
        tool: chopsticks-test
        build:
          marker: {id: 6, size: 0.014}
        session:
          angle_opne_deg: 76.5
        """,
    )
    gu = load_gripper_unit(unit)  # core carries the block; it does not judge it
    with pytest.raises(ValueError, match=r"unit-m6\.yaml: unknown session key"):
        validate_session_block("chopsticks", gu.session, str(unit))


# --- resolve_params: override > config > builtin ---------------------------


def test_resolve_params_builtin_defaults():
    p = resolve_params(None)
    assert p.polarity == "dark"
    assert p.threshold == DEFAULT_THRESHOLD_BY_POLARITY["dark"]
    assert (p.min_area, p.max_area, p.min_circularity) == (70, 700, 0.55)
    assert (p.fov_deg, p.extent_x, p.extent_y) == (110.0, 4.0, 4.0)
    assert p.roi_radius_mm is None and p.roi_semicircle_keep == "-y"
    assert p.max_asymmetry_deg == 30.0 and p.enable_outlier_trim
    assert p.long_gap_min_s == 0.167
    assert not p.enable_drop_dot_retry and not p.enable_long_gap_rescue
    assert p.enable_one_jaw_fallback
    assert p.symmetry_axis_offset_deg is None  # auto-calibrate
    assert p.one_jaw_max_open_deg == 180.0


def test_resolve_params_threshold_default_tracks_polarity():
    assert resolve_params(GripperDotConfig(polarity="red")).threshold == 150
    assert resolve_params(GripperDotConfig(polarity="light")).threshold == 175
    # ...but an explicit threshold at any layer wins.
    assert (
        resolve_params(GripperDotConfig(polarity="red", threshold=142)).threshold == 142
    )


def test_resolve_params_layering():
    cfg = GripperDotConfig(polarity="red", min_area=200, extent_x=5.5)
    p = resolve_params(cfg, {"min_area": 500})
    assert p.min_area == 500  # override beats config
    assert p.polarity == "red"  # config beats builtin
    assert p.max_area == 700  # builtin fills the rest
    assert p.extent_y == 5.5  # extent_y mirrors the RESOLVED extent_x


def test_resolve_params_rejects_unknown_override():
    with pytest.raises(ValueError, match="unknown parameter override"):
        resolve_params(None, {"min_aera": 100})


def test_resolve_params_rejects_bad_polarity_override():
    with pytest.raises(ValueError, match="polarity must be one of"):
        resolve_params(None, {"polarity": "blue"})


def test_resolve_params_validates_the_annulus_early():
    with pytest.raises(ValueError, match="annulus would be empty"):
        resolve_params(GripperDotConfig(roi_radius_mm=50.0, roi_inner_radius_mm=60.0))
    with pytest.raises(ValueError, match="no outer edge"):
        resolve_params(GripperDotConfig(roi_inner_radius_mm=30.0))


# --- gap thresholds: feature-specific validation after resolution ----------


def _direct_gap_params(**thresholds):
    values = asdict(resolve_params())
    del values["gap_short_max_s"]
    del values["gap_medium_max_s"]
    return DotParams(**values, **thresholds)


def test_gap_threshold_defaults():
    cfg = GripperDotConfig()
    assert (cfg.gap_short_max_s, cfg.gap_medium_max_s) == (None, None)
    for params in (resolve_params(), _direct_gap_params()):
        assert (params.gap_short_max_s, params.gap_medium_max_s) == (0.167, 1.0)


def test_gap_threshold_layering_and_none_fallthrough():
    cfg = config_from_block(
        merge_session(
            {"gap_short_max_s": 0.2, "gap_medium_max_s": 2.0},
            {"gap_medium_max_s": 3.0},
            {"gap_short_max_s": 0.3},
        )
    )
    assert (cfg.gap_short_max_s, cfg.gap_medium_max_s) == (0.3, 3.0)
    params = resolve_params(cfg, {"gap_short_max_s": 0, "gap_medium_max_s": None})
    assert (params.gap_short_max_s, params.gap_medium_max_s) == (0, 3.0)
    params = resolve_params(
        config_from_block({"gap_short_max_s": None}), {"gap_medium_max_s": 2}
    )
    assert (params.gap_short_max_s, params.gap_medium_max_s) == (0.167, 2)


@pytest.mark.parametrize("entry", ["config", "override", "direct"])
@pytest.mark.parametrize("name", ["gap_short_max_s", "gap_medium_max_s"])
@pytest.mark.parametrize(
    "value",
    [True, False, "0.2", 0.2j, [], float("nan"), float("inf"), -float("inf"), 10**400],
)
def test_gap_thresholds_reject_nonfinite_or_nonreal_values(entry, name, value):
    values = {name: value}
    with pytest.raises(ValueError, match=rf"{name} must be a finite number"):
        if entry == "config":
            resolve_params(config_from_block(values))
        elif entry == "override":
            resolve_params(None, values)
        else:
            _direct_gap_params(**values)


@pytest.mark.parametrize("entry", ["config", "override", "direct"])
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"gap_short_max_s": -0.1}, "gap_short_max_s must be >= 0"),
        ({"gap_medium_max_s": -0.1}, "gap_medium_max_s must be >= 0"),
        (
            {"gap_short_max_s": 2.0, "gap_medium_max_s": 1.0},
            "gap_short_max_s must be <= gap_medium_max_s",
        ),
    ],
)
def test_gap_thresholds_reject_negative_or_reversed_bounds(entry, values, message):
    with pytest.raises(ValueError, match=message):
        if entry == "config":
            resolve_params(GripperDotConfig(**values))
        elif entry == "override":
            resolve_params(None, values)
        else:
            _direct_gap_params(**values)


@pytest.mark.parametrize("value", [0, 0.5, Fraction(1, 2)])
def test_gap_thresholds_normalize_equal_nonnegative_reals_to_floats(value):
    values = {"gap_short_max_s": value, "gap_medium_max_s": value}
    for params in (
        resolve_params(GripperDotConfig(**values)),
        resolve_params(None, values),
        _direct_gap_params(**values),
    ):
        assert (params.gap_short_max_s, params.gap_medium_max_s) == (value, value)
        assert type(params.gap_short_max_s) is float
        assert type(params.gap_medium_max_s) is float


@pytest.mark.parametrize(
    "values", [{"gap_short_max_s": 2.0}, {"gap_medium_max_s": 0.1}]
)
def test_gap_threshold_partial_layers_validate_order_only_after_resolution(values):
    cfg = config_from_block(values)
    with pytest.raises(ValueError, match="gap_short_max_s must be <= gap_medium_max_s"):
        resolve_params(cfg)
    params = resolve_params(cfg, {"gap_short_max_s": 0, "gap_medium_max_s": 3.0})
    assert (params.gap_short_max_s, params.gap_medium_max_s) == (0, 3.0)


def test_gap_threshold_order_is_checked_across_config_and_override():
    cfg = GripperDotConfig(gap_short_max_s=0.2, gap_medium_max_s=0.5)
    with pytest.raises(ValueError, match="gap_short_max_s must be <= gap_medium_max_s"):
        resolve_params(cfg, {"gap_short_max_s": 0.6})


@pytest.mark.parametrize("short", [2.0, "invalid", True, -1, float("inf")])
def test_gap_threshold_override_can_repair_lower_layer(short):
    cfg = config_from_block({"gap_short_max_s": short, "gap_medium_max_s": 1.0})
    params = resolve_params(cfg, {"gap_short_max_s": 0.2})
    assert (params.gap_short_max_s, params.gap_medium_max_s) == (0.2, 1.0)


def test_gap_thresholds_are_independent_of_the_long_gap_rescue_threshold():
    params = resolve_params(GripperDotConfig(long_gap_min_s=5.0))
    assert (params.gap_short_max_s, params.gap_medium_max_s) == (0.167, 1.0)


@pytest.mark.parametrize("name", ["gap_short_max_s", "gap_medium_max_s"])
def test_direct_gap_params_require_concrete_thresholds(name):
    with pytest.raises(ValueError, match=rf"{name} must be a finite number"):
        _direct_gap_params(**{name: None})


def test_gap_validation_preserves_existing_unrelated_parameter_coercion():
    params = resolve_params(GripperDotConfig(min_area="100"), {"fov_deg": "110"})
    assert params.min_area == 100
    assert params.fov_deg == 110.0

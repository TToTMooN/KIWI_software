"""Contract tests for dense EEF completion and its serializable provenance."""

import io

import numpy as np
import pytest

from kiwi.eef.completion import COMPLETION_POLICY_VERSION, complete_series


def test_linear_values_preserve_original_anchors_and_input():
    raw = np.array([0.0, np.nan, np.nan, np.nan, 8.0, -0.0, 123.456789])
    original = raw.copy()
    raw.flags.writeable = False
    out, meta, _ = complete_series(raw, 30)
    np.testing.assert_array_equal(out, [0, 2, 4, 6, 8, -0.0, 123.456789])
    np.testing.assert_array_equal(raw, original)
    assert out[np.isfinite(raw)].tobytes() == raw[np.isfinite(raw)].tobytes()
    assert not np.shares_memory(out, raw)
    np.testing.assert_array_equal(meta["interpolation_weight"][1:4], [0.25, 0.5, 0.75])


def test_explicit_metadata_for_multiple_and_edge_gaps():
    raw = np.array([np.nan, 2, np.nan, np.nan, 8, 9, np.nan])
    out, meta, notes = complete_series(raw, 2, short_max_s=0.5, medium_max_s=1)
    np.testing.assert_array_equal(out, [2, 2, 4, 6, 8, 9, 9])
    expected = {
        "filled": [True, False, True, True, False, False, True],
        "method": [
            "hold_next",
            "observed",
            "linear",
            "linear",
            "observed",
            "observed",
            "hold_previous",
        ],
        "gap_level": ["short", "none", "medium", "medium", "none", "none", "short"],
        "gap_id": [0, -1, 1, 1, -1, -1, 2],
        "gap_frames": [1, 0, 2, 2, 0, 0, 1],
        "gap_duration_s": [0.5, 0, 1, 1, 0, 0, 0.5],
        "left_anchor_frame": [-1, 1, 1, 1, 4, 5, 5],
        "right_anchor_frame": [1, 1, 4, 4, 4, 5, -1],
        "nearest_anchor_distance_s": [0.5, 0, 0.5, 0.5, 0, 0, 0.5],
        "anchor_span_s": [np.nan, 0, 1.5, 1.5, 0, 0, np.nan],
        "interpolation_weight": [np.nan, np.nan, 1 / 3, 2 / 3, np.nan, np.nan, np.nan],
    }
    for key, value in expected.items():
        np.testing.assert_array_equal(meta[key], value, err_msg=key)
    aggregates = {
        "filled_frames": 4,
        "filled_fraction": 4 / 7,
        "gap_count": 3,
        "max_gap_s": 1,
        "total_filled_s": 2,
        "short_max_s": 0.5,
        "medium_max_s": 1,
        "policy": "linear_interior_nearest_edge",
        "version": 1,
    }
    assert meta.keys() == expected.keys() | aggregates.keys()
    for key, value in aggregates.items():
        assert meta[key].shape == ()
        assert meta[key].item() == value
    assert meta["filled"].dtype.kind == "b"
    for key in ("method", "gap_level", "policy"):
        assert meta[key].dtype.kind == "U"
    for key in (
        "gap_id",
        "gap_frames",
        "left_anchor_frame",
        "right_anchor_frame",
        "filled_frames",
        "gap_count",
        "version",
    ):
        assert meta[key].dtype.kind in "iu"
    for value in meta.values():
        assert isinstance(value, np.ndarray)
        assert value.shape in ((), raw.shape)
        assert not value.dtype.hasobject
    assert all(isinstance(note, str) for note in notes)
    summary = " ".join(notes)
    for text in (
        "4/7",
        "57.1%",
        "3 gaps",
        "longest gap 1 s",
        "observed=3",
        "linear=2",
        "hold_next=1",
        "hold_previous=1",
        "not uncertainty",
        "provided finite anchor",
    ):
        assert text in summary
    buffer = io.BytesIO()
    np.savez(buffer, **meta)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        for key, value in meta.items():
            np.testing.assert_array_equal(saved[key], value)


@pytest.mark.parametrize("anchor", [0, 2, 4])
def test_one_anchor_holds_every_missing_frame(anchor):
    raw = np.full(5, np.nan)
    raw[anchor] = -3.25
    out, meta, _ = complete_series(raw, 2)
    np.testing.assert_array_equal(out, np.full(5, -3.25))
    assert np.all(meta["method"][:anchor] == "hold_next")
    assert np.all(meta["method"][anchor + 1 :] == "hold_previous")
    np.testing.assert_array_equal(
        meta["left_anchor_frame"], [-1] * anchor + [anchor] * (5 - anchor)
    )
    np.testing.assert_array_equal(
        meta["right_anchor_frame"], [anchor] * (anchor + 1) + [-1] * (4 - anchor)
    )
    np.testing.assert_array_equal(
        meta["nearest_anchor_distance_s"], np.abs(np.arange(5) - anchor) / 2
    )
    assert np.isnan(meta["anchor_span_s"][meta["filled"]]).all()
    assert np.isnan(meta["interpolation_weight"]).all()
    assert meta["gap_count"] == (anchor > 0) + (anchor < 4)


@pytest.mark.parametrize(
    "raw",
    [
        [4.5],
        np.array([0, 2**63 - 1], dtype=np.int64),
        np.array([0, 2**64 - 1], dtype=np.uint64),
        np.array([1.25, -0.0, 9], dtype=np.float32),
    ],
)
def test_no_missing_is_an_independent_exact_copy(raw):
    raw = np.asarray(raw)
    out, meta, _ = complete_series(raw, np.float64(30))
    np.testing.assert_array_equal(out, raw)
    assert out.astype(raw.dtype).tobytes() == raw.tobytes()
    assert not np.shares_memory(out, raw)
    assert not meta["filled"].any()
    assert np.all(meta["method"] == "observed")
    assert np.all(meta["gap_level"] == "none")
    assert np.all(meta["gap_id"] == -1)
    for key in (
        "gap_frames",
        "gap_duration_s",
        "nearest_anchor_distance_s",
        "anchor_span_s",
        "filled_frames",
        "filled_fraction",
        "gap_count",
        "max_gap_s",
        "total_filled_s",
    ):
        assert np.all(meta[key] == 0), key
    for key in ("left_anchor_frame", "right_anchor_frame"):
        np.testing.assert_array_equal(meta[key], np.arange(raw.size))
    assert np.isnan(meta["interpolation_weight"]).all()


@pytest.mark.parametrize(
    "fps,count,level",
    [
        (30, 5, "short"),
        (30, 6, "medium"),
        (30, 30, "medium"),
        (30, 31, "long"),
        (120, 20, "short"),
        (120, 21, "medium"),
        (120, 120, "medium"),
        (120, 121, "long"),
    ],
)
def test_default_duration_boundaries_and_all_gap_sizes_fill(fps, count, level):
    raw = np.r_[0.0, np.full(count, np.nan), float(count + 1)]
    out, meta, _ = complete_series(raw, fps)
    np.testing.assert_allclose(out, np.arange(count + 2), rtol=1e-15)
    assert np.all(meta["method"][1:-1] == "linear")
    assert np.all(meta["gap_level"][1:-1] == level)
    assert np.all(meta["gap_duration_s"][1:-1] == count / fps)
    assert np.all(meta["anchor_span_s"][1:-1] == (count + 1) / fps)
    assert meta["max_gap_s"] == count / fps


@pytest.mark.parametrize("fps", [30, 120])
@pytest.mark.parametrize("threshold", ["short_max_s", "medium_max_s"])
def test_thresholds_compare_seconds_exactly_without_rounding(fps, threshold):
    duration = 1 / fps
    config = {"short_max_s": 0, "medium_max_s": 1}
    config[threshold] = duration
    _, at, _ = complete_series([0, np.nan, 2], fps, **config)
    config[threshold] = np.nextafter(duration, 0)
    _, below, _ = complete_series([0, np.nan, 2], fps, **config)
    assert at["gap_level"][1] == ("short" if threshold == "short_max_s" else "medium")
    assert below["gap_level"][1] == ("medium" if threshold == "short_max_s" else "long")


def test_equal_zero_thresholds_are_valid_and_still_fill():
    out, meta, _ = complete_series([0, np.nan, 2], 30, short_max_s=0, medium_max_s=0)
    np.testing.assert_array_equal(out, [0, 1, 2])
    assert meta["gap_level"][1] == "long"


@pytest.mark.parametrize("count", [1, 2, 7, 120])
@pytest.mark.parametrize("signs", [(-1, 1), (1, -1), (1, 1), (-1, -1)])
def test_extreme_finite_anchors_do_not_overflow(count, signs):
    largest = np.finfo(np.float64).max
    left, right = (sign * largest for sign in signs)
    raw = np.r_[left, np.full(count, np.nan), right]
    with np.errstate(over="raise", invalid="raise"):
        out, _, _ = complete_series(raw, 30)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[[0, -1]], [left, right])
    np.testing.assert_allclose(
        out / largest, np.linspace(*signs, count + 2), atol=1e-15
    )


def test_exported_policy_version_matches_scalar_metadata():
    _, meta, _ = complete_series([1], 30)
    assert meta["version"].shape == ()
    assert meta["version"].item() == COMPLETION_POLICY_VERSION == 1


@pytest.mark.parametrize("raw", [[], np.array([], dtype=int), [np.nan], [np.nan] * 5])
def test_no_usable_samples_raises_descriptively(raw):
    with pytest.raises(ValueError, match="no usable EEF samples"):
        complete_series(raw, 30)


@pytest.mark.parametrize(
    "raw",
    [
        [0, np.inf],
        [-np.inf, np.nan],
        [np.inf],
        [True, False],
        ["1", "2"],
        [1 + 0j, 2 + 1j],
        np.array([1, np.nan], dtype=object),
        [None],
        [0, None],
        [[1, 2]],
        np.zeros((2, 2, 1)),
        1.0,
        np.array(["2026-01-01"], dtype="datetime64[D]"),
    ],
)
def test_rejects_invalid_sample_arrays(raw):
    with pytest.raises(ValueError, match="values"):
        complete_series(raw, 30)


@pytest.mark.parametrize("name", ["fps", "short_max_s", "medium_max_s"])
@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        "30",
        1 + 0j,
        [30],
        np.array([30]),
        None,
    ],
)
def test_rejects_nonfinite_or_nonreal_scalar_configuration(name, value):
    config = {"fps": 30, name: value}
    with pytest.raises(ValueError, match=name):
        complete_series([1, np.nan, 2], **config)


@pytest.mark.parametrize(
    "config",
    [
        {"fps": 0},
        {"fps": -30},
        {"short_max_s": -0.1},
        {"medium_max_s": -0.1},
        {"short_max_s": 2, "medium_max_s": 1},
    ],
)
def test_rejects_invalid_configuration_even_without_gaps(config):
    with pytest.raises(ValueError, match="fps|thresholds"):
        complete_series([1], **({"fps": 30} | config))

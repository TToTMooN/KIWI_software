"""Invalid calibration data cannot become a confident video clock."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from kiwi.sync.audio import (
    AV_SYNC_SCHEMA,
    EXTRACTION,
    GATES,
    AVCorrection,
    AudioOffset,
    Clip,
    MediaProbe,
    RawOffset,
    RecordingMode,
    SyncParams,
    read_av_delay,
    resolve_av_correction,
)


MODE = RecordingMode(320, 240, 30.0)


def _unit(tmp_path, name, *, model="Model A", delay=0.0, **metadata):
    unit = tmp_path / name
    unit.mkdir()
    (unit / "camera.yaml").write_text(
        yaml.safe_dump({"model": model, "av_sync": "av-sync.yaml"})
    )
    (unit / "av-sync.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": AV_SYNC_SCHEMA,
                "unit": name,
                "extraction": EXTRACTION,
                "modes": {MODE.key: {"av_delay_s": delay}},
                **metadata,
            }
        )
    )
    return unit


def _probe():
    return MediaProbe(Path("video.mp4"), MODE, 12.0, "aac", 48_000, 1)


@pytest.mark.parametrize(
    "delay", [float("nan"), float("inf"), -float("inf"), None, True, False, "bad", []]
)
def test_invalid_av_delay_is_refused_with_its_source(tmp_path, delay):
    unit = _unit(tmp_path, "invalid", delay=delay)
    with pytest.raises(RuntimeError, match="av_delay_s.*finite number") as exc:
        read_av_delay(unit, MODE)
    assert str(unit / "av-sync.yaml") in str(exc.value)
    assert MODE.key in str(exc.value)


@pytest.mark.parametrize("extraction", ["ffmpeg -different-recipe -", None])
def test_a_declared_different_extraction_recipe_is_refused(tmp_path, extraction):
    unit = _unit(tmp_path, "recipe", extraction=extraction)
    with pytest.raises(RuntimeError, match="extraction.*expected"):
        read_av_delay(unit, MODE)


def test_the_current_recipe_is_readable(tmp_path):
    unit = _unit(tmp_path, "compatible", delay=0.059)
    assert read_av_delay(unit, MODE).seconds == pytest.approx(0.059)


@pytest.mark.parametrize(
    "doc",
    [
        ["not", "a", "table"],
        {
            "schema": AV_SYNC_SCHEMA,
            "extraction": EXTRACTION,
            "modes": ["not", "a", "mapping"],
        },
        {
            "schema": AV_SYNC_SCHEMA,
            "extraction": EXTRACTION,
            "modes": {MODE.key: 0.059},
        },
    ],
)
def test_malformed_av_table_shapes_are_measurement_errors(tmp_path, doc):
    unit = _unit(tmp_path, "malformed")
    (unit / "av-sync.yaml").write_text(yaml.safe_dump(doc))
    with pytest.raises(RuntimeError, match="mapping"):
        read_av_delay(unit, MODE)


@pytest.mark.parametrize("model", [None, "", "  ", True, ["Model A"]])
def test_an_invalid_model_never_establishes_common_delay(tmp_path, model):
    unit = _unit(tmp_path, "unknown", model=model)
    clip = Clip(Path("clip.mp4"), unit)
    with pytest.raises(RuntimeError, match="model"):
        resolve_av_correction(clip, _probe(), clip, _probe())


def test_a_nonexistent_registry_unit_never_establishes_common_delay(tmp_path):
    clip = Clip(Path("clip.mp4"), tmp_path / "missing")
    with pytest.raises(RuntimeError, match="model"):
        resolve_av_correction(clip, _probe(), clip, _probe())


def test_one_unknown_camera_cannot_borrow_the_others_identity(tmp_path):
    unit = _unit(tmp_path, "known")
    with pytest.raises(RuntimeError, match="no camera unit"):
        resolve_av_correction(
            Clip(Path("known.mp4"), unit), _probe(), Clip(Path("unknown.mp4")), _probe()
        )


def test_matching_containers_with_different_known_models_apply_calibration(tmp_path):
    ref = _unit(tmp_path, "reference", model="Model A", delay=0.0)
    other = _unit(tmp_path, "other", model="Model B", delay=0.059)
    correction = resolve_av_correction(
        Clip(Path("a.mp4"), ref), _probe(), Clip(Path("b.mp4"), other), _probe()
    )
    assert correction.seconds == pytest.approx(0.059)


def test_finite_calibrations_cannot_overflow_their_difference(tmp_path):
    ref = _unit(tmp_path, "reference", model="Model A", delay=-1e308)
    other = _unit(tmp_path, "other", model="Model B", delay=1e308)
    with pytest.raises(RuntimeError, match="difference must be finite"):
        resolve_av_correction(
            Clip(Path("a.mp4"), ref), _probe(), Clip(Path("b.mp4"), other), _probe()
        )


def _healthy_offset():
    raw = RawOffset(
        offset_s=1.0,
        voted_offset_s=1.0,
        votes=30,
        runner_up=5,
        onsets_matched=3,
        onsets_total=5,
        gates={name: True for name in GATES},
        diagnostics={},
        ref=_probe(),
        other=_probe(),
        params=SyncParams(),
    )
    return AudioOffset(1.0, raw, AVCorrection(0.0, "known calibration"))


@pytest.mark.parametrize("field", ["video", "raw", "correction"])
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), None, True]
)
def test_nonfinite_result_cannot_be_confident_or_required(field, value):
    result = _healthy_offset()
    if field == "video":
        result = dataclasses.replace(result, offset_s=value)
    elif field == "raw":
        result = dataclasses.replace(
            result, raw=dataclasses.replace(result.raw, offset_s=value)
        )
    else:
        result = dataclasses.replace(
            result, correction=dataclasses.replace(result.correction, seconds=value)
        )
    assert not result.confident
    assert "finite_video_offset" in result.failed_gates
    with pytest.raises(RuntimeError, match="finite_video_offset"):
        result.require_confident()


def test_valid_result_keeps_confidence_and_existing_failures():
    healthy = _healthy_offset()
    assert healthy.require_confident() is healthy
    failed = dataclasses.replace(
        healthy,
        raw=dataclasses.replace(
            healthy.raw, gates={**healthy.raw.gates, "vote_margin": False}
        ),
    )
    assert failed.failed_gates == ["vote_margin"]
    assert not failed.confident

"""Tests for kiwi.calib.kb4 - the DS -> KB4 fit and its registry writer.

The fit is judged on its actual purpose, not a smoke test: KB4 must match a
realistic X5-like DS model to sub-pixel RMS inside the fit FOV and degrade
monotonically past it - that degradation number is what an engine's FOV-crop
decision consumes.

Run: pixi run test
"""

from __future__ import annotations

import numpy as np
import pytest

from kiwi.core.camera.kb4 import read_kb4
from kiwi.calib.kb4 import (
    Kb4Fit,
    fit_kb4,
    project_ds,
    project_kb4,
    write_kb4_json,
)

# An X5-front-like DS model (xi ~ 0, alpha ~ 0.57, fx ~ 500 at 1920x1920).
DS = np.array([-0.023, 0.571, 499.64, 499.54, 962.30, 961.43])
RES = (1920, 1920)


@pytest.fixture(scope="module")
def fit() -> Kb4Fit:
    return fit_kb4(DS, RES)


def test_fit_matches_ds_inside_fit_fov(fit):
    # Hold a 0.5 px band-RMS bound inside the fit region.
    assert fit.rms_within_fit_fov() < 0.5
    inside = [r for r in fit.rms_by_fov if r[1] <= fit.fit_max_deg]
    assert inside, "no bands inside the fit FOV"


def test_error_grows_monotonically_past_the_fit_edge(fit):
    rms = [r[2] for r in fit.rms_by_fov]
    # The last band (beyond the fit edge) must be the worst.
    assert rms[-1] == max(rms)
    assert rms[-1] > 2 * min(rms)


def test_projection_models_agree_at_the_optical_axis():
    # theta -> 0: both models project to the principal point.
    th = np.array([1e-6])
    ph = np.array([0.7])
    ray = np.column_stack(
        [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    )
    ds_px = project_ds(ray, *DS)
    kb4_px = project_kb4(th, ph, np.array([*DS[2:], 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(ds_px, kb4_px, atol=1e-3)
    np.testing.assert_allclose(ds_px[0], DS[4:6], atol=1e-2)


def test_kb4_json_round_trip_and_schema(fit, tmp_path):
    out = write_kb4_json(
        tmp_path / "intrinsics-kb4-2026-08-19.json", fit, unit="u1", lens="front"
    )
    params, doc = read_kb4(out)
    np.testing.assert_allclose(params, fit.params)
    assert doc["schema"] == "kb4_intrinsics"
    assert doc["unit"] == "u1" and doc["lens"] == "front"
    assert doc["resolution"] == [1920, 1920]
    assert doc["rms_by_fov"][0].keys() == {"lo", "hi", "rms_px", "max_px"}
    (tmp_path / "bad.json").write_text('{"schema": "nope"}')
    with pytest.raises(ValueError, match="schema"):
        read_kb4(tmp_path / "bad.json")


def test_the_lens_gate_refuses_a_wrong_lens_fit(fit, tmp_path):
    """A front fit attached to the back pointer must FAIL."""
    front = write_kb4_json(tmp_path / "f.json", fit, unit="u1", lens="front")
    back = write_kb4_json(tmp_path / "b.json", fit, unit="u1", lens="back")
    read_kb4(front, expected_lens="front")
    read_kb4(back, expected_lens="back")
    with pytest.raises(ValueError, match="front-lens fit"):
        read_kb4(front, expected_lens="back")
    with pytest.raises(ValueError, match="back-lens fit"):
        read_kb4(back, expected_lens="front")


def test_any_other_schema_tag_is_refused(fit, tmp_path):
    """One canonical document, no compatibility branch: a file with any
    other schema tag is refused, not read through a second code path."""
    import json

    out = write_kb4_json(tmp_path / "other.json", fit, unit="u1", lens="front")
    doc = json.loads(out.read_text())
    doc["schema"] = "other_schema"
    del doc["lens"]
    out.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="schema"):
        read_kb4(out)


def test_broken_parameters_and_resolution_are_refused(fit, tmp_path):
    import json

    out = write_kb4_json(tmp_path / "a.json", fit, unit="u1", lens="front")
    doc = json.loads(out.read_text())
    doc["kb4"]["fx"] = float("nan")
    (tmp_path / "nan.json").write_text(json.dumps(doc).replace("NaN", "NaN"))
    with pytest.raises(ValueError, match="finite"):
        read_kb4(tmp_path / "nan.json")
    doc = json.loads(out.read_text())
    doc["resolution"] = [1920, 0]
    (tmp_path / "res.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="resolution"):
        read_kb4(tmp_path / "res.json")


def test_the_writer_requires_lens_and_unit(fit, tmp_path):
    with pytest.raises(ValueError, match="lens"):
        write_kb4_json(tmp_path / "x.json", fit, unit="u1", lens="rear")
    with pytest.raises(ValueError, match="unit"):
        write_kb4_json(tmp_path / "x.json", fit, unit="", lens="back")

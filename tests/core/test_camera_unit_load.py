"""CameraUnit.load refuses a broken camera.yaml as a ValueError naming it.

Every stage and CLI verb turns a ValueError into a refusal; the parser's own
error, or the AttributeError a non-mapping meets at ``.artifact()``, escapes
them as a traceback that names no file.
"""

from __future__ import annotations

import pytest
import yaml

from kiwi.core.camera.registry import CameraUnit


def test_a_camera_yaml_that_is_not_yaml_names_the_file(tmp_path):
    (tmp_path / "camera.yaml").write_text("serial: [unclosed\n")
    with pytest.raises(ValueError, match=r"camera\.yaml: not valid YAML: ") as err:
        CameraUnit.load(tmp_path)
    assert str(tmp_path / "camera.yaml") in str(err.value)
    assert isinstance(err.value.__cause__, yaml.YAMLError)


@pytest.mark.parametrize("text", ["- serial\n- S1\n", "just a string\n"])
def test_a_camera_yaml_that_is_not_a_mapping_names_the_file(tmp_path, text):
    (tmp_path / "camera.yaml").write_text(text)
    with pytest.raises(ValueError, match=r"camera\.yaml: must be a YAML mapping"):
        CameraUnit.load(tmp_path)


@pytest.mark.parametrize("text", [None, "", "# only a comment\n"])
def test_an_absent_or_empty_camera_yaml_is_an_empty_unit(tmp_path, text):
    if text is not None:
        (tmp_path / "camera.yaml").write_text(text)
    unit = CameraUnit.load(tmp_path)
    assert unit.meta == {} and unit.artifact("intrinsics") is None

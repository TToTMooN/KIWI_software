"""Explicit synthetic optical calibration for export integration fixtures."""

import pytest
import yaml


@pytest.fixture
def calibrated_dc_description(tmp_path, monkeypatch):
    """Isolated fixture, never a calibration fallback for shipped hardware."""
    root = tmp_path / "test-descriptions"
    family = root / "synthetic"
    family.mkdir(parents=True)
    (family / "model.urdf").write_text("""<robot name="synthetic">
      <link name="mount"/><link name="camera"/><link name="tcp"/>
      <joint name="camera_joint" type="fixed">
        <parent link="mount"/><child link="camera"/>
        <origin xyz="0.01 0.055 0.056" rpy="0.1 -0.2 0.3"/>
      </joint>
      <joint name="tcp_joint" type="fixed">
        <parent link="mount"/><child link="tcp"/>
        <origin xyz="0.188 0 -0.003" rpy="1.5708 0 1.5708"/>
      </joint>
    </robot>""")
    (family / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "robot_descriptions_v1",
                "family": "synthetic",
                "descriptions": {
                    "test_calibrated_dc": {
                        "urdf": "synthetic/model.urdf",
                        "end_effectors": {
                            role: {"urdf_link": link}
                            for role, link in (
                                ("default", "mount"),
                                ("camera", "camera"),
                                ("tcp", "tcp"),
                            )
                        },
                    }
                },
            }
        )
    )
    monkeypatch.setenv("KIWI_DESCRIPTIONS", str(root))

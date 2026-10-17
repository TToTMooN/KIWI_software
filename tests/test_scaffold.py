"""Package contracts: importability, stage consistency and shared types."""

import importlib
from pathlib import Path

import numpy as np
import pytest

import kiwi
from kiwi.core.clock import GlobalClock
from kiwi.core.trajectory import Trajectory
from kiwi.export import schema
from kiwi.episode import stages


def _subpackages() -> list[str]:
    pkg_root = Path(kiwi.__file__).parent
    return sorted(p.name for p in pkg_root.iterdir() if (p / "__init__.py").is_file())


def test_all_packages_import():
    names = _subpackages()
    assert names, "no kiwi subpackages found"
    for name in names:
        importlib.import_module(f"kiwi.{name}")
    assert kiwi.__version__


def test_stage_registry_covers_pipeline():
    assert set(stages.REGISTRY) == set(stages.STAGES)


def test_every_stage_is_dispatchable(tmp_path):
    # A direct ground request dispatches its entry even though the default
    # profile does not request it inside a chain.
    assert all(callable(fn) for fn in stages.REGISTRY.values())
    res = stages.run_stage("ground", tmp_path)
    assert res.stage == "ground" and not res.ok and "kiwi init" in res.summary
    [r] = stages.run_pipeline(tmp_path, from_stage="ground", to_stage="ground")
    assert (r.stage, r.ok) == ("ground", False) and "kiwi init" in r.summary
    with pytest.raises(ValueError):
        stages.run_pipeline(tmp_path, from_stage="eef", to_stage="convert")


def test_pipeline_halts_on_the_first_failed_stage(tmp_path):
    # An empty episode dir has nothing for sync to place, and every later stage
    # consumes its clock, so the run stops there.
    results = stages.run_pipeline(tmp_path)
    assert [r.stage for r in results] == ["sync"]
    assert results[0].ok is False


def test_trajectory_contract():
    n = 5
    traj = Trajectory(
        timestamps=np.linspace(0.0, 1.0, n),
        positions=np.zeros((n, 3)),
        quaternions=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        frame="world",
        body="camera",
    ).validate()
    assert len(traj) == n
    with pytest.raises(ValueError):
        Trajectory(
            timestamps=np.array([0.0, 0.0]),  # not strictly increasing
            positions=np.zeros((2, 3)),
            quaternions=np.zeros((2, 4)),
            frame="world",
            body="camera",
        ).validate()
    with pytest.raises(ValueError):
        Trajectory(
            timestamps=np.zeros(3),
            positions=np.zeros((2, 3)),  # shape mismatch
            quaternions=np.zeros((3, 4)),
            frame="world",
            body="camera",
        ).validate()


def test_global_clock_contract():
    GlobalClock(origin="ego", offsets_s={"ego": 0.0, "left": -1.2}).validate()
    with pytest.raises(ValueError):
        GlobalClock(origin="ego", offsets_s={"left": -1.2}).validate()
    with pytest.raises(ValueError):
        GlobalClock(origin="ego", offsets_s={"ego": 0.0, "left": 0.5}).validate()


def test_export_schema_names():
    assert schema.wrist_image_key("left") == "observation.images.left_wrist"
    # a single unnamed arm exports under the "main" label
    assert schema.wrist_image_key("main") == "observation.images.main_wrist"
    assert schema.wrist_image_key("left").startswith(schema.IMAGE_KEY_PREFIX)

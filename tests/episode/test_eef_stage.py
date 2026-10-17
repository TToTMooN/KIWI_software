"""The eef stage entry: job assembly, reuse, and the failure contract.

run_clip is monkeypatched at the component seam (the _sync_entry test
pattern): the fake captures the assembled ClipJob and writes a REAL
eef_state_v1 npz into the job's ``out_dir``, so the reuse path reads
genuine artifacts at the layout's fixed names.

The job is the meeting point of the tool_v1 layers, so the assembly tests
pin which layer each field came from: the ``design:`` block of the tool file
(estimator, marker dictionary), the ``build:`` block of THIS arm's gripper
unit (its tag id and size, its calibrated fixed_pose), and the merged
``session:`` recipe. Config that refuses to load is a stage RESULT, never a
traceback; the refusals are covered here.

Run: pixi run test
"""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from kiwi.core import layout
from kiwi.eef import run as run_mod
from kiwi.eef.completion import complete_series
from kiwi.eef.state import EefState, write_eef_state
from kiwi.episode import generation, stages
from kiwi.episode.manifest import MANIFEST_NAME

UNIT_A, UNIT_B = "insta360-X5-001", "insta360-X5-002"
# Two arms, two PHYSICAL grippers: each build carries its own tag, and the
# tag id and size are unit facts under tool_v1, not tool facts.
GRIPPER_LEFT, GRIPPER_RIGHT = "chopstick-v2-m3", "chopstick-v2-m4"

TOOL_YAML = {
    "schema": "tool_v1",
    "name": "chopsticks-v2",
    "design": {
        "estimator": "chopsticks",
        "marker": {"dictionary": "DICT_4X4_50"},
    },
    "session": {"polarity": "red", "angle_closed_deg": 70.0, "angle_open_deg": 165.0},
}

FIXED_POSE = {
    "rvec": [2.7, 0.1, 0.0],
    "tvec": [0.0, 0.02, 0.26],
    "fov_deg": 160.0,
    "out_size": [1920, 1920],
    "lens": "front",
}


def _unit_yaml(marker_id: int, *, session=None, tool="chopsticks-v2", pose=None) -> str:
    """One chopstick_unit_v1 build: its tag, its calibration, its overrides."""
    doc = {
        "schema": "chopstick_unit_v1",
        "tool": tool,
        "build": {
            "marker": {"id": marker_id, "size": 0.014},
            "fixed_pose": pose or FIXED_POSE,
        },
    }
    if session:
        doc["session"] = session
    return yaml.safe_dump(doc, sort_keys=False)


def _write_manifest(ep, *, left=None, right=None):
    """Write ``episode.yaml``, merging per-arm extras into each arm's block."""
    arms = {
        "left": {
            "camera": UNIT_A,
            "recording": "raw/A.insv",
            "gripper_unit": GRIPPER_LEFT,
        },
        "right": {
            "camera": UNIT_B,
            "recording": "raw/B.insv",
            "gripper_unit": GRIPPER_RIGHT,
        },
    }
    arms["left"].update(left or {})
    arms["right"].update(right or {})
    doc = {
        "schema": "episode_v1",
        "name": "ep1",
        "tool": "chopsticks-v2",
        "arms": arms,
    }
    (ep / MANIFEST_NAME).write_text(yaml.safe_dump(doc, sort_keys=False))
    return ep / MANIFEST_NAME


def _write_intrinsics(cfg, unit, fx=900.0):
    """A real npz: the stage hashes the intrinsics by CONTENT to key the reuse,
    so a placeholder byte string would make every arm unhashable."""
    path = cfg / "cameras" / unit / "intrinsics.npz"
    np.savez(path, K=np.array([[fx, 0.0, 960.0], [0.0, fx, 960.0], [0.0, 0.0, 1.0]]))
    return path


@pytest.fixture
def episode(tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    for unit in (UNIT_A, UNIT_B):
        (cfg / "cameras" / unit).mkdir(parents=True)
        _write_intrinsics(cfg, unit)
    (cfg / "gripper" / "chopsticks-v2" / "units").mkdir(parents=True)
    (cfg / "gripper" / "chopsticks-v2" / "tool.yaml").write_text(
        yaml.safe_dump(TOOL_YAML, sort_keys=False)
    )
    units = cfg / "gripper" / "chopsticks-v2" / "units"
    (units / f"{GRIPPER_LEFT}.yaml").write_text(_unit_yaml(3))
    (units / f"{GRIPPER_RIGHT}.yaml").write_text(_unit_yaml(4))
    monkeypatch.setenv("KIWI_CONFIG", str(cfg))

    # The tree the convert stage leaves behind: recordings flat in raw/, the
    # demuxed front lens in each arm's own stream directory.
    ep = tmp_path / "ep1"
    (ep / "raw").mkdir(parents=True)
    for label, name in (("left", "A.insv"), ("right", "B.insv")):
        (ep / "raw" / name).write_bytes(b"media")
        (ep / label).mkdir()
        (ep / label / "lens0.mp4").write_bytes(b"mp4")
    _write_manifest(ep)
    return ep


def _state(n=10, n_valid=10) -> EefState:
    value = np.full(n, 40.0)
    value[n_valid:] = np.nan
    output, completion, _ = complete_series(value, 30.0)
    return EefState(
        tool="chopsticks-v2",
        estimator="chopsticks",
        unit="deg",
        fps=30.0,
        source="x_lens0.mp4",
        pose_mode="fixed_pose",
        # A calibration is a build fact: the pose is written into the gripper
        # unit's build.fixed_pose, so that is the file the artifact names.
        pose_source=f"config/gripper/chopsticks-v2/units/{GRIPPER_LEFT}.yaml",
        value=value,
        rejected_value=np.full(n, np.nan),
        has_pose=np.ones(n, dtype=bool),
        n_detected=np.full(n, 6, dtype=np.int32),
        n_kept=np.full(n, 6, dtype=np.int32),
        error=np.array([""] * n, dtype=str),
        detail={},
        value_completed=output,
        completion=completion,
    )


def _fake_run_clip(monkeypatch, calls, n_valid_by_stem=None):
    def fake(job, **kw):
        calls.append(job)
        n_valid = (n_valid_by_stem or {}).get(job.recording.stem, 10)
        st = _state(10, n_valid)
        # Inside an episode the artifact is a fixed name in the job's out_dir,
        # never a stem suffix beside the recording - raw/ takes no derived file.
        path = write_eef_state(job.out_dir / layout.EEF_STATE, st)
        return run_mod.ClipResult(
            state_path=path, state=st, trace_path=None, overlay_path=None, notes=[]
        )

    monkeypatch.setattr(run_mod, "run_clip", fake)


def test_runs_both_arms_and_assembles_the_jobs(episode, monkeypatch):
    calls = []
    _fake_run_clip(monkeypatch, calls)
    r = stages.run_stage("eef", episode)

    assert r.ok, r.summary
    assert r.summary.startswith("2 arm(s) estimated (2 run, 0 reused)")
    assert len(calls) == 2
    by_label = {j.out_dir.name: j for j in calls}
    job = by_label["left"]
    # From the tool file's frozen design: block.
    assert job.estimator_key == "chopsticks"
    assert job.tool == "chopsticks-v2"
    assert job.marker_dictionary == "DICT_4X4_50"
    # From the tool file's session: recipe (no unit or arm override here).
    assert job.session_block["polarity"] == "red"
    # From THIS arm's gripper unit's build: block - the tag and the pose that
    # was calibrated against that tag's placement.
    assert job.marker_id == 3 and job.marker_size_m == 0.014
    assert job.fixed_pose["fov_deg"] == 160.0
    assert job.fixed_pose_source.endswith(f"units/{GRIPPER_LEFT}.yaml")
    assert job.intrinsics.name == "intrinsics.npz" and UNIT_A in str(job.intrinsics)
    assert job.recording == episode / "raw" / "A.insv"
    assert job.lens0_mp4 == episode / "left" / "lens0.mp4"
    assert job.out_dir == episode / "left"
    # Each arm gets ITS OWN build's tag - the two grippers are two objects.
    assert by_label["right"].marker_id == 4
    assert by_label["right"].fixed_pose_source.endswith(f"{GRIPPER_RIGHT}.yaml")
    npzs = {p.relative_to(episode) for p in r.artifacts}
    assert npzs == {Path("left/eef.npz"), Path("right/eef.npz")}


def test_the_session_recipe_layers_tool_then_unit_then_arm(
    episode, tmp_path, monkeypatch
):
    # The session: ladder - each layer overrides the one above it and leaves
    # the rest alone.
    (
        tmp_path
        / "config"
        / "gripper"
        / "chopsticks-v2"
        / "units"
        / f"{GRIPPER_LEFT}.yaml"
    ).write_text(_unit_yaml(3, session={"angle_open_deg": 158.0, "min_area": 100}))
    _write_manifest(episode, left={"session": {"min_area": 120}})
    calls = []
    _fake_run_clip(monkeypatch, calls)
    assert stages.run_stage("eef", episode).ok

    by_label = {j.out_dir.name: j for j in calls}
    left = by_label["left"].session_block
    assert left["polarity"] == "red", "untouched tool default"
    assert left["angle_closed_deg"] == 70.0, "untouched tool default"
    assert left["angle_open_deg"] == 158.0, "the unit's measured hinge wins"
    assert left["min_area"] == 120, "the episode-arm block wins over the unit"
    # The other arm's unit states no override, so it keeps the tool defaults.
    assert by_label["right"].session_block["angle_open_deg"] == 165.0
    assert "min_area" not in by_label["right"].session_block


def test_an_arm_without_a_gripper_unit_runs_dictionary_only(episode, monkeypatch):
    # `gripper_unit:` is optional: an arm may name only its tool design, and
    # then no build facts exist at all - no tag id, no size, no calibrated
    # pose. The stage must assemble that job rather than invent them.
    _write_manifest(episode)
    doc = yaml.safe_load((episode / MANIFEST_NAME).read_text())
    for arm in doc["arms"].values():
        del arm["gripper_unit"]
    (episode / MANIFEST_NAME).write_text(yaml.safe_dump(doc, sort_keys=False))
    calls = []
    _fake_run_clip(monkeypatch, calls)
    assert stages.run_stage("eef", episode).ok

    job = {j.out_dir.name: j for j in calls}["left"]
    assert job.marker_dictionary == "DICT_4X4_50", "the design still applies"
    assert job.marker_id is None and job.marker_size_m is None
    assert job.fixed_pose is None and job.fixed_pose_source == ""


def test_fresh_state_is_reused(episode, monkeypatch):
    calls = []
    _fake_run_clip(monkeypatch, calls)
    assert stages.run_stage("eef", episode).ok
    assert len(calls) == 2

    r = stages.run_stage("eef", episode)
    assert r.ok
    assert len(calls) == 2  # nothing re-ran
    assert r.summary.startswith("2 arm(s) estimated (0 run, 2 reused)")
    assert "left: reused eef.npz (10/10 valid)" in r.summary


def _left_unit(tmp_path):
    return (
        tmp_path
        / "config"
        / "gripper"
        / "chopsticks-v2"
        / "units"
        / f"{GRIPPER_LEFT}.yaml"
    )


def test_a_retuned_session_recipe_invalidates_the_reuse(episode, tmp_path, monkeypatch):
    # The recipe IS the estimator's settings, so a retune changes the series.
    calls = []
    _fake_run_clip(monkeypatch, calls)
    assert stages.run_stage("eef", episode).ok
    _left_unit(episode).write_text(_unit_yaml(3, session={"angle_open_deg": 158.0}))

    r = stages.run_stage("eef", episode)
    assert r.ok, r.summary
    assert len(calls) == 3, "the retuned arm must re-run"
    assert r.summary.startswith("2 arm(s) estimated (1 run, 1 reused)")
    assert "right: reused eef.npz" in r.summary, "the untouched arm still reuses"


def test_a_recalibrated_fixed_pose_invalidates_the_reuse(
    episode, tmp_path, monkeypatch
):
    # A moved pose shifts the output by a constant offset.
    calls = []
    _fake_run_clip(monkeypatch, calls)
    assert stages.run_stage("eef", episode).ok
    pose = {**FIXED_POSE, "tvec": [0.0, 0.02, 0.31]}
    _left_unit(episode).write_text(_unit_yaml(3, pose=pose))

    r = stages.run_stage("eef", episode)
    assert r.ok, r.summary
    assert len(calls) == 3
    assert "right: reused eef.npz" in r.summary


def test_republished_intrinsics_invalidate_the_reuse(episode, tmp_path, monkeypatch):
    # Hashed by array, not by file: republishing the same numbers is free,
    # while a new fit re-estimates the arm that reads it.
    calls = []
    _fake_run_clip(monkeypatch, calls)
    assert stages.run_stage("eef", episode).ok
    _write_intrinsics(episode / "config", UNIT_A)  # identical arrays
    assert stages.run_stage("eef", episode).ok
    assert len(calls) == 2, "a byte-different, array-identical npz is not a change"

    _write_intrinsics(episode / "config", UNIT_A, fx=901.0)
    r = stages.run_stage("eef", episode)
    assert r.ok, r.summary
    assert len(calls) == 3
    assert "right: reused eef.npz" in r.summary


def test_moving_the_episode_without_the_deployment_config_still_reuses(
    episode, tmp_path, monkeypatch
):
    # Nothing semantic moved. The key names WHICH build file the pose came
    # from, relative to the config root.
    calls = []
    _fake_run_clip(monkeypatch, calls)
    assert stages.run_stage("eef", episode).ok
    assert len(calls) == 2
    moved = tmp_path / "moved-episode"
    shutil.move(episode, moved)
    shutil.rmtree(tmp_path / "config")

    r = stages.run_stage("eef", moved)
    assert r.ok, r.summary
    assert len(calls) == 2, "an identical config tree at another path re-estimated"
    assert r.summary.startswith("2 arm(s) estimated (0 run, 2 reused)")


def test_snapshot_preserves_the_existing_eef_key_and_ignores_external_retuning(
    episode, tmp_path, monkeypatch
):
    from kiwi.core.camera.registry import CameraUnit
    from kiwi.episode.manifest import EpisodeConfig

    cfg = EpisodeConfig.load(episode / "episode.yaml")
    arm = cfg.arms["left"]
    previous_key = stages._eef_inputs_hash(
        arm,
        CameraUnit.load(arm.camera_dir).artifact("intrinsics"),
        config_dir=cfg.config_dir,
    )
    calls = []
    _fake_run_clip(monkeypatch, calls)
    assert stages.run_stage("eef", episode).ok
    local = EpisodeConfig.load(episode / "episode.yaml")
    local_arm = local.arms["left"]
    assert (
        stages._eef_inputs_hash(
            local_arm,
            CameraUnit.load(local_arm.camera_dir).artifact("intrinsics"),
            config_dir=local.config_dir,
        )
        == previous_key
    )

    _left_unit(tmp_path).write_text(_unit_yaml(3, session={"angle_open_deg": 158.0}))
    _write_intrinsics(tmp_path / "config", UNIT_A, fx=901.0)
    result = stages.run_stage("eef", episode)
    assert result.ok, result.summary
    assert len(calls) == 2, "deployment updates must not change an existing episode"


def test_a_malformed_camera_yaml_is_a_result_not_a_traceback(
    episode, tmp_path, monkeypatch
):
    # The camera unit is parsed on the REUSE path too (its intrinsics are
    # an input to eef.npz, not just to the run), so a hand-edit gone wrong
    # reaches every eef run - and this stage owes a result either way.
    (tmp_path / "config" / "cameras" / UNIT_A / "camera.yaml").write_text(
        "intrinsics: [unclosed\n"
    )
    _fake_run_clip(monkeypatch, [])
    r = stages.run_stage("eef", episode)
    assert not r.ok and r.summary.startswith("left: ")


def test_an_eef_npz_without_a_sidecar_is_re_estimated(episode, monkeypatch):
    # Nothing records what produced it, so nothing can say it is still valid.
    calls = []
    _fake_run_clip(monkeypatch, calls)
    assert stages.run_stage("eef", episode).ok
    (episode / "left" / "eef.inputs.json").unlink()

    r = stages.run_stage("eef", episode)
    assert r.ok, r.summary
    assert len(calls) == 3
    assert "right: reused eef.npz" in r.summary


def test_completed_single_anchor_runs_and_reuses_with_fill_disclosure(
    episode, monkeypatch
):
    calls = []
    _fake_run_clip(monkeypatch, calls, n_valid_by_stem={"A": 1})
    for attempt in range(2):
        result = stages.run_stage("eef", episode)
        assert result.ok, result.summary
        assert (
            "left: completion filled 9/10 frames (90.0%); raw anchors 1/10"
            in result.summary
        )
        assert len(calls) == 2
        if attempt:
            assert "left: reused eef.npz (1/10 valid)" in result.summary
    record = json.loads((episode / "left" / layout.EEF_INPUTS).read_text())
    assert record["source_fingerprint"]["version"] == "eef_completion_v1"
    assert set(record["source_fingerprint"]["sources"]) >= {
        "completion.py",
        "state.py",
        "run.py",
    }
    assert record["completion"]["filled_frames"] == 9


def test_zero_anchors_fail_instead_of_publishing_a_reusable_artifact(
    episode, monkeypatch
):
    calls = []
    _fake_run_clip(monkeypatch, calls, n_valid_by_stem={"A": 0})
    result = stages.run_stage("eef", episode)
    assert not result.ok
    assert "no usable EEF samples" in result.summary
    assert not (episode / "left" / layout.EEF_INPUTS).exists()


def test_an_artifact_without_completed_output_is_refused(episode, monkeypatch):
    """An eef.npz with no value_completed has no consumer output; the stage
    names the file and the remedy instead of reusing or reading around it."""
    calls = []
    _fake_run_clip(monkeypatch, calls)
    assert stages.run_stage("eef", episode).ok
    out = episode / "left" / layout.EEF_STATE
    with np.load(out) as data:
        payload = {
            key: data[key]
            for key in data.files
            if key != "value_completed" and not key.startswith("completion/")
        }
    np.savez_compressed(out, **payload)
    result = stages.run_stage("eef", episode)
    assert not result.ok
    assert "no value_completed channel" in result.summary
    assert "re-run the eef stage" in result.summary


@pytest.mark.parametrize("changed", generation._EEF_SOURCES)
def test_generating_source_change_invalidates_reuse_without_config_change(
    episode, tmp_path, monkeypatch, changed
):
    sources = tmp_path / "eef_source"
    shutil.copytree(generation._EEF_SOURCE_DIR, sources)
    monkeypatch.setattr(generation, "_EEF_SOURCE_DIR", sources)
    calls = []
    _fake_run_clip(monkeypatch, calls)
    assert stages.run_stage("eef", episode).ok
    before = json.loads((episode / "left" / layout.EEF_INPUTS).read_text())
    with (sources / changed).open("a") as stream:
        stream.write("\n# changed generating source\n")
    result = stages.run_stage("eef", episode)
    assert result.ok, result.summary
    assert len(calls) == 4
    after = json.loads((episode / "left" / layout.EEF_INPUTS).read_text())
    assert before["sha256"] != after["sha256"]
    assert before["input_values"] == after["input_values"]
    assert stages.run_stage("eef", episode).ok
    assert len(calls) == 4


# Where the eef stage enters kiwi.eef: run_clip estimates, read_eef_state
# reads a reused eef.npz back.
EEF_STAGE_ENTRIES = ("kiwi.eef.run", "kiwi.eef.state")


def _kiwi_eef_import_closure(entries) -> set[str]:
    """Source files, relative to kiwi/eef, of every kiwi.eef module that
    ``entries`` reach through kiwi.eef imports alone.

    Every import statement counts, including the lazy ones inside functions
    (the estimator registry and the dot detector load that way), and
    importing a submodule counts as importing its parent package too.
    """
    root = generation._EEF_SOURCE_DIR

    def source(module: str) -> Path | None:
        if module != "kiwi.eef" and not module.startswith("kiwi.eef."):
            return None
        base = root.joinpath(*module.split(".")[2:])
        for path in (base / "__init__.py", base.with_suffix(".py")):
            if path.is_file():
                return path
        return None

    def reached(name: str) -> list[Path]:
        parts = name.split(".")
        found = (source(".".join(parts[:i])) for i in range(1, len(parts) + 1))
        return [path for path in found if path is not None]

    seen: set[Path] = set()
    todo = [path for name in entries for path in reached(name)]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        package = ".".join(("kiwi", "eef", *path.relative_to(root).parent.parts))
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
                names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            else:
                continue
            todo.extend(path for name in names for path in reached(name))
    return {path.relative_to(root).as_posix() for path in seen}


def test_the_eef_source_fingerprint_is_the_stage_import_closure():
    closure = _kiwi_eef_import_closure(EEF_STAGE_ENTRIES)
    hashed = set(generation.eef_source_fingerprint()["sources"])
    assert not closure - hashed, (
        f"kiwi.eef modules the eef stage imports but the reuse key does not "
        f"hash: {sorted(closure - hashed)}"
    )
    assert not hashed - closure, (
        f"kiwi.eef modules the reuse key hashes but the eef stage never "
        f"imports: {sorted(hashed - closure)}"
    )


def test_an_absent_tool_config_is_actionable(episode, tmp_path, monkeypatch):
    # An absent tool file still LOADS (an episode does not require every tool
    # config to exist), so the refusal has to come from the stage that
    # actually needs the design: block.
    (tmp_path / "config" / "gripper" / "chopsticks-v2" / "tool.yaml").unlink()
    _fake_run_clip(monkeypatch, [])
    r = stages.run_stage("eef", episode)
    assert not r.ok
    assert "no tool config at" in r.summary and "design:" in r.summary


def test_a_tool_file_missing_its_estimator_is_refused(episode, tmp_path, monkeypatch):
    # A PRESENT file, in contrast, must be valid: half a design: block is a
    # hand-edit gone wrong.
    doc = {"schema": "tool_v1", "design": {"marker": {"dictionary": "DICT_4X4_50"}}}
    (tmp_path / "config" / "gripper" / "chopsticks-v2" / "tool.yaml").write_text(
        yaml.safe_dump(doc, sort_keys=False)
    )
    _fake_run_clip(monkeypatch, [])
    r = stages.run_stage("eef", episode)
    assert not r.ok and "design.estimator is required" in r.summary


def test_a_gripper_unit_naming_no_file_is_refused(episode, monkeypatch):
    _write_manifest(episode, left={"gripper_unit": "chopstick-v2-m9"})
    _fake_run_clip(monkeypatch, [])
    r = stages.run_stage("eef", episode)
    assert not r.ok
    assert "chopstick-v2-m9" in r.summary and "names no config" in r.summary


def test_a_unit_built_for_another_tool_is_refused(episode, tmp_path, monkeypatch):
    # The file sits under chopsticks-v2/units/ but declares chopsticks-v3: a
    # miscopied unit, refused at load - the parent directory is authoritative.
    (
        tmp_path
        / "config"
        / "gripper"
        / "chopsticks-v2"
        / "units"
        / f"{GRIPPER_LEFT}.yaml"
    ).write_text(_unit_yaml(3, tool="chopsticks-v3"))
    _fake_run_clip(monkeypatch, [])
    r = stages.run_stage("eef", episode)
    assert not r.ok
    assert "chopsticks-v2" in r.summary and "chopsticks-v3" in r.summary


def test_an_unknown_session_key_is_refused_naming_its_own_layer(episode, monkeypatch):
    # The session block is strict, and validated per LAYER.
    _write_manifest(episode, left={"session": {"min_ara": 120}})
    _fake_run_clip(monkeypatch, [])
    r = stages.run_stage("eef", episode)
    assert not r.ok
    assert "arms.left.session" in r.summary
    assert "unknown session key(s): min_ara" in r.summary
    assert "did you mean 'min_area'?" in r.summary


def test_missing_intrinsics_is_actionable(episode, tmp_path, monkeypatch):
    (tmp_path / "config" / "cameras" / UNIT_A / "intrinsics.npz").unlink()
    _fake_run_clip(monkeypatch, [])
    r = stages.run_stage("eef", episode)
    assert not r.ok
    assert "no intrinsics artifact" in r.summary and "kiwi calib" in r.summary


def test_run_clip_errors_become_results(episode, monkeypatch):
    def boom(job, **kw):
        raise FileNotFoundError(f"no demuxed stream at {job.lens0_mp4}")

    monkeypatch.setattr(run_mod, "run_clip", boom)
    r = stages.run_stage("eef", episode)
    assert not r.ok and "no demuxed stream" in r.summary


def test_uninitialized_directory_is_a_result(tmp_path):
    r = stages.run_stage("eef", tmp_path)
    assert not r.ok and "kiwi init" in r.summary

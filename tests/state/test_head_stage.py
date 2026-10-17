"""Pipeline wiring for the optional post-fusion head-marker stage."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kiwi.episode.head import HeadPublication
from kiwi.episode.manifest import EpisodeConfig
from kiwi.episode.stages import REGISTRY, STAGES, run_stage


@pytest.fixture
def episode(tmp_path, monkeypatch):
    config = tmp_path / "config"
    (config / "gripper" / "chopsticks-v2").mkdir(parents=True)
    (config / "gripper" / "chopsticks-v2" / "tool.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "tool_v1",
                "name": "chopsticks-v2",
                "design": {
                    "estimator": "chopsticks",
                    "marker": {"dictionary": "DICT_4X4_50"},
                },
            }
        )
    )
    for unit in ("unit-a", "unit-b"):
        (config / "cameras" / unit).mkdir(parents=True)
    monkeypatch.setenv("KIWI_CONFIG", str(config))

    root = tmp_path / "episode"
    (root / "raw").mkdir(parents=True)
    for name in ("A.insv", "B.insv"):
        (root / "raw" / name).write_bytes(b"media")
    manifest = {
        "schema": "episode_v1",
        "name": "episode",
        "tool": "chopsticks-v2",
        "arms": {
            "left": {
                "camera": "unit-a",
                "recording": "raw/A.insv",
            },
            "right": {
                "camera": "unit-b",
                "recording": "raw/B.insv",
            },
        },
        "sync": {
            "method": "audio_gcc_phat",
            "origin": "right",
            "offsets": {"left": -0.1, "right": 0.0},
        },
    }
    (root / "episode.yaml").write_text(yaml.safe_dump(manifest))
    return root, manifest


def test_head_stage_is_immediately_after_state_and_dispatched():
    assert STAGES.index("head") == STAGES.index("state") + 1
    assert REGISTRY["head"] is not None


def test_absent_head_config_is_a_typed_successful_skip(episode):
    root, _ = episode
    result = run_stage("head", root)
    assert result.ok
    assert result.data == {"status": "skipped_no_config"}
    assert "no cameras.*.head_marker" in result.summary
    assert "no head/head_marker.tum will be produced" in result.summary


def test_head_input_resolution_refuses_a_missing_arm_sync_offset(episode):
    root, manifest = episode
    del manifest["sync"]["offsets"]["left"]
    (root / "episode.yaml").write_text(yaml.safe_dump(manifest))
    cfg = EpisodeConfig.load(root / "episode.yaml")

    from kiwi.episode.head import _resolve_inputs

    with pytest.raises(ValueError, match="sync offset for 'left'"):
        _resolve_inputs(root, cfg)


def test_configured_stage_publishes_without_touching_arm_state(episode, monkeypatch):
    root, manifest = episode
    ego = root / "raw" / "ego.mp4"
    ego.write_bytes(b"ego")
    manifest["cameras"] = {
        "ego": {
            "recording": "raw/ego.mp4",
            "head_marker": {
                "dictionary": "DICT_4X4_100",
                "id": 50,
                "size": 0.049,
            },
        }
    }
    manifest["sync"]["offsets"]["ego"] = -0.2
    (root / "episode.yaml").write_text(yaml.safe_dump(manifest))
    arm_state = {}
    for arm in ("left", "right"):
        path = root / arm / "scene" / "state_latent.npz"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"unchanged-{arm}".encode())
        arm_state[path] = path.read_bytes()

    report_path = root / "head" / "head_marker_report.json"
    trajectory_path = root / "head" / "head_marker.tum"

    def fake_publish(path: Path):
        assert path == root.resolve()
        return HeadPublication(
            status="computed_unvalidated",
            reason="",
            artifacts=(trajectory_path, report_path),
            reused=False,
            report={"output": {"rows": 321}, "segments": [{}, {}]},
        )

    import kiwi.episode.head as head_module

    monkeypatch.setattr(head_module, "publish_head_marker", fake_publish)
    result = run_stage("head", root)

    assert result.ok
    assert "321 observations in 2 segments" in result.summary
    assert result.data == {"status": "computed_unvalidated", "reused": False}
    assert all(path.read_bytes() == contents for path, contents in arm_state.items())


def test_a_configured_noncomputed_result_fails_only_the_head_stage(
    episode, monkeypatch
):
    root, manifest = episode
    (root / "raw" / "ego.mp4").write_bytes(b"ego")
    manifest["cameras"] = {
        "ego": {
            "recording": "raw/ego.mp4",
            "head_marker": {
                "dictionary": "DICT_4X4_100",
                "id": 50,
                "size": 0.049,
            },
        }
    }
    (root / "episode.yaml").write_text(yaml.safe_dump(manifest))

    import kiwi.episode.head as head_module

    monkeypatch.setattr(
        head_module,
        "publish_head_marker",
        lambda _path: HeadPublication(
            status="insufficient_observations",
            reason="detections=12 (need 200)",
            artifacts=(root / "head" / "head_marker_report.json",),
            reused=False,
            report={},
        ),
    )
    result = run_stage("head", root)
    assert not result.ok
    assert result.stage == "head"
    assert "detections=12" in result.summary

"""The core-owned `convert` stage: registered, and honest about failure.

The stage entry lives in kiwi.episode.stages (episode may import core, never
the reverse), lazily importing kiwi.core.camera.convert. The stage is
manifest-driven: the labels in episode.yaml are what name each stream's
directory, so the demuxed lenses land as `<label>/lens0.mp4` + `lens1.mp4`
(kiwi.core.layout) and never beside the recording in raw/. ffmpeg itself is
monkeypatched out - CI is hardware- and binary-free.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from kiwi.core.camera import convert as convert_mod
from kiwi.episode import stages


def _episode(tmp_path, arms=("left", "right"), cameras=(), name="ep1"):
    """A manifest-carrying episode with one raw .insv per arm.

    ``cameras`` entries are ``(label, filename)`` - a filename ending .insv
    gets demuxed like an arm's, anything else is a plain container.
    """
    ep = tmp_path / name
    (ep / "raw").mkdir(parents=True)
    doc = {"schema": "episode_v1", "name": name, "tool": "chopsticks-v2", "arms": {}}
    for i, label in enumerate(arms):
        clip = f"VID_00{i}_{label}.insv"
        (ep / "raw" / clip).write_bytes(b"insv")
        doc["arms"][label] = {
            "camera": f"unit-{label}",
            "recording": f"raw/{clip}",
        }
    if cameras:
        doc["cameras"] = {}
        for label, clip in cameras:
            (ep / "raw" / clip).write_bytes(b"media")
            doc["cameras"][label] = {"recording": f"raw/{clip}"}
    (ep / "episode.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    return ep


def _fake_convert(monkeypatch, calls=None):
    def fake(inp: Path, out0: Path, out1: Path, force: bool = False):
        if calls is not None:
            calls.append(force)
        out0.parent.mkdir(parents=True, exist_ok=True)
        out0.write_bytes(b"lens0")
        out1.write_bytes(b"lens1")
        return out0, out1

    monkeypatch.setattr(convert_mod, "convert", fake)


def test_convert_stage_is_registered():
    assert stages.REGISTRY["convert"] is not None


def test_an_uninitialized_episode_reports_not_crashes(tmp_path):
    # No manifest means no stream labels, so nothing says where a demux
    # should land - the answer is `kiwi init`, not a guess.
    r = stages.run_stage("convert", tmp_path)
    assert isinstance(r, stages.StageResult)
    assert r.stage == "convert" and not r.ok
    assert "kiwi init" in r.summary


def test_pipeline_skips_not_requested_stages_instead_of_dying(tmp_path, monkeypatch):
    # A successful convert lets the chain run state (which refuses: no scene yet).
    # `ground` is not_requested by the default profile, but narrowing the range
    # to that one stage is a direct instruction: the real entry runs and refuses
    # on this episode (unknown camera units, no state products).
    ep = _episode(tmp_path)
    _fake_convert(monkeypatch)
    results = stages.run_pipeline(ep, from_stage="convert", to_stage="state")
    assert [r.stage for r in results] == ["convert", "state"]
    assert results[0].ok and not results[1].ok  # state requires a published scene
    [r] = stages.run_pipeline(ep, from_stage="ground", to_stage="ground")
    assert r.stage == "ground" and not r.ok and r.summary
    direct = stages.run_stage("ground", ep)
    assert not direct.ok and direct.summary == r.summary


def test_a_failed_convert_halts_before_the_later_stages(tmp_path):
    # The mirror image: convert fails on a bare directory, and the run stops
    # there. Later stages decode what convert produced, so there is nothing
    # for them to do.
    results = stages.run_pipeline(tmp_path, from_stage="convert")
    assert [r.stage for r in results] == ["convert"]
    assert not results[0].ok


def test_demuxes_every_declared_stream_into_its_directory(tmp_path, monkeypatch):
    # Both arms AND an extra .insv stream (a third-person X5); the ego GO3
    # mp4 needs no demux and is only counted.
    ep = _episode(
        tmp_path,
        cameras=[("tripod", "VID_009_tripod.insv"), ("ego_video", "GO_1.mp4")],
    )
    _fake_convert(monkeypatch)
    r = stages.run_stage("convert", ep)
    assert r.ok
    assert "3 stream(s) demuxed" in r.summary
    assert "1 plain-container stream(s)" in r.summary
    assert len(r.artifacts) == 6
    for label in ("left", "right", "tripod"):
        assert (ep / label / "lens0.mp4").exists()
        assert (ep / label / "lens1.mp4").exists()
    # raw/ holds exactly what it held before the stage ran.
    assert all(p.parent.name == "raw" for p in (ep / "raw").iterdir())
    assert not any(
        p.suffix == ".mp4" and "lens" in p.name for p in (ep / "raw").iterdir()
    )


def test_fresh_outputs_reused_stale_redemuxed(tmp_path, monkeypatch):
    import os

    ep = _episode(tmp_path, arms=("left",))
    clip = ep / "raw" / "VID_000_left.insv"
    calls: list[bool] = []
    _fake_convert(monkeypatch, calls)
    lens0 = ep / "left" / "lens0.mp4"
    lens1 = ep / "left" / "lens1.mp4"

    r = stages.run_stage("convert", ep)  # first run: demux
    assert r.ok and "1 stream(s) demuxed, 0 reused" in r.summary and len(calls) == 1

    # Outputs newer than the clip -> reused, no new demux call.
    os.utime(clip, (1000, 1000))
    for o in (lens0, lens1):
        os.utime(o, (2000, 2000))
    r = stages.run_stage("convert", ep)
    assert r.ok and "0 stream(s) demuxed, 1 reused" in r.summary and len(calls) == 1

    # Clip re-copied (newer than outputs) -> stale, re-demuxed with force.
    os.utime(clip, (3000, 3000))
    r = stages.run_stage("convert", ep)
    assert r.ok and "1 stream(s) demuxed, 0 reused" in r.summary
    assert len(calls) == 2 and calls[-1] is True


def test_a_missing_recording_is_a_result_not_a_traceback(tmp_path, monkeypatch):
    # This stage loads the manifest with check_exists=False, so a recording
    # the manifest names but raw/ does not hold gets past the load. With the
    # demux already on disk, stat()ing the missing clip for the freshness test
    # would end the run as a FileNotFoundError - the one thing the stage's
    # contract says it never does, and the shape a pruned raw/ takes.
    ep = _episode(tmp_path, arms=("left",))
    _fake_convert(monkeypatch)
    assert stages.run_stage("convert", ep).ok
    (ep / "raw" / "VID_000_left.insv").unlink()

    r = stages.run_stage("convert", ep)
    assert isinstance(r, stages.StageResult) and not r.ok
    assert "left: no recording at" in r.summary
    assert "VID_000_left.insv" in r.summary


def test_demux_failure_is_a_result_not_an_exception(tmp_path, monkeypatch):
    ep = _episode(tmp_path, arms=("left",))

    def broken_convert(inp, out0, out1, force=False):
        raise RuntimeError("ffmpeg not found on PATH")

    monkeypatch.setattr(convert_mod, "convert", broken_convert)
    r = stages.run_stage("convert", ep)
    assert not r.ok
    assert "VID_000_left.insv" in r.summary and "ffmpeg" in r.summary
    assert "left:" in r.summary  # the failing stream is named by label


def test_conversion_record_binds_raw_and_lenses_and_is_unchanged_on_reuse(
    tmp_path, monkeypatch
):
    import json

    from kiwi.core.digest import digest_bytes

    episode = _episode(tmp_path, arms=("left",))
    _fake_convert(monkeypatch)
    assert stages.run_stage("convert", episode).ok
    sidecar = episode / "left" / "convert.inputs.json"
    before = sidecar.read_bytes()
    record = json.loads(before)
    raw = next((episode / "raw").iterdir())
    assert record["inputs"]["recording"]["digest"] == digest_bytes(raw)
    assert record["producer"]["algorithm"] == "ffmpeg_demux"
    for i in range(2):
        ref = record["outputs"][f"lens{i}"]
        assert ref["path"] == f"lens{i}.mp4"
        assert ref["digest"] == digest_bytes(sidecar.parent / ref["path"])
    assert stages.run_stage("convert", episode).ok
    assert sidecar.read_bytes() == before


def test_conversion_record_write_error_is_reported(tmp_path, monkeypatch):
    episode = _episode(tmp_path, arms=("left",))
    _fake_convert(monkeypatch)

    def unwritable(*args, **kwargs):
        raise PermissionError("read only")

    monkeypatch.setattr(stages, "_write_inputs", unwritable)
    result = stages.run_stage("convert", episode)
    assert not result.ok and "read only" in result.summary

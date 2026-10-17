"""The sync stage: manifest-driven, writing the clock into episode.yaml.

Stream discovery is not this stage's job - the manifest names which clip
belongs to which arm, and `kiwi init` is what establishes that (see
tests/episode/test_init.py). What is tested here is that the stage honours the
manifest rather than the clips' own names, persists into the manifest's `sync:`
block, and turns every refusal into a result instead of an exception.

The estimator itself is monkeypatched out: the recipe is covered by
test_audio_sync.py and the algebra by test_clock.py, so CI stays binary-free.

Run: pixi run test
"""

from __future__ import annotations

import pytest
import yaml

from kiwi.core.clock import GlobalClock
from kiwi.episode import stages
from kiwi.episode.manifest import MANIFEST_NAME
from kiwi.sync import ClockEstimate, SyncGateError
from kiwi.sync.audio import EXTRACTION

UNIT_A, UNIT_B = "insta360-X5-001", "insta360-X5-002"
UNIT_EGO = "insta360-GO3-001"

# Recordings live flat in raw/ (kiwi.core.layout); the tag in the filename is
# what the operator typed at import, and here it is wrong on both clips.
CLIP_LEFT = "A_right.insv"  # the LEFT arm's clip, tagged _right
CLIP_RIGHT = "B_left.insv"  # the RIGHT arm's clip, tagged _left

# The minimal valid tool_v1 design. Sync reads no tool fact, but a PRESENT
# tool file must be valid tool_v1 or the manifest refuses to load.
TOOL_CONFIG = (
    "schema: tool_v1\n"
    "design:\n"
    "  estimator: chopsticks\n"
    "  marker: {dictionary: DICT_4X4_50}\n"
)


@pytest.fixture
def episode(tmp_path, monkeypatch):
    """An initialized episode whose filenames contradict the manifest.

    A swapped-tag episode in the flat-``raw/`` layout: the LEFT
    arm's clip is the one tagged ``_right``, because `kiwi init` bound the
    arms by embedded serial and the tag the operator typed at import was
    wrong. Every test here inherits it, so any regression to labelling by
    name shows up as a swapped clock rather than passing quietly.
    """
    cfg = tmp_path / "config"
    for unit in (UNIT_A, UNIT_B, UNIT_EGO):
        (cfg / "cameras" / unit).mkdir(parents=True)
    (cfg / "gripper" / "chopsticks-v2").mkdir(parents=True)
    (cfg / "gripper" / "chopsticks-v2" / "tool.yaml").write_text(TOOL_CONFIG)
    monkeypatch.setenv("KIWI_CONFIG", str(cfg))

    ep = tmp_path / "ep1"
    (ep / "raw").mkdir(parents=True)
    for name in (CLIP_LEFT, CLIP_RIGHT):
        (ep / "raw" / name).write_bytes(b"media")
    return ep


def _manifest(ep, *, arms=None, cameras=None, sync=None, extra_text=""):
    doc = {
        "schema": "episode_v1",
        "name": "ep1",
        "tool": "chopsticks-v2",
        "arms": arms
        or {
            "left": {
                "camera": UNIT_A,
                "recording": f"raw/{CLIP_LEFT}",
            },
            "right": {
                "camera": UNIT_B,
                "recording": f"raw/{CLIP_RIGHT}",
            },
        },
    }
    if cameras:
        doc["cameras"] = cameras
    if sync:
        doc["sync"] = sync
    (ep / MANIFEST_NAME).write_text(extra_text + yaml.safe_dump(doc, sort_keys=False))
    return ep / MANIFEST_NAME


def _fake_estimate(monkeypatch, offsets, origin, *, capture=None):
    def fake(arms, extras=None, *, params=None):
        if capture is not None:
            capture.append((dict(arms), dict(extras or {})))
        return ClockEstimate(
            clock=GlobalClock(origin=origin, offsets_s=offsets).validate(),
            method="audio_gcc_phat",
            reports=("(stub)",),
            loop_residuals={},
        )

    monkeypatch.setattr("kiwi.sync.resolve_clock", fake)


def test_sync_stage_is_registered():
    assert stages.REGISTRY["sync"] is not None
    assert stages.STAGES[0] == "sync", "sync must run before convert strips the audio"


# --- the manifest is the authority ------------------------------------------


def test_an_uninitialized_directory_is_refused_with_the_init_route(episode):
    r = stages.run_stage("sync", episode)
    assert not r.ok
    assert "no episode.yaml" in r.summary
    assert "kiwi init" in r.summary


def test_streams_come_from_the_manifest_not_the_filenames(episode, monkeypatch):
    # The headline. raw/B_left.insv is the clip the manifest binds to `right`.
    # Labelling by name would hand the estimator the opposite pairing.
    _manifest(episode)
    seen = []
    _fake_estimate(monkeypatch, {"left": 0.0, "right": -2.5}, "left", capture=seen)
    assert stages.run_stage("sync", episode).ok

    arms, extras = seen[0]
    assert set(arms) == {"left", "right"} and not extras
    assert arms["left"].path.name == CLIP_LEFT, "left must be the clip tagged _right"
    assert arms["right"].path.name == CLIP_RIGHT


def test_each_stream_carries_its_registry_unit(episode, monkeypatch):
    # camera_dir is what lets the A/V correction resolve a real av_sync table,
    # which is what makes a mixed-model pair measurable at all.
    _manifest(episode)
    seen = []
    _fake_estimate(monkeypatch, {"left": 0.0, "right": -2.5}, "left", capture=seen)
    stages.run_stage("sync", episode)
    arms, _ = seen[0]
    assert arms["left"].camera_dir.name == UNIT_A
    assert arms["right"].camera_dir.name == UNIT_B


def test_an_extra_stream_is_passed_through_as_an_extra(episode, monkeypatch):
    (episode / "raw" / "GO_ego.mp4").write_bytes(b"media")
    _manifest(
        episode, cameras={"ego": {"camera": UNIT_EGO, "recording": "raw/GO_ego.mp4"}}
    )
    seen = []
    _fake_estimate(
        monkeypatch, {"left": -1.0, "right": -3.0, "ego": 0.0}, "ego", capture=seen
    )
    assert stages.run_stage("sync", episode).ok
    arms, extras = seen[0]
    assert set(arms) == {"left", "right"} and set(extras) == {"ego"}
    assert extras["ego"].camera_dir.name == UNIT_EGO


# --- persistence ------------------------------------------------------------


def test_the_clock_lands_in_the_manifest_and_keeps_its_comments(episode, monkeypatch):
    header = "# the clip tagged _right holds the LEFT arm - verified by serial\n"
    m = _manifest(episode, extra_text=header)
    _fake_estimate(monkeypatch, {"left": 0.0, "right": -2.5}, "left")

    r = stages.run_stage("sync", episode)
    assert r.ok and r.artifacts == [m]
    doc = yaml.safe_load(m.read_text())
    assert doc["sync"]["method"] == "audio_gcc_phat"
    assert doc["sync"]["origin"] == "left"
    assert doc["sync"]["offsets"] == {"left": 0.0, "right": -2.5}
    # A machine write must not cost the operator their account of the capture.
    assert "verified by serial" in m.read_text()
    assert not (episode / "sync.yaml").exists(), "the clock lives only in episode.yaml"


def test_the_written_clock_satisfies_the_core_contract(episode, monkeypatch):
    m = _manifest(episode)
    _fake_estimate(monkeypatch, {"left": 0.0, "right": -2.5}, "left")
    stages.run_stage("sync", episode)
    doc = yaml.safe_load(m.read_text())["sync"]
    GlobalClock(origin=doc["origin"], offsets_s=doc["offsets"]).validate()


def test_explicit_resync_replaces_a_complete_saved_clock(episode, monkeypatch):
    manifest = _manifest(
        episode,
        sync={
            "method": "manual",
            "origin": "left",
            "offsets": {"left": 0.0, "right": -2.5},
        },
    )
    calls = []
    _fake_estimate(monkeypatch, {"left": -4.0, "right": 0.0}, "right", capture=calls)

    result = stages.run_stage("sync", episode, resync=True)
    assert result.ok and len(calls) == 1
    doc = yaml.safe_load(manifest.read_text())["sync"]
    assert doc == {
        "method": "audio_gcc_phat",
        "origin": "right",
        "offsets": {"left": -4.0, "right": 0.0},
    }


def test_incomplete_saved_clock_requires_explicit_resync(episode, monkeypatch):
    (episode / "raw" / "GO_ego.mp4").write_bytes(b"media")
    manifest = _manifest(
        episode,
        cameras={"ego": {"camera": UNIT_EGO, "recording": "raw/GO_ego.mp4"}},
        sync={
            "method": "audio_gcc_phat",
            "origin": "left",
            "offsets": {"left": 0.0, "right": -2.5},
        },
    )
    before = manifest.read_bytes()
    calls = []
    _fake_estimate(
        monkeypatch, {"left": -1.0, "right": -3.0, "ego": 0.0}, "ego", capture=calls
    )

    result = stages.run_stage("sync", episode)
    assert not result.ok and "--resync" in result.summary
    assert not calls
    assert manifest.read_bytes() == before
    assert not (episode / "sync.inputs.json").exists()

    result = stages.run_stage("sync", episode, resync=True)
    assert result.ok and len(calls) == 1
    assert set(yaml.safe_load(manifest.read_text())["sync"]["offsets"]) == {
        "left",
        "right",
        "ego",
    }


@pytest.mark.parametrize(
    "dependency", ["task_window", "curation", "local_curation", "malformed_curation"]
)
def test_explicit_resync_changes_the_clock_and_leaves_windows_human_managed(
    episode, monkeypatch, dependency
):
    import json

    manifest = _manifest(episode)
    _fake_estimate(monkeypatch, {"left": 0.0, "right": -2.0}, "left")
    assert stages.run_stage("sync", episode).ok
    sidecar = episode / "sync.inputs.json"
    inputs_before = sidecar.read_bytes()
    doc = yaml.safe_load(manifest.read_text())
    curation = None
    curation_before = None
    if dependency == "task_window":
        doc["task_window"] = [12.0, 13.0]
    else:
        directory = (
            episode if dependency == "local_curation" else episode.parent
        ) / "curation"
        directory.mkdir()
        curation = directory / "kept.json"
        curation.write_text(
            "{broken"
            if dependency == "malformed_curation"
            else json.dumps(
                {
                    "schema": "curation_v1",
                    "episodes": [
                        {"source": episode.name, "keep_windows": [[12.0, 13.0]]}
                    ],
                }
            )
        )
        curation_before = curation.read_bytes()
    (episode / "raw" / "ego.mp4").write_bytes(b"ego audio")
    doc["cameras"] = {"ego": {"camera": UNIT_EGO, "recording": "raw/ego.mp4"}}
    manifest.write_text(yaml.safe_dump(doc))
    _fake_estimate(monkeypatch, {"left": -10.0, "right": -12.0, "ego": 0.0}, "ego")

    result = stages.run_stage("sync", episode, resync=True)
    assert result.ok, result.summary
    saved = yaml.safe_load(manifest.read_text())
    assert saved["sync"]["offsets"] == {"left": -10.0, "right": -12.0, "ego": 0.0}
    assert sidecar.read_bytes() != inputs_before
    if dependency == "task_window":
        assert saved["task_window"] == [12.0, 13.0]
    else:
        assert curation.read_bytes() == curation_before


def test_absent_clock_is_computed_without_consulting_an_older_sidecar(
    episode, monkeypatch
):
    manifest = _manifest(episode, extra_text="task_window: [12.0, 13.0]\n")
    _fake_estimate(monkeypatch, {"left": 0.0, "right": -2.0}, "left")
    assert stages.run_stage("sync", episode).ok
    doc = yaml.safe_load(manifest.read_text())
    del doc["sync"]
    manifest.write_text(yaml.safe_dump(doc))
    _fake_estimate(monkeypatch, {"left": -3.0, "right": 0.0}, "right")

    result = stages.run_stage("sync", episode)
    assert result.ok, result.summary
    doc = yaml.safe_load(manifest.read_text())
    assert doc["sync"]["offsets"] == {"left": -3.0, "right": 0.0}
    assert doc["task_window"] == [12.0, 13.0]


# --- refusal ----------------------------------------------------------------


def test_gate_failure_is_a_result_and_leaves_the_manifest_alone(episode, monkeypatch):
    m = _manifest(episode)
    before = m.read_text()

    def boom(arms, extras=None, *, params=None):
        raise SyncGateError(
            "audio sync (left vs right) gates FAILED: vote_margin",
            failed_gates=("vote_margin",),
        )

    monkeypatch.setattr("kiwi.sync.resolve_clock", boom)
    r = stages.run_stage("sync", episode)
    assert not r.ok and "vote_margin" in r.summary
    assert m.read_text() == before, "a refused clock must not be persisted"


def test_a_refused_clock_halts_the_pipeline(episode, monkeypatch):
    _manifest(episode)

    def boom(arms, extras=None, *, params=None):
        raise SyncGateError("gates FAILED: vote_margin", failed_gates=("vote_margin",))

    monkeypatch.setattr("kiwi.sync.resolve_clock", boom)
    results = stages.run_pipeline(episode)
    assert [r.stage for r in results] == ["sync"], "convert ran despite no clock"


def test_missing_ffmpeg_is_a_result_not_a_traceback(episode, monkeypatch):
    # The stage's half of the contract; that kiwi.sync really raises RuntimeError
    # here - and not subprocess's own exceptions - is pinned in
    # test_audio_sync.py's subprocess-boundary section.
    _manifest(episode)

    def boom(arms, extras=None, *, params=None):
        raise RuntimeError("ffmpeg not found on PATH")

    monkeypatch.setattr("kiwi.sync.resolve_clock", boom)
    r = stages.run_stage("sync", episode)
    assert not r.ok and "ffmpeg" in r.summary


def test_a_broken_manifest_is_a_result_not_a_traceback(episode):
    (episode / MANIFEST_NAME).write_text("schema: episode_v1\narms: {left: {}}\n")
    r = stages.run_stage("sync", episode)
    assert not r.ok and "arms.left.camera is required" in r.summary


def test_a_missing_clip_is_a_result_not_a_traceback(episode):
    _manifest(episode)
    (episode / "raw" / CLIP_LEFT).unlink()
    r = stages.run_stage("sync", episode)
    assert not r.ok and str(episode / "raw" / CLIP_LEFT) in r.summary


X5_FPS, GO_FPS = 120000 / 1001, 30000 / 1001


def _stub_estimator(monkeypatch, *, other_fps):
    """Stub only the DECODE, so the registry lookup below runs for real.

    ``other_fps`` picks the branch of `resolve_av_correction`: a differing mode
    forces the A/V table lookup, an identical one cancels and never opens the
    table - but reads each unit's model first, either way.
    """
    from kiwi.sync import audio

    def probe(path, fps):
        return audio.MediaProbe(
            path=path,
            mode=audio.RecordingMode(1920, 1920, fps),
            video_duration_s=10.0,
            audio_codec="aac",
            audio_sample_rate=48_000,
            audio_channels=1,
        )

    def fake_raw(
        ref, other, *, params, ffmpeg="ffmpeg", ffprobe="ffprobe", _cache=None
    ):
        return audio.RawOffset(
            offset_s=-2.5,
            voted_offset_s=-2.5,
            votes=100,
            runner_up=2,
            onsets_matched=5,
            onsets_total=8,
            gates={g: True for g in audio.GATES},
            diagnostics={},
            ref=probe(ref.path, X5_FPS),
            other=probe(other.path, other_fps),
            params=params,
        )

    monkeypatch.setattr(audio, "estimate_raw_offset", fake_raw)


def test_a_dangling_av_sync_pointer_is_a_result_not_a_traceback(
    episode, tmp_path, monkeypatch
):
    """camera.yaml naming an av_sync table that does not exist: the registry
    raises FileNotFoundError for it, and the stage reports it as a result."""
    (tmp_path / "config" / "cameras" / UNIT_A / "camera.yaml").write_text(
        yaml.safe_dump({"model": "Insta360 X5", "av_sync": "av-sync-2026-01-01.yaml"})
    )
    (tmp_path / "config" / "cameras" / UNIT_B / "camera.yaml").write_text(
        yaml.safe_dump({"model": "Insta360 GO 3"})
    )
    _manifest(episode)
    _stub_estimator(monkeypatch, other_fps=GO_FPS)
    r = stages.run_stage("sync", episode)
    assert not r.ok
    assert "av-sync-2026-01-01.yaml" in r.summary


def test_an_extra_stream_without_a_unit_is_refused_before_any_decode(
    episode, tmp_path, monkeypatch
):
    """What `kiwi init` writes for a GO3 .mp4 no --camera named. The stage
    refuses it before decoding the arm pair, naming the `kiwi init` remedy."""
    for unit in (UNIT_A, UNIT_B):
        (tmp_path / "config" / "cameras" / unit / "camera.yaml").write_text(
            yaml.safe_dump({"model": "Insta360 X5"})
        )
    (episode / "raw" / "C_ego_video.mp4").write_bytes(b"media")
    _manifest(episode, cameras={"ego_video": {"recording": "raw/C_ego_video.mp4"}})
    _stub_estimator(monkeypatch, other_fps=X5_FPS)
    from kiwi.sync import audio

    decoded = []
    stub = audio.estimate_raw_offset
    monkeypatch.setattr(
        audio,
        "estimate_raw_offset",
        lambda *a, **kw: decoded.append(a) or stub(*a, **kw),
    )
    r = stages.run_stage("sync", episode)
    assert not r.ok and not decoded
    assert r.summary.startswith("ego_video: no camera unit in episode.yaml")
    assert "Set cameras.ego_video.camera in episode.yaml" in r.summary
    assert (
        f"`kiwi init --episode {episode} --left {UNIT_A} --right {UNIT_B} "
        "--camera ego_video=<unit> --force`" in r.summary
    )
    assert "kiwi sync pair" not in r.summary
    assert not (episode / "sync.inputs.json").exists()


@pytest.mark.parametrize("other_fps", [GO_FPS, X5_FPS], ids=["mixed-mode", "same-mode"])
def test_a_corrupt_camera_yaml_is_a_result_not_a_traceback(
    episode, tmp_path, monkeypatch, other_fps
):
    """The case a unit test of `read_av_delay` cannot reach:
    `resolve_av_correction` loads each unit for its MODEL before it ever looks
    up a table. The same-mode pair needs no A/V table at all and still reads
    the model."""
    (tmp_path / "config" / "cameras" / UNIT_A / "camera.yaml").write_text(
        "model: [unterminated\n"
    )
    _manifest(episode)
    _stub_estimator(monkeypatch, other_fps=other_fps)
    r = stages.run_stage("sync", episode)
    assert not r.ok
    assert UNIT_A in r.summary


def test_a_corrupt_episode_yaml_is_a_result_not_a_traceback(episode):
    """A hand edit that breaks the manifest YAML is a refusal with the file
    named."""
    _manifest(episode).write_text("schema: [unterminated\n")
    r = stages.run_stage("sync", episode)
    assert not r.ok
    assert MANIFEST_NAME in r.summary


@pytest.mark.parametrize(
    "exc", [PermissionError, yaml.YAMLError], ids=["oserror", "yamlerror"]
)
def test_an_untranslated_error_is_a_result_not_a_traceback(episode, monkeypatch, exc):
    # The last line of defence, as the track/register entries have it: an I/O
    # or parse failure the estimator did not translate still lands as a result.
    # kiwi.sync.audio translates both at the load; this is the belt behind it.
    _manifest(episode)

    def boom(arms, extras=None, *, params=None):
        raise exc(f"raw/{CLIP_LEFT}: permission denied")

    monkeypatch.setattr("kiwi.sync.resolve_clock", boom)
    r = stages.run_stage("sync", episode)
    assert not r.ok and "permission denied" in r.summary


@pytest.mark.parametrize(
    "text", ["model: [unterminated\n", "- model: Insta360 X5\n"], ids=["yaml", "list"]
)
def test_a_corrupt_camera_yaml_survives_the_whole_run_not_just_the_stage(
    episode, tmp_path, monkeypatch, text
):
    """`kiwi run` reads the registry BEFORE the stage body: RunRecord.start
    stamps each stage's declared inputs, and for sync those are the units'
    av_sync tables. So a corrupt camera.yaml is met in the record, upstream of
    every guard in `_sync_entry`, and run_status.json must still carry a failed
    sync entry. Only a run through `run_pipeline` reaches that code;
    `run_stage` does not.
    """
    (tmp_path / "config" / "cameras" / UNIT_A / "camera.yaml").write_text(text)
    _manifest(episode)
    _stub_estimator(monkeypatch, other_fps=GO_FPS)
    results = stages.run_pipeline(episode, from_stage="sync", to_stage="sync")
    assert [r.stage for r in results] == ["sync"]
    assert not results[0].ok
    import json

    doc = json.loads((episode / "run_status.json").read_text())
    [entry] = doc["stages"]
    assert entry["stage"] == "sync" and entry["status"] == "failed"


# --- what the run record keeps -----------------------------------------------


def _write_av_table(cfg_cameras, unit, *, model="Insta360 X5"):
    d = cfg_cameras / unit
    d.mkdir(parents=True, exist_ok=True)
    dated = "av-sync-2026-07-25.yaml"
    (d / "camera.yaml").write_text(yaml.safe_dump({"model": model, "av_sync": dated}))
    (d / dated).write_text(
        yaml.safe_dump(
            {
                "schema": "av_sync_v1",
                "unit": unit,
                "extraction": EXTRACTION,
                "modes": {"1920x1920@119.88": {"av_delay_s": 0.0}},
            }
        )
    )
    return d / dated


def test_stage_inputs_names_every_streams_av_table(episode, tmp_path):
    """Retuning a delay must move a recorded hash: the tables the sync stage
    can consume are attributed to it - arms and extra cameras alike."""
    from kiwi.episode.run_record import stage_inputs

    cfg_cameras = tmp_path / "config" / "cameras"
    t_a = _write_av_table(cfg_cameras, UNIT_A)
    t_b = _write_av_table(cfg_cameras, UNIT_B)
    t_ego = _write_av_table(cfg_cameras, "insta360-GO3-001", model="Insta360 GO 3")

    (episode / "raw" / "GO_ego.mp4").write_bytes(b"media")
    _manifest(
        episode,
        cameras={"ego": {"camera": "insta360-GO3-001", "recording": "raw/GO_ego.mp4"}},
    )
    assert stage_inputs("sync", episode) == sorted([t_a, t_b, t_ego])


def test_stage_inputs_attributes_nothing_when_no_tables_exist(episode):
    """Unit dirs with no av-sync table published (e.g. an X5-X5 pair)."""
    from kiwi.episode.run_record import stage_inputs

    _manifest(episode)
    assert stage_inputs("sync", episode) == []


def test_the_stage_records_the_clock_machine_readably(episode, monkeypatch):
    """The structured twin of the summary: offsets, method, origin, and each
    measured edge's A/V correction land in StageResult.data, so run_status
    can answer 'which rule fired' without parsing prose."""
    from kiwi.sync import SyncEdge

    _manifest(episode)

    def fake(arms, extras=None, *, params=None):
        return ClockEstimate(
            clock=GlobalClock(
                origin="right", offsets_s={"left": -2.5, "right": 0.0}
            ).validate(),
            method="audio_gcc_phat",
            reports=("(stub)",),
            loop_residuals={},
            edges=(
                SyncEdge(
                    kind="arms",
                    ref="left",
                    other="right",
                    offset_s=2.5,
                    av_correction_s=0.0,
                    av_provenance="identical recording mode (stub)",
                    modes=("1920x1920@119.88", "1920x1920@119.88"),
                ),
            ),
        )

    monkeypatch.setattr("kiwi.sync.resolve_clock", fake)
    r = stages.run_stage("sync", episode)
    assert r.ok
    assert r.data["method"] == "audio_gcc_phat"
    assert r.data["origin"] == "right"
    assert r.data["offsets_s"] == {"left": -2.5, "right": 0.0}
    [edge] = r.data["edges"]
    assert edge["kind"] == "arms" and edge["av_correction_s"] == 0.0
    assert "identical recording mode" in edge["av_provenance"]
    import json

    json.dumps(r.data)  # the record must be persistable as-is


def test_failed_explicit_resync_preserves_manifest_and_previous_generation(
    episode, tmp_path, monkeypatch
):
    import json

    manifest = _manifest(episode)
    table = _write_av_table(tmp_path / "config" / "cameras", UNIT_A)
    _fake_estimate(monkeypatch, {"left": 0.0, "right": -2.5}, "left")
    assert stages.run_stage("sync", episode).ok
    manifest_before = manifest.read_bytes()
    sidecar = episode / "sync.inputs.json"
    produced = json.loads(sidecar.read_text())
    table = episode / "config" / "cameras" / UNIT_A / table.name
    table.write_text(table.read_text().replace("av_delay_s: 0.0", "av_delay_s: 0.02"))
    calls = []

    def failed(*args, **kwargs):
        calls.append(True)
        raise SyncGateError(
            "ambiguous audio", evidence={"votes": [30, 29], "spread_s": float("inf")}
        )

    monkeypatch.setattr("kiwi.sync.resolve_clock", failed)
    assert not stages.run_stage("sync", episode, resync=True).ok
    record = json.loads(sidecar.read_text())
    assert record["status"] == "failed"
    assert record["sha256"] is None
    assert record["previous_generation"] == produced
    assert record["diagnostics"] == {"votes": [30, 29], "spread_s": None}
    assert manifest.read_bytes() == manifest_before

    failure_record = sidecar.read_bytes()
    result = stages.run_stage("sync", episode)
    assert result.ok and "reused" in result.summary
    assert len(calls) == 1
    assert sidecar.read_bytes() == failure_record
    assert manifest.read_bytes() == manifest_before

    assert not stages.run_stage("sync", episode, resync=True).ok
    assert len(calls) == 2
    assert json.loads(sidecar.read_text())["previous_generation"] == produced
    assert manifest.read_bytes() == manifest_before


def test_failed_sync_record_write_is_a_result(episode, monkeypatch):
    _manifest(episode)

    def failed(*args, **kwargs):
        raise SyncGateError("ambiguous audio")

    def unwritable(*args, **kwargs):
        raise PermissionError("read only")

    monkeypatch.setattr("kiwi.sync.resolve_clock", failed)
    monkeypatch.setattr(stages, "_write_inputs", unwritable)
    result = stages.run_stage("sync", episode)
    assert not result.ok
    assert "ambiguous audio" in result.summary
    assert "could not save sync diagnostics" in result.summary

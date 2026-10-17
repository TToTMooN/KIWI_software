"""`build_and_write` end to end: spec -> dataset dir, provenance, card.

Builds on the same synthetic artifacts as test_build_episode; needs a cv2
encoder for the source clip and ffmpeg for the writer's encode - both are in
the pixi env, and the test skips cleanly where they are absent.
"""

from __future__ import annotations

import dataclasses
import json
import shutil

import numpy as np
import pytest


cv2 = pytest.importorskip("cv2")

from kiwi.eef.completion import complete_series  # noqa: E402
from kiwi.eef.state import EefState, write_eef_state  # noqa: E402
from kiwi.export.source import ArmSource, EpisodeSource  # noqa: E402
from kiwi.export.write import build_and_write  # noqa: E402

FPS = 30.0
N = 60

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


pytestmark = pytest.mark.usefixtures("calibrated_dc_description")


def _make_clip(path, fps=FPS, size=64):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not vw.isOpened():
        pytest.skip("no mp4v encoder available in this build of cv2")
    img = np.zeros((size, size, 3), np.uint8)
    for i in range(N):
        img[:, :] = (i * 3 % 255, 128, 255 - i * 3 % 255)
        vw.write(img)
    vw.release()
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return _make_clip(tmp_path_factory.mktemp("bw") / "rec_lens0.mp4")


@pytest.fixture(scope="module")
def clip60(tmp_path_factory):
    return _make_clip(tmp_path_factory.mktemp("bw60") / "rec60_lens0.mp4", fps=60.0)


@pytest.fixture(scope="module")
def clip32(tmp_path_factory):
    return _make_clip(tmp_path_factory.mktemp("bw32") / "rec32_lens0.mp4", size=32)


DS_FRONT = [-0.023, 0.571, 499.64, 499.54, 962.30, 961.43]


def _write_intrinsics(path, ds_front=DS_FRONT):
    """A minimal kiwi.calib.intrinsics bundle: DS front lens + image size."""
    np.savez(
        path,
        ds_front=np.asarray(ds_front, dtype=np.float64),
        image_size=np.asarray([1920, 1920], dtype=np.int32),
    )
    return path


SCENE_FRAME = "shared_rear_metric_gravity_aligned"
MAP_SET = {"front": "attempts/2026-09-01-front", "rear": "attempts/2026-09-01-rear"}


def _episode_source(
    tmp_path,
    clip,
    name="vid7",
    frame=SCENE_FRAME,
    scene="kitchen-a",
    map_set=None,
) -> EpisodeSource:
    ep_dir = tmp_path / name
    (ep_dir / "left" / "scene").mkdir(parents=True)
    (ep_dir / "episode.yaml").write_text(f"schema: episode_v1\nname: {name}\n")
    header = (
        f"# t tx ty tz qx qy qz qw  (frame={frame} body=camera; "
        "t on episode global clock (origin=left))\n"
    )
    t = np.arange(0.0, 2.01, 0.1)
    (ep_dir / "left" / "scene" / "state.tum").write_text(
        header + "".join(f"{v:.9f} {v:.9f} 0 0 0 0 0 1\n" for v in t)
    )
    (ep_dir / "left" / "scene" / "state.inputs.json").write_text(
        json.dumps({"map_set": map_set or MAP_SET})
    )
    n = N
    completed, audit, _ = complete_series(np.full(n, 60.0), FPS)
    state = EefState(
        tool="chopsticks-v3",
        estimator="chopsticks",
        unit="deg",
        fps=FPS,
        source=clip.name,
        pose_mode="fixed_pose",
        pose_source="config/gripper/chopsticks-v3/units/chopstick-v3-m5.yaml",
        value=np.full(n, 60.0),
        rejected_value=np.full(n, np.nan),
        has_pose=np.ones(n, dtype=bool),
        n_detected=np.full(n, 6, dtype=np.int32),
        n_kept=np.full(n, 6, dtype=np.int32),
        error=np.asarray([""] * n, dtype=str),
        value_completed=completed,
        completion=audit,
    )
    eef = write_eef_state(ep_dir / "rec_eef.npz", state)
    arm = ArmSource(
        label="left",
        lens0=clip,
        state_tum=ep_dir / "left" / "scene" / "state.tum",
        state_inputs=ep_dir / "left" / "scene" / "state.inputs.json",
        eef_state=eef,
        sync_off=0.0,
        session_block={"angle_closed_deg": 20.0, "angle_open_deg": 100.0},
        tool="chopsticks-v3",
        description="test_calibrated_dc",
        camera="test-unit-001",
        # DS model only, no KB4: the DS model IS the calibration, so a unit
        # with no KB4 fit yet still publishes.
        intrinsics=_write_intrinsics(ep_dir / "intrinsics.npz"),
    )
    return EpisodeSource(
        name=name,
        episode_dir=ep_dir,
        arms=(arm,),
        task="stack the cubes",
        scene=scene,
    )


@needs_ffmpeg
def test_v21_dataset_provenance_and_card(tmp_path, clip):
    src = _episode_source(tmp_path, clip)
    out = tmp_path / "ds"
    result = build_and_write([src], out, format="v2.1", repo_id="me/demo")
    assert result.episodes == 1 and result.skipped == []
    assert result.card_error is None

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["features"]["observation.state"]["names"][0] == "left_ee_x"
    assert (out / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    mp4 = out / "videos" / "chunk-000" / "observation.images.left_wrist"
    assert (mp4 / "episode_000000.mp4").stat().st_size > 0

    prov = json.loads((out / "meta" / "export_provenance.json").read_text())
    assert prov["ee_frame"] == "tcp" and prov["frame"] == SCENE_FRAME
    ep0 = prov["episodes"][0]
    assert ep0["source"] == "vid7"
    assert "T_camera_tcp" in ep0["extrinsics"]["left"]
    assert ep0["tool_state"]["left"]["endpoints"] == [20.0, 100.0]
    assert "sha256" in ep0["source_stamps"]["episode_yaml"]

    card = (out / "README.md").read_text()
    assert "me/demo" in card and "`vid7`" in card


@needs_ffmpeg
def test_curation_fans_out_and_reports_skips(tmp_path, clip):
    src7 = _episode_source(tmp_path, clip, "vid7")
    src8 = _episode_source(tmp_path, clip, "vid8")
    cur = tmp_path / "takes.json"
    cur.write_text(
        json.dumps(
            {
                "schema": "curation_v1",
                "episodes": [
                    {"source": "vid7", "keep_windows": [[0.2, 0.8], [1.0, 1.6]]},
                    {"source": "vid8", "keep_windows": []},
                ],
            }
        )
    )
    out = tmp_path / "ds_cur"
    result = build_and_write([src7, src8], out, format="v2.1", curation=cur)
    assert result.episodes == 2
    assert result.skipped == [
        {"source": "vid8", "reason": "curated out (no kept spans)"}
    ]
    prov = json.loads((out / "meta" / "export_provenance.json").read_text())
    assert [e["span_index"] for e in prov["episodes"]] == [0, 1]
    assert prov["curation"]["file"] == "takes.json"
    # both output episodes chain back to the same source
    assert {e["source"] for e in prov["episodes"]} == {"vid7"}


def test_no_sources_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no episodes"):
        build_and_write([], tmp_path / "out", format="v2.1")


# One dataset carries ONE schema, and meta/info.json plus the provenance record
# take it from the FIRST episode. Everything they read from obs[0] has to agree
# across the list, or the dataset describes only its first episode. None of
# these need ffmpeg: the gate runs before the writer opens.
def test_mixed_frame_rates_are_refused(tmp_path, clip, clip60):
    a = _episode_source(tmp_path, clip, "vid7")
    b = _episode_source(tmp_path, clip60, "vid8")
    with pytest.raises(ValueError, match="disagree on the frame rate"):
        build_and_write([a, b], tmp_path / "ds", format="v2.1")


def test_mixed_world_frames_are_refused(tmp_path, clip):
    a = _episode_source(tmp_path, clip, "vid7", frame=SCENE_FRAME)
    b = _episode_source(tmp_path, clip, "vid8", frame="vio_world")
    with pytest.raises(ValueError, match="disagree on the world they live in"):
        build_and_write([a, b], tmp_path / "ds", format="v2.1")


# One dataset, one scene. Under the scene world the frame NAME is a single
# constant for every scene there is, so these three pin the gate that the
# frame string alone cannot hold: same frame + different scene is refused,
# same frame + different map set is refused, and the honest case still passes.
def test_two_scenes_cannot_share_one_dataset(tmp_path, clip):
    a = _episode_source(tmp_path, clip, "vid7", scene="kitchen-a")
    b = _episode_source(tmp_path, clip, "vid8", scene="lab-bench")
    with pytest.raises(ValueError, match="disagree on the world they live in") as e:
        build_and_write([a, b], tmp_path / "ds", format="v2.1")
    # The refusal has to NAME both scenes; "the frames differ" would be a lie
    # here, since they do not.
    assert "'kitchen-a'" in str(e.value) and "'lab-bench'" in str(e.value)
    assert SCENE_FRAME in str(e.value)


def test_one_scene_rebuilt_maps_cannot_share_one_dataset(tmp_path, clip):
    """Same scene, different published map set: a re-built map set freezes a
    new frame, so the poses are no longer comparable."""
    a = _episode_source(tmp_path, clip, "vid7")
    b = _episode_source(
        tmp_path, clip, "vid8", map_set={"front": "attempts/rebuilt", "rear": "x"}
    )
    with pytest.raises(ValueError, match="disagree on the world they live in") as e:
        build_and_write([a, b], tmp_path / "ds", format="v2.1")
    assert "map set" in str(e.value)


def test_one_scene_exports_as_one_dataset(tmp_path, clip):
    """The honest case: two episodes of one scene, one map set - accepted,
    and the shared world identity is stamped into the provenance."""
    a = _episode_source(tmp_path, clip, "vid7")
    b = _episode_source(tmp_path, clip, "vid8")
    try:
        build_and_write([a, b], tmp_path / "ds", format="v2.1")
    except ValueError as e:
        pytest.fail(f"two episodes of one scene were refused: {e}")
    except (RuntimeError, FileNotFoundError):
        pass  # no ffmpeg in this env; reaching the encoder is the point


@needs_ffmpeg
def test_the_world_identity_reaches_the_provenance(tmp_path, clip):
    out = tmp_path / "ds"
    build_and_write([_episode_source(tmp_path, clip)], out, format="v2.1")
    prov = json.loads((out / "meta" / "export_provenance.json").read_text())
    ws = prov["world_scene"]
    assert ws["frame"] == SCENE_FRAME and ws["scene"] == "kitchen-a"
    assert len(ws["map_set_digest"]) == 64


def test_a_marker_export_names_no_scene(tmp_path, clip):
    """The marker world is per-episode, so it has nothing to be comparable
    to - both scene fields stay None and the gate falls back to the frame."""
    from kiwi.export.build import build_episode

    src = _episode_source(tmp_path, clip, "vid7")
    marker = tmp_path / "vid7" / "left" / "register.tum"
    marker.write_text(
        "# t tx ty tz qx qy qz qw  (frame=marker body=camera; "
        "t on episode global clock (origin=left))\n"
        + "".join(f"{v:.9f} {v:.9f} 0 0 0 0 0 1\n" for v in np.arange(0.0, 2.01, 0.1))
    )
    arm = dataclasses.replace(src.arms[0], register_tum=marker)
    oe = build_episode(dataclasses.replace(src, arms=(arm,)), world_source="marker")
    assert oe.meta["world_scene"] == {
        "frame": "marker",
        "scene": None,
        "map_set_digest": None,
    }


def test_mixed_feature_schemas_are_refused(tmp_path, clip, clip32):
    """A resolution change is a column-shape change, so the schema gate catches
    it here - upstream of the v3.0 shard guard, which backstops the writer."""
    a = _episode_source(tmp_path, clip, "vid7")
    b = _episode_source(tmp_path, clip32, "vid8")
    with pytest.raises(ValueError, match="disagree on the feature schema"):
        build_and_write([a, b], tmp_path / "ds", format="v2.1")


def test_the_gates_do_not_fire_on_an_agreeing_list(tmp_path, clip):
    """A valid list must still reach the writer.

    The end-to-end cases above need ffmpeg and so skip in CI; this one gets
    there on the pre-writer path alone, so the gates stay pinned everywhere.
    """
    a = _episode_source(tmp_path, clip, "vid7")
    b = _episode_source(tmp_path, clip, "vid8")
    try:
        build_and_write([a, b], tmp_path / "ds", format="v2.1")
    except ValueError as e:  # only a gate raises this early
        pytest.fail(f"a valid episode list was refused: {e}")
    except (RuntimeError, FileNotFoundError):
        # Reaching the encoder at all is the point. Without ffmpeg on PATH
        # that is FileNotFoundError out of Popen (CI); with a broken one it is
        # the writer's RuntimeError. Either way the gates let the list past.
        pass


def test_changed_scene_placement_is_refused_even_when_map_set_is_unchanged(
    tmp_path, clip
):
    sources = [_episode_source(tmp_path, clip, name) for name in ("vid7", "vid8")]
    for i, source in enumerate(sources):
        for arm in source.arms:
            record = json.loads(arm.state_inputs.read_text())
            record["world_scene"] = {"scene_frame_digest": f"placement-{i}"}
            arm.state_inputs.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="scene_frame_digest"):
        build_and_write(sources, tmp_path / "dataset", format="v2.1")

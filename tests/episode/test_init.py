"""``kiwi init``: discovery, serial-verified arm binding, and the refusals.

The registry lookup is monkeypatched, so CI needs no telemetry parser and no
media - the binding logic is what is under test.

Recordings live flat in ``<episode>/raw/`` (kiwi.core.layout); extra streams
are labelled by the ``_<tag>`` filename suffix ``kiwi import`` stamps.

Run: pixi run test
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kiwi.core.camera.registry import CameraMatch
from kiwi.episode import init as init_mod
from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig

# Serials as they actually read off the two X5 bodies in config/cameras/.
A = ("insta360-X5-001", "IAHYA2511CWJEA")
B = ("insta360-X5-002", "IAHYA2511DVC37")

# Camera-shaped stems, as import leaves them: the camera's own stem plus the
# unit (or --label) tag. The sequence number is the needle the fake registry
# matches on.
CLIP_A = "VID_20260820_100000_00_001_left.insv"  # seq 001 -> body A
CLIP_B = "VID_20260820_100002_00_002_right.insv"  # seq 002 -> body B
EGO = "PRO_VID_20260820_100005_00_046_ego_video.mp4"  # tagged extra


def _registry(monkeypatch, mapping):
    """Resolve a clip to a unit by filename substring, standing in for telemetry."""

    def fake(video: Path, cameras_dir: Path):
        for needle, (unit, serial) in mapping.items():
            if needle in video.name:
                return CameraMatch(unit, serial, None)
        return None

    monkeypatch.setattr(init_mod, "resolve_camera_for_video", fake)


def _episode(tmp_path, files, name="ep1"):
    """An episode directory with ``files`` flat in its raw/."""
    ep = tmp_path / name
    (ep / "raw").mkdir(parents=True, exist_ok=True)
    for f in files:
        (ep / "raw" / f).write_bytes(b"media")
    return ep


# --- the headline: the filename does not decide anything ---------------------


def test_arms_bind_by_serial_even_when_the_filename_tags_are_swapped(
    tmp_path, monkeypatch
):
    # A swapped-tag episode in filename form: the body that recorded the
    # LEFT arm carries a clip tagged _right.
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_002": A, "00_001": B})  # tags lie; serials do not

    out, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
        tool="chopsticks-v2",
    )
    assert not problem
    doc = yaml.safe_load(out.read_text())
    assert doc["arms"]["left"]["camera"] == A[0]
    assert doc["arms"]["left"]["recording"] == f"raw/{CLIP_B}"
    assert doc["arms"]["right"]["camera"] == B[0]
    assert doc["arms"]["right"]["recording"] == f"raw/{CLIP_A}"


def test_the_header_records_the_serial_each_binding_came_from(tmp_path, monkeypatch):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_002": A, "00_001": B})
    out, _ = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "c",
    )
    text = out.read_text()
    assert f"serial {A[1]}" in text and f"serial {B[1]}" in text
    assert f"raw/{CLIP_B}" in text.split("schema:")[0]
    assert "EMBEDDED SERIAL" in text


def test_the_skeleton_loads_back_through_the_parser(tmp_path, monkeypatch):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    cfg_dir = tmp_path / "config"
    (cfg_dir / "cameras" / A[0]).mkdir(parents=True)
    (cfg_dir / "cameras" / B[0]).mkdir(parents=True)
    out, _ = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=cfg_dir / "cameras",
        tool="chopsticks-v2",
    )
    loaded = EpisodeConfig.load(out, check_exists=True, config_dir=cfg_dir)
    assert sorted(loaded.arms) == ["left", "right"]
    assert loaded.global_clock() is None, "init never writes a clock"


# --- discovery --------------------------------------------------------------


def test_discovery_reads_raw_only_so_derived_files_cannot_shadow_a_recording(
    tmp_path, monkeypatch
):
    # Proxies, overlays and demuxed lenses are media files too; they live in
    # the stream directories, which discovery never reads, so none of them
    # can shadow or be mistaken for a recording.
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    for name in ("lens0.mp4", "lens1.mp4", "proxy.mp4", "eef.mp4"):
        (ep / "left").mkdir(exist_ok=True)
        (ep / "left" / name).write_bytes(b"derived")
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    clips, problem = init_mod.discover_clips(ep, tmp_path / "cams")
    assert not problem
    assert {c.path.name for c in clips} == {CLIP_A, CLIP_B}


def test_appledouble_junk_is_skipped(tmp_path, monkeypatch):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B, f"._{CLIP_B}"])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    clips, problem = init_mod.discover_clips(ep, tmp_path / "cams")
    assert not problem
    assert {c.path.name for c in clips} == {CLIP_A, CLIP_B}


def test_a_missing_raw_directory_says_what_it_expected(tmp_path, monkeypatch):
    ep = tmp_path / "ep1"
    ep.mkdir()
    (ep / CLIP_A).write_bytes(b"media")  # flat in the episode root: not raw/
    _, problem = init_mod.discover_clips(ep, tmp_path / "cams")
    assert "no raw/" in problem and "kiwi import" in problem


def test_an_empty_raw_says_what_it_expected(tmp_path, monkeypatch):
    ep = tmp_path / "ep1"
    (ep / "raw").mkdir(parents=True)
    _, problem = init_mod.discover_clips(ep, tmp_path / "cams")
    assert "no recordings in" in problem and "flat" in problem


def test_the_label_is_the_filename_tag_or_the_whole_stem(tmp_path, monkeypatch):
    untagged = "PRO_VID_20260820_100005_00_046.mp4"
    ep = _episode(tmp_path, [CLIP_A, EGO, untagged])
    _registry(monkeypatch, {"00_001": A})
    clips, problem = init_mod.discover_clips(ep, tmp_path / "cams")
    assert not problem
    by_name = {c.path.name: c.label for c in clips}
    assert by_name[CLIP_A] == "left"  # the tag, though arms never use it
    assert by_name[EGO] == "ego_video"
    assert by_name[untagged] == "PRO_VID_20260820_100005_00_046"


def test_find_episode_dirs_lists_a_sitting(tmp_path):
    for name in ("vid1", "vid2"):
        _episode(tmp_path, [CLIP_A], name=name)
    (tmp_path / "not_an_episode").mkdir()
    (tmp_path / "no_raw" / "left").mkdir(parents=True)
    (tmp_path / "no_raw" / "left" / "a.insv").write_bytes(b"x")  # media, not in raw/
    (tmp_path / "loose.insv").write_bytes(b"x")
    assert [d.name for d in init_mod.find_episode_dirs(tmp_path)] == ["vid1", "vid2"]


# --- extra streams ----------------------------------------------------------


def test_a_clip_matching_neither_arm_becomes_an_extra_camera(tmp_path, monkeypatch):
    # The GO3 ego view: an mp4 with no readable telemetry serial. It needs no
    # device-specific rule, only "not one of the two named bodies".
    ep = _episode(tmp_path, [CLIP_A, CLIP_B, EGO])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    out, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
    )
    assert not problem
    doc = yaml.safe_load(out.read_text())
    assert doc["cameras"] == {"ego_video": {"recording": f"raw/{EGO}"}}
    assert "(extra stream, no arm)" in out.read_text()


def test_an_extra_stream_keeps_its_unit_when_the_serial_resolves(tmp_path, monkeypatch):
    ego = ("insta360-GO3-001", "GOSERIAL")
    ep = _episode(tmp_path, [CLIP_A, CLIP_B, EGO])
    _registry(monkeypatch, {"00_001": A, "00_002": B, "00_046": ego})
    out, _ = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
    )
    doc = yaml.safe_load(out.read_text())
    # Sync refuses an extra stream with no unit (it must establish the camera
    # model and A/V delay), so a unit resolved from the serial is written.
    assert doc["cameras"]["ego_video"]["camera"] == ego[0]


def test_two_extra_streams_sharing_a_tag_are_refused(tmp_path, monkeypatch):
    ep = _episode(
        tmp_path,
        [
            CLIP_A,
            CLIP_B,
            "PRO_VID_20260820_100005_00_046_ego_video.mp4",
            "PRO_VID_20260820_110005_00_047_ego_video.mp4",
        ],
    )
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    _, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
    )
    assert "share the label 'ego_video'" in problem


def test_a_reserved_tag_on_an_extra_stream_is_refused(tmp_path, monkeypatch):
    # A GO3 clip someone tagged _left: it cannot become an arm (no serial),
    # and the left/ stream directory holds the left arm's artifacts.
    ep = _episode(tmp_path, [CLIP_A, CLIP_B, "PRO_VID_20260820_100005_00_046_left.mp4"])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    _, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
    )
    assert "reserved" in problem and "'left'" in problem


# --- refusing bindings the serials cannot verify ----------------------------


def test_a_named_unit_with_no_clip_is_refused(tmp_path, monkeypatch):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    _, problem = init_mod.init_episode(
        ep,
        left="insta360-X5-009",
        right=B[0],
        config_dir=tmp_path / "cams",
    )
    assert "no clip resolves to --left insta360-X5-009" in problem
    # It shows what it did find.
    assert f"raw/{CLIP_A} -> insta360-X5-001" in problem
    assert "config/cameras/ has no camera.yaml carrying those serials" in problem


def test_two_clips_resolving_to_one_unit_are_refused(tmp_path, monkeypatch):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_001": A, "00_002": A})
    _, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
    )
    assert "both resolve to" in problem
    assert "cannot record two arms at once" in problem


def test_naming_one_unit_for_both_arms_is_refused(tmp_path, monkeypatch):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    _, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=A[0],
        config_dir=tmp_path / "cams",
    )
    assert "name the same unit" in problem


def test_a_clip_with_no_readable_serial_never_becomes_an_arm(tmp_path, monkeypatch):
    # It lands as an extra stream instead, and the missing arm is what is
    # reported.
    ep = _episode(tmp_path, [CLIP_A, "VID_20260820_100002_00_099_mystery.insv"])
    _registry(monkeypatch, {"00_001": A})
    _, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
    )
    assert f"no clip resolves to --right {B[0]}" in problem
    assert "no readable serial" in problem


# --- not clobbering hand-written context ------------------------------------


def test_an_existing_manifest_is_never_overwritten_by_default(tmp_path, monkeypatch):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    (ep / MANIFEST_NAME).write_text("schema: episode_v1\nnotes: hand written\n")

    out, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
    )
    assert out is None and "already exists" in problem
    assert "hand written" in (ep / MANIFEST_NAME).read_text()


def test_force_overwrites(tmp_path, monkeypatch):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    (ep / MANIFEST_NAME).write_text("schema: episode_v1\nnotes: hand written\n")
    out, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
        force=True,
    )
    assert not problem and "hand written" not in out.read_text()


# --- the skeleton's placeholders --------------------------------------------


def test_unset_fields_are_written_as_TODO(tmp_path, monkeypatch):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    out, _ = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
    )
    doc = yaml.safe_load(out.read_text())
    assert doc["description"] == "TODO" and doc["date"] == "TODO"
    assert doc["tool"] == "TODO"
    assert set(doc["arms"]["left"]) == {"camera", "recording"}
    # No anchor skeleton: the table marker belongs to the on-demand register
    # stage, and a normal capture has none. The header offers the shape,
    # commented, for the episodes recorded against a marker on purpose.
    assert "anchor" not in doc
    header = out.read_text().split("schema:")[0]
    assert "anchor: {marker: {id: <aruco id>, size: <metres>}}" in header
    assert "optional" in header and "register" in header


def test_each_arm_gets_a_commented_gripper_unit_placeholder(tmp_path, monkeypatch):
    # The skeleton cannot know the physical build; it marks the spot under each
    # arm as a comment, so the document itself is unchanged until it is filled.
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    out, _ = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
    )
    text = out.read_text()
    assert text.count("# gripper_unit: <unit>") == 2
    left = text.split("  left:\n")[1].split("  right:\n")[0]
    assert "# gripper_unit: <unit>" in left
    doc = yaml.safe_load(text)
    assert "gripper_unit" not in doc["arms"]["left"]
    assert "gripper_unit" not in doc["arms"]["right"]


@pytest.mark.parametrize(
    "clip",
    [
        # a hand-named clip with spaces whose recording: line passes 80 columns
        "VID_20260820_100000_00_001 VID_20260820_100000_00_001 "
        "VID_20260820_100000_00_001 VID_20260820_100000_00_001_left.insv",
        # a hand-named clip with no spaces: non-ASCII is dumped quoted and escaped
        "VID_20260820_100000_00_001_录像_左臂_二零二六年八月二十日_十点零分零秒_第一段_left.insv",
    ],
)
def test_the_placeholder_never_splits_a_folded_recording_path(
    tmp_path, monkeypatch, clip
):
    ego = "PRO_VID_20260820_100005_00_046_ego_video.mp4"
    ep = _episode(tmp_path, [clip, CLIP_B, ego])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    cfg = tmp_path / "config"
    (cfg / "cameras" / A[0]).mkdir(parents=True)
    (cfg / "cameras" / B[0]).mkdir(parents=True)
    out, problem = init_mod.init_episode(
        ep, left=A[0], right=B[0], config_dir=cfg / "cameras", tool="chopsticks-v2"
    )
    assert not problem
    text = out.read_text()
    assert text.count("# gripper_unit: <unit>") == 2
    doc = yaml.safe_load(text)
    assert doc["arms"]["left"]["recording"] == f"raw/{clip}"
    assert doc["cameras"] == {"ego_video": {"recording": f"raw/{ego}"}}
    EpisodeConfig.load(out, check_exists=True, config_dir=cfg)


@pytest.mark.parametrize("marker_id", [-1, True, "50"])
def test_head_marker_id_must_be_a_non_negative_integer(
    tmp_path, monkeypatch, marker_id
):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    out, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        head_marker_id=marker_id,
        config_dir=tmp_path / "cams",
    )
    assert out is None
    assert "--head-marker-id must be a non-negative integer" in problem
    assert not (ep / MANIFEST_NAME).exists()


# --- naming an extra stream's unit ------------------------------------------


def test_an_extra_streams_unit_can_be_named(tmp_path, monkeypatch):
    # A GO3 mp4 carries no telemetry serial, so nothing on disk supplies its
    # unit - and without one sync cannot establish its camera model or A/V
    # delay and refuses the stream, so init lets the operator name it.
    ego = ("insta360-GO3-001", None)
    ep = _episode(tmp_path, [CLIP_A, CLIP_B, EGO])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    out, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
        extra_units={"ego_video": ego[0]},
    )
    assert not problem
    doc = yaml.safe_load(out.read_text())
    assert doc["cameras"]["ego_video"]["camera"] == ego[0]


def test_head_marker_is_written_on_the_extra_camera_and_read_by_head(
    tmp_path, monkeypatch
):
    from kiwi.episode.head import configured_head_camera

    ep = _episode(tmp_path, [CLIP_A, CLIP_B, EGO])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    config = tmp_path / "config"
    out, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        head_marker_id=50,
        config_dir=config / "cameras",
    )
    assert not problem
    doc = yaml.safe_load(out.read_text())
    assert "anchor" not in doc
    assert doc["cameras"]["ego_video"]["head_marker"] == {
        "dictionary": "DICT_4X4_100",
        "id": 50,
        "size": 0.049,
    }

    cfg = EpisodeConfig.load(out, check_exists=False, config_dir=config)
    camera = configured_head_camera(cfg)
    assert camera.label == "ego_video"
    assert camera.head_marker == doc["cameras"]["ego_video"]["head_marker"]


def test_head_marker_requires_an_extra_camera(tmp_path, monkeypatch):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    out, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        head_marker_id=50,
        config_dir=tmp_path / "cams",
    )
    assert out is None
    assert "extra head-camera stream" in problem


def test_head_camera_selects_one_of_multiple_extra_streams(tmp_path, monkeypatch):
    observer = "PRO_VID_20260820_100006_00_047_observer.mp4"
    ep = _episode(tmp_path, [CLIP_A, CLIP_B, EGO, observer])
    _registry(monkeypatch, {"00_001": A, "00_002": B})

    out, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        head_marker_id=30,
        config_dir=tmp_path / "cams",
    )
    assert out is None
    assert "pass --head-camera" in problem

    out, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        head_marker_id=30,
        head_camera="ego_video",
        head_marker_dictionary="DICT_4X4_50",
        head_marker_size=0.055,
        config_dir=tmp_path / "cams",
    )
    assert not problem
    cameras = yaml.safe_load(out.read_text())["cameras"]
    assert "head_marker" not in cameras["observer"]
    assert cameras["ego_video"]["head_marker"] == {
        "dictionary": "DICT_4X4_50",
        "id": 30,
        "size": 0.055,
    }


def test_naming_a_label_that_is_not_an_extra_stream_is_refused(tmp_path, monkeypatch):
    ep = _episode(tmp_path, [CLIP_A, CLIP_B, EGO])
    _registry(monkeypatch, {"00_001": A, "00_002": B})
    _, problem = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
        extra_units={"ego": "insta360-GO3-001"},
    )
    assert "--camera named ego" in problem
    assert "extra streams found: ego_video" in problem
    assert "_<tag>" in problem  # the fix is spelled: the label is the tag


def test_a_named_unit_overrides_a_resolved_one(tmp_path, monkeypatch):
    # An operator correcting a mis-registered serial must win over the registry.
    ep = _episode(tmp_path, [CLIP_A, CLIP_B, EGO])
    _registry(monkeypatch, {"00_001": A, "00_002": B, "00_046": ("wrong-unit", "S")})
    out, _ = init_mod.init_episode(
        ep,
        left=A[0],
        right=B[0],
        config_dir=tmp_path / "cams",
        extra_units={"ego_video": "insta360-GO3-001"},
    )
    assert yaml.safe_load(out.read_text())["cameras"]["ego_video"]["camera"] == (
        "insta360-GO3-001"
    )

"""scene.yaml: strict, portable, digest-pinned.

The two load-bearing properties: a fixture scene keeps loading after its
root directory moves (everything is relative), and every invalid reference
fails at load - before any component would run.
"""

from __future__ import annotations

import shutil

import pytest
import yaml

from kiwi.episode import scene as scene_mod
from kiwi.episode.scene import (
    SCHEMA_MIN_USABLE_S,
    effective_min_usable_s,
    init_scene,
    load_scene,
)

EPISODE_DOC = {
    "schema": "episode_v1",
    "name": "e",
    "arms": {},
}


def _episode(root, name):
    d = root / name
    d.mkdir(parents=True)
    (d / "episode.yaml").write_text(yaml.safe_dump({**EPISODE_DOC, "name": name}))
    return d


@pytest.fixture
def config(tmp_path, monkeypatch):
    """A config root carrying the builtin default profile (copied verbatim
    from the shipped one, so the template itself is under test)."""
    cfg = tmp_path / "config"
    dst = cfg / "scene" / "mapping-profiles"
    dst.mkdir(parents=True)
    from kiwi.core.paths import repo_root

    shutil.copyfile(
        repo_root() / "config" / "scene" / "mapping-profiles" / "default.yaml",
        dst / "default.yaml",
    )
    monkeypatch.setenv("KIWI_CONFIG", str(cfg))
    return cfg


@pytest.fixture
def scene_root(tmp_path, config):
    root = tmp_path / "workbench"
    root.mkdir()
    _episode(root, "mapping-session")
    _episode(root / "episodes", "evaluation-session")
    return root


def _init(root, **over):
    kw = dict(root=root, mapping_episode="mapping-session")
    kw.update(over)
    return init_scene(**kw)


# -- init + portability ------------------------------------------------------


def test_init_discovers_both_layouts_and_reloads_itself(scene_root):
    cfg = _init(scene_root)
    assert set(cfg.episodes) == {"mapping-session", "evaluation-session"}
    assert cfg.episodes["evaluation-session"] == "episodes/evaluation-session"
    assert cfg.builder_episode == "mapping-session"
    assert cfg.profile.name == "default"
    assert cfg.profile_digest.startswith("canonical_yaml_sha256:")


@pytest.mark.parametrize(
    "profile_name", ["default", "bootstrap", "reference", "custom"]
)
def test_a_scene_survives_moving_without_builtin_profiles(
    scene_root, tmp_path, config, profile_name
):
    """Scene-local copies remain authoritative, whatever the profile is named."""
    profiles = config / "scene" / "mapping-profiles"
    if profile_name != "default":
        doc = yaml.safe_load((profiles / "default.yaml").read_text())
        doc["name"] = profile_name
        doc["engine"]["map_fps"] = 15.0
        (profiles / f"{profile_name}.yaml").write_text(yaml.safe_dump(doc))
    initial = _init(scene_root, mapping_profile=profile_name)
    shutil.rmtree(profiles)
    moved = tmp_path / "elsewhere" / "workbench-copy"
    shutil.copytree(scene_root, moved)
    cfg = load_scene(moved / "scene.yaml")
    assert cfg.root == moved.resolve()
    assert set(cfg.episodes) == {"mapping-session", "evaluation-session"}
    assert cfg.profile == initial.profile
    assert cfg.profile.name == profile_name
    assert cfg.profile_digest == initial.profile_digest


def test_init_never_overwrites(scene_root):
    _init(scene_root)
    with pytest.raises(FileExistsError, match="never overwritten"):
        _init(scene_root)


def test_init_refuses_unknown_builder_and_missing_profile(scene_root):
    with pytest.raises(ValueError, match="mapping-episode"):
        _init(scene_root, mapping_episode="nope")
    with pytest.raises(FileNotFoundError, match="available"):
        _init(scene_root, mapping_profile="production")


def test_init_refuses_an_empty_root(tmp_path, config):
    empty = tmp_path / "bare"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no episode directories"):
        _init(empty)


# -- the floor merge ---------------------------------------------------------


def test_floor_merges_monotonically(scene_root):
    cfg = _init(scene_root, min_usable_s=90.0)
    assert cfg.min_usable_duration_s == 90.0
    assert effective_min_usable_s(cfg) == 90.0  # profile stays at the floor

    # A profile demanding more than the scene wins the merge.
    profile = scene_root / "profiles" / "default.yaml"
    doc = yaml.safe_load(profile.read_text())
    doc["selection"]["min_usable_duration_s"] = 120.0
    profile.write_text(yaml.safe_dump(doc))
    manifest = yaml.safe_load((scene_root / "scene.yaml").read_text())
    from kiwi.core.digest import digest_canonical_yaml

    manifest["mapping"]["profile"]["digest"] = digest_canonical_yaml(profile)
    (scene_root / "scene.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    cfg = load_scene(scene_root / "scene.yaml")
    assert effective_min_usable_s(cfg) == 120.0


def test_a_value_below_the_schema_floor_is_refused(scene_root):
    with pytest.raises(ValueError, match="floor"):
        _init(scene_root, min_usable_s=30.0)
    assert not (scene_root / "scene.yaml").exists()


# -- invalid references fail at load ------------------------------------------


def _mutate(scene_root, fn):
    manifest = scene_root / "scene.yaml"
    doc = yaml.safe_load(manifest.read_text())
    fn(doc)
    manifest.write_text(yaml.safe_dump(doc, sort_keys=False))
    return manifest


def test_unknown_keys_are_refused(scene_root):
    _init(scene_root)
    m = _mutate(scene_root, lambda d: d.update(notes="remember to sweep"))
    with pytest.raises(ValueError, match="unknown key"):
        load_scene(m)


def test_absolute_and_escaping_paths_are_refused(scene_root, tmp_path):
    _init(scene_root)
    m = _mutate(
        scene_root,
        lambda d: d["episodes"].update({"evil": str(tmp_path / "outside")}),
    )
    with pytest.raises(ValueError, match="POSIX-relative"):
        load_scene(m)
    _mutate(scene_root, lambda d: d["episodes"].update({"evil": "../outside"}))
    with pytest.raises(ValueError, match="POSIX-relative"):
        load_scene(m)


def test_an_unknown_selection_mode_is_refused(scene_root):
    _init(scene_root)
    m = _mutate(
        scene_root,
        lambda d: d["mapping"]["selection"].update(mode="ranked_candidates"),
    )
    with pytest.raises(ValueError, match="explicit"):
        load_scene(m)


def test_map_builder_arm_and_trajectory_mode_are_validated(scene_root):
    _init(scene_root)
    m = _mutate(scene_root, lambda d: d["mapping"].update(map_builder_arm="top"))
    with pytest.raises(ValueError, match="map_builder_arm"):
        load_scene(m)
    _mutate(scene_root, lambda d: d["mapping"].update(map_builder_arm="right"))
    _mutate(
        scene_root,
        lambda d: d["processing"].update(
            trajectory_mode="alignment_only", state_profile="dual-map-imu"
        ),
    )
    with pytest.raises(ValueError, match="never use"):
        load_scene(m)
    _mutate(
        scene_root,
        lambda d: d["processing"].pop("state_profile"),
    )
    _mutate(
        scene_root,
        lambda d: d["processing"].update(trajectory_mode="global_state"),
    )
    with pytest.raises(ValueError, match="state_profile"):
        load_scene(m)
    _mutate(
        scene_root,
        lambda d: d["processing"].update(
            trajectory_mode="global_state", state_profile="dual-map-imu"
        ),
    )
    cfg = load_scene(m)
    assert cfg.trajectory_mode == "global_state"
    assert cfg.state_profile == "dual-map-imu"


def test_an_episode_failing_the_manifest_contract_is_refused(scene_root):
    _init(scene_root)
    bad = scene_root / "mapping-session" / "episode.yaml"
    bad.write_text("schema: something_else\nname: x\n")
    with pytest.raises(ValueError, match="manifest contract"):
        load_scene(scene_root / "scene.yaml")


def test_init_is_retryable_over_its_own_leftovers(scene_root, tmp_path):
    """A failed init must not strand the scene: a leftover profile copy that
    matches the template is reused; foreign content still refuses."""
    profiles = scene_root / "profiles"
    profiles.mkdir()
    import shutil as _sh

    from kiwi.episode.scene import builtin_profile_dir

    _sh.copyfile(builtin_profile_dir() / "default.yaml", profiles / "default.yaml")
    cfg = _init(scene_root)  # retry over the leftover succeeds
    assert cfg.profile.name == "default"
    (scene_root / "scene.yaml").unlink()
    (profiles / "default.yaml").write_text(
        "schema: scene_mapping_profile\nname: default\n"
        "selection: {min_usable_duration_s: 61.0}\n"
    )
    with pytest.raises(FileExistsError, match="different content"):
        _init(scene_root)


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"map_builder_arm": "top"}, "map_builder_arm"),
        ({"trajectory_mode": "both"}, "trajectory_mode"),
        ({"pipeline_profile": "../escape"}, "pipeline_profile"),
        (
            {"trajectory_mode": "alignment_only", "state_profile": "scene-fusion"},
            "never use",
        ),
    ],
)
def test_a_refused_init_writes_nothing_and_the_retry_succeeds(scene_root, over, match):
    """init runs load_scene's checks on the manifest before writing it: a
    refused value leaves no scene.yaml or profile copy behind, so the
    corrected retry is not refused as 'never overwritten'."""
    with pytest.raises(ValueError, match=match):
        _init(scene_root, **over)
    assert not (scene_root / "scene.yaml").exists()
    assert not (scene_root / "profiles").exists()
    assert _init(scene_root).map_builder_arm == "left"


def test_init_checks_back_references_before_writing(scene_root):
    edir = scene_root / "mapping-session"
    doc = yaml.safe_load((edir / "episode.yaml").read_text())
    doc["scene"] = {"manifest": "../scene.yaml", "episode": "evaluation-session"}
    (edir / "episode.yaml").write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="scene manifest lists"):
        _init(scene_root)
    assert not (scene_root / "scene.yaml").exists()

    # An agreeing back-reference resolves against the manifest init is
    # about to write.
    doc["scene"]["episode"] = "mapping-session"
    (edir / "episode.yaml").write_text(yaml.safe_dump(doc))
    _init(scene_root)
    load_scene(scene_root / "scene.yaml")


def test_init_links_every_episode_back_to_the_scene(scene_root):
    """The state stage finds the published map set through the episode's
    own `scene:` block; init writes it, in both layouts, and it agrees with
    the manifest in both directions (load_scene checks that)."""
    cfg = _init(scene_root)
    for episode_id, rel in cfg.episodes.items():
        manifest = scene_root / rel / "episode.yaml"
        doc = yaml.safe_load(manifest.read_text())
        assert doc["scene"]["episode"] == episode_id
        assert (manifest.parent / doc["scene"]["manifest"]).resolve() == cfg.path
    assert (
        yaml.safe_load(
            (
                scene_root / "episodes" / "evaluation-session" / "episode.yaml"
            ).read_text()
        )["scene"]["manifest"]
        == "../../scene.yaml"
    )
    load_scene(cfg.path)


def test_init_keeps_an_agreeing_back_reference_and_the_operator_comments(scene_root):
    edir = scene_root / "mapping-session"
    manifest = edir / "episode.yaml"
    manifest.write_text(
        "# operator notes stay\n"
        + manifest.read_text()
        + "scene:\n  manifest: ../scene.yaml  # hand-written\n  episode: mapping-session\n"
    )
    before = manifest.read_text()
    _init(scene_root)
    assert manifest.read_text() == before
    other = scene_root / "episodes" / "evaluation-session" / "episode.yaml"
    assert "# operator notes stay" not in other.read_text()
    assert yaml.safe_load(other.read_text())["scene"]["episode"] == "evaluation-session"


def test_a_missing_episode_dir_fails_at_load(scene_root):
    _init(scene_root)
    shutil.rmtree(scene_root / "episodes" / "evaluation-session")
    with pytest.raises(FileNotFoundError, match="evaluation-session"):
        load_scene(scene_root / "scene.yaml")


def test_an_edited_profile_fails_the_digest_pin(scene_root):
    _init(scene_root)
    profile = scene_root / "profiles" / "default.yaml"
    doc = yaml.safe_load(profile.read_text())
    doc["selection"]["min_usable_duration_s"] = 61.0
    profile.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="mismatch"):
        load_scene(scene_root / "scene.yaml")


def test_a_comment_edit_does_not_break_the_pin(scene_root):
    """canonical_yaml digests parse content: fixing a typo in a comment must
    not invalidate every scene pinning the profile."""
    _init(scene_root)
    profile = scene_root / "profiles" / "default.yaml"
    profile.write_text(profile.read_text() + "\n# documentation, edited freely\n")
    load_scene(scene_root / "scene.yaml")


@pytest.mark.parametrize("broken", ["scene.yaml", "profiles/default.yaml"])
def test_a_scene_file_that_is_not_yaml_is_a_value_error_naming_it(scene_root, broken):
    """Every caller - the state stage, `kiwi scene inspect/build/localize` -
    reports a ValueError as a refusal; the parser's own error escapes them as
    a traceback naming "<unicode string>"."""
    _init(scene_root)
    path = scene_root / broken
    path.write_text(path.read_text() + "episodes: [unclosed\n")
    with pytest.raises(ValueError, match="not valid YAML: ") as err:
        load_scene(scene_root / "scene.yaml")
    assert str(err.value).startswith(f"{path.resolve()}: ")
    assert isinstance(err.value.__cause__, yaml.YAMLError)


def test_init_refuses_a_leftover_profile_copy_that_is_not_yaml(scene_root):
    copy = scene_root / "profiles" / "default.yaml"
    copy.parent.mkdir()
    copy.write_text("selection: [unclosed\n")
    with pytest.raises(ValueError, match="not valid YAML: ") as err:
        _init(scene_root)
    assert str(err.value).startswith(f"{copy.resolve()}: ")
    assert not (scene_root / "scene.yaml").exists()


# -- back-references ----------------------------------------------------------


def test_agreeing_back_references_load(scene_root):
    _init(scene_root)
    edir = scene_root / "mapping-session"
    doc = yaml.safe_load((edir / "episode.yaml").read_text())
    doc["scene"] = {"manifest": "../scene.yaml", "episode": "mapping-session"}
    (edir / "episode.yaml").write_text(yaml.safe_dump(doc))
    load_scene(scene_root / "scene.yaml")


def test_a_disagreeing_back_reference_is_refused(scene_root):
    _init(scene_root)
    edir = scene_root / "mapping-session"
    doc = yaml.safe_load((edir / "episode.yaml").read_text())
    doc["scene"] = {"manifest": "../scene.yaml", "episode": "evaluation-session"}
    (edir / "episode.yaml").write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="scene manifest lists"):
        load_scene(scene_root / "scene.yaml")


# -- profile schema strictness ------------------------------------------------


def test_profile_unknown_keys_and_low_floor_are_refused(tmp_path):
    import copy

    base = {
        "schema": "scene_mapping_profile",
        "name": "p",
        "selection": {"min_usable_duration_s": 60.0},
    }
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(base))
    scene_mod.load_mapping_profile(p)

    doc = copy.deepcopy(base)
    doc["engine"] = "surprise"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="engine must be a mapping"):
        scene_mod.load_mapping_profile(p)

    doc = copy.deepcopy(base)
    doc["engine"] = {"map_fps": 30.0, "surprise": 1}
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="unknown key"):
        scene_mod.load_mapping_profile(p)

    doc = copy.deepcopy(base)
    doc["souvenirs"] = "surprise"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="unknown key"):
        scene_mod.load_mapping_profile(p)

    doc = copy.deepcopy(base)
    doc["selection"]["min_usable_duration_s"] = 5.0
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="floor"):
        scene_mod.load_mapping_profile(p)

    doc = copy.deepcopy(base)
    doc["engine"] = {"map_fps": -1}
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="positive"):
        scene_mod.load_mapping_profile(p)


def test_schema_floor_constant_is_sixty():
    assert SCHEMA_MIN_USABLE_S == 60.0


# -- the masking mode --------------------------------------------------------


def test_masking_defaults_explicit_and_strict(scene_root):
    cfg = _init(scene_root)
    assert cfg.masking == "learned", "the packaged model is the default"
    doc = yaml.safe_load((scene_root / "scene.yaml").read_text())
    assert doc["processing"]["masking"] == "learned", (
        "the default is written, not implied"
    )

    doc["processing"]["masking"] = "model_masks"
    (scene_root / "scene.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError, match="masking"):
        load_scene(scene_root / "scene.yaml")

    del doc["processing"]["masking"]
    (scene_root / "scene.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError, match="masking"):
        load_scene(scene_root / "scene.yaml")


def test_the_default_profile_carries_the_mapping_recipe(config):
    from kiwi.episode.scene import load_mapping_profile

    profile = load_mapping_profile(
        config / "scene" / "mapping-profiles" / "default.yaml"
    )
    assert profile.name == "default"
    assert profile.min_usable_duration_s == 60.0
    assert profile.map_fps == 30.0 and profile.map_image_size == 960


def test_an_omitted_engine_section_means_the_default_recipe(tmp_path):
    """A profile that says nothing about the engine maps at 30 fps and 960 px,
    not at the recording's native ~120 fps and 1920 px."""
    doc = {
        "schema": "scene_mapping_profile",
        "name": "p",
        "selection": {"min_usable_duration_s": 60.0},
    }
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(doc))
    silent = scene_mod.load_mapping_profile(p)
    assert (silent.map_fps, silent.map_image_size) == (30.0, 960)

    # a profile that states a recipe still gets exactly that one, including
    # the native rate and size it has to name to ask for them
    doc["engine"] = {"map_fps": 10.0, "map_image_size": 480}
    p.write_text(yaml.safe_dump(doc))
    stated = scene_mod.load_mapping_profile(p)
    assert (stated.map_fps, stated.map_image_size) == (10.0, 480)

    doc["engine"] = {"map_fps": 1000.0, "map_image_size": 1920}
    p.write_text(yaml.safe_dump(doc))
    native = scene_mod.load_mapping_profile(p)
    assert (native.map_fps, native.map_image_size) == (1000.0, 1920)

    # one field stated, the other defaulted - they default independently
    doc["engine"] = {"map_fps": 15.0}
    p.write_text(yaml.safe_dump(doc))
    half = scene_mod.load_mapping_profile(p)
    assert (half.map_fps, half.map_image_size) == (15.0, 960)

"""Pipeline profiles: complete, typed, and honest about dependencies."""

from __future__ import annotations

import pytest

from kiwi.episode.pipeline_profile import (
    DEFAULT_PROFILE,
    PREPARE_PROFILE,
    GLOBAL_STATE_GROUND_PROFILE,
    GLOBAL_STATE_HEAD_GROUND_PROFILE,
    PROFILES,
    PipelineProfile,
)
from kiwi.episode.stages import STAGES


def _dispositions(**over):
    d = dict(PREPARE_PROFILE.dispositions)
    d.update(over)
    return d


def test_default_profile_states_every_stage():
    assert set(DEFAULT_PROFILE.dispositions) == set(STAGES)
    assert not DEFAULT_PROFILE.requires("ground")
    assert DEFAULT_PROFILE.requires("state")
    assert not DEFAULT_PROFILE.requires("register")
    assert not DEFAULT_PROFILE.requires("head")


def test_the_core_chain_is_sync_convert_state_eef():
    """The core chain a plain `kiwi run` walks. `register` is the on-demand
    ground-truth anchor and is not part of it."""
    assert [s for s in STAGES if DEFAULT_PROFILE.requires(s)] == [
        "sync",
        "convert",
        "state",
        "eef",
    ]


def test_no_builtin_profile_requests_register():
    from kiwi.episode.pipeline_profile import PROFILES

    assert PROFILES, "there must be builtin profiles to make this claim about"
    for name, profile in PROFILES.items():
        assert profile.dispositions["register"] == "not_requested", (
            f"{name} requests the register stage; the ArUco table marker is a "
            f"ground-truth instrument, run on demand only"
        )


def test_the_state_stage_does_not_depend_on_register():
    """The scene estimator localizes against the map set and opens nothing
    register produces, so a profile may request state while skipping register
    - which GLOBAL_STATE_PROFILE does."""
    from kiwi.episode.pipeline_profile import _DEPENDS

    assert "register" not in _DEPENDS["state"]
    PipelineProfile(
        name="p",
        dispositions=_dispositions(register="not_requested", state="required"),
        trajectory_mode="global_state",
        state_profile="dual-map-imu",
    )


def test_incomplete_coverage_is_refused():
    d = _dispositions()
    del d["eef"]
    with pytest.raises(ValueError, match="missing"):
        PipelineProfile(name="p", dispositions=d)


def test_unknown_stage_is_refused():
    with pytest.raises(ValueError, match="unknown"):
        PipelineProfile(name="p", dispositions=_dispositions(warp="required"))


def test_bad_disposition_is_refused():
    with pytest.raises(ValueError, match="disposition"):
        PipelineProfile(name="p", dispositions=_dispositions(ground="skipped"))


def test_a_required_stage_cannot_read_from_a_skipped_one():
    """A profile is validated against `_DEPENDS`, and the edges checked here
    are refusals the stage entries raise themselves: register's own entry
    refuses without scene/state.tum."""
    with pytest.raises(ValueError, match="which the profile skips"):
        PipelineProfile(
            name="p",
            dispositions=_dispositions(register="required", state="not_requested"),
        )

    with pytest.raises(ValueError, match="which the profile skips"):
        PipelineProfile(name="p", dispositions=_dispositions(head="required"))


def test_empty_name_is_refused():
    with pytest.raises(ValueError, match="name"):
        PipelineProfile(name="", dispositions=_dispositions())


def test_global_state_mode_requires_state_stage_and_profile():
    with pytest.raises(ValueError, match="requires the state stage"):
        PipelineProfile(
            name="p",
            dispositions=_dispositions(),
            trajectory_mode="global_state",
            state_profile="dual-map-imu",
        )
    with pytest.raises(ValueError, match="named state profile"):
        PipelineProfile(
            name="p",
            dispositions=_dispositions(state="required"),
            trajectory_mode="global_state",
        )
    PipelineProfile(
        name="p",
        dispositions=_dispositions(state="required", register="not_requested"),
        trajectory_mode="global_state",
        state_profile="dual-map-imu",
    )


def test_alignment_only_mode_forbids_state():
    with pytest.raises(ValueError, match="must not require the state stage"):
        PipelineProfile(name="p", dispositions=_dispositions(state="required"))
    with pytest.raises(ValueError, match="never use"):
        PipelineProfile(
            name="p", dispositions=_dispositions(), state_profile="dual-map-imu"
        )


def test_a_global_state_profile_schedules_the_state_stage(tmp_path, monkeypatch):
    """Both the default and an explicit global-state profile schedule fusion."""
    from kiwi.episode import stages

    calls = []
    reg = dict(stages.REGISTRY)
    for name in stages.STAGES:
        reg[name] = lambda d, _n=name, **_kw: (
            calls.append(_n) or stages.StageResult(_n, ok=True, summary="stub")
        )
    monkeypatch.setattr(stages, "REGISTRY", reg)

    stages.run_pipeline(tmp_path)
    assert "state" in calls and "ground" not in calls
    assert "register" not in calls

    calls.clear()
    profile = PipelineProfile(
        name="scene-global-state",
        dispositions=_dispositions(state="required", register="not_requested"),
        trajectory_mode="global_state",
        state_profile="dual-map-imu",
    )
    stages.run_pipeline(tmp_path, profile=profile)
    assert "state" in calls, (
        "an explicit global-state profile must dispatch the state stage"
    )


def test_a_single_stage_range_beats_the_profile_disposition(tmp_path, monkeypatch):
    """A profile states what a CHAIN asks for. Narrowing the range to one
    stage is a direct instruction instead, which is the only thing
    keeping the on-demand stages reachable through `kiwi run`."""
    from kiwi.episode import stages

    calls = []
    reg = dict(stages.REGISTRY)
    for name in stages.STAGES:
        reg[name] = lambda d, _n=name, **_kw: (
            calls.append(_n) or stages.StageResult(_n, ok=True, summary="stub")
        )
    monkeypatch.setattr(stages, "REGISTRY", reg)

    stages.run_pipeline(tmp_path, from_stage="register", to_stage="register")
    assert calls == ["register"]

    # A wider range is a chain again, and stays profile-governed.
    calls.clear()
    stages.run_pipeline(tmp_path, from_stage="state")
    assert calls == ["state", "eef"]

    # A range narrowed to `ground` alone is the same direct instruction as
    # `register`, and dispatches even though the default profile leaves it
    # not_requested inside a chain.
    calls.clear()
    [r] = stages.run_pipeline(tmp_path, from_stage="ground", to_stage="ground")
    assert calls == ["ground"] and r.ok


def test_builtin_profiles_are_named_and_distinct():
    from kiwi.episode.pipeline_profile import DEFAULT_PROFILE, PROFILES

    assert PROFILES["global-state"] is DEFAULT_PROFILE
    assert PROFILES["prepare"] is PREPARE_PROFILE
    assert DEFAULT_PROFILE.requires("state"), (
        "the default profile must publish the scene state consumed by export"
    )
    gs = PROFILES["global-state"]
    assert gs.requires("state") and gs.trajectory_mode == "global_state"
    assert not gs.requires("register")
    assert not gs.requires("head")
    assert gs.state_profile == "scene-fusion"
    with_head = PROFILES["global-state-head"]
    assert with_head.requires("state") and with_head.requires("head")
    assert not with_head.requires("register")
    assert with_head.state_profile == "scene-fusion"


def test_the_state_profile_carries_the_estimator_noise_policy():
    from kiwi.state import STATE_PROFILES

    # the Allan artifact stays RAW; inflation is estimator policy
    assert STATE_PROFILES["scene-fusion"].imu_noise_scale == 10.0


def test_ground_reads_from_state():
    """The ground entry refuses without scene/state.tum."""
    with pytest.raises(ValueError, match="which the profile skips"):
        PipelineProfile(name="p", dispositions=_dispositions(ground="required"))
    for prof in (GLOBAL_STATE_GROUND_PROFILE, GLOBAL_STATE_HEAD_GROUND_PROFILE):
        assert prof.requires("ground") and prof.requires("state")
        assert prof.trajectory_mode == "global_state"
        assert PROFILES[prof.name] is prof
    assert not GLOBAL_STATE_GROUND_PROFILE.requires("head")
    assert GLOBAL_STATE_HEAD_GROUND_PROFILE.requires("head")


def test_builtin_profile_dependencies_are_ordered():
    from kiwi.episode.pipeline_profile import _DEPENDS

    assert _DEPENDS["state"] == ("sync", "convert")
    for profile in PROFILES.values():
        for stage, deps in _DEPENDS.items():
            assert all(STAGES.index(dep) < STAGES.index(stage) for dep in deps)


def test_prepare_never_dispatches_pose_engines(tmp_path, monkeypatch):
    from kiwi.episode import stages

    calls = []
    for name in ("sync", "convert", "eef"):
        monkeypatch.setitem(
            stages.REGISTRY,
            name,
            lambda d, _name=name: (
                calls.append(_name) or stages.StageResult(_name, ok=True)
            ),
        )
    # Real state/head/ground entries have no scene or media and would fail.
    results = stages.run_pipeline(tmp_path, profile=PREPARE_PROFILE)
    assert all(result.ok for result in results)
    assert calls == ["sync", "convert", "eef"]

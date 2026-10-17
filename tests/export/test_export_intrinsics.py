"""The camera model travels with the dataset, or nothing ships.

Three claims, each with teeth: the resolved model VALUES (not just a hash)
land in the episode meta and the provenance beside the extrinsic chain; a
unit with no KB4 fit exports on its DS model alone (the DS model is the
calibration, the KB4 a derived convenience); and an arm with no model at all
is refused at ``build_and_write`` - the one seam every publishing export
passes through - before any build or encode work.

Run: pixi run test
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest


cv2 = pytest.importorskip("cv2")

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")

from kiwi.eef.completion import complete_series  # noqa: E402
from kiwi.eef.state import EefState, write_eef_state  # noqa: E402
from kiwi.export.build import _intrinsics_block, build_episode  # noqa: E402
from kiwi.export.schema import IMAGE_KEY_PREFIX, wrist_image_key  # noqa: E402
from kiwi.export.source import ArmSource, CameraSource, EpisodeSource  # noqa: E402
from kiwi.export.write import build_and_write  # noqa: E402

# An extra camera passed with key="ego" lands under this feature key.
EGO_KEY = IMAGE_KEY_PREFIX + "ego"

MAP_SET = {"front": "attempts/2026-09-01-front", "rear": "attempts/2026-09-01-rear"}
FPS = 30.0
N = 60  # 2 s clip

DS = [-0.023, 0.571, 499.64, 499.54, 962.30, 961.43]


pytestmark = pytest.mark.usefixtures("calibrated_dc_description")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("intrinsics") / "rec_lens0.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 64))
    if not vw.isOpened():
        pytest.skip("no mp4v encoder available in this build of cv2")
    img = np.zeros((64, 64, 3), np.uint8)
    for i in range(N):
        img[:, :] = (i, 255 - i, i)
        vw.write(img)
    vw.release()
    return p


def _arm(tmp_path, clip, label):
    tum = tmp_path / label / "scene" / "state.tum"
    tum.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, 2.01, 0.1)
    tum.write_text(
        "# t tx ty tz qx qy qz qw  (frame=shared_rear_metric_gravity_aligned body=camera; "
        "t on episode global clock (origin=left))\n"
        + "".join(f"{v:.9f} {v:.9f} 0 0 0 0 0 1\n" for v in t)
    )
    inputs = tum.parent / "state.inputs.json"
    inputs.write_text(json.dumps({"map_set": MAP_SET}))
    completed, audit, _ = complete_series(np.full(N, 60.0), FPS)
    state = EefState(
        tool="chopsticks-v3",
        estimator="chopsticks",
        unit="deg",
        fps=FPS,
        source="rec_lens0.mp4",
        pose_mode="fixed_pose",
        pose_source="config/gripper/chopsticks-v3/units/chopstick-v3-m5.yaml",
        value=np.full(N, 60.0),
        rejected_value=np.full(N, np.nan),
        has_pose=np.ones(N, dtype=bool),
        n_detected=np.full(N, 6, dtype=np.int32),
        n_kept=np.full(N, 6, dtype=np.int32),
        error=np.asarray([""] * N, dtype=str),
        value_completed=completed,
        completion=audit,
    )
    eef = write_eef_state(tmp_path / f"{label}_eef.npz", state)
    return ArmSource(
        label=label,
        lens0=clip,
        state_tum=tum,
        state_inputs=inputs,
        eef_state=eef,
        sync_off=0.0,
        session_block={"angle_closed_deg": 20.0, "angle_open_deg": 100.0},
        tool="chopsticks-v3",
        description="test_calibrated_dc",
    )


def _source(tmp_path, clip, **kw):
    if "arms" not in kw:
        kw["arms"] = (_arm(tmp_path, clip, "left"), _arm(tmp_path, clip, "right"))
    return EpisodeSource(
        name="demo", episode_dir=tmp_path, task="stack the cubes", **kw
    )


def _npz(path, **arrays):
    np.savez(path, **arrays)
    return path


def _ds_npz(path, ds=DS, size=(1920, 1920)):
    return _npz(
        path,
        ds_front=np.asarray(ds, dtype=np.float64),
        ds_back=np.asarray(ds, dtype=np.float64),
        image_size=np.asarray(size, dtype=np.int32),
    )


def _kb4(path):
    body = {
        "schema": "kb4_intrinsics",
        "unit": "test-unit-001",
        "kb4": {
            "fx": 511.4,
            "fy": 511.3,
            "cx": 962.3,
            "cy": 961.4,
            "k1": 0.0458,
            "k2": 0.0003,
            "k3": -0.0002,
            "k4": -0.0001,
        },
        "resolution": [1920, 1920],
    }
    path.write_text(json.dumps(body))
    return path, body


def _arm_with_model(tmp_path, clip, label, *, kb4=True):
    a = _arm(tmp_path, clip, label)
    npz = _ds_npz(tmp_path / f"{label}-intr.npz")
    kb4_path = _kb4(tmp_path / f"{label}-kb4.json")[0] if kb4 else None
    import dataclasses

    return dataclasses.replace(
        a, camera="test-unit-001", intrinsics=npz, intrinsics_kb4=kb4_path
    )


# --- the reader --------------------------------------------------------------


def test_ds_values_and_stamp_are_embedded(tmp_path):
    npz = _ds_npz(tmp_path / "i.npz")
    block = _intrinsics_block("unit-9", npz, None, None)
    assert block["camera"] == "unit-9" and block["lens"] == "front"
    assert block["model"] == "double_sphere"
    assert block["ds"] == {
        "xi": DS[0],
        "alpha": DS[1],
        "fx": DS[2],
        "fy": DS[3],
        "cx": DS[4],
        "cy": DS[5],
    }
    assert block["calibration_size"] == [1920, 1920]
    # Values AND identity: the stamp says which registry file supplied them.
    assert block["source"]["sha256"]
    # No KB4 artifact -> kb4 is null, and only null - a reader must never
    # confuse "no fit published" with "model unknown".
    assert block["kb4"] is None and "kb4_source" not in block


def test_kb4_body_travels_verbatim(tmp_path):
    npz = _ds_npz(tmp_path / "i.npz")
    kb4_path, body = _kb4(tmp_path / "k.json")
    block = _intrinsics_block("u", npz, kb4_path, None)
    assert block["kb4"] == body
    assert block["kb4_source"]["sha256"]


def test_pinhole_equi_bundles_are_supported(tmp_path):
    npz = _npz(
        tmp_path / "i.npz",
        K_front=np.eye(3) * 500.0,
        D_front=np.asarray([0.1, 0.01, 0.001, 0.0001]),
        image_size=np.asarray([1920, 1920], dtype=np.int32),
    )
    block = _intrinsics_block("u", npz, None, None)
    assert block["model"] == "pinhole_equi"
    assert block["K"][0][0] == 500.0 and len(block["D"]) == 4


def test_a_bundle_with_no_front_model_is_refused(tmp_path):
    npz = _npz(tmp_path / "i.npz", other=np.zeros(3))
    with pytest.raises(ValueError, match="no front-lens model"):
        _intrinsics_block("u", npz, None, None)


# --- through build_episode ---------------------------------------------------


def test_model_lands_in_meta_keyed_by_image_feature(tmp_path, clip):
    arms = (
        _arm_with_model(tmp_path, clip, "left"),
        _arm_with_model(tmp_path, clip, "right", kb4=False),
    )
    oe = build_episode(_source(tmp_path, clip, arms=arms), resize=(32, 32))
    intr = oe.meta["intrinsics"]
    left = intr[wrist_image_key("left")]
    assert left["ds"]["fx"] == DS[2] and left["kb4"] is not None
    assert intr[wrist_image_key("right")]["kb4"] is None
    # The block states what the consumer actually receives -
    # the unrectified stream at its delivered size, which is what tells them
    # to rescale the calibration-resolution model.
    assert left["delivered"] == {"width": 32, "height": 32, "rectified": False}


def test_an_arm_without_a_model_records_nothing_here(tmp_path, clip):
    # build_episode is the builder, not the publishing gate - a scripted or
    # test build without registry artifacts still works; build_and_write is
    # where absence refuses.
    oe = build_episode(_source(tmp_path, clip))
    assert "intrinsics" not in oe.meta


# --- the gate ----------------------------------------------------------------


def test_build_and_write_refuses_an_arm_with_no_model(tmp_path, clip):
    src = _source(tmp_path, clip)  # fixture arms carry no intrinsics
    with pytest.raises(ValueError, match="carries the camera model that made it") as e:
        build_and_write([src], tmp_path / "ds", format="v2.1")
    # The refusal is actionable and names the offenders (by episode DIR name
    # - the same key curation and the funnel use - and arm label).
    assert ":left" in str(e.value) and "intrinsics:" in str(e.value)
    assert not (tmp_path / "ds").exists()  # refused before any bytes


@needs_ffmpeg
def test_the_model_reaches_export_provenance(tmp_path, clip):
    arms = (
        _arm_with_model(tmp_path, clip, "left"),
        _arm_with_model(tmp_path, clip, "right", kb4=False),
    )
    src = _source(tmp_path, clip, arms=arms)
    out = tmp_path / "ds"
    build_and_write([src], out, format="v2.1", repo_id="me/demo")
    doc = json.loads((out / "meta" / "export_provenance.json").read_text())
    ep = doc["episodes"][0]
    left = ep["intrinsics"][wrist_image_key("left")]
    # Beside the extrinsic chain, with the same stamp discipline.
    assert "extrinsics" in ep
    assert left["ds"]["xi"] == DS[0]
    assert left["source"]["sha256"]
    assert ep["intrinsics"][wrist_image_key("right")]["kb4"] is None


# --- the extra-camera path (best-effort, never gated) ------------------------


def test_an_extra_camera_without_a_model_builds_and_records_nothing(tmp_path, clip):
    cam = CameraSource(label="ego", video=clip, sync_off=0.0, key="ego")
    oe = build_episode(_source(tmp_path, clip, cameras=(cam,)))
    assert EGO_KEY in oe.features
    assert EGO_KEY not in oe.meta.get("intrinsics", {})


def test_an_extra_camera_with_a_model_records_it(tmp_path, clip):
    npz = _ds_npz(tmp_path / "ego-intr.npz")
    cam = CameraSource(
        label="ego",
        video=clip,
        sync_off=0.0,
        key="ego",
        camera="test-unit-go3",
        intrinsics=npz,
        intrinsics_kb4=None,
    )
    oe = build_episode(_source(tmp_path, clip, cameras=(cam,)))
    block = oe.meta["intrinsics"][EGO_KEY]
    assert block["camera"] == "test-unit-go3"
    assert block["model"] == "double_sphere"
    assert block["kb4"] is None
    assert block["delivered"] == {"width": 64, "height": 64, "rectified": False}

"""The UMI-style replay-buffer ``.zarr.zip`` side export.

Opened back with zarr itself: keys, shapes, dtypes, the axis-angle
conversion, ``episode_ends``, and the zip round-trip (the directory store
must be gone afterwards - only the zip ships).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np
import pytest
import zarr
from scipy.spatial.transform import Rotation

from kiwi.core.observation_episode import VIDEO_DTYPE, FeatureColumn, ObservationEpisode
from kiwi.export.schema import arm_state_names
from kiwi.export.writers import DatasetInfo, get_writer

NAMES = arm_state_names("left") + arm_state_names("right")


@dataclasses.dataclass
class _FakeFrames:
    n: int
    height: int = 32
    width: int = 32

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.n):
            yield np.full((self.height, self.width, 3), (i * 11) % 255, dtype=np.uint8)


def _episode(n: int, seed: int = 3) -> ObservationEpisode:
    rng = np.random.default_rng(seed)
    state = np.zeros((n, 16), dtype=np.float32)
    for off in (0, 8):
        state[:, off : off + 3] = rng.normal(size=(n, 3))
        state[:, off + 3 : off + 7] = Rotation.random(
            n, random_state=seed + off
        ).as_quat()
        state[:, off + 7] = rng.uniform(0, 1, size=n)
    feats = {
        "observation.state": FeatureColumn(
            "observation.state", "float32", (16,), list(NAMES), state
        ),
        "observation.images.left_wrist": FeatureColumn(
            "observation.images.left_wrist",
            VIDEO_DTYPE,
            (32, 32, 3),
            ["height", "width", "channels"],
            _FakeFrames(n),
        ),
    }
    return ObservationEpisode(
        fps=30.0,
        timestamps=np.arange(n) / 30.0,
        task="pick",
        frame="marker",
        features=feats,
    )


def _write(tmp_path, episodes):
    w = get_writer("umi-zarr")
    info = DatasetInfo(
        repo_id="me/umi",
        robot_type="portable_bimanual",
        fps=30.0,
        features=episodes[0].features,
    )
    w.open(tmp_path, info)
    for i, ep in enumerate(episodes):
        w.add_episode(ep, i, 0)
    w.close()
    return tmp_path / "replay_buffer.zarr.zip"


def test_layout_episode_ends_and_zip_roundtrip(tmp_path):
    eps = [_episode(40), _episode(25, seed=9)]
    zip_path = _write(tmp_path, eps)
    assert zip_path.is_file()
    assert not (tmp_path / "replay_buffer.zarr").exists()  # dir packed + removed

    root = zarr.open_group(zarr.ZipStore(str(zip_path), mode="r"))
    data = root["data"]
    assert sorted(data.array_keys()) == [
        "camera0_rgb",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper",
        "robot1_eef_pos",
        "robot1_eef_rot_axis_angle",
        "robot1_gripper",
    ]
    assert data["robot0_eef_pos"].shape == (65, 3)
    assert data["robot0_gripper"].shape == (65, 1)
    assert data["camera0_rgb"].shape == (65, 32, 32, 3)
    assert data["camera0_rgb"].dtype == np.uint8
    assert data["camera0_rgb"].chunks == (1, 32, 32, 3)
    np.testing.assert_array_equal(root["meta"]["episode_ends"][:], [40, 65])

    # The mapping is explicit, never implied.
    attrs = root.attrs["kiwi_export"]
    assert attrs["arms"] == {"robot0": "left", "robot1": "right"}
    assert attrs["cameras"] == {"camera0_rgb": "observation.images.left_wrist"}


def test_pose_channels_match_the_state(tmp_path):
    ep = _episode(20)
    zip_path = _write(tmp_path, [ep])
    root = zarr.open_group(zarr.ZipStore(str(zip_path), mode="r"))
    state = ep.features["observation.state"].data
    np.testing.assert_allclose(
        root["data"]["robot1_eef_pos"][:], state[:, 8:11], atol=1e-6
    )
    want_rotvec = Rotation.from_quat(state[:, 11:15].astype(np.float64)).as_rotvec()
    np.testing.assert_allclose(
        root["data"]["robot1_eef_rot_axis_angle"][:], want_rotvec, atol=1e-6
    )
    np.testing.assert_allclose(
        root["data"]["robot0_gripper"][:, 0], state[:, 7], atol=1e-7
    )
    # Pixels survive the block-streamed append byte-exact.
    np.testing.assert_array_equal(
        root["data"]["camera0_rgb"][7], np.full((32, 32, 3), (7 * 11) % 255, np.uint8)
    )


def test_unnamed_state_is_refused(tmp_path):
    w = get_writer("umi-zarr")
    feats = {
        "observation.state": FeatureColumn("observation.state", "float32", (16,), None)
    }
    with pytest.raises(ValueError, match="named 'observation.state'"):
        w.open(
            tmp_path,
            DatasetInfo(repo_id="x", robot_type="r", fps=30.0, features=feats),
        )

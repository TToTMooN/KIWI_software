"""Write a small LeRobot v3 dataset with KIWI's writer, for the load-back check.

    pixi run loadback-fixture                  # writes build/loadback
    pixi run -e lerobot loadback               # loads it with stock lerobot

The dataset has no videos: two episodes with the 16-value state, the action
the writer synthesizes from it, and a shape-(1,) pose-gap column, which the
writer stores as a scalar parquet column, the type lerobot derives from the
declared shape. The reader side lives in load_fixture.py and imports nothing
from kiwi.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

from kiwi.core.observation_episode import FeatureColumn, ObservationEpisode
from kiwi.export.writers import DatasetInfo, get_writer

DIM = 16
FPS = 30.0


def episode(n: int, seed: int) -> ObservationEpisode:
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((n, DIM)).astype(np.float32)
    gap = (np.arange(n, dtype=np.float32) / 1000.0)[:, None]
    names = [f"s{i}" for i in range(DIM)]
    features = {
        "observation.state": FeatureColumn(
            "observation.state", "float32", (DIM,), names, state
        ),
        "observation.pose_gap.left": FeatureColumn(
            "observation.pose_gap.left", "float32", (1,), ["left_pose_gap_s"], gap
        ),
    }
    return ObservationEpisode(
        fps=FPS,
        timestamps=np.arange(n) / FPS,
        task="pick",
        frame="scene",
        features=features,
    )


def main(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    episodes = [episode(40, 0), episode(25, 1)]
    writer = get_writer("v3.0")
    writer.open(
        root,
        DatasetInfo(
            repo_id="kiwi/loadback",
            robot_type="portable_bimanual",
            fps=FPS,
            features=episodes[0].features,
        ),
    )
    for index, ep in enumerate(episodes):
        writer.add_episode(ep, index, 0)
    writer.close()
    print(
        f"wrote {sum(len(e.timestamps) for e in episodes)} frames in {len(episodes)} episodes to {root}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "build/loadback"))

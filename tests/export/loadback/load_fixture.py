"""Load a LeRobot v3 dataset with stock lerobot and check it against its metadata.

Runs in the isolated `lerobot` environment (pixi env or a plain venv with
`lerobot>=0.4,<0.5`); it imports nothing from kiwi. Exit code 0 and a final
LOADBACK_OK line mean every frame loads and every declared tabular feature
comes back with its declared shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main(root: Path) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    ds = LeRobotDataset(repo_id=info.get("repo_id", "local/loadback"), root=root)
    assert len(ds) == info["total_frames"], (len(ds), info["total_frames"])
    assert ds.num_episodes == info["total_episodes"], (
        ds.num_episodes,
        info["total_episodes"],
    )
    for index in range(len(ds)):
        sample = ds[index]
        for name, spec in info["features"].items():
            if spec["dtype"] in ("video", "image"):
                continue
            assert name in sample, f"{name} missing from sample {index}"
            got = tuple(sample[name].shape)
            want = tuple(spec["shape"])
            assert got == want or (want == (1,) and got == ()), (name, got, want)
    print("LOADBACK_OK", len(ds), "frames", ds.num_episodes, "episodes")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "build/loadback").resolve())

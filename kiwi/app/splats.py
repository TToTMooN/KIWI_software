"""Associate an aligned Gaussian reconstruction with its source episode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kiwi.core.atomic import atomic_write_text
from kiwi.core.digest import verify_digest

ENTITY_PATH = "world/gaussian_scene"


def attachment_path(episode: Path, cache: Path) -> Path:
    key = hashlib.sha256(str(episode.resolve()).encode()).hexdigest()[:24]
    return cache / "scenes" / f"{key}.json"


def attached_scene(episode: Path, cache: Path) -> dict | None:
    path = attachment_path(episode, cache)
    return json.loads(path.read_text()) if path.is_file() else None


def ply_count(path: Path) -> int:
    """Check the header without reading the large Gaussian payload."""
    with path.open("rb") as source:
        header = source.read(65536).replace(b"\r\n", b"\n").split(b"end_header\n", 1)
    if len(header) != 2:
        raise ValueError("The Gaussian PLY has no readable header.")
    lines = header[0].decode("ascii").splitlines()
    if not lines or lines[0] != "ply":
        raise ValueError("Choose a Gaussian PLY file.")
    properties = {line.split()[-1] for line in lines if line.startswith("property ")}
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    if required - properties:
        raise ValueError(
            "This PLY is missing Gaussian scale, rotation, color or opacity fields."
        )
    counts = [
        int(line.split()[-1]) for line in lines if line.startswith("element vertex ")
    ]
    if len(counts) != 1 or counts[0] <= 0:
        raise ValueError("The Gaussian PLY has no vertices.")
    return counts[0]


def validate_scene(episode: Path, scene: dict) -> dict:
    """Require the training frame and exact source trajectories before overlaying."""
    ply = Path(scene["ply"])
    manifest = Path(scene["manifest"])
    doc = json.loads(manifest.read_text())
    if (
        doc.get("frame") != "shared_rear_metric_gravity_aligned"
        or doc.get("units") != "metres"
    ):
        raise ValueError(
            "The scene must already use shared_rear_metric_gravity_aligned in metres."
        )
    streams = [
        stream["source"]
        for stream in doc.get("streams", [])
        if Path(stream["source"]["episode"]).resolve() == episode.resolve()
    ]
    if not streams:
        raise ValueError("The training manifest does not reference this episode.")
    checked = set()
    for source in streams:
        arm = source["arm"]
        if arm not in ("left", "right"):
            raise ValueError(f"Unexpected training arm: {arm!r}")
        digest = source["trajectory"]["digest"]
        if (arm, digest) not in checked:
            verify_digest(digest, episode / arm / "scene" / "state.tum")
            checked.add((arm, digest))
    layer = {
        "path": ENTITY_PATH,
        "label": "Gaussian scene",
        "default": True,
        "count": ply_count(ply),
        "frame": doc["frame"],
        "ply": str(ply),
        "manifest": str(manifest),
    }
    # Frame the capture region. This moves only the observer; it never
    # transforms the reconstruction or poses.
    frames = [frame for frame in doc.get("frames", []) if frame.get("lens") == "front"]
    if frames:
        import numpy as np

        center = np.median(
            [np.asarray(frame["T_world_camera"])[:3, 3] for frame in frames], axis=0
        )
        layer["eye"] = {
            "position": (center + [1.2, 1.5, 0.8]).tolist(),
            "look_target": (center + [0, 0, 0.15]).tolist(),
            "eye_up": [0, 0, 1],
        }
    return layer


def attach_scene(episode: Path, cache: Path, ply: Path, manifest: Path) -> dict:
    scene = {
        "ply": str(ply.expanduser().resolve()),
        "manifest": str(manifest.expanduser().resolve()),
    }
    layer = validate_scene(episode, scene)
    target = attachment_path(episode, cache)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(scene, indent=2) + "\n")
    return layer


def log_scene(rr, episode: Path, cache: Path) -> list[dict]:
    scene = attached_scene(episode, cache)
    if scene is None:
        return []
    layer = validate_scene(episode, scene)
    print(f"Loading {layer['count']:,} Gaussians from {scene['ply']}…", flush=True)
    # The native importer converts log-scales, opacity logits, wxyz rotations
    # and SH coefficients. Keep all Gaussians and their view-dependent color.
    # Both this scene and state.tum already use the same metric world frame.
    rr.log_file_from_path(scene["ply"], entity_path_prefix=ENTITY_PATH, static=True)
    return [layer]

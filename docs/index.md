# KIWI

**Kinematic Interface for the Wild** — *off the shelf, into the wild.*

Bimanual and egocentric robot data collection with consumer cameras: two Insta360 X5 on printed Chopstick grippers and an optional Insta360 GO 3 on a hat.
Recording starts and stops by voice; everything else is computed offline:

```
sync → convert → state → eef
```

`state` localizes the wrist videos against shared scene maps and fuses them with each camera's IMU; `eef` reads the gripper opening from the video.

## Outputs

| Per episode | Contents |
| --- | --- |
| `<arm>/scene/state.tum` | each wrist camera's pose per video frame, metric and gravity-aligned, on the episode clock |
| `<arm>/eef.npz` | the gripper opening per video frame |
| `head/head_marker.tum` (optional) | the hat marker's pose per detection |
| `ground/` (optional) | the floor height |

`kiwi export run` turns episodes into LeRobot v3 / v2.1, GR00T or UMI-zarr datasets with the tool-tip pose and opening of both arms and the camera videos.

## Requirements

- Linux x86-64 with [Pixi](https://pixi.sh) 0.82 or newer and Git.
- An NVIDIA GPU with CUDA for the gripper masks of `kiwi scene build`.
- Per five-minute take: 12.5 GB of recordings, about 27 GB of scratch under `scene_map/` during the scene build and 0.8 GB per export; about 1.5 hours of processing on a 24-thread CPU with a recent GPU.
- Two calibrated Insta360 X5 cameras and two Chopstick grippers (see [Hardware and cameras](hardware.md)); an Insta360 GO 3 and a hat marker for the head pose.

## What runs when

| When | Step | Command |
| --- | --- | --- |
| once per camera | intrinsics, camera-IMU, IMU noise, A/V delay | `kiwi calib ...` |
| once per gripper build | camera-to-marker pose | `kiwi eef calibrate` |
| every take | record, import, write `episode.yaml` | `kiwi import`, `kiwi init` |
| once per place | build the scene maps from one episode | `kiwi scene init`, `kiwi scene build` |
| every episode | sync, convert, state, eef | `kiwi run` |
| per dataset | export | `kiwi export run` |

## Data layout

```
data/scene/
  scene.yaml          scene manifest written by kiwi scene init
  profiles/           copy of the mapping profile, written by kiwi scene init and pinned by digest in scene.yaml
  scene_map/          maps published by kiwi scene build
  demo/               one episode
    episode.yaml      streams, cameras, tool, clock, scene link
    raw/              recordings
    config/           copy of the configuration used by this episode
    left/  right/     lens videos, scene/state.tum, eef.npz
    head/  ground/    optional results
```

## Environments

Commands run in Pixi environments selected with `pixi run -e <env>` or `pixi shell -e <env>`:

| Environment | Used for |
| --- | --- |
| `state` | `kiwi import`, `kiwi init`, `kiwi run`, the scene commands, `kiwi export`, `kiwi real2simviz` |
| `slam-offline` | building the ORB-SLAM3 engine |
| `mask-gpu` | `kiwi scene build` with the segmentation model on CUDA |
| `viz` | `kiwi viz` |
| `tune` | `kiwi eef calibrate`, `kiwi eef tune` |
| `ground` | the optional `ground` stage |
| `replay` | `kiwi replay` |
| `lerobot` | loading an export with LeRobot |

## Quick start

```bash
pixi install
git submodule update --init third_party/ORB_SLAM3
pixi run -e slam-offline build-orbslam3
pixi shell -e state

kiwi import --dest data/scratch
mkdir -p data/scene/demo/raw && mv data/scratch/*.insv data/scene/demo/raw/
kiwi init --episode data/scene/demo --left insta360-X5-001 --right insta360-X5-002
# fill in tool, gripper_unit and the capture metadata in episode.yaml
kiwi run --episode data/scene/demo --profile prepare
kiwi scene init --root data/scene --mapping-episode demo
pixi run -e mask-gpu -- kiwi scene build --scene data/scene/scene.yaml
kiwi run --episode data/scene/demo
pixi run -e viz -- kiwi viz play data/scene/demo
kiwi export run --episodes data/scene/demo --output data/demo_lerobot
```

## Pages

| Step | Page |
| --- | --- |
| Build the rig and calibrate the cameras | [Hardware and cameras](hardware.md) |
| Record, initialize and process episodes | [Collect and process](processing.md) |
| Scene maps, camera state, tool state, ground | [Estimation](estimation.md) |
| Output files and dataset export | [Data and export](data.md) |
| Robot replay and Gaussian scenes | [Replay and Gaussian scenes](replay.md) |

# Hardware and cameras

## Hardware

KIWI records with two Insta360 X5 cameras on passive Chopstick grippers, and optionally an Insta360 GO 3 on a hat.
The mechanical files are in [KIWI_hardware](https://github.com/TToTMooN/KIWI_hardware).

| Configuration | Cameras | Views |
| --- | --- | --- |
| 1 | 2× Insta360 X5 | wrists |
| 2 | 2× Insta360 X5 + 1× Insta360 GO 3 | wrists + ego |

### Parts

| Item | Quantity | Files |
| --- | --- | --- |
| Insta360 X5 | 2 | |
| Printed Chopstick gripper with red dot pattern and marker | 2 | [chopsticks](https://github.com/TToTMooN/KIWI_hardware/tree/main/chopsticks) |
| Bolts and heat-set inserts | as required | |
| Insta360 GO 3 (configuration 2) | 1 | |
| GO 3 hat mount with tag plate, stiff-brim hat | 1 | [ego_cam](https://github.com/TToTMooN/KIWI_hardware/tree/main/ego_cam/assets) |
| Hat fiducial, printed at true size | 1 | [tags](https://github.com/TToTMooN/KIWI_hardware/tree/main/tags) |

### Assembly checks

1. Both jaws move through their full travel without binding.
2. Each X5 stays fixed relative to its gripper while the jaws open and close.
3. The jaw dots and the gripper marker are visible in the X5 front view over the full travel.
4. The hat camera and tag do not move relative to each other, and the wrist cameras see the tag during the task.
5. Measure the printed tag edge; `kiwi init` defaults to `DICT_4X4_100` and a 0.049 m edge (`--head-marker-dictionary`, `--head-marker-size`). The tag id has no default: pass it with `--head-marker-id` (the examples use 50), or no `head_marker` block is written and the head stage is skipped.

### Camera setup

- Keep each camera's recording mode fixed after calibration; the A/V delay table is per mode.
- Enable voice control and check that every camera responds to "start recording" and "stop recording".
- Play a shared sound that every camera hears during recording; `sync` aligns the streams on it.

Calibrate each camera ([Cameras](#cameras)) and each gripper build ([Tool state](estimation.md#calibrate-a-gripper-build)) before collecting data.
The two shipped gripper units `chopstick-m0` and `chopstick-m1` carry the same camera-to-marker pose; treat it as a starting point and calibrate each build you print.

### First episode

1. Start the shared sound, say "start recording", and check that every camera records.
2. Perform a short task that opens and closes both grippers, then say "stop recording".
3. Connect each camera in file-transfer mode and import its clip with `kiwi import --dest data/first_session`.
4. Group the clips into an episode and run `kiwi init` ([Collect](processing.md#collect)).

## Cameras

### Registry

`config/cameras/<unit>/` holds one physical camera: `camera.yaml` and the calibration files it points to.

```yaml
serial: "IAHYA2511CWJEA"
model: Insta360 X5
nickname: insta360-X5-001
intrinsics: intrinsics.npz
intrinsics_kb4: intrinsics-kb4.json
intrinsics_kb4_back: intrinsics-kb4-back.json
extrinsic_imu: extrinsic-imu.yaml
extrinsic_imu_back: extrinsic-imu-back.yaml
imu_noise: imu-noise.yaml
av_sync: av-sync.yaml
```

| Pointer | Contents | Written by |
| --- | --- | --- |
| `intrinsics` | DoubleSphere model per lens (GO 3: pinhole-equi, one file per video mode) | `kiwi calib intrinsics publish` |
| `intrinsics_kb4`, `intrinsics_kb4_back` | KB4 fit of the front / rear lens | `kiwi calib intrinsics kb4` |
| `extrinsic_imu`, `extrinsic_imu_back` | Kalibr camera-IMU chain of the front / rear lens | `kiwi calib imu cam`, copied by hand after `kiwi calib imu check` |
| `imu_noise` | Allan noise densities | `kiwi calib imu noise` |
| `av_sync` | audio-vs-video delay per recording mode | `kiwi calib qr av-delay` |

`kiwi init` binds each arm to its unit by the serial embedded in the `.insv` clip.
The GO 3 records no readable serial (`serial: null`) and is selected by name.
A unit lacks the pointers for calibrations it does not have.

Each publishing verb (`intrinsics publish`, `intrinsics kb4`, `imu noise`, `qr av-delay`) writes a new dated file into the unit directory and prints the pointer line; set that pointer in `camera.yaml` to use it.
`imu cam` leaves Kalibr's camchain in its `--output-dir`; copy it in by hand (see [Camera-IMU](#camera-imu-per-lens)).
An existing episode keeps its own `<episode>/config/` copy.

### Setup

Intrinsics and camera-IMU calibration run Kalibr in Docker:

```bash
git submodule update --init third_party/kalibr
docker build -t kalibr:latest -f third_party/kalibr/Dockerfile_ros1_20_04 third_party/kalibr
```

`KIWI_KALIBR_IMAGE` selects another image tag.
Record the X5 at 4K, 120 fps, with clean lenses.
The default target is the AprilGrid in `config/markers/calibio_kalibr_apriltag_board.yaml` (`--marker-config` selects another).

### Intrinsics (per lens)

Point the lens at the board and cover the fisheye periphery; one recording can serve both the intrinsics and the camera-IMU solve.

```bash
kiwi calib intrinsics run --insv <rec>.insv --output-dir data/kalibr/<unit>_front --lenses front
kiwi calib intrinsics publish --front-run data/kalibr/<unit>_front --camera <unit> \
    --base <current intrinsics.npz>     # keeps the lens that was not re-solved
```

Each lens needs its own recording and `run` directory; `publish` takes `--front-run` and/or `--back-run`.
Accept a solve with reprojection std around 0.4–0.5 px and `fx` near 505–517 px (X5).

### Camera-IMU (per lens)

Hold one lens on the board for 2–4 minutes: 5 s still, 15 s slow figure-8, 30 s of rotations (±30° per axis), 30 s of translations (±30 cm per axis), 15 s of fast shaking without blur.

```bash
kiwi calib imu cam --insv <rec>.insv --camera <unit> --lens back \
    --output-dir data/kalibr/<unit>_imucam_back --frame-stride 20 --bag-from-to 5 95
kiwi calib imu check --units <unit> --back data/kalibr/<unit>_imucam_back/imucam_calib-camchain-imucam.yaml
```

`--frame-stride` is in frames (20 at 120 fps, 5 at 30 fps); `--bag-from-to` is the window in which the board is visible.
Accept with reprojection around 0.7 px, gravity 9.8065 m/s² and a time offset of −4 to −8 ms.
`kiwi calib imu check` composes the two lenses: `T_back_front` should be about 180° and 20–40 mm.
`imu cam` writes nothing into the unit directory: Kalibr's result stays at `<output-dir>/imucam_calib-camchain-imucam.yaml`. Once `kiwi calib imu check` passes, copy that file into `config/cameras/<unit>/` and set `extrinsic_imu:` (front) or `extrinsic_imu_back:` (rear) in `camera.yaml` to its name.

### GO 3

The GO 3 has one lens and one intrinsics file per video mode (`intrinsics: {2688x2688: ..., 2720x1536: ..., 2560x1440: ...}`), solved with the `pinhole-equi` model:

```bash
kiwi calib intrinsics run --insv PRO_VID_<...>.mp4 --model pinhole-equi --output-dir data/kalibr/<unit>_front
kiwi calib intrinsics publish --front-run data/kalibr/<unit>_front --camera <unit>
kiwi calib imu cam --insv PRO_VID_<...>.mp4 --camera <unit> --output-dir data/kalibr/<unit>_imucam --frame-stride 5
```

`--focal-init-px` sets Kalibr's initial focal length (default: half the image width).

### KB4

```bash
kiwi calib intrinsics kb4 --camera <unit> --lens front
kiwi calib intrinsics kb4 --camera <unit> --lens back
```

Fits KB4 to the published DoubleSphere model; no new recording is needed.

### IMU noise

Record about 2 h of video with the camera still, then:

```bash
kiwi calib imu extract --insv <dir-of-parts>
kiwi calib imu noise --imu-npz <dir>/imu_raw.npz --camera <unit> \
    --discard-seconds 1200 --trim-end-seconds 30
```

### Audio/video delay

The delay between a camera's audio and video tracks is a constant per recording mode.
Film a monitor that shows the rolling QR clock with both cameras at once, then:

```bash
kiwi calib qr clock                                       # on the machine driving the monitor
kiwi calib qr sync L_lens0.mp4 R.mp4 --left-undistort <intrinsics.npz> \
    --left-insv L.insv --right-insv R.insv --audio --json-output run1.json
kiwi calib qr av-delay L.insv R.insv --camera <unit> --qr-json run1.json
```

With three cameras, film all three at once and run `sync` and `av-delay` for each camera against the same reference clip; the reference defines zero delay (`--ref-delay-s`).
Repeated runs of a mode are averaged into the table.

For the gripper's camera-to-marker pose, see [Tool state](estimation.md#calibrate-a-gripper-build).

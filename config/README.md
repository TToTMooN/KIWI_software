# config/

Deployment configuration, resolved by name.

| Path | Contents |
|---|---|
| `cameras/<unit>/camera.yaml` | The unit's `serial:`, `model` and `nickname`, and one pointer key per calibration file: `intrinsics`, `intrinsics_kb4`, `intrinsics_kb4_back`, `extrinsic_imu`, `extrinsic_imu_back`, `imu_noise`, `av_sync`. A recording is matched to its unit by serial. |
| `cameras/<unit>/*.npz, *.json, *.yaml` | The calibration files the pointers name. |
| `gripper/<tool>/tool.yaml` | One tool design (`tool_v1`): `design:` (estimator, marker dictionary, `capture:` and `replay:` descriptions) and the `session:` recipe defaults. |
| `gripper/<tool>/units/<unit>.yaml` | One physical build (`chopstick_unit_v1`): `build:` (`marker: {id, size}`, calibrated `fixed_pose`, optional `mount_tcp:` / `mount_cam:`) and `session:` overrides. |
| `markers/` | Calibration board definitions. |
| `scene/mapping-profiles/` | Scene mapping recipes. |

A unit that records in several video modes (the GO 3) has one intrinsics file per mode; its `intrinsics:` pointer is a mapping keyed by `"<W>x<H>"`.

Session settings layer as builtin < tool < unit < the episode arm's `session:` block < `kiwi eef run --set KEY=VALUE`.
Unknown keys and schema tags are refused.

Shipped units: `insta360-X5-001` to `-003` (full calibration), `insta360-X5-004` (registry-only: it binds recordings by serial and syncs, but has no validated front-lens fit and no IMU-noise file, so scene mapping and `state` cannot use it), `insta360-GO3-001` (ego camera), and the `chopstick` tool with builds `chopstick-m0` and `chopstick-m1`.

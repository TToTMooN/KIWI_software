# gripper-unet - keypoint-mask segmentation

The production model behind scene mapping's `masking: learned` mode: per-frame semantic segmentation whose union with the vignette ring excludes the operator's own hardware and body from ORB keypoint extraction during map builds.

- **Architecture / format**: UNet, exported to ONNX (opset 18, ~31 MB).
- **Distribution**: the weights are a GitHub release asset, not repo content.
  Release tag `model/gripper-unet-v1` of `TToTMooN/KIWI_software`, asset `gripper-unet-v1.onnx`, 31,126,465 bytes, sha256 `ef94000bbc65ec33c6f802beb8751f4d6d4f6a8e082f97e5763b06c226e8e8b0`.
  This directory keeps the contract - the sidecar, this card, and the golden record - while `kiwi models fetch` puts the bytes in the local model cache (`$KIWI_MODEL_CACHE`, else `$XDG_CACHE_HOME/kiwi/models`, else `~/.cache/kiwi/models`), verified against the pinned hash before anything is published there.
- **Classes**: `background(0)`, `own_gripper(1)`, `other_gripper(2)`, `person(3)`; classes 1–3 are masked out (`mask_out`).
- **Input**: 640x640 RGB, NCHW, ImageNet normalization, declared in `gripper-unet.onnx.meta.json`.
  The loader validates required sidecar fields and checks the model bytes against the hash declared there.
- **Output convention**: argmax over class logits; published masks are 0/255 uint8, nonzero = excluded (the engine convention).
- **Training data**: frames from dual-arm chopstick-gripper sessions recorded by the maintainers, annotated for the classes above.
  No third-party datasets.
- **Intended use**: keypoint exclusion for scene map building on the dual-arm chopstick-gripper camera rig represented in the training footage.
  It is not a general segmentation model; behavior on other rigs, optics, or scenes is unevaluated.
- **Known limits**: CPU and CUDA outputs are not byte-identical, so they never share a cache identity.
- **Determinism**: a golden CPU inference (`gripper-unet.golden.json`) pins the exact output hash for a fixed input; the test suite verifies it on every run with onnxruntime and opencv installed and the model cached (`kiwi models fetch`).

"""Learned gripper/person segmentation for keypoint masking.

The production model (a UNet exported to ONNX, ~31 MB, trained on
dual-arm chopstick-gripper capture footage) is declared in-repo by a
sidecar that pins the model hash,
while the weights themselves are a GitHub release asset fetched once into
the local model cache (``kiwi models fetch``, or on first use). Everything
about how a mask is produced - preprocess schema, class mapping, polarity,
the supported onnxruntime range, and the ACTUAL execution provider - is
validated and recorded; CPU and CUDA outputs never share a cache identity.

Component rules: imports ``kiwi.core`` and itself only. onnxruntime is an
optional dependency (the ``mask`` / ``mask-gpu`` pixi environments); every
other pipeline path never imports it.
"""

from kiwi.perception.gripper_segmentation.artifact import (
    default_model_path,
    load_artifact,
)
from kiwi.perception.gripper_segmentation.inference import (
    open_runtime,
    preprocess,
    resolve_device,
    segment,
)
from kiwi.perception.gripper_segmentation.store import (
    PACKAGED_SIDECAR,
    model_cache_dir,
    resolve_model,
)
from kiwi.perception.gripper_segmentation.types import (
    ArtifactError,
    DeviceError,
    DevicePlan,
    MaskRuntime,
    SegmentationArtifact,
)

__all__ = [
    "PACKAGED_SIDECAR",
    "ArtifactError",
    "DeviceError",
    "DevicePlan",
    "MaskRuntime",
    "SegmentationArtifact",
    "default_model_path",
    "load_artifact",
    "model_cache_dir",
    "open_runtime",
    "preprocess",
    "resolve_device",
    "resolve_model",
    "segment",
]

"""Deterministic gripper-segmentation inference over a validated artifact.

Device policy: 'cuda' FAILS CLOSED when the CUDA provider does not actually
load (onnxruntime lists providers optimistically AND session creation
silently falls back to CPU, so the check is against the session's ACTUAL
provider); 'auto' prefers CUDA and warns loudly on CPU fallback; 'cpu' is
explicit. The actual provider and onnxruntime version are part of the
runtime identity; downstream caches never mix outputs across devices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifact import load_artifact
from .types import (
    CPU_EP,
    CUDA_EP,
    MASK_DEVICES,
    ArtifactError,
    DeviceError,
    DevicePlan,
    MaskRuntime,
    SegmentationArtifact,
)

AUTO_CPU_FALLBACK_WARNING = (
    "WARNING: mask_device=auto but the CUDA execution provider is unavailable; "
    "falling back to CPU inference. Pass --mask-device cpu to "
    "silence, or fix the CUDA runtime."
)


def _onnxruntime():
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - env misconfiguration
        raise DeviceError(
            f"onnxruntime is not importable in this environment ({exc}); an "
            "unavailable runtime is an invalid inference setup, never a silent "
            "downgrade"
        ) from exc
    return onnxruntime


def resolve_device(requested: str) -> DevicePlan:
    if requested not in MASK_DEVICES:
        raise DeviceError(f"mask_device {requested!r} not in {MASK_DEVICES}")
    if requested == "cpu":
        return DevicePlan("cpu", (CPU_EP,))
    if requested == "cuda":
        return DevicePlan("cuda", (CUDA_EP,))
    if CUDA_EP in _onnxruntime().get_available_providers():
        return DevicePlan("auto", (CUDA_EP, CPU_EP))
    print(AUTO_CPU_FALLBACK_WARNING)
    return DevicePlan("auto", (CPU_EP,), warning=AUTO_CPU_FALLBACK_WARNING)


def _check_runtime_supported(supported: str, actual: str) -> None:
    """ENFORCE the sidecar's supported onnxruntime range - an out-of-range
    runtime is a typed failure, never a warning."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    try:
        spec = SpecifierSet(supported)
        version = Version(actual)
    except Exception as exc:  # noqa: BLE001 - malformed spec/version is fatal too
        raise ArtifactError(
            f"cannot evaluate supported_onnxruntime {supported!r} against "
            f"onnxruntime {actual!r}: {exc}"
        ) from exc
    if version not in spec:
        raise DeviceError(
            f"onnxruntime {actual} is outside the artifact's supported range "
            f"{supported!r}; install a supported runtime or re-export/qualify "
            "the model - an unqualified runtime never runs silently"
        )


def open_runtime(model: Path | SegmentationArtifact, device: DevicePlan) -> MaskRuntime:
    """Validate the artifact, enforce its supported runtime range, open one
    ONNX session honoring the device plan, and pin the ACTUAL execution
    identity."""
    ort = _onnxruntime()
    artifact = (
        model if isinstance(model, SegmentationArtifact) else load_artifact(model)
    )
    supported = artifact.sidecar["supported_onnxruntime"]
    _check_runtime_supported(supported, ort.__version__)
    try:
        ort.preload_dlls()  # locate pip-installed CUDA libs; harmless on CPU
    except Exception:  # noqa: BLE001 - best effort
        pass
    try:
        sess = ort.InferenceSession(
            str(artifact.model), providers=list(device.providers)
        )
    except Exception as exc:  # noqa: BLE001 - typed: callers surface SCENE-FATAL
        raise DeviceError(
            f"onnxruntime session creation failed for {artifact.model} "
            f"(providers {device.providers}): {exc}"
        ) from exc
    actual = sess.get_providers()[0]
    if device.requested == "cuda" and actual != CUDA_EP:
        raise DeviceError(
            "mask_device=cuda but the CUDA execution provider failed to load "
            f"(actual: {actual}); fix the CUDA runtime"
        )
    if device.requested == "auto" and CUDA_EP in device.providers and actual != CUDA_EP:
        print(AUTO_CPU_FALLBACK_WARNING)
    return MaskRuntime(
        artifact=artifact,
        device=device,
        session=sess,
        actual_provider=actual,
        ort_version=f"{ort.__version__} (supported: {supported})",
    )


def preprocess(img_bgr: np.ndarray, schema: dict) -> np.ndarray:
    import cv2

    h, w = schema["input_size"]
    if (img_bgr.shape[0], img_bgr.shape[1]) != (h, w):
        img_bgr = cv2.resize(img_bgr, (w, h))
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if schema["channel_order"] == "rgb":
        img_bgr = img_bgr[:, :, ::-1]
    arr = img_bgr.astype(np.float32)
    norm = schema["normalize"]
    if norm == "unit":
        arr /= 255.0
    else:
        arr = (arr / 255.0 - np.asarray(norm["mean"], np.float32)) / np.asarray(
            norm["std"], np.float32
        )
    if schema["layout"] == "nchw":
        arr = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(arr[None])


def segment(runtime: MaskRuntime, img_bgr: np.ndarray) -> np.ndarray:
    """One frame -> 0/255 uint8 semantic EXCLUDE mask at the input's size
    (255 = masked out, the engine convention)."""
    import cv2

    x = preprocess(img_bgr, runtime.artifact.preprocess_schema)
    out = runtime.session.run(None, {runtime.session.get_inputs()[0].name: x})[0]
    cls = np.asarray(out).squeeze()
    if cls.ndim == 3:  # class logits/probs -> argmax over the channel axis
        cls = cls.argmax(axis=0 if cls.shape[0] < cls.shape[-1] else -1)
    seg = np.isin(cls, runtime.artifact.mask_out).astype(np.uint8) * 255
    if seg.shape != img_bgr.shape[:2]:
        seg = cv2.resize(seg, img_bgr.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return seg

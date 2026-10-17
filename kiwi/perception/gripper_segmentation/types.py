"""Typed contracts for gripper-segmentation inference."""

from __future__ import annotations

import dataclasses
from pathlib import Path


class ArtifactError(RuntimeError):
    """The model artifact (ONNX + sidecar) is missing, tampered, or invalid."""


class DeviceError(RuntimeError):
    """The requested execution provider is unavailable (fail-closed)."""


MASK_DEVICES = ("auto", "cuda", "cpu")
CUDA_EP = "CUDAExecutionProvider"
CPU_EP = "CPUExecutionProvider"


@dataclasses.dataclass(frozen=True)
class SegmentationArtifact:
    """A validated, self-contained model artifact: the ONNX file plus its
    sidecar contract. Never construct directly - ``artifact.load_artifact``
    is the only validated path."""

    model: Path
    sidecar: dict

    @property
    def model_sha256(self) -> str:
        return self.sidecar["model_sha256"]

    @property
    def preprocess_schema(self) -> dict:
        return self.sidecar["preprocess_schema"]

    @property
    def mask_out(self) -> list[int]:
        return list(self.sidecar["class_mapping"]["mask_out"])


@dataclasses.dataclass(frozen=True)
class DevicePlan:
    requested: str  # auto | cuda | cpu
    providers: tuple[str, ...]
    warning: str | None = None


@dataclasses.dataclass(frozen=True)
class MaskRuntime:
    """An OPEN inference runtime: artifact + session + the identity of what
    is actually executing. ``actual_provider`` and ``ort_version`` are part
    of downstream cache/provenance identity - CPU and CUDA outputs never
    share a cache."""

    artifact: SegmentationArtifact
    device: DevicePlan
    session: object
    actual_provider: str
    ort_version: str

    def identity(self) -> dict:
        """The runtime-identity block bound into mask recipes/provenance."""
        return {
            "model_sha256": self.artifact.model_sha256,
            "model_version": self.artifact.sidecar.get("model_version"),
            "sidecar_schema": self.artifact.sidecar.get("schema"),
            "onnxruntime_version": self.ort_version,
            "execution_provider": self.actual_provider,
        }

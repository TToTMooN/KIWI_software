"""Model-artifact loading: the sidecar declares the ONNX hash and inference
contract. A bare checkpoint, a missing/invalid sidecar, or a model hash
mismatch is an ArtifactError - never a silent downgrade.

The sidecar is repo content; the weights it pins are not. ``store`` fetches
them from the release the sidecar names into the local model cache, and
``load_artifact`` then validates that pair like any other."""

from __future__ import annotations

import json
from pathlib import Path

from .store import PACKAGED_SIDECAR, resolve_model, sha256_file
from .types import ArtifactError, SegmentationArtifact

SIDECAR_SCHEMA = "segmentation_model"
SUPPORTED_MODEL_FORMATS = ("onnx",)
REQUIRED_SIDECAR_KEYS = (
    "model_format",
    "model_sha256",
    "model_version",
    "onnx_opset",
    "supported_onnxruntime",
    "preprocess_schema",
    "class_mapping",
    "mask_polarity",
)
REQUIRED_PREPROCESS_KEYS = ("input_size", "channel_order", "normalize", "layout")


def default_model_path() -> Path:
    """The declared production artifact, in the local model cache - fetched
    from its release on first use (see ``store.resolve_model``)."""
    return resolve_model(PACKAGED_SIDECAR, fetch=True)


def load_artifact(model: Path) -> SegmentationArtifact:
    """Validate and load one self-contained model artifact (fail-closed)."""
    model = Path(model)
    if not model.exists():
        raise ArtifactError(f"segmentation model missing: {model}")
    sidecar_path = model.with_suffix(model.suffix + ".meta.json")
    if not sidecar_path.exists():
        raise ArtifactError(
            f"segmentation model {model} has no {sidecar_path.name} sidecar; a bare "
            "checkpoint is not a self-contained inference artifact"
        )
    try:
        sidecar = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{sidecar_path}: unreadable model sidecar: {exc}") from exc
    if sidecar.get("schema") != SIDECAR_SCHEMA:
        raise ArtifactError(
            f"{sidecar_path}: schema {sidecar.get('schema')!r} != {SIDECAR_SCHEMA}"
        )
    for k in REQUIRED_SIDECAR_KEYS:
        if k not in sidecar or sidecar[k] in (None, ""):
            raise ArtifactError(f"{sidecar_path}: sidecar missing '{k}'")
    if sidecar["model_format"] not in SUPPORTED_MODEL_FORMATS:
        raise ArtifactError(
            f"{sidecar_path}: model_format {sidecar['model_format']!r} not in "
            f"{SUPPORTED_MODEL_FORMATS}"
        )
    pp = sidecar["preprocess_schema"]
    if not isinstance(pp, dict) or any(k not in pp for k in REQUIRED_PREPROCESS_KEYS):
        raise ArtifactError(
            f"{sidecar_path}: preprocess_schema must declare {REQUIRED_PREPROCESS_KEYS}"
        )
    cm = sidecar["class_mapping"]
    if not isinstance(cm, dict) or "mask_out" not in cm or not cm["mask_out"]:
        raise ArtifactError(
            f"{sidecar_path}: class_mapping must declare 'mask_out' classes"
        )
    if sidecar["mask_polarity"] != "nonzero_excluded_0_255":
        raise ArtifactError(
            f"{sidecar_path}: mask_polarity {sidecar['mask_polarity']!r} is not the "
            "engine convention 'nonzero_excluded_0_255'"
        )
    if sha256_file(model) != sidecar["model_sha256"]:
        raise ArtifactError(f"{model}: content does not match sidecar model_sha256")
    return SegmentationArtifact(model=model, sidecar=sidecar)

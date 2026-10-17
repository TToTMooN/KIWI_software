"""perception.gripper_segmentation: artifact fail-closed rules, device
policy, and a golden inference regression over the DECLARED model.

The inference tests here run onnxruntime on the unet_v1 artifact to check
that the published weights produce the recorded mask. They need
onnxruntime, opencv and a populated model cache (`kiwi models fetch`, for
example in the `mask` or `mask-gpu` pixi environment); they never download
the release weights themselves. Missing dependencies or cached weights
cause affected inference tests to skip, except that KIWI_REQUIRE_CUDA=1
requires OpenCV and CUDA availability in the CUDA check.

The fail-closed rules are about the SIDECAR contract, not the weights, so
they run against a synthetic artifact and must never skip anywhere."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from kiwi.perception.gripper_segmentation.types import CPU_EP, CUDA_EP
from kiwi.perception.gripper_segmentation import (
    PACKAGED_SIDECAR,
    ArtifactError,
    DeviceError,
    DevicePlan,
    load_artifact,
    open_runtime,
    preprocess,
    resolve_device,
    resolve_model,
    segment,
)
from kiwi.perception.gripper_segmentation.store import cached_model_path


@pytest.fixture(scope="module")
def model_path():
    """The cached production weights - never fetched from a test.

    An absent cache entry is a skip (a fresh checkout has none); an entry
    whose bytes do NOT match the sidecar is left to fail."""
    doc = json.loads(PACKAGED_SIDECAR.read_text())
    if not cached_model_path(doc).exists():
        pytest.skip("segmentation model not cached; run 'kiwi models fetch'")
    return resolve_model(fetch=False)


# ------------------------------------------------------------- artifact


@pytest.fixture
def artifact(tmp_path):
    """A self-contained artifact needing neither the cache nor real weights:
    the REPO's sidecar, re-pinned to throwaway bytes. load_artifact's rules
    are sidecar rules, so they must run everywhere - including the CI lane,
    which has no model cache and no onnxruntime."""
    model = tmp_path / "unet.onnx"
    model.write_bytes(b"stand-in-weights" * 64)
    doc = json.loads(PACKAGED_SIDECAR.read_text())
    doc["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    sidecar = model.with_suffix(model.suffix + ".meta.json")
    sidecar.write_text(json.dumps(doc))
    return model, sidecar


def test_the_repo_sidecar_is_a_valid_contract(artifact):
    model, _ = artifact
    a = load_artifact(model)
    assert a.sidecar["model_version"] == "unet_v1"
    assert a.mask_out == [1, 2, 3]
    assert a.preprocess_schema["input_size"] == [640, 640]


def test_packaged_artifact_loads(model_path):
    a = load_artifact(model_path)
    assert a.sidecar["model_version"] == "unet_v1"
    assert a.mask_out == [1, 2, 3]
    assert a.preprocess_schema["input_size"] == [640, 640]


def test_tampered_model_fails_closed(artifact):
    model, _ = artifact
    model.write_bytes(model.read_bytes() + b"x")
    with pytest.raises(ArtifactError, match="model_sha256"):
        load_artifact(model)


def test_tampered_sidecar_fails_closed(artifact):
    model, sidecar = artifact
    doc = json.loads(sidecar.read_text())
    doc["class_mapping"] = {"mask_out": []}
    sidecar.write_text(json.dumps(doc))
    with pytest.raises(ArtifactError, match="class_mapping"):
        load_artifact(model)
    doc = json.loads(sidecar.read_text())
    del doc["supported_onnxruntime"]
    doc["class_mapping"] = {"mask_out": [1]}
    sidecar.write_text(json.dumps(doc))
    with pytest.raises(ArtifactError, match="supported_onnxruntime"):
        load_artifact(model)


def test_missing_sidecar_fails_closed(artifact):
    model, sidecar = artifact
    sidecar.unlink()
    with pytest.raises(ArtifactError, match="sidecar"):
        load_artifact(model)


def test_wrong_polarity_fails_closed(artifact):
    model, sidecar = artifact
    doc = json.loads(sidecar.read_text())
    doc["mask_polarity"] = "nonzero_included"
    sidecar.write_text(json.dumps(doc))
    with pytest.raises(ArtifactError, match="mask_polarity"):
        load_artifact(model)


# ------------------------------------------------------------- device policy


def test_resolve_device_rejects_unknown():
    with pytest.raises(DeviceError, match="mask_device"):
        resolve_device("gpu")


def test_resolve_device_auto_falls_back_with_loud_warning(monkeypatch, capsys):
    onnxruntime = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: [CPU_EP])
    plan = resolve_device("auto")
    assert plan.providers == (CPU_EP,)
    assert plan.warning is not None
    assert "falling back to CPU" in capsys.readouterr().out


def test_cuda_fails_closed_on_silent_cpu_fallback(monkeypatch, model_path):
    onnxruntime = pytest.importorskip("onnxruntime")

    class _CpuOnlySession:
        def get_providers(self):
            return [CPU_EP]

    monkeypatch.setattr(
        onnxruntime, "InferenceSession", lambda *a, **k: _CpuOnlySession()
    )
    monkeypatch.setattr(onnxruntime, "preload_dlls", lambda: None, raising=False)
    with pytest.raises(DeviceError, match="mask_device=cuda"):
        open_runtime(model_path, DevicePlan("cuda", (CUDA_EP,)))


# ------------------------------------------------- real CPU inference (golden)


def _golden_input():
    """Deterministic synthetic BGR frame: fixed-seed noise + a bright blob."""
    rng = np.random.default_rng(0)
    img = (rng.random((960, 960, 3)) * 255).astype(np.uint8)
    img[300:600, 300:600] = (40, 40, 200)  # red-ish blob (BGR)
    return img


def test_cpu_inference_runs_and_is_deterministic(model_path):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    rt = open_runtime(model_path, DevicePlan("cpu", (CPU_EP,)))
    assert rt.actual_provider == CPU_EP
    img = _golden_input()
    a, b = segment(rt, img), segment(rt, img)
    assert a.dtype == np.uint8 and a.shape == (960, 960)
    assert set(np.unique(a)) <= {0, 255}
    assert (a == b).all()  # deterministic on the same session
    # golden regression: pin the exact CPU output for this fixed input. If a
    # model/preprocess/runtime change alters it, this hash MUST be reviewed
    # and updated deliberately (runtime identity contributes to the mask cache key).
    digest = hashlib.sha256(a.tobytes()).hexdigest()
    golden = json.loads(
        (PACKAGED_SIDECAR.parent / "gripper-unet.golden.json").read_text()
    )
    assert digest == golden["cpu_golden_sha256"], (
        f"CPU inference output changed: {digest} != recorded {golden['cpu_golden_sha256']}"
    )


def test_preprocess_matches_contract(artifact):
    # preprocess reads the sidecar's schema and never the graph, so this runs
    # off the synthetic artifact - and therefore everywhere.
    pytest.importorskip("cv2")
    a = load_artifact(artifact[0])
    x = preprocess(_golden_input(), a.preprocess_schema)
    assert x.shape == (1, 3, 640, 640)
    assert x.dtype == np.float32


REQUIRE_CUDA = os.environ.get("KIWI_REQUIRE_CUDA") == "1"


def _cuda_unavailable(reason: str):
    """Report unavailable CUDA as a failure when KIWI_REQUIRE_CUDA=1,
    so a detected CUDA failure cannot pass as a successful test.
    Otherwise skip the test."""
    if REQUIRE_CUDA:
        pytest.fail(f"KIWI_REQUIRE_CUDA=1 but {reason}")
    pytest.skip(reason)


def test_cuda_inference_matches_cpu_within_boundary_tolerance(model_path):
    # cv2 FIRST: its Qt needs the env's libstdc++, and letting the NVIDIA
    # wheels load first can bind the process to the older system libstdc++.
    # With KIWI_REQUIRE_CUDA=1 an unavailable cv2 is a failure.
    if REQUIRE_CUDA:
        try:
            import cv2  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"KIWI_REQUIRE_CUDA=1 but cv2 is unavailable: {exc}")
    else:
        pytest.importorskip("cv2")
    onnxruntime = pytest.importorskip("onnxruntime")

    if CUDA_EP not in onnxruntime.get_available_providers():
        _cuda_unavailable("the CUDA execution provider is not available")
    cpu = open_runtime(model_path, DevicePlan("cpu", (CPU_EP,)))
    try:
        gpu = open_runtime(model_path, DevicePlan("cuda", (CUDA_EP,)))
    except DeviceError as exc:
        # listed-but-unloadable OR session silently fell back to CPU
        _cuda_unavailable(f"the CUDA runtime failed to load: {exc}")
    assert gpu.actual_provider == CUDA_EP  # never a silent CPU fallback
    img = _golden_input()
    a, b = segment(cpu, img), segment(gpu, img)  # a REAL CUDA inference ran
    assert a.shape == b.shape == (960, 960)
    # devices are equivalent, not byte-identical: tolerate boundary argmax
    # flips only
    assert (a != b).mean() < 0.001


def test_runtime_version_below_supported_fails_closed(monkeypatch, model_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(onnxruntime, "__version__", "1.20.0")
    with pytest.raises(DeviceError, match="outside the artifact's supported range"):
        open_runtime(model_path, DevicePlan("cpu", (CPU_EP,)))


def test_runtime_version_above_supported_fails_closed(monkeypatch, model_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(onnxruntime, "__version__", "2.1.0")
    with pytest.raises(DeviceError, match="outside the artifact's supported range"):
        open_runtime(model_path, DevicePlan("cpu", (CPU_EP,)))


def test_runtime_version_in_supported_range_opens(monkeypatch, model_path):
    pytest.importorskip("onnxruntime")
    # the environment's real version must sit inside the sidecar's range -
    # open_runtime enforcing it is exactly what the CPU golden test relies on
    rt = open_runtime(model_path, DevicePlan("cpu", (CPU_EP,)))
    assert rt.actual_provider == CPU_EP

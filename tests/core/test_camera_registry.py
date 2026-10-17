"""Unit tests for kiwi.core.camera.registry - serial -> camera-unit resolution.

Run: pixi run test
Pure yaml/path logic; the .insv telemetry probe itself is monkeypatched.
"""

from __future__ import annotations

import datetime

import pytest
from pathlib import Path

from kiwi.core.camera import registry
from kiwi.core.camera.registry import find_camera_by_serial, resolve_camera_for_video


def _make_registry(tmp_path: Path) -> Path:
    cams = tmp_path / "cameras"
    for unit, serial in [
        ("insta360-X5-001", "IAHYA2511CWJEA"),
        ("insta360-X5-002", "IAHYA2511DVC37"),
    ]:
        d = cams / unit
        d.mkdir(parents=True)
        (d / "camera.yaml").write_text(f'serial: "{serial}"\n')
    (cams / "insta360-X5-001" / "intrinsics.npz").write_bytes(b"")
    return cams


def test_find_camera_by_serial_exact(tmp_path):
    cams = _make_registry(tmp_path)
    assert find_camera_by_serial(cams, "IAHYA2511DVC37").name == "insta360-X5-002"


def test_find_camera_by_serial_substring_both_ways(tmp_path):
    cams = _make_registry(tmp_path)
    # probed serial carries an extra prefix over the registry's
    assert find_camera_by_serial(cams, "XXIAHYA2511CWJEAYY").name == "insta360-X5-001"
    # registry carries more than the probed serial
    assert find_camera_by_serial(cams, "2511DVC37").name == "insta360-X5-002"


def test_find_camera_by_serial_no_match(tmp_path):
    cams = _make_registry(tmp_path)
    assert find_camera_by_serial(cams, "UNKNOWN0000000") is None
    assert find_camera_by_serial(cams, "") is None


def test_resolve_camera_for_video(tmp_path, monkeypatch):
    cams = _make_registry(tmp_path)
    monkeypatch.setattr(registry, "read_insv_serial", lambda p: "IAHYA2511CWJEA")
    m = resolve_camera_for_video(tmp_path / "clip.insv", cams)
    assert m is not None
    assert m.unit == "insta360-X5-001"
    assert m.serial == "IAHYA2511CWJEA"
    assert m.intrinsics == cams / "insta360-X5-001" / "intrinsics.npz"


def test_resolve_camera_for_video_missing_intrinsics(tmp_path, monkeypatch):
    cams = _make_registry(tmp_path)
    monkeypatch.setattr(registry, "read_insv_serial", lambda p: "IAHYA2511DVC37")
    m = resolve_camera_for_video(tmp_path / "clip.insv", cams)
    assert m is not None and m.unit == "insta360-X5-002"
    assert m.intrinsics is None  # unit matched but no intrinsics.npz deployed


def test_resolve_camera_for_video_non_insv(tmp_path, monkeypatch):
    cams = _make_registry(tmp_path)
    monkeypatch.setattr(
        registry,
        "read_insv_serial",
        lambda p: (_ for _ in ()).throw(AssertionError("probed")),
    )
    assert resolve_camera_for_video(tmp_path / "clip_lens0.mp4", cams) is None


def test_find_camera_by_serial_serial_number_key(tmp_path):
    # camera.yaml may spell the field `serial_number:` (the telemetry's name).
    cams = tmp_path / "cameras"
    d = cams / "insta360-GO3-001"
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text('serial_number: "GO3ABCD1234"\n')
    assert find_camera_by_serial(cams, "GO3ABCD1234").name == "insta360-GO3-001"


def test_find_camera_by_serial_ambiguous_is_sorted_first(tmp_path):
    # Two units whose serials substring-overlap: the sorted() walk makes the
    # winner deterministic. Pin it so a resolver change is a visible decision,
    # not silent behavior drift.
    cams = tmp_path / "cameras"
    for unit, serial in [("unit-b", "ABCD123"), ("unit-a", "ABCD")]:
        d = cams / unit
        d.mkdir(parents=True)
        (d / "camera.yaml").write_text(f'serial: "{serial}"\n')
    assert find_camera_by_serial(cams, "ABCD123").name == "unit-a"


def test_read_insv_serial_none_on_probe_failure(tmp_path, monkeypatch):
    # A truncated/telemetry-less file must mean "cannot auto-resolve", never
    # an exception out of the probe.
    import sys
    import types

    class _Boom:
        def __init__(self, path):
            raise RuntimeError("truncated file")

    monkeypatch.setitem(
        sys.modules, "telemetry_parser", types.SimpleNamespace(Parser=_Boom)
    )
    assert registry.read_insv_serial(tmp_path / "clip.insv") is None


def test_read_insv_serial_none_on_empty_serial(tmp_path, monkeypatch):
    import sys
    import types

    class _Empty:
        def __init__(self, path):
            pass

        def telemetry(self):
            return [{"Default": {"Metadata": {"serial_number": ""}}}]

    monkeypatch.setitem(
        sys.modules, "telemetry_parser", types.SimpleNamespace(Parser=_Empty)
    )
    assert registry.read_insv_serial(tmp_path / "clip.insv") is None


def test_camera_unit_pointer_wins_over_unversioned(tmp_path):
    d = tmp_path / "cameras" / "insta360-X5-003"
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text(
        'serial: "S3"\nintrinsics: intrinsics-2026-08-03.npz\n'
    )
    (d / "intrinsics-2026-08-03.npz").write_bytes(b"dated")
    (d / "intrinsics.npz").write_bytes(b"stale unversioned copy")  # must NOT win
    unit = registry.CameraUnit.load(d)
    assert unit.artifact("intrinsics") == d / "intrinsics-2026-08-03.npz"


def test_camera_unit_dangling_pointer_raises(tmp_path):
    # A pointer naming a missing file is a broken registry - loud, no fallback.
    d = tmp_path / "cameras" / "u"
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text("intrinsics: intrinsics-2099-01-01.npz\n")
    (d / "intrinsics.npz").write_bytes(b"would-be fallback")
    with pytest.raises(FileNotFoundError, match="2099-01-01"):
        registry.CameraUnit.load(d).artifact("intrinsics")


def test_camera_unit_fallback_to_unversioned(tmp_path):
    # No pointer key: the conventional unversioned filename is used when
    # present, so a hand-assembled unit dir works without pointers.
    d = tmp_path / "cameras" / "u"
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text('serial: "S"\n')
    (d / "intrinsics.npz").write_bytes(b"")
    unit = registry.CameraUnit.load(d)
    assert unit.artifact("intrinsics") == d / "intrinsics.npz"
    assert unit.artifact("av_sync") is None  # neither pointer nor file


def test_camera_unit_unknown_artifact_key_raises(tmp_path):
    d = tmp_path / "cameras" / "u"
    d.mkdir(parents=True)
    with pytest.raises(KeyError):
        registry.CameraUnit.load(d).artifact("no_such_artifact")


def test_resolve_camera_for_video_follows_pointer(tmp_path, monkeypatch):
    cams = tmp_path / "cameras"
    d = cams / "insta360-X5-003"
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text(
        'serial: "IAHYA2511ZZZZZ"\nintrinsics: intrinsics-2026-08-03.npz\n'
    )
    (d / "intrinsics-2026-08-03.npz").write_bytes(b"")
    monkeypatch.setattr(registry, "read_insv_serial", lambda p: "IAHYA2511ZZZZZ")
    m = resolve_camera_for_video(tmp_path / "clip.insv", cams)
    assert m is not None
    assert m.intrinsics == d / "intrinsics-2026-08-03.npz"


def test_artifact_tables_are_consistent():
    # Every schema-tagged artifact must be a registered artifact, and the
    # calibration keys must exist: components resolve their artifacts through
    # these tables, so a missing key is an unusable artifact.
    assert set(registry.ARTIFACT_SCHEMAS) <= set(registry.ARTIFACT_NAMES)
    for key in ("imu_noise", "intrinsics_kb4"):
        assert key in registry.ARTIFACT_NAMES
    # Filenames follow the dated-artifact convention: an extension, no path.
    for name in registry.ARTIFACT_NAMES.values():
        assert "/" not in name and "." in name


def test_artifact_keys_resolve_through_pointer_and_fallback(tmp_path):
    d = tmp_path / "cameras" / "u"
    d.mkdir(parents=True)
    (d / "camera.yaml").write_text(
        'serial: "S"\nimu_noise: imu-noise-2026-08-19.yaml\n'
    )
    (d / "imu-noise-2026-08-19.yaml").write_text("schema: imu_noise_v1\n")
    (d / "av-sync.yaml").write_text("schema: av_sync_v1\n")
    unit = registry.CameraUnit.load(d)
    # Pointer wins for the pointed key; fallback filename serves the other.
    assert unit.artifact("imu_noise") == d / "imu-noise-2026-08-19.yaml"
    assert unit.artifact("av_sync") == d / "av-sync.yaml"
    assert unit.artifact("intrinsics_kb4") is None


# ---- default_artifact_out: dated publish defaults + the published-file guard


def test_default_artifact_out_dated_name(tmp_path):
    unit = tmp_path / "cameras" / "insta360-X5-001"
    today = datetime.date.today()
    out = registry.default_artifact_out(unit, "imu_noise")
    assert out == unit / f"imu-noise-{today}.yaml"
    assert unit.is_dir()  # ensured for the caller
    # the extension follows the artifact kind
    out = registry.default_artifact_out(unit, "intrinsics")
    assert out.name == f"intrinsics-{today}.npz"


def test_default_artifact_out_refuses_published_today(tmp_path):
    unit = tmp_path / "cameras" / "insta360-X5-001"
    unit.mkdir(parents=True)
    today = datetime.date.today()
    (unit / f"imu-noise-{today}.yaml").write_text("")
    (unit / "camera.yaml").write_text(f"imu_noise: imu-noise-{today}.yaml\n")
    with pytest.raises(FileExistsError, match="published artifact"):
        registry.default_artifact_out(unit, "imu_noise")


def test_default_artifact_out_allows_earlier_pointer(tmp_path):
    unit = tmp_path / "cameras" / "insta360-X5-001"
    unit.mkdir(parents=True)
    today = datetime.date.today()
    # the pointer names an earlier publication: today's default is a candidate
    (unit / "camera.yaml").write_text("imu_noise: imu-noise-2026-01-01.yaml\n")
    out = registry.default_artifact_out(unit, "imu_noise")
    assert out.name == f"imu-noise-{today}.yaml"
    # an existing un-pointed candidate from earlier today is fair to replace
    (unit / f"imu-noise-{today}.yaml").write_text("")
    out = registry.default_artifact_out(unit, "imu_noise")
    assert out.name == f"imu-noise-{today}.yaml"


def test_default_artifact_out_no_camera_yaml(tmp_path):
    # a unit dir with no camera.yaml has no pointer to guard against
    unit = tmp_path / "cameras" / "insta360-X5-001"
    out = registry.default_artifact_out(unit, "av_sync")
    assert out.name.startswith("av-sync-")


# ---- mode-keyed pointers: one intrinsics profile per video mode ------------


def _modal_unit(tmp_path: Path) -> Path:
    unit = tmp_path / "cameras" / "insta360-GO3-001"
    unit.mkdir(parents=True)
    (unit / "camera.yaml").write_text(
        "serial: null\n"
        "intrinsics:\n"
        "  2688x2688: intrinsics-2688x2688-2026-09-12.npz\n"
        "  2720x1536: intrinsics-2720x1536-2026-09-12.npz\n"
    )
    (unit / "intrinsics-2688x2688-2026-09-12.npz").write_bytes(b"")
    # the 2720x1536 entry is deliberately dangling for the broken-registry test
    return unit


def test_artifact_refuses_a_mode_keyed_pointer(tmp_path):
    unit = registry.CameraUnit.load(_modal_unit(tmp_path))
    with pytest.raises(ValueError, match="artifact_for_size"):
        unit.artifact("intrinsics")


def test_artifact_for_size_resolves_by_frame_size(tmp_path):
    unit = registry.CameraUnit.load(_modal_unit(tmp_path))
    p = unit.artifact_for_size("intrinsics", (2688, 2688))
    assert p is not None and p.name == "intrinsics-2688x2688-2026-09-12.npz"


def test_artifact_for_size_unknown_mode_names_the_calibrated_ones(tmp_path):
    unit = registry.CameraUnit.load(_modal_unit(tmp_path))
    with pytest.raises(ValueError, match="2688x2688, 2720x1536"):
        unit.artifact_for_size("intrinsics", (1920, 1080))


def test_artifact_for_size_dangling_mode_entry_is_a_broken_registry(tmp_path):
    unit = registry.CameraUnit.load(_modal_unit(tmp_path))
    with pytest.raises(FileNotFoundError, match=r"intrinsics\[2720x1536\]"):
        unit.artifact_for_size("intrinsics", (2720, 1536))


def test_artifact_for_size_scalar_pointer_matches_artifact(tmp_path):
    # Single-mode units: both methods agree, size is not consulted here
    # (their consumers keep the image_size guard).
    unit_dir = tmp_path / "cameras" / "insta360-X5-001"
    unit_dir.mkdir(parents=True)
    (unit_dir / "camera.yaml").write_text("intrinsics: intrinsics-2026-05-19.npz\n")
    (unit_dir / "intrinsics-2026-05-19.npz").write_bytes(b"")
    unit = registry.CameraUnit.load(unit_dir)
    assert unit.artifact_for_size("intrinsics", (1, 1)) == unit.artifact("intrinsics")


def test_default_artifact_out_mode_tag_and_mapping_guard(tmp_path):
    unit = _modal_unit(tmp_path)
    today = datetime.date.today()
    out = registry.default_artifact_out(unit, "intrinsics", mode="2560x1440")
    assert out.name == f"intrinsics-2560x1440-{today}.npz"
    # the guard sees THROUGH the mapping: an entry naming today's candidate
    # is a published artifact, not a scratch file
    (unit / "camera.yaml").write_text(
        f"intrinsics:\n  2560x1440: intrinsics-2560x1440-{today}.npz\n"
    )
    with pytest.raises(FileExistsError, match="published artifact"):
        registry.default_artifact_out(unit, "intrinsics", mode="2560x1440")


def test_resolve_camera_for_video_tolerates_a_mode_keyed_unit(tmp_path, monkeypatch):
    unit = _modal_unit(tmp_path)
    (unit / "camera.yaml").write_text(
        'serial: "IBKEC230648SW4"\n'
        "intrinsics:\n"
        "  2688x2688: intrinsics-2688x2688-2026-09-12.npz\n"
    )
    monkeypatch.setattr(registry, "read_insv_serial", lambda p: "IBKEC230648SW4")
    clip = tmp_path / "VID_1.insv"
    clip.write_bytes(b"x")
    m = resolve_camera_for_video(clip, tmp_path / "cameras")
    # matched by serial, but no single active intrinsics file exists to name
    assert m is not None and m.unit == "insta360-GO3-001"
    assert m.intrinsics is None

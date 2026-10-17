"""Typed digest references: golden canonical-JSON bytes, spelling, verify."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from kiwi.core.digest import (
    BYTES_SHA256,
    CANONICAL_JSON_SHA256,
    NPZ_SEMANTIC_SHA256,
    canonical_json_bytes,
    digest_bytes,
    digest_canonical_json,
    digest_npz,
    parse_digest,
    verify_digest,
)

# The golden document exercises every encoding rule at once: key sorting,
# nested containers, unicode (must land as UTF-8, not \u escapes), null/bool
# spelling, and shortest round-trip floats (0.1 stays "0.1", 1e300 "1e+300").
GOLDEN_DOC = {
    "schema": "golden",
    "z_last": [1, 2.5, -0.1],
    "a_first": {"nested": True, "text": "café"},
    "n": None,
    "big": 1e300,
    "tenth": 0.1,
}
GOLDEN_BYTES = (
    b'{"a_first":{"nested":true,"text":"caf\xc3\xa9"},"big":1e+300,"n":null,'
    b'"schema":"golden","tenth":0.1,"z_last":[1,2.5,-0.1]}'
)
GOLDEN_DIGEST = (
    "canonical_json_sha256:"
    "20bfb7bee46f90458708cb20a60d9ff8252e27a5f2b7c6b48be44043cbfa4fa9"
)


# -- canonical JSON: the one encoding ----------------------------------------


def test_canonical_json_golden_bytes():
    """Byte-exact pin. If this moves, every content-addressed ID moves - that
    must be a reviewed schema revision, never an accidental encoding drift."""
    assert canonical_json_bytes(GOLDEN_DOC) == GOLDEN_BYTES


def test_canonical_json_golden_digest():
    assert digest_canonical_json(GOLDEN_DOC) == GOLDEN_DIGEST
    assert digest_canonical_json({}) == (
        "canonical_json_sha256:"
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )


def test_canonical_json_is_key_order_independent():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json_bytes({"v": float("nan")})
    with pytest.raises(ValueError):
        canonical_json_bytes({"v": float("inf")})


def test_canonical_json_rejects_foreign_types():
    """No default=str coercion: two writers must not disagree on the bytes of
    'the same' identity document because one held a Path and one a str."""
    from pathlib import Path

    with pytest.raises(TypeError):
        canonical_json_bytes({"p": Path("/tmp/x")})


def test_canonical_json_rejects_non_str_keys():
    """json.dumps would coerce {2: ..} to {"2": ..} but sort by the ORIGINAL
    keys, so {2: .., 10: ..} encodes "10" after "2" - bytes its own parsed
    round-trip re-encodes differently. Refused, never coerced."""
    with pytest.raises(TypeError, match="str mapping keys"):
        canonical_json_bytes({2: "b", 10: "a"})
    with pytest.raises(TypeError, match="str mapping keys"):
        canonical_json_bytes({True: 1})
    with pytest.raises(TypeError, match="str mapping keys"):
        canonical_json_bytes({"outer": [{"deep": {None: 1}}]})


# -- spelling ----------------------------------------------------------------


def test_parse_digest_round_trips_every_algorithm(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"payload")
    n = tmp_path / "arrays.npz"
    np.savez(n, k=np.zeros(3))
    for ref, algorithm in (
        (digest_bytes(p), BYTES_SHA256),
        (digest_canonical_json({"k": 1}), CANONICAL_JSON_SHA256),
        (digest_npz(n), NPZ_SEMANTIC_SHA256),
    ):
        algo, value = parse_digest(ref)
        assert algo == algorithm
        assert len(value) == 64


@pytest.mark.parametrize(
    "bad",
    [
        "deadbeef",  # no algorithm
        "sha256:" + "0" * 64,  # bare 'sha256' is exactly the ambiguity banned
        "bytes_sha256:" + "0" * 63,  # short hex
        "bytes_sha256:" + "A" * 64,  # uppercase hex
        "bytes_sha256:" + "0" * 64 + "\n",  # '$' would admit this newline
        " bytes_sha256:" + "0" * 64,  # leading whitespace
        "unknown_algo:" + "0" * 64,  # unknown algorithm fails closed
    ],
)
def test_parse_digest_rejects(bad):
    with pytest.raises(ValueError):
        parse_digest(bad)


# -- computing and verifying -------------------------------------------------


def test_digest_bytes_streams_to_the_same_hash(tmp_path):
    """Chunked streaming must equal the one-shot hash, across a chunk boundary."""
    p = tmp_path / "big.bin"
    payload = bytes(range(256)) * 5000 + b"tail"  # ~1.28 MB > one 1 MiB chunk
    p.write_bytes(payload)
    assert digest_bytes(p) == f"bytes_sha256:{hashlib.sha256(payload).hexdigest()}"


def test_digest_npz_is_regeneration_stable(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    arrays = {"t": np.arange(4, dtype=np.float64)}
    np.savez(a, **arrays)
    np.savez(b, **arrays)
    assert digest_npz(a) == digest_npz(b)


def test_verify_digest_accepts_and_refuses_bytes(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"payload")
    ref = digest_bytes(p)
    verify_digest(ref, p)
    p.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        verify_digest(ref, p)


def test_verify_digest_npz_sees_content_not_zip_bytes(tmp_path):
    p = tmp_path / "arrays.npz"
    np.savez(p, k=np.arange(3, dtype=np.float64))
    ref = digest_npz(p)
    np.savez(p, k=np.arange(3, dtype=np.float64))  # regenerated, new zip mtimes
    verify_digest(ref, p)
    np.savez(p, k=np.arange(4, dtype=np.float64))
    with pytest.raises(ValueError, match="mismatch"):
        verify_digest(ref, p)


def test_verify_digest_canonical_json_survives_a_pretty_print(tmp_path):
    p = tmp_path / "doc.json"
    p.write_text(json.dumps(GOLDEN_DOC), encoding="utf-8")
    verify_digest(GOLDEN_DIGEST, p)
    # Reformatted + reordered, same content: still the same identity.
    p.write_text(
        json.dumps(GOLDEN_DOC, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    verify_digest(GOLDEN_DIGEST, p)
    changed = {**GOLDEN_DOC, "tenth": 0.2}
    p.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_digest(GOLDEN_DIGEST, p)


def test_canonical_yaml_digest_is_comment_insensitive(tmp_path):
    from kiwi.core.digest import digest_canonical_yaml

    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("# one comment\nkey: 1\n")
    b.write_text("# a different comment\nkey: 1\n")
    ref = digest_canonical_yaml(a)
    assert ref.startswith("canonical_yaml_sha256:")
    assert ref == digest_canonical_yaml(b)
    verify_digest(ref, b)
    b.write_text("key: 2\n")
    with pytest.raises(ValueError, match="mismatch"):
        verify_digest(ref, b)

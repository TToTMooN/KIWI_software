"""Typed content digests: the algorithm travels inside the reference.

:mod:`kiwi.core.config_hash` computes three *different* hashes behind one
field name: a ``.yaml`` hashes as its parsed document, an ``.npz`` as its
arrays, and anything else as its bytes.

References that cross an artifact boundary spell their algorithm into the
value: ``<algorithm>:<hex>``. Four algorithms exist; three of them are the
hash kinds :mod:`~kiwi.core.config_hash` computes:

* ``bytes_sha256`` - the file's bytes, streamed (recordings are multi-GB).
* ``canonical_json_sha256`` - a JSON document's *content*, via the one
  canonical encoding below.
* ``npz_semantic_sha256`` - an ``.npz``'s arrays (dtype + shape + buffer
  per sorted key), i.e. :func:`kiwi.core.config_hash.hash_npz`.
* ``canonical_yaml_sha256`` - a YAML file's *parsed content*, i.e.
  :func:`kiwi.core.config_hash.hash_yaml` - comment and layout edits do not
  move it.

Canonical JSON gives content-addressed identities (a frame or map ID hashed
from its definition document) one byte-exact encoding, pinned by golden
fixtures. The encoding is UTF-8, lexicographically sorted keys, no
insignificant whitespace, ``allow_nan=False``, and Python's shortest
round-trip float representation. Unlike the observational ``config_hash``
stamps, everything here fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from kiwi.core.config_hash import hash_npz, hash_yaml

__all__ = [
    "BYTES_SHA256",
    "CANONICAL_JSON_SHA256",
    "CANONICAL_YAML_SHA256",
    "NPZ_SEMANTIC_SHA256",
    "canonical_json_bytes",
    "digest_bytes",
    "digest_canonical_json",
    "digest_canonical_yaml",
    "digest_npz",
    "parse_digest",
    "verify_digest",
]

BYTES_SHA256 = "bytes_sha256"
CANONICAL_JSON_SHA256 = "canonical_json_sha256"
CANONICAL_YAML_SHA256 = "canonical_yaml_sha256"
NPZ_SEMANTIC_SHA256 = "npz_semantic_sha256"

_ALGORITHMS = frozenset(
    {BYTES_SHA256, CANONICAL_JSON_SHA256, CANONICAL_YAML_SHA256, NPZ_SEMANTIC_SHA256}
)

# <algorithm>:<64 lowercase hex>, matched with fullmatch. Uppercase hex and
# surrounding whitespace are rejected rather than normalized.
_DIGEST_RE = re.compile(r"([a-z0-9_]+):([0-9a-f]{64})")


def _require_str_keys(doc: object) -> None:
    """Refuse any non-str mapping key anywhere in ``doc``.

    ``json.dumps`` would otherwise coerce int/float/bool/None keys to
    strings - and sort by the *original* keys, so ``{2: .., 10: ..}``
    encodes with ``"10"`` after ``"2"``, an ordering its own parsed
    round-trip re-encodes differently.
    """
    if isinstance(doc, dict):
        for key, value in doc.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"canonical JSON requires str mapping keys, got {key!r} "
                    f"({type(key).__name__}) - a coerced key would not "
                    f"round-trip to the same encoding"
                )
            _require_str_keys(value)
    elif isinstance(doc, (list, tuple)):
        for item in doc:
            _require_str_keys(item)


def canonical_json_bytes(doc: object) -> bytes:
    """The one canonical JSON encoding of ``doc``, as UTF-8 bytes.

    Strict, unlike :func:`kiwi.core.config_hash._canonical` (whose
    ``default=str`` serves observational stamps): a foreign type or non-str
    mapping key raises ``TypeError`` and a NaN/Infinity raises
    ``ValueError``. Floats use Python's shortest round-trip representation;
    golden fixtures in the test suite pin the exact bytes.
    """
    _require_str_keys(doc)
    return json.dumps(
        doc,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _stream_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_bytes(path: str | Path) -> str:
    """``bytes_sha256:<hex>`` of a file's bytes, streamed (multi-GB safe)."""
    return f"{BYTES_SHA256}:{_stream_sha256(Path(path))}"


def digest_canonical_json(doc: object) -> str:
    """``canonical_json_sha256:<hex>`` of a document's canonical encoding."""
    return f"{CANONICAL_JSON_SHA256}:{hashlib.sha256(canonical_json_bytes(doc)).hexdigest()}"


def digest_canonical_yaml(path: str | Path) -> str:
    """``canonical_yaml_sha256:<hex>`` of a YAML file's parsed content."""
    return f"{CANONICAL_YAML_SHA256}:{hash_yaml(path)}"


def digest_npz(path: str | Path) -> str:
    """``npz_semantic_sha256:<hex>`` of an npz's arrays (not its zip bytes)."""
    return f"{NPZ_SEMANTIC_SHA256}:{hash_npz(path)}"


def parse_digest(ref: str) -> tuple[str, str]:
    """Split a typed reference into ``(algorithm, hex)``, failing closed.

    An unknown algorithm is refused here rather than at comparison time.
    """
    m = _DIGEST_RE.fullmatch(ref)
    if m is None:
        raise ValueError(
            f"malformed digest reference {ref!r}: expected "
            f"'<algorithm>:<64 lowercase hex>'"
        )
    algorithm, value = m.groups()
    if algorithm not in _ALGORITHMS:
        raise ValueError(
            f"unknown digest algorithm {algorithm!r} in {ref!r}: this reader "
            f"can recompute {sorted(_ALGORITHMS)}"
        )
    return algorithm, value


def verify_digest(ref: str, path: str | Path) -> None:
    """Recompute ``ref``'s algorithm over ``path``; raise on any mismatch.

    ``canonical_json_sha256`` verifies a ``.json`` *file* by parsing it
    and re-encoding canonically, so an indent/key-order rewrite of the same
    content still verifies - the reference names content, not bytes.
    """
    algorithm, expected = parse_digest(ref)
    path = Path(path)
    if algorithm == BYTES_SHA256:
        actual = _stream_sha256(path)
    elif algorithm == NPZ_SEMANTIC_SHA256:
        actual = hash_npz(path)
    elif algorithm == CANONICAL_YAML_SHA256:
        actual = hash_yaml(path)
    else:  # CANONICAL_JSON_SHA256 - parse_digest admits nothing else
        doc = json.loads(path.read_text(encoding="utf-8"))
        actual = hashlib.sha256(canonical_json_bytes(doc)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{path}: {algorithm} mismatch - expected {expected}, got {actual}"
        )

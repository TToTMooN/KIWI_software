"""The model store: where the production weights live and how they arrive.

The ONNX weights are NOT repo content - they are a GitHub release asset,
pinned by sha256 in the in-repo sidecar and fetched once into a local
cache. The sidecar is the contract; the store only guarantees that
the bytes next to it are exactly the bytes it declares. Every failure is a
typed ArtifactError carrying the remedy: a missing model is a refusal,
never a silent downgrade to unmasked inference.

Cache layout - one directory per model hash, the sidecar copied in beside
the weights so ``artifact.load_artifact`` finds its sibling unchanged::

    <cache>/<model_sha256>/<asset>
    <cache>/<model_sha256>/<asset>.meta.json
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

from kiwi.core.atomic import atomic_write_text

from .types import ArtifactError

#: Sidecar of the declared production model (the weights it pins are fetched).
PACKAGED_SIDECAR = (
    Path(__file__).resolve().parent / "models" / "gripper-unet.onnx.meta.json"
)

#: Environment variables consulted for a GitHub token, in precedence order.
TOKEN_ENV_VARS = ("KIWI_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")

SIDECAR_SUFFIX = ".meta.json"
REQUIRED_SOURCE_KEYS = ("repo", "release_tag", "asset", "size_bytes")
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_AUTH_CODES = (401, 403, 404)
_CHUNK = 1 << 20
_TIMEOUT_S = 60.0
_MAX_REDIRECTS = 5
_HEX = frozenset("0123456789abcdef")
#: Mode of every file published into the cache (not mkstemp's 0600).
_PUBLISH_MODE = 0o644


# --------------------------------------------------------------- cache paths


def model_cache_dir() -> Path:
    """Where fetched model weights live on this machine."""
    override = os.environ.get("KIWI_MODEL_CACHE")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "kiwi" / "models"
    return Path.home() / ".cache" / "kiwi" / "models"


def cached_model_path(sidecar_doc: dict) -> Path:
    """The cache path the sidecar's weights belong at (hash-addressed)."""
    source = sidecar_doc["source"]
    return model_cache_dir() / sidecar_doc["model_sha256"] / source["asset"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ sidecar


def _read_sidecar(sidecar: Path) -> dict:
    try:
        doc = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{sidecar}: unreadable model sidecar: {exc}") from exc
    if not isinstance(doc, dict):
        raise ArtifactError(f"{sidecar}: model sidecar is not a JSON object")
    if doc.get("schema") != "segmentation_model":
        raise ArtifactError(
            f"{sidecar}: schema {doc.get('schema')!r} != 'segmentation_model'"
        )
    if not doc.get("model_sha256"):
        raise ArtifactError(f"{sidecar}: sidecar missing 'model_sha256'")
    digest = doc["model_sha256"]
    # Both of the fields below become path components of the cache entry, so
    # each is constrained to what it claims to be.
    if not isinstance(digest, str) or len(digest) != 64 or set(digest) - _HEX:
        raise ArtifactError(
            f"{sidecar}: model_sha256 {digest!r} is not a 64-character lowercase "
            "hex digest"
        )
    source = doc.get("source")
    if not isinstance(source, dict):
        raise ArtifactError(
            f"{sidecar}: sidecar has no 'source' block; a resolvable model must "
            f"declare where its weights are published ({', '.join(REQUIRED_SOURCE_KEYS)})"
        )
    for key in REQUIRED_SOURCE_KEYS:
        if key not in source or source[key] in (None, ""):
            raise ArtifactError(f"{sidecar}: sidecar source missing '{key}'")
    asset = source["asset"]
    if (
        not isinstance(asset, str)
        or asset in (".", "..")
        or "\\" in asset
        or Path(asset).name != asset
    ):
        raise ArtifactError(
            f"{sidecar}: sidecar source asset {asset!r} is not a plain file name"
        )
    size = source["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ArtifactError(
            f"{sidecar}: sidecar source size_bytes {size!r} is not a positive int"
        )
    return doc


def _sync_sidecar_copy(sidecar: Path, model: Path) -> Path:
    """Keep the cache's sidecar copy byte-identical to the repo's.

    ``load_artifact`` reads the sibling sidecar, so the cached copy is the
    contract the runtime actually validates against."""
    copy = model.with_suffix(model.suffix + SIDECAR_SUFFIX)
    want = sidecar.read_bytes()
    try:
        if not copy.exists() or copy.read_bytes() != want:
            atomic_write_text(copy, want.decode("utf-8"))
            copy.chmod(_PUBLISH_MODE)
    except OSError as exc:
        raise ArtifactError(
            f"{copy}: the cached sidecar copy could not be read or written "
            f"({exc}); the model cache directory must be readable and writable "
            "by this user, or KIWI_MODEL_CACHE must point at a cache whose "
            "sidecar copy is already in place"
        ) from exc
    return copy


# ------------------------------------------------------------------ messages


def _unavailable(source: dict, dest: Path, reason: str) -> ArtifactError:
    """The one actionable message for 'the weights are not here'."""
    return ArtifactError(
        f"segmentation model {source['asset']} is not in the model cache "
        f"({dest}) and was not fetched from release {source['release_tag']} of "
        f"{source['repo']}: {reason}. Remedies: run 'kiwi models fetch' on a "
        "connected machine; or point KIWI_MODEL_CACHE at a directory that "
        "already holds the model (an offline copy of the cache); or, for "
        "authenticated access, provide a token via KIWI_GITHUB_TOKEN, GITHUB_TOKEN "
        "or GH_TOKEN, or run 'gh auth login'."
    )


def _availability(token: str | None) -> str:
    return "a GitHub token was available" if token else "no GitHub token was available"


def _log_stderr(message: str) -> None:
    print(message, file=sys.stderr)


# -------------------------------------------------------------------- tokens


def github_token() -> str | None:
    """A GitHub token for authenticated downloads, or None. Never logged."""
    for var in TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip()
    gh = shutil.which("gh")
    if gh is None:
        return None
    try:
        proc = subprocess.run(
            [gh, "auth", "token"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


# ------------------------------------------------------------------ download


def browser_url(source: dict) -> str:
    """The public release-asset URL."""
    return (
        f"https://github.com/{source['repo']}/releases/download/"
        f"{source['release_tag']}/{source['asset']}"
    )


def release_api_url(source: dict) -> str:
    return (
        f"https://api.github.com/repos/{source['repo']}/releases/tags/"
        f"{source['release_tag']}"
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to auto-follow: returning None turns the 3xx into an HTTPError
    the caller can inspect (see ``open_authenticated``)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _no_redirect_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirect)


def _drain(exc: urllib.error.HTTPError) -> None:
    if getattr(exc, "fp", None) is not None:
        exc.close()


def open_authenticated(url: str, token: str, accept: str, opener=None):
    """GET ``url`` with a bearer token, following redirects BY HAND.

    GitHub answers an asset download with a 302 to signed object storage,
    and that backend REJECTS a request carrying two auth mechanisms - so the
    redirected GET carries no Authorization header, and neither do later
    hops."""
    opener = opener or _no_redirect_opener()
    headers = {"Authorization": f"Bearer {token}", "Accept": accept}
    for _ in range(_MAX_REDIRECTS):
        try:
            return opener.open(
                urllib.request.Request(url, headers=headers), timeout=_TIMEOUT_S
            )
        except urllib.error.HTTPError as exc:
            if exc.code not in _REDIRECT_CODES:
                raise
            location = exc.headers.get("Location") if exc.headers else None
            _drain(exc)
            if not location:
                raise
            # No Authorization from here on: the signed URL carries its own.
            url, headers = location, {}
    raise ArtifactError(f"more than {_MAX_REDIRECTS} redirects fetching {url}")


def open_asset_authenticated(source: dict, token: str, opener=None):
    """Resolve the release asset through the API and open its bytes."""
    response = open_authenticated(
        release_api_url(source), token, "application/vnd.github+json", opener=opener
    )
    with response:
        release = json.loads(response.read().decode("utf-8"))
    for asset in release.get("assets", []):
        if asset.get("name") == source["asset"]:
            return open_authenticated(
                asset["url"], token, "application/octet-stream", opener=opener
            )
    raise ArtifactError(
        f"release {source['release_tag']} of {source['repo']} has no asset named "
        f"{source['asset']}"
    )


def _stream_asset(source: dict, dest: Path) -> Iterator[bytes]:
    """Yield the asset's bytes: public URL first, authenticated API second.

    Every way this can fail - including a connection that dies halfway
    through the 31 MB - surfaces as the one actionable ArtifactError."""
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(browser_url(source)), timeout=_TIMEOUT_S
        )
    except urllib.error.HTTPError as exc:
        code = exc.code
        _drain(exc)
        # A token is looked up only after the public request fails; finding
        # one may shell out to `gh`.
        token = github_token()
        if code not in _AUTH_CODES or token is None:
            raise _unavailable(
                source,
                dest,
                f"the public asset URL returned HTTP {code} and {_availability(token)}",
            ) from exc
        try:
            response = open_asset_authenticated(source, token)
        except urllib.error.HTTPError as exc2:
            _drain(exc2)
            raise _unavailable(
                source,
                dest,
                f"the authenticated GitHub API returned HTTP {exc2.code}",
            ) from exc2
        except urllib.error.URLError as exc2:
            raise _unavailable(
                source,
                dest,
                f"the authenticated GitHub API was unreachable ({exc2.reason})",
            ) from exc2
        except ArtifactError as exc2:
            raise _unavailable(source, dest, str(exc2)) from exc2
    except urllib.error.URLError as exc:
        raise _unavailable(
            source,
            dest,
            f"the release asset was unreachable ({exc.reason}) and "
            f"{_availability(github_token())}",
        ) from exc
    with response:
        while True:
            try:
                chunk = response.read(_CHUNK)
            except (http.client.HTTPException, OSError) as exc:
                raise _unavailable(
                    source, dest, f"the transfer failed mid-stream ({exc})"
                ) from exc
            if not chunk:
                return
            yield chunk


def _download(
    doc: dict,
    sidecar: Path,
    dest: Path,
    *,
    log: Callable[[str], None],
    fetcher: Callable[[dict, Path], Iterator[bytes]],
) -> Path:
    """Stream the asset to a ``.part`` beside ``dest``, verify it, publish it.

    Nothing lands at ``dest`` that has not matched both the declared size
    and the declared sha256 - and the sha256 is taken from the finished file
    on disk, not from the stream. The temp name is UNIQUE per download (the
    rule ``kiwi.core.atomic`` states). A ``.part`` left by an earlier crash
    is dropped."""
    source = doc["source"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_name(dest.name + ".part").unlink(missing_ok=True)
    size = int(source["size_bytes"])
    log(
        f"fetching {source['asset']} ({size / 1e6:.1f} MB) from release "
        f"{source['release_tag']} of {source['repo']}"
    )
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    part = Path(tmp)
    written = 0
    try:
        with os.fdopen(fd, "wb") as f:
            for chunk in fetcher(source, dest):
                f.write(chunk)
                written += len(chunk)
        if written != size:
            raise ArtifactError(
                f"{source['asset']}: download size check failed (expected {size} "
                f"bytes, got {written}); nothing was written to {dest}"
            )
        actual = sha256_file(part)
        if actual != doc["model_sha256"]:
            raise ArtifactError(
                f"{source['asset']}: download sha256 check failed (expected "
                f"{doc['model_sha256']}, got {actual}); nothing was written to {dest}"
            )
        part.chmod(_PUBLISH_MODE)
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    _sync_sidecar_copy(sidecar, dest)
    log(f"verified sha256 {actual[:12]}... -> {dest}")
    return dest


# ------------------------------------------------------------------- resolve


def resolve_model(
    sidecar: Path = PACKAGED_SIDECAR,
    *,
    fetch: bool = True,
    log: Callable[[str], None] | None = None,
    fetcher: Callable[[dict, Path], Iterator[bytes]] | None = None,
) -> Path:
    """Return the local path of the weights this sidecar declares.

    Cache hit: the bytes are re-verified against the sidecar (a mismatch is
    a refusal naming the file, never a silent re-download). Cache miss:
    fetched when ``fetch``, otherwise a refusal carrying the remedies."""
    sidecar = Path(sidecar)
    doc = _read_sidecar(sidecar)
    source = doc["source"]
    dest = cached_model_path(doc)
    if dest.exists():
        actual = sha256_file(dest)
        if actual != doc["model_sha256"]:
            raise ArtifactError(
                f"{dest}: cached model bytes do not match the sidecar "
                f"model_sha256 (expected {doc['model_sha256']}, got {actual}); "
                f"delete {dest}, then run 'kiwi models fetch' - a mismatched "
                "cache entry is never silently replaced"
            )
        _sync_sidecar_copy(sidecar, dest)
        return dest
    if not fetch:
        raise _unavailable(source, dest, "fetching is disabled (fetch=False)")
    return _download(
        doc,
        sidecar,
        dest,
        log=log or _log_stderr,
        fetcher=fetcher or _stream_asset,
    )

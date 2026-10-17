"""The ``kiwi import`` picker: a page on localhost that plays the clips.

The picker sits where the repo's other interactive tools live (viser serves
the gripper tuner and the replay viewer on localhost): one page, served by
the standard library, listing the candidates newest first beside a
``<video>`` of the highlighted one.

The video is the camera's own ``LRV_*.lrv`` proxy, served as it is. It is an
MP4 with H.264 inside, which every browser decodes natively, so playback and
scrubbing are the browser's job and nothing here touches a codec. With
``Range`` honoured, seeking costs one read at the sought offset. A clip
with no proxy (the GO3 mp4) is served itself, and whether the browser can
play it is the browser's affair.

When no card was found where :func:`ingest.find_mounts` looks (or two were),
the page opens on a folder browser instead of the run refusing: the operator
walks to the card - an SD reader under ``/mnt``, a copy of a card on disk -
and the list fills from there.

The page needs a browser somewhere, not here: the URL is printed, and over
ssh ``-L 8765:localhost:8765`` brings it to the laptop. The server binds
127.0.0.1 only; media is served by candidate index, nothing else on disk.
"""

from __future__ import annotations

import errno
import json
import os
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from kiwi.core.camera.ingest import (
    DEFAULT_PORT,
    MOUNT_GLOBS,
    ROLE_TAGS,
    Candidate,
    CopyJob,
    CopyReporter,
    IngestError,
    Selection,
    find_mounts,
    lrv_for,
    newest_clips,
)

__all__ = [
    "browse_roots",
    "clip_count",
    "list_dir",
    "pick_in_browser",
    "preview_source",
]

CHUNK = 1 << 20
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")

ACK_WAIT = 3.0
"""Seconds the server stays up after the copy for the page to fetch the
final state and acknowledge it, before it is shut down regardless."""


def preview_source(clip: Path) -> tuple[Path, bool]:
    """What the page plays for ``clip``: its LRV proxy when the camera wrote
    one (True), else the clip itself (False)."""
    proxy = lrv_for(clip)
    return (proxy, True) if proxy is not None else (clip, False)


# --- the folder browser -----------------------------------------------------


def clip_count(d: Path) -> int:
    """How many clips :func:`ingest.newest_clips` would find under ``d``."""
    try:
        return len(newest_clips(d, 10_000))
    except OSError:
        return 0


def browse_roots() -> list[Path]:
    """Where the folder browser starts: any card that is mounted, the places
    cards mount, then home and the root. Existing directories only, in that
    order, without repeats."""
    roots: list[Path] = list(find_mounts())
    for g in MOUNT_GLOBS:
        roots.extend(sorted(Path("/").glob(g.rsplit("/", 1)[0])))  # /media/<user>
    roots += [Path("/mnt"), Path.home(), Path("/")]
    seen: list[Path] = []
    for r in roots:
        if r.is_dir() and r not in seen:
            seen.append(r)
    return seen


def list_dir(path: Path) -> dict:
    """One level of the folder browser: ``path``'s subdirectories (hidden ones
    skipped), each with how many clips it holds, plus how many ``path`` itself
    holds - that count is what makes "use this folder" available."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    dirs = []
    with os.scandir(path) as it:
        for e in sorted(it, key=lambda e: e.name.lower()):
            if e.name.startswith("."):
                continue
            try:
                if not e.is_dir():
                    continue
            except OSError:
                continue
            dirs.append(
                {
                    "name": e.name,
                    "path": str(path / e.name),
                    "clips": clip_count(path / e.name),
                }
            )
    return {
        "path": str(path),
        "parent": str(path.parent) if path.parent != path else None,
        "clips": clip_count(path),
        "dirs": dirs,
    }


# --- the session ------------------------------------------------------------


class _Session:
    """One picker run: the candidates, what the page decided, and the signal
    that it has. ``clips`` may start empty (nothing mounted): the page then
    opens on the folder browser and :meth:`use_folder` fills the list."""

    def __init__(
        self, clips: list[Candidate], n: int, note: str, dest: Path | None
    ) -> None:
        self.n = n
        self.note = note
        self.dest = dest
        self.picked: list[int] | None = None
        self.role: str | None = None
        self.done = threading.Event()
        self.acked = threading.Event()
        self.lock = threading.Lock()
        self.copy: dict = {"phase": "picking", "message": "", "jobs": []}
        self._load(clips)

    def _load(self, clips: list[Candidate]) -> None:
        self.clips = clips
        self.sources = [preview_source(c.path) for c in clips]
        self.mount = str(clips[0].path.parent) if clips else None

    def use_folder(self, path: Path) -> str | None:
        """Take the newest clips under ``path`` as the candidates; the reason
        when there are none."""
        path = path.expanduser()
        if not path.is_dir():
            return f"{path} is not a directory"
        clips = newest_clips(path, self.n)
        if not clips:
            return (
                f"no VID_* clips under {path} (looked in its DCIM/*/ when it has "
                f"one, else in the folder itself and its immediate subfolders)"
            )
        with self.lock:
            self._load(clips)
            self.note = ""  # whatever kept the card from being found is moot now
        return None

    def page_state(self) -> dict:
        with self.lock:
            return {
                "mount": self.mount,
                "note": self.note,
                "dest": str(self.dest) if self.dest else None,
                "roots": [str(r) for r in browse_roots()],
                "roles": ROLE_TAGS,
                "clips": [
                    {
                        "name": c.path.name,
                        "stamp": f"{c.start:%Y-%m-%d %H:%M:%S}",
                        "size": c.size,
                        "proxy": has_proxy,
                    }
                    for c, (_, has_proxy) in zip(self.clips, self.sources)
                ],
            }


class _Handler(BaseHTTPRequestHandler):
    session: _Session  # set on the per-run subclass

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib name
        pass  # the terminal shows the copy plan, not the request log

    # --- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        url = urlsplit(self.path)
        if url.path in ("/", "/index.html"):
            self._send_bytes(PAGE.encode(), "text/html; charset=utf-8")
            return
        if url.path == "/state":
            self._send_json(self.session.page_state())
            return
        if url.path == "/progress":
            with self.session.lock:
                self._send_json(json.loads(json.dumps(self.session.copy)))
            return
        if url.path == "/dirs":
            path = parse_qs(url.query).get("path", [""])[0]
            try:
                self._send_json(list_dir(Path(path) if path else Path.home()))
            except (OSError, ValueError) as e:
                self._send_json(
                    {"error": f"cannot list {path or '~'}: {e}"}, HTTPStatus.NOT_FOUND
                )
            return
        m = re.fullmatch(r"/media/(\d+)", url.path)
        if m:
            with self.session.lock:
                sources = self.session.sources
            if int(m[1]) < len(sources):
                self._send_media(sources[int(m[1])][0])
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            data = json.loads(body or b"{}")
            if not isinstance(data, dict):
                raise TypeError(data)
        except (ValueError, TypeError):
            self.send_error(HTTPStatus.BAD_REQUEST, "a JSON object")
            return
        if self.path == "/mount":
            path = data.get("path")
            if not isinstance(path, str) or not path:
                self.send_error(HTTPStatus.BAD_REQUEST, "path: a directory")
                return
            why = self.session.use_folder(Path(path))
            if why is not None:
                self._send_json({"error": why}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(self.session.page_state())
            return
        if self.path == "/select":
            try:
                with self.session.lock:
                    n = len(self.session.clips)
                picked = sorted(
                    {int(i) for i in data.get("picked", []) if 0 <= int(i) < n}
                )
                role = data.get("role")
                if role is not None and role not in ROLE_TAGS:
                    raise ValueError(role)
            except (ValueError, TypeError, AttributeError):
                self.send_error(
                    HTTPStatus.BAD_REQUEST,
                    f"picked: a list of indices; role: one of {sorted(ROLE_TAGS)} or null",
                )
                return
            self.session.picked = picked
            self.session.role = role
        elif self.path == "/cancel":
            self.session.picked = None
        elif self.path == "/ack":
            self.session.acked.set()
            self._send_json({})
            return
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_json({})
        self.session.done.set()

    # --- replies ----------------------------------------------------------

    def _send_bytes(
        self, body: bytes, ctype: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(data).encode(), "application/json", status)

    def _send_media(self, path: Path) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        start, end = 0, size - 1
        m = RANGE_RE.match(self.headers.get("Range", ""))
        if m:
            if m[1]:
                start = int(m[1])
                if m[2]:
                    end = min(int(m[2]), size - 1)
            elif m[2]:  # a suffix range: the last N bytes (the moov atom lives there)
                start = max(0, size - int(m[2]))
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    buf = f.read(min(CHUNK, left))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    left -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser abandons a fetch every time it seeks


class _PageReporter(CopyReporter):
    """The copy, as the page sees it: one entry per job with its bar, then a
    closing message. :meth:`close` gives the page a moment to fetch the final
    state before the server goes away."""

    def __init__(self, server: ThreadingHTTPServer, session: _Session) -> None:
        self.server = server
        self.session = session
        self.closed = False

    def _set(self, **fields) -> None:
        with self.session.lock:
            self.session.copy.update(fields)

    def _job(self, i: int, **fields) -> None:
        with self.session.lock:
            self.session.copy["jobs"][i].update(fields)

    def planned(self, jobs: list[CopyJob], dest: Path) -> None:
        self._set(
            phase="planned",
            dest=str(dest),
            jobs=[
                {
                    "src": j.src.name,
                    "dst": j.dst.name,
                    "file": j.dst.name,
                    "size": j.src.stat().st_size,
                    "copied": 0,
                    "elapsed": 0.0,
                    "state": "pending",
                }
                for j in jobs
            ],
        )

    def started(self, i: int, src: Path, dst: Path, size: int) -> None:
        self._set(phase="copying")
        self._job(i, file=dst.name, size=size, copied=0, elapsed=0.0, state="copying")

    def progress(self, i: int, copied: int, size: int, elapsed: float) -> None:
        self._job(i, copied=copied, size=size, elapsed=elapsed)

    def finished(self, i: int, outcome: str) -> None:
        self._job(i, state=outcome)

    def done(self, message: str) -> None:
        self._set(phase="done", message=message)

    def failed(self, message: str) -> None:
        self._set(phase="failed", message=message)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with self.session.lock:
            if self.session.copy["phase"] in ("picking", "planned", "copying"):
                self.session.copy.update(phase="failed", message="interrupted")
        self.session.acked.wait(ACK_WAIT)
        self.server.shutdown()
        self.server.server_close()


def pick_in_browser(
    clips: list[Candidate],
    *,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    n: int = 30,
    note: str = "",
    dest: Path | None = None,
) -> Selection:
    """Serve the picker for ``clips`` until the page imports or cancels, and
    return the chosen candidates in the order they were offered, with the
    role tag the page picked for them. With no ``clips`` the page opens on
    the folder browser, ``note`` saying why, and ``n`` is how many of the
    newest clips a chosen folder contributes; ``dest`` is shown on the way
    out, so the operator knows where to look. Raises :class:`IngestError` on
    cancel (the page's button, or Ctrl-C here) and when ``port`` is taken.

    On a selection the server stays up: the returned :class:`Selection`
    carries a :class:`CopyReporter` that draws the copy on the page, and
    whose ``close`` is what finally stops the server."""
    session = _Session(clips, n, note, dest)
    handler = type("PickHandler", (_Handler,), {"session": session})
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            raise IngestError(
                f"port {port} is busy (another kiwi import?) - pass --port to choose another"
            ) from e
        raise
    server.daemon_threads = True
    url = f"http://localhost:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    what = "pick clips" if clips else "locate the card"
    print(f"{what} at {url}  (Ctrl-C cancels)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass  # the URL is printed; opening it is a convenience
    try:
        while not session.done.wait(0.25):
            pass
    except KeyboardInterrupt:
        session.picked = None
    if session.picked is None:
        server.shutdown()
        server.server_close()
        raise IngestError("cancelled")
    tag = ROLE_TAGS[session.role] if session.role else None
    with session.lock:
        chosen = [session.clips[i] for i in session.picked]
    return Selection(chosen, tag, _PageReporter(server, session))


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kiwi import</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f6f6f4; --panel: #ffffff; --ink: #1c1c1a; --muted: #6b6b66;
    --line: #dededa; --accent: #2f6f4e; --accent-ink: #ffffff; --tint: #e8f2ec;
    --hi: #eef0ea; --warn: #b3552b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #171716; --panel: #1f1f1d; --ink: #ececea; --muted: #9a9a94;
      --line: #33332f; --accent: #6fbf94; --accent-ink: #10201a; --tint: #1f2e26;
      --hi: #26262a; --warn: #e08a5c;
    }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  header {
    display: flex; align-items: baseline; gap: 16px; padding: 12px 20px;
    border-bottom: 1px solid var(--line); background: var(--panel);
    position: sticky; top: 0; z-index: 2;
  }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  header .where { color: var(--muted); }
  header .where code { color: var(--ink); }
  header .where a { color: var(--accent); cursor: pointer; margin-left: 8px; }
  header .keys { color: var(--muted); font-size: 12px; margin-left: auto; }
  header kbd {
    font: 11px ui-monospace, SFMono-Regular, Menlo, monospace;
    border: 1px solid var(--line); border-radius: 4px; padding: 0 4px;
    background: var(--bg);
  }
  main {
    display: grid; grid-template-columns: max-content minmax(320px, 1fr);
    gap: 20px; padding: 20px; align-items: start;
  }
  @media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
  .list { overflow-x: auto; }
  table { border-collapse: collapse; background: var(--panel);
    border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  th, td { padding: 8px 10px; text-align: left; white-space: nowrap; }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
    color: var(--muted); border-bottom: 1px solid var(--line); font-weight: 600; }
  tbody tr { cursor: pointer; border-top: 1px solid var(--line); }
  tbody tr:hover { background: var(--hi); }
  tbody tr.cur { box-shadow: inset 3px 0 0 var(--accent); background: var(--hi); }
  tbody tr.on td.name { font-weight: 600; }
  tbody tr.on { background: var(--tint); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
  td.name { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
  td.chk { width: 34px; }
  input[type=checkbox] { width: 16px; height: 16px; accent-color: var(--accent); margin: 0; vertical-align: -3px; }
  aside { position: sticky; top: 64px; }
  video { width: 100%; aspect-ratio: 2 / 1; background: #000; border-radius: 8px; display: block; }
  .caption { display: flex; gap: 12px; margin-top: 8px; color: var(--muted); font-size: 13px; }
  .caption .file { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--ink); }
  .caption .warn { color: var(--warn); }
  footer {
    position: sticky; bottom: 0; display: flex; align-items: center; gap: 12px;
    padding: 12px 20px; border-top: 1px solid var(--line); background: var(--panel);
  }
  footer .count { color: var(--muted); min-width: 90px; }
  .roles { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
  .roles label { padding: 6px 14px; cursor: pointer; user-select: none; border-left: 1px solid var(--line); }
  .roles label:first-child { border-left: 0; }
  .roles label:has(input:checked) { background: var(--accent); color: var(--accent-ink); }
  .roles input { position: absolute; opacity: 0; width: 0; height: 0; }
  .tagnote { color: var(--muted); font-size: 13px; }
  .tagnote code { color: var(--ink); }
  button {
    font: inherit; padding: 7px 14px; border-radius: 6px; cursor: pointer;
    border: 1px solid var(--line); background: var(--panel); color: var(--ink);
  }
  button.primary { background: var(--accent); color: var(--accent-ink); border-color: transparent; }
  button:disabled { opacity: .5; cursor: default; }
  .done { max-width: 760px; margin: 0 auto; padding: 60px 20px; color: var(--muted); font-size: 15px; }
  .done h2 { font-size: 15px; margin: 0 0 12px; color: var(--ink); font-weight: 600; }
  .done h2 code { font-weight: 400; }
  .done ul { list-style: none; margin: 0 0 16px; padding: 0; background: var(--panel);
    border: 1px solid var(--line); border-radius: 8px; }
  .done li { padding: 8px 12px; border-top: 1px solid var(--line); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13px; color: var(--ink); }
  .done li:first-child { border-top: 0; }
  .done code { color: var(--ink); }
  .done li { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 12px; align-items: center; }
  .done li > span:first-child { overflow: hidden; text-overflow: ellipsis; }
  .done li .stat { color: var(--muted); font-family: inherit; font-size: 13px; white-space: nowrap; }
  .done li .stat.ok { color: var(--accent); }
  .done li .stat.bad { color: var(--warn); }
  .done li .bar { grid-column: 1 / -1; height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; }
  .done li .bar i { display: block; height: 100%; width: 0; background: var(--accent); transition: width .3s; }
  .done li.pending .bar, .done li.skipped .bar { display: none; }
  .done .message { color: var(--ink); }
  .done .message.bad { color: var(--warn); }

  /* the folder browser */
  #locate { max-width: 760px; margin: 0 auto; padding: 20px; }
  #locate h2 { font-size: 15px; margin: 0 0 4px; }
  #locate .note { color: var(--warn); margin: 0 0 16px; }
  .roots { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
  .roots button { padding: 3px 10px; font-size: 13px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .pathbar { display: flex; gap: 8px; margin-bottom: 12px; }
  .pathbar input { flex: 1; font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; padding: 7px 10px;
    border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--ink); }
  .dirs { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .dirs .row { display: flex; align-items: center; gap: 10px; padding: 8px 12px; cursor: pointer;
    border-top: 1px solid var(--line); }
  .dirs .row:first-child { border-top: 0; }
  .dirs .row:hover { background: var(--hi); }
  .dirs .row .n { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
  .dirs .row .badge { margin-left: auto; font-size: 12px; color: var(--accent); font-weight: 600; }
  .dirs .row.up { color: var(--muted); }
  .dirs .empty { padding: 14px 12px; color: var(--muted); }
  .locate-foot { display: flex; align-items: center; gap: 12px; margin-top: 14px; }
  .locate-foot .err { color: var(--warn); }
  .locate-foot .here { color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>kiwi import</h1>
  <span class="where" id="where"></span>
  <span class="keys" id="keys">
    <kbd>↑</kbd>/<kbd>↓</kbd> move · <kbd>space</kbd> check · <kbd>a</kbd> all/none ·
    <kbd>i</kbd> import · <kbd>q</kbd> cancel
  </span>
</header>
<section id="locate" hidden>
  <h2>Locate the card</h2>
  <p class="note" id="note"></p>
  <div class="roots" id="roots"></div>
  <div class="pathbar">
    <input id="path" spellcheck="false" placeholder="/media/you/Insta360 X5">
    <button id="go">Go</button>
  </div>
  <div class="dirs" id="dirs"></div>
  <div class="locate-foot">
    <span class="here" id="here"></span>
    <span class="err" id="lerr"></span>
    <span style="margin-left:auto"></span>
    <button id="lcancel">Cancel</button>
    <button id="use" class="primary" disabled>Use this folder</button>
  </div>
</section>
<main id="pick" hidden>
  <div class="list"><table>
    <thead><tr><th></th><th>recorded</th><th class="num">size</th><th>clip</th></tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
  <aside>
    <video id="video" controls autoplay muted loop playsinline></video>
    <div class="caption">
      <span class="file" id="file"></span>
      <span id="dur"></span>
      <span class="warn" id="warn"></span>
    </div>
  </aside>
</main>
<footer id="foot" hidden>
  <span class="count" id="count">0 checked</span>
  <span class="roles" id="roles">
    <label><input type="radio" name="role" value="left">Left</label>
    <label><input type="radio" name="role" value="right">Right</label>
    <label><input type="radio" name="role" value="ego">Ego</label>
  </span>
  <span class="tagnote" id="tagnote"></span>
  <span style="margin-left:auto"></span>
  <button id="cancel">Cancel</button>
  <button id="import" class="primary">Import</button>
</footer>
<script>
(() => {
  const $ = id => document.getElementById(id);
  const rows = $('rows'), video = $('video');
  let clips = [], ROLES = {}, mount = null, dest = null;
  const checked = new Set();
  let cur = 0, shown = -1, finished = false, role = null;

  const mb = n => n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' : (n / 1e6).toFixed(1) + ' MB';
  const hms = s => {
    if (!isFinite(s)) return '';
    const m = Math.floor(s / 60), r = Math.round(s % 60);
    return m + ':' + String(r).padStart(2, '0');
  };
  const esc = s => s.replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const post = (path, body) => fetch(path, { method: 'POST', body: JSON.stringify(body) });

  // --- which view -------------------------------------------------------
  let view = 'pick';
  function showView(v) {
    view = v;
    $('locate').hidden = v !== 'locate';
    $('pick').hidden = v !== 'pick';
    $('foot').hidden = v !== 'pick';
    $('keys').hidden = v !== 'pick';
    if (v === 'pick') {
      $('where').innerHTML = `${clips.length} ${clips.length === 1 ? 'clip' : 'clips'}, newest first, from ` +
        `<code>${esc(mount)}</code><a id="change">change folder</a>`;
      $('change').addEventListener('click', () => { video.pause(); openLocate(mount); });
    } else {
      $('where').innerHTML = mount ? `<code>${esc(mount)}</code>` : '';
    }
  }

  function applyState(s) {
    clips = s.clips; ROLES = s.roles; mount = s.mount; dest = s.dest;
    checked.clear(); cur = 0; shown = -1;
    $('note').textContent = s.note || '';
    $('roots').innerHTML = '';
    s.roots.forEach(r => {
      const b = document.createElement('button');
      b.textContent = r;
      b.addEventListener('click', () => browse(r));
      $('roots').appendChild(b);
    });
    if (clips.length) { buildRows(); showView('pick'); render(); }
    else openLocate(s.roots[0] || '/');
  }

  // --- the folder browser -----------------------------------------------
  let here = null, hereClips = 0;
  function openLocate(start) { showView('locate'); browse(start || '/'); }
  async function browse(path) {
    $('lerr').textContent = '';
    const r = await fetch('/dirs?path=' + encodeURIComponent(path));
    const d = await r.json();
    if (d.error) { $('lerr').textContent = d.error; return; }
    here = d.path; hereClips = d.clips;
    $('path').value = d.path;
    const list = $('dirs');
    list.innerHTML = '';
    if (d.parent) {
      const up = document.createElement('div');
      up.className = 'row up'; up.innerHTML = '<span class="n">..</span>';
      up.addEventListener('click', () => browse(d.parent));
      list.appendChild(up);
    }
    d.dirs.forEach(e => {
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = `<span class="n">${esc(e.name)}/</span>` +
        (e.clips ? `<span class="badge">${e.clips} clip${e.clips === 1 ? '' : 's'}</span>` : '');
      row.addEventListener('click', () => browse(e.path));
      list.appendChild(row);
    });
    if (!d.dirs.length) list.innerHTML += '<div class="empty">no folders here</div>';
    $('here').textContent = hereClips
      ? `${hereClips} clip${hereClips === 1 ? '' : 's'} in this folder`
      : 'no clips in this folder';
    $('use').disabled = !hereClips;
  }
  async function useFolder() {
    if (!here || !hereClips) return;
    const r = await post('/mount', { path: here });
    const d = await r.json();
    if (d.error) { $('lerr').textContent = d.error; return; }
    applyState(d);
  }
  $('go').addEventListener('click', () => browse($('path').value));
  $('path').addEventListener('keydown', e => { if (e.key === 'Enter') browse($('path').value); });
  $('use').addEventListener('click', useFolder);
  $('lcancel').addEventListener('click', () => doCancel());

  // --- the clip list ----------------------------------------------------
  function buildRows() {
    rows.innerHTML = '';
    clips.forEach((c, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td class="chk"><input type="checkbox"></td>` +
        `<td>${c.stamp}</td><td class="num">${mb(c.size)}</td><td class="name">${esc(c.name)}</td>`;
      tr.addEventListener('click', e => {
        if (e.target.tagName === 'INPUT') { toggle(i); } else { cur = i; render(); }
      });
      rows.appendChild(tr);
    });
  }

  // One role at most: radios enforce that; a second click on the chosen one
  // clears it, back to naming the files by camera unit.
  document.querySelectorAll('input[name=role]').forEach(r => {
    r.addEventListener('click', () => {
      if (role === r.value) { r.checked = false; role = null; } else { role = r.value; }
      r.blur();
      renderTag();
    });
  });
  function renderTag() {
    $('tagnote').innerHTML = role
      ? `files get <code>_${ROLES[role]}</code>` : 'files get the camera unit';
  }

  function toggle(i) { checked.has(i) ? checked.delete(i) : checked.add(i); cur = i; render(); }

  function render() {
    [...rows.children].forEach((tr, i) => {
      tr.classList.toggle('cur', i === cur);
      tr.classList.toggle('on', checked.has(i));
      tr.querySelector('input').checked = checked.has(i);
    });
    $('count').textContent = checked.size + ' checked';
    renderTag();
    if (shown !== cur) show(cur);
    if (rows.children[cur]) rows.children[cur].scrollIntoView({ block: 'nearest' });
  }

  function show(i) {
    shown = i;
    const c = clips[i];
    $('file').textContent = c.name;
    $('dur').textContent = '';
    $('warn').textContent = c.proxy ? '' : 'no LRV proxy: playing the clip itself';
    video.src = '/media/' + i;
    video.play().catch(() => {});
  }
  video.addEventListener('loadedmetadata', () => { $('dur').textContent = hms(video.duration); });
  video.addEventListener('error', () => {
    if (clips[cur]) $('warn').textContent = 'this browser cannot play ' + clips[cur].name;
  });

  async function finish(path, body, html) {
    if (finished) return;
    finished = true;
    video.pause();
    await post(path, body).catch(() => {});
    document.body.innerHTML = `<div class="done">${html}<p>back to the terminal.</p></div>`;
  }
  // After Import the server stays up and the copy reports into /progress:
  // one row per planned file, a bar while it streams, then the closing line.
  const rate = j => j.elapsed > 0 ? mb(j.copied / j.elapsed) + '/s' : '';
  const eta = j => {
    if (!j.copied || !j.elapsed || j.copied >= j.size) return '';
    const s = (j.size - j.copied) / (j.copied / j.elapsed);
    return s >= 60 ? `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, '0')}s` : `${s.toFixed(0)}s`;
  };
  function drawProgress(p) {
    const where = p.dest ? ` to <code>${esc(p.dest)}</code>` : '';
    const rowsHtml = p.jobs.map(j => {
      const pct = j.size ? Math.min(100, 100 * j.copied / j.size) : 0;
      let stat = '';
      if (j.state === 'pending') stat = `<span class="stat">${mb(j.size)}</span>`;
      else if (j.state === 'copying') {
        // The file is named only when it is not the row's own (the LRV proxy).
        const which = j.file !== j.dst ? `${esc(j.file)} · ` : '';
        stat = `<span class="stat">${which}${pct.toFixed(0)}% · ${rate(j)} · ${eta(j)}</span>`;
      }
      else if (j.state === 'copied') stat = `<span class="stat ok">done · ${mb(j.size)}</span>`;
      else if (j.state === 'skipped') stat = `<span class="stat">already imported</span>`;
      const width = j.state === 'copied' ? 100 : pct;
      return `<li class="${j.state}"><span>${esc(j.dst)}</span>${stat}<span class="bar"><i style="width:${width}%"></i></span></li>`;
    }).join('');
    const tail = p.phase === 'done' ? `<p class="message">${esc(p.message)}</p><p>back to the terminal.</p>`
      : p.phase === 'failed' ? `<p class="message bad">${esc(p.message)}</p><p>back to the terminal.</p>`
      : p.phase === 'picking' ? '<p>preparing the copy plan…</p>' : '';
    document.body.innerHTML = `<div class="done"><h2>Importing${where}</h2>` +
      (p.jobs.length ? `<ul>${rowsHtml}</ul>` : '') + tail + '</div>';
  }
  async function watchProgress() {
    for (;;) {
      let p;
      try { p = await (await fetch('/progress')).json(); } catch (e) { return; }  // server gone
      drawProgress(p);
      if (p.phase === 'done' || p.phase === 'failed') { post('/ack', {}).catch(() => {}); return; }
      await new Promise(r => setTimeout(r, 400));
    }
  }
  async function doImport() {
    if (finished) return;
    finished = true;
    video.pause();
    const picked = checked.size ? [...checked].sort((a, b) => a - b) : [cur];
    await post('/select', { picked, role }).catch(() => {});
    watchProgress();
  }
  function doCancel() { finish('/cancel', {}, '<h2>Cancelled</h2>'); }
  $('import').addEventListener('click', doImport);
  $('cancel').addEventListener('click', doCancel);

  document.addEventListener('keydown', e => {
    if (finished || e.metaKey || e.ctrlKey || e.altKey) return;
    if (view !== 'pick') {
      if ((e.key === 'q' || e.key === 'Escape') && e.target.tagName !== 'INPUT') doCancel();
      return;
    }
    const n = clips.length, page = 10;
    switch (e.key) {
      case 'ArrowDown': cur = Math.min(n - 1, cur + 1); break;
      case 'ArrowUp': cur = Math.max(0, cur - 1); break;
      case 'PageDown': cur = Math.min(n - 1, cur + page); break;
      case 'PageUp': cur = Math.max(0, cur - page); break;
      case 'Home': cur = 0; break;
      case 'End': cur = n - 1; break;
      case ' ': case 'Enter': toggle(cur); break;
      case 'a': case 'A':
        if (checked.size === n) checked.clear(); else clips.forEach((_, i) => checked.add(i));
        break;
      case 'i': case 'I': doImport(); return;
      case 'q': case 'Escape': doCancel(); return;
      default: return;
    }
    e.preventDefault();
    render();
  });

  fetch('/state').then(r => r.json()).then(applyState);
})();
</script>
</body>
</html>
"""

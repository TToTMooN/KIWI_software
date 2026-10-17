"""``kiwi import``: pull clips off a USB-mounted camera into a directory.

In U-disk mode the camera mounts as plain mass storage with clips under
``DCIM/Camera01/``. This copies the newest N clips off the connected camera
into ``--dest``, naming each after the **camera-registry unit** the clip's
telemetry serial resolves to.

Two design rules:

* **No arm flag.** A serial identifies the camera *body*, never the arm -
  which arm a body rides is a property of the sitting, and
  :mod:`kiwi.episode.init` is *told* (``--left`` / ``--right``) and
  re-probes the serial itself. So the name written here is never arm
  authority.
* **Atomic copy, with progress.** Written in chunks to a ``.part`` sidecar
  and renamed on success, an interrupted run leaves nothing to mistake for a
  clip; the copy reports a share, a rate and an eta as it goes.

Choosing happens on a page served at localhost (:mod:`kiwi.core.camera.picker`):
the candidates newest first beside a ``<video>`` playing the highlighted
clip's LRV proxy, which the browser decodes itself - nothing here touches a
codec, and over ssh the page is one ``-L`` port forward away. Picking is the
default when stdout is a terminal; the newest clip is taken blind with
``--no-pick``, or in any run whose stdout is not a terminal to print the URL
on.

Every selected clip lands directly in ``--dest``, with the unit on the name::

    <dest>/VID_20260820_155520_00_126_insta360-X5-001.insv
    <dest>/VID_20260820_155518_00_070_insta360-X5-003.insv

Import writes no episode structure. It offloads a card and says what each
clip is; sorting clips into episodes is the operator's - each episode's
clips go flat into its ``raw/`` (:mod:`kiwi.core.layout`), where ``kiwi
init`` reads them and takes an extra stream's label from the ``_<tag>``
suffix stamped here. Running again with the other camera plugged in simply
lands its clips beside the first's.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

from kiwi.core.camera.registry import find_camera_by_serial, read_insv_serial

__all__ = [
    "CLIP_PREFIXES",
    "CLIP_RE",
    "CLIP_SUFFIXES",
    "MOUNT_GLOBS",
    "Candidate",
    "CopyJob",
    "CopyReporter",
    "DEFAULT_PORT",
    "IngestError",
    "ROLE_TAGS",
    "Result",
    "Selection",
    "clip_key",
    "clip_label",
    "clip_start",
    "copy_one",
    "find_mounts",
    "ingest",
    "lrv_for",
    "newest_clips",
    "pick_clips",
    "plan_jobs",
    "progress_line",
    "resolve_unit",
]

MOUNT_GLOBS = ("media/*/*", "run/media/*/*")

# Clip stems: VID_20260724_173320_00_050 on both the X5 and the GO 3, plus the
# GO 3's PRO_ capture mode (PRO_VID_20231218_021737_00_048, paired with a
# PRO_LRV_* proxy).
CLIP_RE = re.compile(
    r"(?P<pro>PRO_)?VID_(?P<date>\d{8})_(?P<time>\d{6})_\d+_(?P<seq>\d+)\Z"
)

# An imported clip's stem: the camera's own stem plus the optional ``_<tag>``
# suffix plan_jobs appends (the resolved unit, or --label). The tag may itself
# contain underscores (ego_video, insta360-X5-001), so it cannot be split off
# the right; it is whatever follows a camera-shaped stem.
TAGGED_RE = re.compile(
    r"(?P<pro>PRO_)?(?:VID|LRV)_\d{8}_\d{6}_\d+_\d+(?:_(?P<tag>.+))?\Z"
)

# The X5 records .insv; the single-lens GO 3 records .mp4.
CLIP_SUFFIXES = (".insv", ".mp4")

# Anchored at the start of the name, so the AppleDouble sidecars macOS leaves
# beside every clip on an exFAT card (._VID_*.mp4) do not match.
CLIP_PREFIXES = ("VID_", "PRO_VID_")


class IngestError(Exception):
    """A refusal the operator can act on; the CLI prints it and exits 2."""


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One clip on the card, with what the picker needs to describe it."""

    path: Path
    size: int
    start: dt.datetime  # from the filename stamp, else the file mtime


@dataclasses.dataclass(frozen=True)
class CopyJob:
    """One planned copy: ``src`` on the card to ``dst`` in ``--dest``."""

    src: Path
    dst: Path

    def rel_to(self, dest: Path) -> str:
        return str(self.dst.relative_to(dest))


@dataclasses.dataclass(frozen=True)
class Result:
    """What an ingest wrote, and what the operator does with it next."""

    written: list[Path]
    dest: Path

    @property
    def next_step(self) -> str:
        """What follows an offload - a human step, not a command to paste."""
        return (
            f"sort the clips in {self.dest} into <episode>/raw/, then: "
            f"kiwi init --episode <episode> --left <unit> --right <unit>"
        )


# --- finding clips on the card ---------------------------------------------


def find_mounts() -> list[Path]:
    # Detect by layout, not volume label: firmware always puts clips under
    # DCIM/Camera01/.
    root = Path("/")
    return sorted(
        p
        for g in MOUNT_GLOBS
        for p in root.glob(g)
        if (p / "DCIM" / "Camera01").is_dir()
    )


def clip_key(clip: Path) -> tuple[str, str, str, str]:
    """Chronological sort key, insensitive to the ``PRO_`` prefix.

    Falls back to the name for stems that do not parse. Equivalent to name
    order on an all-``VID_`` card.
    """
    m = CLIP_RE.match(clip.stem)
    if m is None:
        return ("", "", "", clip.name)
    return (m["date"], m["time"], m["seq"].zfill(6), clip.name)


def clip_label(clip: Path) -> str:
    """The stream label an imported clip's filename carries.

    The ``_<tag>`` suffix :func:`plan_jobs` stamped on the stem, when there is
    one; the whole stem otherwise (a clip imported untagged, or a file that
    never went through import). This is what names an extra stream's directory
    at ``kiwi init`` time - arms are named by side there, from the embedded
    serial, so their tags are provenance only.
    """
    m = TAGGED_RE.match(clip.stem)
    if m is not None and m["tag"]:
        return m["tag"]
    return clip.stem


def clip_start(clip: Path) -> dt.datetime:
    """When the clip started: its filename stamp, else its mtime.

    The stamp is the camera's own clock, which is what pairs a take across two
    bodies; mtime is the copy time on a card that has been through a computer.
    """
    m = CLIP_RE.match(clip.stem)
    if m is not None:
        try:
            return dt.datetime.strptime(f"{m['date']} {m['time']}", "%Y%m%d %H%M%S")
        except ValueError:
            pass  # a well-shaped name carrying an impossible date
    return dt.datetime.fromtimestamp(clip.stat().st_mtime)


def newest_clips(mount: Path, n: int) -> list[Candidate]:
    """The newest ``n`` clips under ``mount`` (its ``DCIM/*/`` when it has
    one, else the folder itself, else its immediate subfolders), newest first.

    Newest first is the order the picker shows them in. The copy plan
    re-sorts chronologically itself (:func:`plan_jobs`).
    """
    # A card keeps its clips under DCIM/<folder>/; a copy of a card, or a
    # folder someone gathered clips into, holds them directly; and DCIM/
    # itself, one level above the clips, counts too.
    if (mount / "DCIM").is_dir():
        layouts = [(mount / "DCIM", "*/")]
    else:
        layouts = [(mount, ""), (mount, "*/")]
    clips: set[Path] = set()
    for root, pattern in layouts:
        clips = {
            p
            for prefix in CLIP_PREFIXES
            for suffix in CLIP_SUFFIXES
            for p in root.glob(f"{pattern}{prefix}*{suffix}")
        }
        if clips:
            break
    newest = sorted(clips, key=clip_key)[-n:]
    return [Candidate(p, p.stat().st_size, clip_start(p)) for p in reversed(newest)]


def lrv_for(clip: Path) -> Path | None:
    # VID_20260701_191600_00_100.insv pairs with LRV_20260701_191600_01_100.lrv
    # (the second-to-last field differs), so glob for it. The GO 3's PRO_ mode
    # pairs PRO_VID_* with PRO_LRV_*.
    m = CLIP_RE.match(clip.stem)
    if m is None:
        return None
    pro = m["pro"] or ""
    hits = sorted(
        clip.parent.glob(f"{pro}LRV_{m['date']}_{m['time']}_*_{m['seq']}.lrv")
    )
    return hits[0] if hits else None


# --- which registry unit shot them -----------------------------------------


def resolve_unit(clip: Path, cameras_dir: Path) -> str:
    """The registry unit that shot ``clip``, or raise naming what to fix.

    The two failures get different messages: a container with no readable
    serial (the GO3 ``.mp4``) needs ``--label``, while a serial that matches
    no unit needs a registry entry. The caller probes ONE clip - one
    invocation reads one camera, so the rest of the card is the same body.
    """
    serial = read_insv_serial(clip) if clip.suffix.lower() == ".insv" else None
    if serial is None:
        raise IngestError(
            f"{clip.name} carries no readable telemetry serial, so the camera "
            f"cannot be named from the recording (the GO3 records a plain "
            f"mp4). Pass --label to tag the filename with a name of your own, "
            f"or --unit to name the registry body it is:\n"
            f"    kiwi import --dest ... --label ego_video\n"
            f"    kiwi import --dest ... --unit insta360-GO3-001"
        )
    unit_dir = find_camera_by_serial(cameras_dir, serial)
    if unit_dir is None:
        raise IngestError(
            f"{clip.name} was shot by serial {serial!r}, which matches no unit "
            f"under {cameras_dir}. Register the body (a directory with a "
            f"camera.yaml naming that serial - see config/README.md) or pass "
            f"--label to tag the filename with a name of your own."
        )
    return unit_dir.name


# --- planning where they land ----------------------------------------------


def plan_jobs(
    clips: list[Candidate],
    dest: Path,
    tag: str | None,
    name: str | None = None,
) -> list[CopyJob]:
    """Decide every destination path. Pure: touches no file it does not stat.

    Every clip lands directly in ``dest``; ``tag`` (the resolved unit, or
    ``--label`` when no serial can be read) goes on the filename, which is
    what says who shot it once the clip has been moved somewhere else.
    """
    if not clips:
        raise IngestError("nothing selected")
    # Chronological whatever order they were picked in: --name numbers them in
    # shooting order.
    clips = sorted(clips, key=lambda c: clip_key(c.path))

    jobs = []
    for i, clip in enumerate(clips):
        stem = clip.path.stem
        if name:
            stem = name if len(clips) == 1 else f"{name}_{i + 1:02d}"
        if tag:
            stem = f"{stem}_{tag}"
        # .insv (X5) or .mp4 (GO 3), never assumed
        jobs.append(CopyJob(clip.path, dest / f"{stem}{clip.path.suffix}"))
    return jobs


# --- copying ---------------------------------------------------------------


#: Read size for the copy.
COPY_CHUNK = 8 << 20

#: Seconds between progress redraws.
PROGRESS_EVERY = 0.25


def _human_bytes(n: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n:.0f} B"


def _human_time(seconds: float) -> str:
    if seconds < 10:  # a fast card, or an eta about to expire
        return f"{max(0.0, seconds):.1f}s"
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def progress_line(copied: int, size: int, elapsed: float) -> str:
    """One redraw of the copy progress: share, bytes, rate, eta.

    The rate is the average over the whole transfer, not the last chunk.
    """
    share = copied / size if size else 1.0
    rate = copied / elapsed if elapsed > 0 else 0.0
    eta = _human_time((size - copied) / rate) if rate > 0 else "--"
    return (
        f"    {share:4.0%}  {_human_bytes(copied)} / {_human_bytes(size)}"
        f"  {_human_bytes(rate)}/s  eta {eta}"
    )


Tick = Callable[[int, int, float], None]
"""A progress sink: ``(copied, size, elapsed)``, every :data:`PROGRESS_EVERY`."""


def _stream(src: Path, dst: Path, size: int, tick: Tick | None = None) -> float:
    """Copy the bytes chunk by chunk, drawing progress. Returns the seconds.

    Metadata is copied after the bytes, so this is ``shutil.copy2`` with a
    hook between the chunks. The line is drawn only at a terminal.
    ``tick`` hears the same beats whatever stdout is - that is the page's
    bar.
    """
    show = sys.stdout.isatty()
    started = drawn = time.monotonic()
    copied = width = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        while chunk := fin.read(COPY_CHUNK):
            fout.write(chunk)
            copied += len(chunk)
            now = time.monotonic()
            if now - drawn < PROGRESS_EVERY:
                continue
            drawn = now
            if tick is not None:
                tick(copied, size, now - started)
            if show:
                line = progress_line(copied, size, now - started)
                # Padded to the widest line drawn so far: the text shrinks
                # (999.9 MB -> 1.0 GB, 1m02s -> 9.4s).
                print("\r" + line.ljust(width), end="", flush=True)
                width = max(width, len(line))
    if tick is not None:
        tick(copied, size, time.monotonic() - started)
    shutil.copystat(src, dst)
    if show:  # leave the line to the summary
        print("\r" + " " * width + "\r", end="", flush=True)
    return time.monotonic() - started


def copy_one(
    src: Path, dst: Path, overwrite: bool = False, tick: Tick | None = None
) -> bool:
    """Copy ``src`` to ``dst`` atomically. Returns False when already there.

    Written to a ``.part`` sidecar and renamed on success (the demux pattern
    from :mod:`kiwi.core.camera.convert`): an interrupted copy leaves
    nothing under the canonical name.
    """
    size = src.stat().st_size
    if dst.exists():
        if dst.stat().st_size == size:
            print(f"  = {dst} already imported, skipping")
            return False
        if not overwrite:
            raise IngestError(
                f"{dst} exists with a different size ({dst.stat().st_size} on "
                f"disk vs {size} on the card) - pass --overwrite to re-copy."
            )
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  > {src.name} -> {dst} ({_human_bytes(size)})", flush=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        elapsed = _stream(src, tmp, size, tick)
        if tmp.stat().st_size != size:
            raise IngestError(f"size mismatch after copy: {dst}")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    rate = _human_bytes(size / elapsed) + "/s" if elapsed > 0 else "instant"
    print(f"    {_human_bytes(size)} in {_human_time(elapsed)} ({rate})")
    return True


# --- the picker ------------------------------------------------------------

DEFAULT_PORT = 8765
"""Where the picker's page is served."""

ROLE_TAGS = {"left": "left", "right": "right", "ego": "ego_video"}
"""The page's one-of-three role choice and the filename tag each stamps in
place of the unit. ``ego_video`` is the label the ``kiwi init --camera``
examples use; ``left`` / ``right`` on an arm's clip are provenance only,
since init binds arms by serial."""


class CopyReporter:
    """Where the copy tells its story besides the terminal: the picker page
    stays up through the copy and draws these as bars. Every method is a
    no-op here, so a run without a page reports into the void."""

    def planned(self, jobs: list[CopyJob], dest: Path) -> None:
        pass

    def started(self, i: int, src: Path, dst: Path, size: int) -> None:
        pass

    def progress(self, i: int, copied: int, size: int, elapsed: float) -> None:
        pass

    def finished(self, i: int, outcome: str) -> None:
        """``outcome`` is ``copied`` or ``skipped`` (already there)."""

    def done(self, message: str) -> None:
        pass

    def failed(self, message: str) -> None:
        pass

    def close(self) -> None:
        pass


@dataclasses.dataclass(frozen=True)
class Selection:
    """What the picker decided: the clips, the tag to put on their names
    instead of the resolved unit (a value of :data:`ROLE_TAGS`, or None to
    name them by unit), and the page that wants to watch the copy, if one
    is still up."""

    clips: list[Candidate]
    tag: str | None = None
    page: CopyReporter | None = None


def pick_clips(
    clips: list[Candidate],
    *,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    n: int = 30,
    note: str = "",
    dest: Path | None = None,
) -> Selection:
    """Choose on the localhost page (:mod:`kiwi.core.camera.picker`). With no
    ``clips`` the page opens on its folder browser, ``note`` saying why, and
    a chosen folder contributes its newest ``n``. Imported here so a run
    that never picks never builds a server."""
    from kiwi.core.camera.picker import pick_in_browser

    return pick_in_browser(
        clips, port=port, open_browser=open_browser, n=n, note=note, dest=dest
    )


# --- the verb --------------------------------------------------------------


def ingest(
    dest: Path,
    cameras_dir: Path,
    *,
    mount: Path | None = None,
    latest: int | None = None,
    pick: bool = True,
    label: str | None = None,
    unit: str | None = None,
    name: str | None = None,
    lrv: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> Result:
    """Offload one camera into ``dest``. Raises :class:`IngestError` to refuse.

    A clip whose unit cannot be resolved - no telemetry serial, or a serial
    registered nowhere - is imported under its own name rather than refused.
    """
    # The picker announces its URL on stdout. With stdout redirected to a
    # file or a pipe the run takes the newest clip; the note goes to stderr.
    if pick and not sys.stdout.isatty():
        print(
            "no terminal to pick on: taking the newest clip (--latest N for more)",
            file=sys.stderr,
        )
        pick = False
    n = latest if latest is not None else (30 if pick else 1)
    if n < 1:
        raise IngestError("--latest must be >= 1")

    # Whatever stops the card from being found is a refusal without a page
    # to fix it on, and the page's folder browser with it: the operator
    # walks to the card (an SD reader under /mnt, a copy of a card) and the
    # list fills from there.
    clips: list[Candidate] = []
    note = ""
    if mount is not None:
        mounts = [mount]
    else:
        mounts = find_mounts()
        if not mounts:
            note = (
                "no camera mount found under /media or /run/media - is the "
                "camera plugged in and in U-disk (file-transfer) mode? Pass "
                "--mount to name one explicitly."
            )
        elif len(mounts) > 1:
            names = ", ".join(str(m) for m in mounts)
            note = (
                f"{len(mounts)} cameras are mounted ({names}). One invocation "
                f"reads one body, so that every clip is named by the same "
                f"unit - pass --mount to choose, then run again for the other."
            )
    if not note:
        if not (mounts[0] / "DCIM").is_dir() and mount is not None:
            note = f"{mounts[0]} has no DCIM/ - not a camera mount?"
        else:
            clips = newest_clips(mounts[0], n)
            if not clips:
                exts = "/".join(CLIP_SUFFIXES)
                note = f"no VID_*{exts} under {mounts[0]}"
    if note and not pick:
        raise IngestError(note)

    if pick:
        chosen = pick_clips(
            clips, port=port, open_browser=open_browser, n=n, note=note, dest=dest
        )
        clips = chosen.clips
        if chosen.tag is not None:
            label = chosen.tag  # the page's role stands where --label would

    if unit is None:
        try:
            unit = resolve_unit(clips[0].path, cameras_dir)
        except IngestError as why:
            # An unnamed clip keeps the name the camera gave it, and the
            # reason goes to stderr.
            if label is None:
                print(f"note: importing untagged - {why}", file=sys.stderr)
            unit = None
    jobs = plan_jobs(clips, dest, label or unit, name)
    page = chosen.page if pick and chosen.page is not None else CopyReporter()
    try:
        return _copy_all(
            jobs, dest, page, lrv=lrv, overwrite=overwrite, dry_run=dry_run
        )
    except BaseException as e:
        page.failed(str(e) or type(e).__name__)
        raise
    finally:
        page.close()


def _copy_all(
    jobs: list[CopyJob],
    dest: Path,
    page: CopyReporter,
    *,
    lrv: bool,
    overwrite: bool,
    dry_run: bool,
) -> Result:
    page.planned(jobs, dest)
    for job in jobs:
        print(f"{job.src} -> {job.dst}")
    if dry_run:
        message = f"dry run: {len(jobs)} clip(s), nothing copied"
        print(message)
        page.done(message)
        return Result([], dest)

    written = []
    for i, job in enumerate(jobs):

        def tick(copied: int, size: int, elapsed: float, i: int = i) -> None:
            page.progress(i, copied, size, elapsed)

        page.started(i, job.src, job.dst, job.src.stat().st_size)
        copied = copy_one(job.src, job.dst, overwrite, tick)
        if copied:
            written.append(job.dst)
        if lrv:
            proxy = lrv_for(job.src)
            if proxy is None:
                print(f"  ! no LRV proxy found for {job.src.name}")
            else:
                page.started(
                    i, proxy, job.dst.with_suffix(".lrv"), proxy.stat().st_size
                )
                copy_one(proxy, job.dst.with_suffix(".lrv"), overwrite, tick)
        page.finished(i, "copied" if copied else "skipped")
    message = f"imported {len(jobs)} clip(s) to {dest}"
    print(message)
    page.done(message)
    return Result(written, dest)

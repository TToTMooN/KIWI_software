"""Render a dataset card (``README.md``) for an exported LeRobot dataset.

``meta/export_provenance.json`` records every output episode's source dir and
global-clock window. HF shows ``README.md`` on the dataset page; this module
renders the provenance sidecar into that file.

The card is derived purely from the provenance dict (plus the optional
curation file), never from the build state, so it can be regenerated for an
export that already exists::

    kiwi export card --dataset data/<dataset>/lerobot_v3
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

__all__ = ["build_card", "write_card"]


def _fmt_window(w: list[float] | tuple[float, float]) -> str:
    return f"{float(w[0]):.2f}-{float(w[1]):.2f}"


def build_card(
    prov: dict, *, repo_id: str | None = None, curation: dict | None = None
) -> str:
    """Render the card body from an ``export_provenance.json`` dict.

    ``curation`` is the raw curation file, used only for the windows a source
    was cut with; the per-episode truth still comes from ``prov`` (a window
    can be narrowed by real pose/video/tool-state coverage after the cut).
    """
    eps = prov.get("episodes") or []
    by_source: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_source[str(e.get("source", "?"))].append(e)

    total_frames = sum(int(e.get("frames", 0)) for e in eps)
    name = repo_id or "LeRobot dataset"

    out: list[str] = []
    out.append(f"# {name}\n")
    out.append(
        f"{len(eps)} episodes from {len(by_source)} source recordings "
        f"({total_frames} frames total), exported from raw portable-gripper "
        f"captures.\n"
    )

    # --- the two things a consumer cannot recover from the parquet alone ----
    wy, wo = prov.get("world_yaw"), prov.get("world_offset")
    out.append("## Coordinate frame\n")
    out.append(
        f"Poses are in the **{prov.get('frame', '?')} frame**, "
        f"body frame `{prov.get('ee_frame', '?')}`.\n"
    )
    world_source = prov.get("world_source")
    if world_source == "marker":
        out.append(
            "That frame is this episode's own table-marker pose, so it is "
            "comparable within an episode but not across them.\n"
        )
    elif world_source == "markerless":
        out.append(
            "That frame is the scene's map frame (metric, gravity-aligned), "
            "shared by every episode recorded in the same scene, so poses "
            "are directly comparable across them.\n"
        )
    if wy is None and wo is None:
        out.append("No world transform is baked in: positions are raw world-frame.\n")
    elif wy is None:
        # Translation only: the replay still applies its own world yaw, and
        # the offset is stated in the UNROTATED frame, so a reader composing
        # it with a yaw applies the yaw afterwards.
        out.append(
            f"A world **translation** of **{wo}** m is baked into every "
            "position. Orientations are untouched, and no rotation is baked "
            "in.\n"
        )
        out.append(
            "A replay whose frame convention differs from this dataset's world frame "
            "must still apply its own world yaw, which rotates this offset "
            "with the rest of the data.\n"
        )
    else:
        out.append(
            "A world transform is **baked into every pose**, applied as "
            "`Translate(world_offset) @ Rz(world_yaw)` - rotate about the "
            "world origin first, then translate in the rotated frame:\n"
        )
        out.append("")
        out.append(f"- `world_yaw`: **{wy}** deg about +Z (rotates orientations too)")
        out.append(
            f"- `world_offset`: **{wo if wo is not None else [0.0, 0.0, 0.0]}** m"
        )
        out.append("")
        out.append(
            "A replay must therefore apply **no** additional yaw or workspace "
            "translation - the alignment is already in the data.\n"
        )

    # --- the curation cuts ---------------------------------------------------
    curated = prov.get("curation") or {}
    out.append("## Episode windows\n")
    if curated:
        src_file = curated.get("file")
        out.append(
            "Each source recording holds several takes of the task; every "
            "kept window below was cut into its own episode"
            + (f" (curation file `{src_file}`).\n" if src_file else ".\n")
        )
    else:
        out.append("One episode per source recording.\n")
    out.append(
        "Windows are seconds on that episode's **global synced clock** (the "
        "shared multi-camera time base), which is also what the source "
        "`episode.yaml` `task_window`/`sync` are expressed in. A window may "
        "be marginally narrower than the requested cut where real "
        "pose/video/tool-state coverage ends sooner.\n"
    )
    out.append("")
    out.append("| episode | source | window (s) | frames |")
    out.append("|---|---|---|---|")
    for e in eps:
        out.append(
            f"| {e.get('episode_index')} | `{e.get('source')}` | "
            f"{_fmt_window(e.get('window', [0, 0]))} | {e.get('frames')} |"
        )
    out.append("")

    if curation:
        out.append("### Requested cuts, by source\n")
        out.append("| source | takes | requested windows (s) |")
        out.append("|---|---|---|")
        for entry in curation.get("episodes", []):
            ws = entry.get("keep_windows") or []
            out.append(
                f"| `{entry.get('source')}` | {len(ws)} | "
                f"{', '.join(_fmt_window(w) for w in ws)} |"
            )
        out.append("")

    skipped = prov.get("skipped") or []
    if skipped:
        out.append("### Skipped sources\n")
        for s in skipped:
            out.append(f"- `{s.get('source')}`: {s.get('reason')}")
        out.append("")

    out.append("## Provenance\n")
    out.append(f"- generated: {prov.get('generated')}")
    git = prov.get("kiwi_git") or {}
    if isinstance(git, dict) and git.get("commit"):
        out.append(f"- kiwi commit: `{git['commit']}`")
    out.append(
        "- full per-episode source/window and the per-arm extrinsic chain: "
        "`meta/export_provenance.json`"
    )
    out.append("")
    return "\n".join(out)


def write_card(dataset_dir: str | Path, *, repo_id: str | None = None) -> Path:
    """Write ``README.md`` into an exported dataset dir; returns the path.

    Reads ``meta/export_provenance.json`` and, when the export was curated and
    the curation file is still resolvable, that file too.
    """
    dataset_dir = Path(dataset_dir)
    prov = json.loads((dataset_dir / "meta" / "export_provenance.json").read_text())
    curation = None
    cur_meta = prov.get("curation") or {}
    if cur_meta.get("file"):
        # The curation file lives with the dataset it curated, not with the
        # export; search the usual data/<dataset>/curation/ location upward.
        # A moved export renders without the requested-cuts table.
        for base in dataset_dir.resolve().parents:
            cand = base / "curation" / cur_meta["file"]
            if cand.is_file():
                curation = json.loads(cand.read_text())
                break
    if repo_id is None:
        # The provenance sidecar records the repo id (LeRobot's meta carries
        # none in v2.1). A sidecar without one falls through to the generic
        # title.
        repo_id = prov.get("repo_id")
    card = build_card(prov, repo_id=repo_id, curation=curation)
    p = dataset_dir / "README.md"
    p.write_text(card)
    return p

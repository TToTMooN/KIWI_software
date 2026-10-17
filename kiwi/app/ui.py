"""NiceGUI workspace: persistent data management beside embedded Rerun."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from nicegui import app, run, ui
from starlette.background import BackgroundTask

from kiwi.app.scene import (
    LAYER_KINDS,
    default_visibility,
    layer_update_bytes,
    manifest_path,
)
from kiwi.app.viewer import Viewer
from kiwi.app.splats import attach_scene, attached_scene
from kiwi.app.worker import review_key
from kiwi.app.workspace import Files, Jobs, import_clips, inside, scan_library
from kiwi.core.camera.ingest import find_mounts, newest_clips
from kiwi.core.camera.picker import preview_source
from kiwi.core.paths import config_root
from kiwi.episode.init import init_episode


class WorkspacePage:
    def __init__(
        self, root: Path, cache: Path, jobs: Jobs, files: Files, viewer: Viewer
    ):
        self.root, self.cache = root, cache
        self.jobs, self.files, self.viewer = jobs, files, viewer
        self.selected = None
        self.clips = []
        self.episodes = []
        self.recordings = []
        self.last_jobs = []
        self.pending_viz = None
        self.open_serial = 0
        self.layer_manifest = None
        self.layer_choices = {}
        self.layer_applied = {}
        self.layer_checkboxes = {}
        self.layer_revision = 0
        self.layer_lock = asyncio.Lock()
        self.layer_syncing = False
        self.recording_url = None
        self.recording_path = None
        self.client = ui.context.client
        self.base_url = str(ui.context.client.request.base_url).rstrip("/")
        ui.colors(primary="#0068e8", secondary="#44546b", accent="#6859db")
        ui.add_css(Path(__file__).with_name("workspace.css").read_text())
        with ui.element("header").classes("app-header"):
            with ui.row().classes("brand items-center"):
                with ui.element("div").classes("brand-mark"):
                    ui.icon("spa", size="21px")
                ui.label("kiwi").classes("brand-name")
                ui.label("workspace").classes("brand-caption")
            with (
                ui.tabs()
                .props("dense inline-label no-caps")
                .classes("main-tabs") as self.tabs
            ):
                self.library_tab = ui.tab("library", label="Overview", icon="grid_view")
                self.viz_tab = ui.tab(
                    "visualization", label="Visualization", icon="view_in_ar"
                )
                self.import_tab = ui.tab("import", label="Import", icon="file_upload")
                self.jobs_tab = ui.tab("jobs", label="Jobs", icon="task_alt")
            with ui.row().classes("local-status items-center gap-2"):
                ui.element("span").classes("status-dot")
                ui.label("Local workspace")
        with (
            ui.splitter(value=300, limits=(240, 460))
            .props("unit=px")
            .classes("workspace-split") as self.split
        ):
            with self.split.before:
                self.build_data_manager()
            with self.split.after:
                self.tabs.set_value(self.library_tab)
                # Keep the iframe mounted across tab switches.
                with ui.element("main").classes("workspace-main"):
                    for tab, page_class, build in (
                        (self.library_tab, "content-page", self.build_library),
                        (self.viz_tab, "visualization-page", self.build_visualization),
                        (self.import_tab, "content-page", self.build_import),
                        (self.jobs_tab, "content-page", self.build_jobs),
                    ):
                        with (
                            ui.element("section")
                            .classes(page_class)
                            .bind_visibility_from(
                                self.tabs,
                                "value",
                                backward=lambda value, target=tab: (
                                    value in (target, target.props["name"])
                                ),
                            )
                        ) as page:
                            page.props["role"] = "tabpanel"
                            page.props["aria-label"] = tab.props["label"]
                            build()
        with ui.element("footer").classes("app-footer"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("folder_open", size="13px")
                self.root_status = ui.label(str(root)).classes("footer-path")
            ui.space()
            self.job_status = ui.label("No active jobs")
        ui.timer(1.0, self.refresh_jobs)
        ui.timer(0.1, self.reload_library, once=True)

    @staticmethod
    def page_heading(eyebrow, title, description):
        ui.label(eyebrow).classes("eyebrow")
        ui.label(title).classes("page-title")
        return ui.label(description).classes("page-description")

    def build_jobs(self):
        self.page_heading(
            "Background activity",
            "Jobs",
            "Imports and visualization preparation keep running while you browse.",
        )
        self.job_cards()

    def build_data_manager(self):
        with (
            ui.element("aside")
            .classes("data-manager")
            .props('aria-label="Data manager"')
        ):
            with ui.row().classes("sidebar-heading"):
                ui.label("Data manager").classes("section-title")
                ui.space()
                ui.button(icon="refresh", on_click=self.reload_library).props(
                    'flat round dense aria-label="Refresh library"'
                ).classes("icon-button").tooltip("Refresh library")
            with (
                ui.button(on_click=self.root_dialog)
                .props('flat no-caps align=left aria-label="Change data root"')
                .classes("root-button")
            ):
                ui.icon("folder_open", size="19px").classes("muted")
                with ui.column().classes("gap-0 min-w-0"):
                    self.root_name = ui.label(self.root.name).classes("root-name")
                    ui.label("Data root").classes("micro muted")
                ui.space()
                ui.icon("unfold_more", size="17px").classes("muted")
            self.search = (
                ui.input(placeholder="Search data…", on_change=self.filter_data)
                .props('outlined dense clearable aria-label="Search data"')
                .classes("sidebar-search")
            )
            with self.search.add_slot("prepend"):
                ui.icon("search", size="17px")
            with (
                ui.tabs()
                .props("dense no-caps")
                .classes("browser-tabs") as self.browser_tabs
            ):
                episodes_tab = ui.tab("episodes", label="Episodes")
                recordings_tab = ui.tab("recordings", label="Recordings")
            with ui.tab_panels(
                self.browser_tabs, value=episodes_tab, animated=False
            ).classes("browser-panels"):
                with ui.tab_panel(episodes_tab).classes("browser-content"):
                    self.tree = (
                        ui.tree([], on_select=self.select_tree)
                        .props("dense no-connectors selected-color=primary")
                        .classes("episode-tree")
                    )
                    self.tree.add_slot(
                        "default-header",
                        """
                        <div class="tree-row" :data-node-id="props.node.id">
                            <q-icon :name="props.node.icon" size="17px" class="tree-icon" />
                            <span class="tree-label">{{ props.node.label }}</span>
                            <span v-if="props.node.count" class="tree-count">{{ props.node.count }}</span>
                            <q-icon v-if="props.node.warning" name="error_outline" size="14px" class="tree-warning">
                                <q-tooltip>{{ props.node.warning }}</q-tooltip>
                            </q-icon>
                        </div>
                    """,
                    )
                    self.tree.props[":filter-method"] = (
                        '(node, filter) => (node.id + " " + node.label)'
                        ".toLowerCase().includes(filter.toLowerCase())"
                    )
                    self.empty_library = ui.label(
                        "No episodes yet. Import recordings to get started."
                    ).classes("empty-sidebar")
                with ui.tab_panel(recordings_tab).classes("browser-content"):
                    self.recording_list()
            self.details()
            with ui.row().classes("sidebar-footer"):
                ui.icon("storage", size="14px")
                self.library_count = ui.label("Reading library…")
                ui.space()
                ui.button(
                    icon="add", on_click=lambda: self.tabs.set_value(self.import_tab)
                ).props('flat round dense aria-label="Import recordings"').classes(
                    "icon-button"
                ).tooltip("Import recordings")

    def build_library(self):
        with ui.row().classes("page-heading-row"):
            with ui.column().classes("gap-0"):
                self.library_description = self.page_heading(
                    "Your workspace",
                    "Data library",
                    "Scenes, episodes and the results you have collected.",
                )
            ui.space()
            ui.button(
                "Import recordings",
                icon="add",
                on_click=lambda: self.tabs.set_value(self.import_tab),
            ).props("unelevated no-caps").classes("primary-button")
        with ui.row().classes("library-toolbar"):
            self.library_header = ui.label("Episodes").classes("section-title")
            self.episode_count = ui.badge("0", color=None).classes("count-badge")
            ui.space()
            ui.label("Select an episode to inspect or visualize").classes(
                "muted text-xs"
            )
        self.table = (
            ui.table(
                columns=[
                    {
                        "name": key,
                        "field": key,
                        "label": label,
                        "align": "left",
                        "sortable": True,
                    }
                    for key, label in (
                        ("id", "Episode"),
                        ("cameras", "Cameras"),
                        ("available", "Results"),
                        ("timing", "Synchronization"),
                        ("status", "Setup"),
                    )
                ],
                rows=[],
                row_key="id",
                selection="single",
                on_select=self.select_episode,
                pagination=12,
            )
            .classes("library-table w-full")
            .props("flat")
        )
        self.table.add_slot(
            "body-cell-id",
            """
            <q-td :props="props">
                <div class="table-episode"><q-icon name="movie" size="17px" />{{ props.value }}</div>
            </q-td>
        """,
        )
        self.table.add_slot(
            "body-cell-timing",
            """
            <q-td :props="props">
                <span :class="['status-pill', /^(Missing|Preview)/.test(props.value) ? 'status-warning' : 'status-neutral']">
                    <span class="pill-dot"></span>{{ props.value }}
                </span>
            </q-td>
        """,
        )
        self.table.add_slot(
            "body-cell-status",
            """
            <q-td :props="props"><span class="setup-value">{{ props.value }}</span></q-td>
        """,
        )
        with ui.element("div").classes("library-bottom"):
            with ui.row().classes("items-center gap-2 mb-3"):
                ui.icon("play_circle_outline", size="18px").classes("muted")
                ui.label("Saved recordings").classes("section-title")
            ui.label(
                "Open a visualization already generated by KIWI or your scene viewer."
            ).classes("muted text-sm mb-4")
            with ui.row().classes("w-full items-center gap-3"):
                self.recording_select = (
                    ui.select({}, label="Recording", with_input=True)
                    .props("outlined dense")
                    .classes("grow")
                )
                ui.button(
                    "Open recording", icon="play_arrow", on_click=self.open_saved
                ).props("outline no-caps").classes("secondary-button")

    def build_visualization(self):
        with ui.row().classes("viz-toolbar"):
            ui.icon("view_in_ar", size="18px").classes("muted")
            self.viz_title = ui.label("Visualization").classes("viz-title")
            ui.space()
            ui.label("Rerun").classes("engine-label")
            self.download = ui.link("Download .rrd", "#").classes("download-link")
            self.download.set_visibility(False)
        with ui.element("div").classes("viz-stage"):
            with ui.column().classes("viz-empty") as self.viz_empty:
                with ui.element("div").classes("empty-visual"):
                    ui.icon("view_in_ar", size="45px")
                self.viz_empty_title = ui.label("See your data in context").classes(
                    "empty-title"
                )
                self.viz_empty_note = ui.label(
                    "Choose an episode in the data manager, then click Visualize. Or open a saved recording."
                ).classes("empty-description")
                with ui.row().classes("items-center gap-5 empty-features"):
                    for icon, label in (
                        ("videocam", "Camera views"),
                        ("route", "3D trajectories"),
                        ("schedule", "One timeline"),
                    ):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon(icon, size="16px")
                            ui.label(label)
            self.frame = (
                ui.element("iframe")
                .props('title="Rerun visualization" allow="fullscreen"')
                .classes("rerun-frame")
            )
            self.frame.set_visibility(False)
            self.frame.on("load", self.viewer_loaded)
        with ui.element("div").classes("layer-controls") as self.layer_panel:
            self.build_layer_controls()
        self.layer_panel.set_visibility(False)
        with ui.row().classes("viz-statusbar"):
            ui.icon("info_outline", size="14px")
            self.viz_note = ui.label(
                "Your data manager stays available while you visualize."
            )

    def build_layer_controls(self):
        self.layer_checkboxes = {}
        with ui.expansion("Data layers", icon="layers", value=True).classes(
            "layers-expansion w-full"
        ):
            if not self.layer_manifest:
                with ui.row().classes("items-center justify-between w-full gap-3 p-3"):
                    ui.label("This saved recording uses its own layer layout.").classes(
                        "muted text-xs"
                    )
                    ui.button(
                        "Open Rerun layer tree", on_click=self.show_rerun_layers
                    ).props("flat dense no-caps")
                return
            tracks = self.layer_manifest["tracks"]
            for layer in self.layer_manifest.get("spatial_layers", []):
                path = layer["path"]
                with ui.row().classes("items-center gap-3 px-3 py-1"):
                    checkbox = ui.checkbox(
                        layer["label"],
                        value=self.layer_choices[path],
                        on_change=lambda e, path=path: self.change_layer(path, e.value),
                    ).props('dense aria-label="Gaussian scene"')
                    ui.label(f"{layer['count']:,} Gaussians").classes("muted text-xs")
                    self.layer_checkboxes[path] = checkbox
                    checkbox.disable()
            if not tracks:
                ui.label(
                    "Video preview. Generate trajectories with the pipeline "
                    "to enable pose and trail layers."
                ).classes("muted text-xs p-3")
                return
            with ui.element("div").classes("layer-grid"):
                with ui.element("div").classes("layer-row layer-heading"):
                    ui.label("Track")
                    ui.label("Pose")
                    ui.label("Trail · 3s")
                    ui.label("Full trajectory")
                for track in tracks:
                    if track["default"]:
                        self.layer_row(track)
                diagnostics = [track for track in tracks if not track["default"]]
                if diagnostics:
                    with ui.expansion("Localization diagnostics", icon="tune").classes(
                        "diagnostic-layers w-full"
                    ):
                        for track in diagnostics:
                            self.layer_row(track)

    def layer_row(self, track):
        with ui.element("div").classes("layer-row"):
            with ui.row().classes("items-center gap-2 min-w-0"):
                color = ",".join(str(int(c)) for c in track["color"])
                ui.element("span").classes("layer-color").style(
                    f"background: rgb({color})"
                )
                ui.label(track["label"]).classes("layer-name")
            for kind in LAYER_KINDS:
                path = track["paths"].get(kind)
                if path is None:
                    ui.label("-").classes("muted text-center").tooltip("Not available")
                    continue
                checkbox = (
                    ui.checkbox(
                        value=self.layer_choices[path],
                        on_change=lambda e, path=path: self.change_layer(path, e.value),
                    )
                    .props(f'dense aria-label="{track["label"]} {kind}"')
                    .classes("layer-checkbox")
                )
                self.layer_checkboxes[path] = checkbox
                checkbox.disable()

    async def viewer_ready(self, serial: int) -> bool:
        """Wait out the embedded viewer's load, however large the recording.

        Poll the viewer's own state, and give up only when it reports an error
        or the page has moved on to another recording.
        """
        while serial == self.open_serial and self.client.has_socket_connection:
            try:
                state = await self.client.run_javascript(
                    f'return document.getElementById("c{self.frame.id}")'
                    '.contentWindow.kiwiViewer?.state ?? "loading"',
                    timeout=10,
                )
            except TimeoutError:
                continue
            except Exception:
                return False
            if state != "loading":
                return state == "ready"
            await asyncio.sleep(0.5)
        return False

    async def viewer_loaded(self):
        serial = self.open_serial
        if not await self.viewer_ready(serial):
            if serial == self.open_serial and self.client.has_socket_connection:
                self.viz_note.set_text(
                    "Rerun did not finish opening. Open the recording again to retry."
                )
            return
        if serial == self.open_serial and self.client.has_socket_connection:
            if self.layer_manifest and self.recording_path:
                try:
                    data = await run.io_bound(
                        layer_update_bytes,
                        self.layer_manifest,
                        self.layer_choices,
                        self.recording_path,
                        initialize=True,
                    )
                    if serial != self.open_serial:
                        return
                    await self.send_layer_update(data)
                except Exception as exc:
                    self.viz_note.set_text(f"Could not initialize data layers: {exc}")
                    return
            for checkbox in self.layer_checkboxes.values():
                checkbox.enable()

    async def send_layer_update(self, data):
        encoded = base64.b64encode(data).decode("ascii")
        applied = await self.client.run_javascript(
            f"""const viewer = document.getElementById("c{self.frame.id}").contentWindow.kiwiViewer;
            if (viewer?.source !== {json.dumps(self.recording_url)}) return false;
            return await viewer.applyLayers({json.dumps(encoded)});""",
            timeout=30,
        )
        if not applied:
            raise RuntimeError("The active recording changed.")

    async def show_rerun_layers(self):
        await self.client.run_javascript(
            f'return await document.getElementById("c{self.frame.id}").contentWindow.kiwiViewer.showBlueprintPanel()',
            timeout=90,
        )

    async def change_layer(self, path, visible):
        if (
            self.layer_syncing
            or not self.layer_manifest
            or not self.client.has_socket_connection
        ):
            return
        self.layer_choices[path] = visible
        self.layer_revision += 1
        revision, serial = self.layer_revision, self.open_serial
        async with self.layer_lock:
            if revision != self.layer_revision or serial != self.open_serial:
                return
            choices = dict(self.layer_choices)
            try:
                changed = {
                    path: shown
                    for path, shown in choices.items()
                    if shown != self.layer_applied.get(path)
                }
                data = await run.io_bound(
                    layer_update_bytes,
                    self.layer_manifest,
                    changed,
                    self.recording_path,
                )
                if revision != self.layer_revision or serial != self.open_serial:
                    return
                await self.send_layer_update(data)
                if serial == self.open_serial:
                    self.layer_applied = choices
            except Exception as exc:
                if (
                    serial == self.open_serial
                    and revision == self.layer_revision
                    and self.client.has_socket_connection
                ):
                    self.layer_syncing = True
                    try:
                        self.layer_choices = dict(self.layer_applied)
                        for key, checkbox in self.layer_checkboxes.items():
                            checkbox.set_value(self.layer_choices[key])
                    finally:
                        self.layer_syncing = False
                    with self.client:
                        ui.notify(f"Could not change layers: {exc}", type="negative")

    def build_import(self):
        self.page_heading(
            "Add to your workspace",
            "Import recordings",
            "Bring recordings from a mounted camera or a folder on this computer.",
        )
        mounts = find_mounts()
        with ui.card().classes("form-section"):
            with ui.row().classes("section-heading"):
                ui.label("1").classes("step-number")
                ui.label("Choose your source").classes("section-title")
            with ui.row().classes("w-full items-center gap-3"):
                self.source_input = (
                    ui.input(
                        "Camera or source folder",
                        value=str(mounts[0]) if mounts else "",
                        placeholder="/media/… or a recordings folder",
                    )
                    .props("outlined dense")
                    .classes("grow")
                )
                ui.button("Find cameras", icon="usb", on_click=self.find_cameras).props(
                    "flat no-caps"
                ).classes("secondary-button")
                ui.button("Scan clips", icon="search", on_click=self.scan_clips).props(
                    "outline no-caps"
                ).classes("secondary-button")
        with ui.element("div").classes("import-media"):
            self.clip_table = (
                ui.table(
                    columns=[
                        {"name": k, "field": k, "label": label, "align": "left"}
                        for k, label in (("name", "Recording"), ("size", "Size"))
                    ],
                    rows=[],
                    row_key="id",
                    selection="multiple",
                    on_select=self.preview_clip,
                    pagination=8,
                )
                .classes("clip-table")
                .props("flat")
            )
            with ui.card().classes("preview-card"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("play_circle_outline", size="17px").classes("muted")
                    ui.label("Clip preview").classes("section-title")
                self.preview = ui.video("").props("controls").classes("clip-preview")
                self.preview_label = ui.label(
                    "Select a recording to preview it."
                ).classes("preview-caption")
        with ui.card().classes("form-section"):
            with ui.row().classes("section-heading"):
                ui.label("2").classes("step-number")
                ui.label("Add to an episode").classes("section-title")
            with ui.row().classes("w-full items-center gap-3"):
                self.destination = (
                    ui.input("Destination inside data root", value="new_episode/raw")
                    .props("outlined dense")
                    .classes("grow")
                )
                self.role = (
                    ui.select(
                        {
                            "": "Camera identity",
                            "left": "Left",
                            "right": "Right",
                            "ego_video": "Ego",
                        },
                        value="",
                        label="Filename label",
                    )
                    .props("outlined dense")
                    .classes("w-44")
                )
            with ui.row().classes("w-full items-center gap-4 mt-2"):
                ui.label(
                    "Use the same episode/raw folder for each camera. Source recordings are preserved."
                ).classes("muted text-xs grow")
                ui.button(
                    "Import selected", icon="file_upload", on_click=self.start_import
                ).props("unelevated no-caps").classes("primary-button")

    def root_dialog(self):
        with ui.dialog() as dialog, ui.card().classes("root-dialog"):
            ui.label("Open a data folder").classes("text-lg font-semibold")
            ui.label("Choose the folder containing your scenes and episodes.").classes(
                "muted text-sm"
            )
            field = (
                ui.input("Data root", value=str(self.root))
                .props("outlined dense")
                .classes("w-full")
            )

            async def load():
                if not (field.value or "").strip():
                    ui.notify("Enter a data folder.", type="negative")
                    return
                if await self.reload_library(Path(field.value).expanduser().resolve()):
                    dialog.close()

            with ui.row().classes("w-full justify-end gap-2 mt-3"):
                ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
                ui.button("Open folder", on_click=load).props("unelevated no-caps")
        dialog.open()

    async def reload_library(self, root: Path | None = None):
        # UI click events must never be mistaken for a filesystem path.
        root = root if isinstance(root, Path) else self.root
        if not root.is_dir():
            ui.notify("That data root is not a directory.", type="negative")
            return False
        previous = self.selected["id"] if self.selected and root == self.root else None
        try:
            self.episodes, self.recordings = await run.io_bound(scan_library, root)
        except Exception as exc:
            ui.notify(str(exc), type="negative")
            return False
        self.root = root
        self.root_name.set_text(root.name)
        self.root_status.set_text(str(root))
        self.episode_count.set_text(str(len(self.episodes)))
        scenes = {}
        for item in self.episodes:
            parent = str(Path(item["id"]).parent)
            node = {
                "id": "episode:" + item["id"],
                "label": Path(item["id"]).name if item["id"] != "." else root.name,
                "icon": "movie",
                "warning": item["timing"]
                if item["timing"].startswith(("Missing", "Preview"))
                else "",
            }
            if parent not in scenes:
                scenes[parent] = {
                    "id": "folder:" + parent,
                    "label": root.name if parent == "." else parent,
                    "icon": "folder_open",
                    "selectable": False,
                    "children": [],
                }
            scenes[parent]["children"].append(node)
        for group in scenes.values():
            group["count"] = len(group["children"])
        self.tree.props["nodes"] = list(scenes.values())
        self.tree.expand([group["id"] for group in scenes.values()])
        self.tree.update()
        self.empty_library.set_visibility(not self.episodes)
        self.library_count.set_text(
            f"{len(scenes)} folders · {len(self.episodes)} episodes"
        )
        self.library_description.set_text(
            f"{len(self.episodes)} episodes across {len(scenes)} folders. Browse your recordings and pipeline results."
        )
        self.table.rows = self.episodes
        self.recording_select.set_options(
            {str(p): str(p.relative_to(root)) for p in self.recordings},
            value=str(self.recordings[0]) if self.recordings else None,
        )
        self.choose_episode(previous)
        self.recording_list.refresh()
        return True

    def filter_data(self):
        query = self.search.value or ""
        self.tree.filter = query
        self.table.filter = query
        self.recording_list.refresh()

    def select_tree(self, event):
        if event.value and event.value.startswith("episode:"):
            self.choose_episode(event.value.removeprefix("episode:"))

    def select_episode(self, event):
        self.choose_episode(event.selection[0]["id"] if event.selection else None)

    def choose_episode(self, episode_id):
        self.selected = next(
            (item for item in self.episodes if item["id"] == episode_id), None
        )
        self.table.selected = [self.selected] if self.selected else []
        self.table.update()
        self.tree.select("episode:" + episode_id if self.selected else None).update()
        self.details.refresh()

    @ui.refreshable
    def recording_list(self):
        query = (self.search.value or "").casefold()
        matches = [
            p
            for p in self.recordings
            if query in str(p.relative_to(self.root)).casefold()
        ]
        if not matches:
            ui.label(
                "No matching recordings."
                if query
                else "Saved .rrd files will appear here."
            ).classes("empty-sidebar")
        for recording in matches:
            with (
                ui.button(on_click=lambda p=recording: self.open_recording(p))
                .props("flat no-caps align=left")
                .classes("recording-row")
            ):
                ui.icon("play_circle_outline", size="18px").classes("muted")
                with ui.column().classes("gap-0 min-w-0"):
                    ui.label(recording.name).classes("recording-name")
                    ui.label(str(recording.parent.relative_to(self.root))).classes(
                        "micro muted"
                    )
                ui.tooltip(str(recording.relative_to(self.root)))

    @ui.refreshable
    def details(self):
        with (
            ui.element("section")
            .classes("selection-details")
            .props('aria-label="Selected episode"')
        ):
            if not self.selected:
                ui.icon("touch_app", size="21px").classes("muted mb-2")
                ui.label("Select an episode").classes("section-title")
                ui.label("Inspect its data and open a visualization here.").classes(
                    "muted text-xs mt-1"
                )
                return
            item = self.selected
            episode = inside(self.root, item["id"]) if item["id"] != "." else self.root
            ui.label("Selected episode").classes("eyebrow")
            ui.label(item["id"]).classes("selected-name")
            ui.label(item["cameras"]).classes("muted text-xs mb-3")
            if item["available"] != "No derived results":
                with ui.row().classes("gap-1 mb-3"):
                    for artifact in item["available"].split(", "):
                        ui.label(artifact).classes("artifact-chip")
            elif item["video_preview"]:
                ui.label(
                    "Videos can be previewed now. Trajectories have not been generated."
                ).classes("muted text-xs mb-3")
            if item["timing"].startswith(("Missing", "Preview")):
                with ui.row().classes("sync-warning"):
                    ui.icon("error_outline", size="15px")
                    ui.label(item["timing"]).classes("text-xs")
                    ui.tooltip(
                        "Preview sync uses an existing camera offset_s.txt for provisional audio alignment. Streams with no explicit offset are omitted."
                    )
            if item["error"]:
                ui.label(item["error"]).classes("text-xs text-red-700 break-words mb-2")
            ui.button(
                "Visualize",
                icon="play_arrow",
                on_click=lambda: self.prepare_visualization(episode),
            ).props("unelevated no-caps").classes("primary-button w-full").set_enabled(
                item["status"] == "Initialized"
            )
            if not (episode / "episode.yaml").exists():
                ui.button(
                    "Initialize episode",
                    icon="settings",
                    on_click=lambda: self.initialize_dialog(episode),
                ).props("outline no-caps").classes("secondary-button w-full mt-2")
            ui.button(
                "Import into episode",
                icon="add",
                on_click=lambda: self.import_into(item["id"]),
            ).props("flat no-caps").classes("secondary-button w-full mt-1")
            ui.button(
                "Gaussian scene",
                icon="blur_on",
                on_click=lambda: self.gaussian_dialog(episode),
            ).props("flat no-caps").classes("secondary-button w-full")

    def gaussian_dialog(self, episode: Path):
        scene = attached_scene(episode, self.cache) or {}
        with ui.dialog() as dialog, ui.card().classes("root-dialog"):
            ui.label("Attach a Gaussian scene").classes("text-lg font-semibold")
            ui.label(
                "Choose the reconstruction and its training manifest. The scene must "
                "already share this episode's world coordinates."
            ).classes("muted text-sm")
            ply = (
                ui.input("Gaussian PLY", value=scene.get("ply", ""))
                .props("outlined dense")
                .classes("w-full")
            )
            manifest = (
                ui.input("Training manifest", value=scene.get("manifest", ""))
                .props("outlined dense")
                .classes("w-full")
            )

            async def load():
                button.disable()
                try:
                    if not ply.value or not manifest.value:
                        raise ValueError(
                            "Choose both the Gaussian PLY and training manifest."
                        )
                    await run.io_bound(
                        attach_scene,
                        episode,
                        self.cache,
                        Path(ply.value),
                        Path(manifest.value),
                    )
                    dialog.close()
                    self.prepare_visualization(episode)
                except Exception as exc:
                    ui.notify(str(exc), type="negative")
                finally:
                    button.enable()

            with ui.row().classes("w-full justify-end gap-2 mt-3"):
                ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
                button = ui.button("Attach and visualize", on_click=load).props(
                    "unelevated no-caps"
                )
        dialog.open()

    def import_into(self, episode_id):
        self.destination.value = str(Path(episode_id) / "raw")
        self.tabs.set_value(self.import_tab)

    async def find_cameras(self):
        mounts = await run.io_bound(find_mounts)
        if mounts:
            self.source_input.value = str(mounts[0])
            ui.notify(f"Found {len(mounts)} camera mount(s).")
        else:
            ui.notify(
                "No camera mount detected. Enter the card or recording folder above."
            )

    async def scan_clips(self):
        source = Path(self.source_input.value).expanduser()
        if not self.source_input.value or not source.is_dir():
            ui.notify("Choose an existing camera or source folder.", type="negative")
            return
        self.clips = await run.io_bound(newest_clips, source, 100)
        self.clip_table.selected = []
        self.clip_table.rows = [
            {"id": i, "name": c.path.name, "size": f"{c.size / 1e6:.1f} MB"}
            for i, c in enumerate(self.clips)
        ]
        self.clip_table.update()
        self.preview.set_source("")
        self.preview_label.set_text(
            f"{len(self.clips)} recordings found. Select clips to import."
        )

    def preview_clip(self, event):
        if event.selection:
            clip = self.clips[event.selection[-1]["id"]]
            source, proxy = preview_source(clip.path)
            self.preview.set_source(self.files.register(source))
            self.preview_label.set_text(
                f"{clip.path.name} · {'Camera proxy' if proxy else 'Original recording; browser playback depends on its codec'}"
            )

    def start_import(self):
        clips = [self.clips[r["id"]] for r in self.clip_table.selected]
        if not clips:
            ui.notify("Select at least one recording.")
            return
        try:
            destination = inside(self.root, self.destination.value)
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
            return
        tag = self.role.value or None
        self.jobs.submit(
            f"Import {len(clips)} recording(s)",
            lambda job: import_clips(
                clips, destination, config_root() / "cameras", tag, self.jobs, job
            ),
        )
        self.tabs.set_value(self.jobs_tab)

    def initialize_dialog(self, episode: Path):
        cameras = sorted(
            p.name
            for p in (config_root() / "cameras").iterdir()
            if (p / "camera.yaml").is_file()
        )
        tools = sorted(
            p.name
            for p in (config_root() / "gripper").iterdir()
            if (p / "tool.yaml").is_file()
        )
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("Initialize episode").classes("text-xl font-semibold")
            ui.label("Camera serials verify the arm assignment.").classes(
                "text-sm text-gray-500"
            )
            left = ui.select(cameras, label="Left camera").classes("w-full")
            right = ui.select(cameras, label="Right camera").classes("w-full")
            tool = ui.select(tools, label="Tool").classes("w-full")
            ego = ui.select(
                {"": "None", **{c: c for c in cameras}},
                value="",
                label="Ego camera (ego_video label)",
            ).classes("w-full")

            def initialize():
                if (
                    not left.value
                    or not right.value
                    or left.value == right.value
                    or not tool.value
                ):
                    ui.notify(
                        "Choose two different arm cameras and a tool.", type="negative"
                    )
                    return
                values = dict(
                    left=left.value,
                    right=right.value,
                    tool=tool.value,
                    extra_units={"ego_video": ego.value} if ego.value else None,
                )

                def operation(job):
                    output, problem = init_episode(episode, **values)
                    if problem:
                        raise ValueError(problem)
                    self.jobs.update(
                        job,
                        message="Episode initialized. Configure its physical gripper build before processing.",
                    )
                    return output

                self.jobs.submit(f"Initialize {episode.name}", operation)
                dialog.close()
                self.tabs.set_value(self.jobs_tab)

            ui.button("Create manifest", on_click=initialize)
        dialog.open()

    def prepare_visualization(self, episode: Path):
        title = f"{episode.parent.name}/{episode.name}"

        def operation(job):
            key = review_key(episode, self.cache)
            output = self.cache / "recordings" / f"{key}.rrd"
            if output.is_file() and manifest_path(output).is_file():
                self.jobs.update(job, message="Opened the cached visualization.")
                return output
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".rrd.part")
            log_path = self.jobs.directory / f"{job.id}.log"
            self.jobs.update(
                job,
                message="Preparing videos and visualization layers. This may take a few minutes.",
            )
            try:
                self.jobs.run_process(
                    [
                        sys.executable,
                        "-m",
                        "kiwi.app.worker",
                        "--episode",
                        str(episode),
                        "--output",
                        str(temporary),
                        "--cache",
                        str(self.cache),
                    ],
                    log_path,
                    on_progress=lambda fraction, message: self.jobs.update(
                        job, progress=fraction, message=message
                    ),
                )
                if review_key(episode, self.cache) != key:
                    raise RuntimeError(
                        "Artifacts changed during visualization preparation. Retry using the latest results."
                    )
                manifest_path(temporary).replace(manifest_path(output))
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
                manifest_path(temporary).unlink(missing_ok=True)
            self.jobs.update(job, message="Visualization ready.")
            return output

        job = self.jobs.submit(f"Visualize {title}", operation)
        self.open_serial += 1
        self.pending_viz = {"id": job.id, "title": title}
        self.tabs.set_value(self.viz_tab)
        self.viz_note.set_text(f"Preparing {title}… You can keep browsing your data.")
        self.viz_empty_title.set_text("Preparing your visualization")
        self.viz_empty_note.set_text(
            "Videos and trajectories are being assembled. This view opens automatically when ready."
        )

    async def open_saved(self):
        if self.recording_select.value:
            await self.open_recording(Path(self.recording_select.value))

    async def open_recording(self, recording: Path, title: str | None = None):
        self.pending_viz = None
        self.recording_path = recording
        self.open_serial += 1
        serial = self.open_serial
        self.tabs.set_value(self.viz_tab)
        self.viz_note.set_text("Opening visualization…")
        if not recording.is_file():
            self.viz_note.set_text(
                "This recording is no longer available. Refresh the data manager."
            )
            return
        try:
            await run.io_bound(self.viewer.start)
        except Exception as exc:
            self.viz_note.set_text(str(exc))
            return
        if serial != self.open_serial:
            return
        self.viz_title.set_text(title or recording.stem)
        url = self.files.register(recording)
        self.recording_url = self.base_url + url
        self.layer_manifest = None
        metadata = manifest_path(recording)
        if metadata.is_file():
            try:
                self.layer_manifest = json.loads(metadata.read_text())
            except (OSError, ValueError):
                pass
        self.layer_choices = (
            default_visibility(self.layer_manifest) if self.layer_manifest else {}
        )
        self.layer_applied = dict(self.layer_choices)
        self.layer_revision += 1
        self.layer_panel.clear()
        with self.layer_panel:
            self.build_layer_controls()
        self.layer_panel.set_visibility(True)
        self.frame.props(f'src="/viewer/?{urlencode({"url": self.recording_url})}"')
        self.frame.set_visibility(True)
        self.viz_empty.set_visibility(False)
        self.download.props(f'href="{url}" download')
        self.download.set_visibility(True)
        self.viz_note.set_text(
            "Use Rerun to play, scrub and explore. Browse or select another episode in the data manager."
        )

    async def refresh_jobs(self):
        latest = self.jobs.snapshot()
        if latest != self.last_jobs:
            previous = {j["id"]: j["status"] for j in self.last_jobs}
            self.last_jobs = latest
            self.job_cards.refresh()
            running = sum(j["status"] in ("running", "queued") for j in latest)
            self.job_status.set_text(
                f"{running} active job(s)" if running else "No active jobs"
            )
            if any(
                j["status"] == "completed"
                and previous.get(j["id"]) != "completed"
                and j["title"].startswith(("Import ", "Initialize "))
                for j in latest
            ):
                await self.reload_library()
        if self.pending_viz:
            pending = self.pending_viz
            job = next((j for j in latest if j["id"] == pending["id"]), None)
            if job and job["status"] in ("running", "queued"):
                # Show which step the running job is on.
                self.viz_empty_note.set_text(job["message"])
            if job and job["status"] not in ("running", "queued"):
                self.pending_viz = None
                if job["status"] == "completed":
                    if self.tabs.value in ("visualization", self.viz_tab):
                        await self.open_recording(Path(job["result"]), pending["title"])
                    else:
                        ui.notify(f"{pending['title']} is ready. Open it from Jobs.")
                else:
                    self.viz_note.set_text(
                        "Visualization could not be prepared. See Jobs for details."
                    )
                    self.viz_empty_title.set_text("This visualization needs attention")
                    self.viz_empty_note.set_text(
                        "Open Jobs for the details, then retry from the data manager."
                    )
                    ui.notify(
                        "Visualization failed. See Jobs for details.", type="negative"
                    )

    @ui.refreshable
    def job_cards(self):
        snapshots = self.jobs.snapshot()
        if not snapshots:
            ui.label(
                "No jobs yet. Import recordings or visualize an episode to get started."
            ).classes("muted text-sm mt-4")
        for job in snapshots[:30]:
            with ui.card().classes("job-card"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.icon(
                        "check" if job["status"] == "completed" else "schedule",
                        size="17px",
                    ).classes("job-icon")
                    ui.label(job["title"]).classes("font-semibold text-xs")
                    ui.space()
                    ui.badge(job["status"], color=None).classes(
                        "job-badge"
                        + (" job-failed" if job["status"] == "failed" else "")
                    )
                ui.label(job["message"]).classes("job-message")
                if job["status"] in ("running", "queued"):
                    ui.linear_progress(job["progress"]).classes("w-full").props(
                        "rounded"
                    )
                if job["status"] == "completed" and job["result"].endswith(".rrd"):
                    ui.button(
                        "Open visualization",
                        icon="play_arrow",
                        on_click=lambda p=Path(job["result"]), title=job["title"].removeprefix("Visualize "): (
                            self.open_recording(p, title)
                        ),
                    ).props("flat no-caps").classes("secondary-button")


def launch(root: Path, cache: Path, *, port: int, open_browser: bool):
    root, cache = root.expanduser().resolve(), cache.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    jobs, files, viewer = Jobs(cache), Files(), Viewer(cache)

    @app.get("/api/files/{token}")
    def media(token: str):
        path = files.paths.get(token)
        if path is None or not path.is_file():
            raise HTTPException(404)
        ctype = (
            "video/mp4"
            if path.suffix.lower() in (".mp4", ".lrv")
            else "application/octet-stream"
        )
        return FileResponse(
            path, media_type=ctype, headers={"Cache-Control": "no-cache"}
        )

    @app.get("/api/jobs")
    def job_status():
        return jobs.snapshot()

    @app.get("/viewer/{asset:path}")
    async def viewer_assets(asset: str, request: Request):
        if viewer.port is None:
            raise HTTPException(503, "Open a recording first.")
        if asset in ("", "index.html"):
            return FileResponse(
                Path(__file__).with_name("viewer.html"), media_type="text/html"
            )
        client = httpx.AsyncClient(timeout=60)
        try:
            upstream = await client.send(
                client.build_request(
                    "GET",
                    f"http://127.0.0.1:{viewer.port}/{asset}",
                    params=request.query_params,
                    headers={"Accept-Encoding": "identity"},
                ),
                stream=True,
            )
        except httpx.HTTPError as exc:
            await client.aclose()
            raise HTTPException(502, "Rerun viewer is unavailable.") from exc

        async def close():
            await upstream.aclose()
            await client.aclose()

        headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower()
            in ("content-type", "content-length", "cache-control", "content-encoding")
        }
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=headers,
            background=BackgroundTask(close),
        )

    @ui.page("/")
    def index():
        WorkspacePage(root, cache, jobs, files, viewer)

    app.on_shutdown(viewer.close)
    app.on_shutdown(jobs.close)
    ui.run(
        host="127.0.0.1",
        port=port,
        show=open_browser,
        reload=False,
        # Keep the GUI session through a delayed browser heartbeat.
        reconnect_timeout=60,
        title="KIWI · Data workspace",
        favicon="🥝",
    )

"""Spot-check a robot description in viser: FK, EE frames, and a TCP ruler.

Serves the URDF at its home pose with one slider per actuated joint, so you
can confirm a description's kinematics and that its declared end-effector
frames sit where the registry claims. Every EE in ``meta.yaml`` gets frame
axes, a name label, and a live ``[x, y, z]`` readout, including frames behind
a fixed joint (``camera_lens0``).

Needs ``viser<1.1``, which the ``replay`` extra pins.

The ``tune TCP`` folder is the measuring tool: it drives a second frame off
a nominal one with x/y/z/roll/pitch/yaw sliders and prints the offset in
URDF ``<origin>`` units, ready to paste into a xacro: nudge the tuned frame
onto the physical point, read off the correction.

MJCF sites are not URDF links and never appear here; open the ``.xml`` with
``python -m mujoco.viewer`` and enable Rendering -> Frame -> Site instead.

viser is optional (``pixi install -e replay``); this module imports without
it and :func:`launch` fails with the install hint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

from kiwi.core.viz.preview import hold_until_interrupt
from kiwi.core.descriptions import RobotDescription, load_robot_description

__all__ = ["launch"]

# Axis thickness that reads at FR3 scale without hiding the frame origin.
_DEFAULT_AXES_RADIUS = 0.002
# Per-joint frames are secondary to the EE overlay, so they draw thinner.
_JOINT_AXES_RADIUS_SCALE = 2.0 / 3.0
# Tuning offset in slider units: x/y/z metres, then roll/pitch/yaw degrees.
_ZERO_OFFSET = (0.0,) * 6
# EE names every registry entry declares as aliases of a more specific frame.
# Bimanual entries point default/tcp at the left arm, so the arm-qualified
# name (`left_tcp`) is the better label for the same link.
_GENERIC_EE_NAMES = frozenset({"default", "tcp"})


def _load_viser():
    try:
        import viser
        import viser.transforms as vtf
        from viser.extras import ViserUrdf
    except ImportError as e:
        raise RuntimeError(
            "the description viewer needs viser, an optional dependency: "
            "`pixi install -e replay` (or `pip install viser`)"
        ) from e
    return viser, vtf, ViserUrdf


def _ee_name_priority(ee_name: str) -> int:
    """Rank an EE name so dedupe keeps the most informative label.

    TCP frames outrank flange frames, and arm-qualified names outrank the
    generic aliases pointing at the same link.
    """
    tcp_score = 2 if ee_name == "tcp" or ee_name.endswith("_tcp") else 0
    return tcp_score + (0 if ee_name in _GENERIC_EE_NAMES else 1)


def _overlay_targets(desc: RobotDescription, urdf) -> list[tuple[str, str]]:
    """The ``(tag, link_name)`` frames to overlay, deduped by link.

    Reads the per-EE table so bimanual descriptions overlay both arms.
    Links the URDF doesn't declare are dropped (an entry may name an
    MJCF-only frame), and a link named by several EEs appears once, tagged
    with the highest-priority name.
    """
    candidates = [(name, ee.urdf_link) for name, ee in desc.end_effectors.items()]
    if not candidates:
        candidates = [("ee", desc.ee_link_name), ("tcp", desc.tcp_link_name)]
    tag_by_link: dict[str, str] = {}
    for tag, link_name in candidates:
        if link_name is None or link_name not in urdf.link_map:
            continue
        current = tag_by_link.get(link_name)
        if current is None or _ee_name_priority(tag) > _ee_name_priority(current):
            tag_by_link[link_name] = tag
    return [(tag, link_name) for link_name, tag in tag_by_link.items()]


def _merged_label_text(
    targets: list[tuple[str, str]], urdf, tolerance_m: float = 1e-4
) -> dict[str, str]:
    """Label text per tag, with frames that share a point merged into one.

    A description can declare several frames at the same physical point.
    Co-located frames get a single
    label naming all of them (``flange+default``), carried by the
    highest-priority tag; the others map to ``""`` and draw no label. Every
    frame still keeps its own axes and its own readout.
    """
    text: dict[str, str] = {tag: "" for tag, _ in targets}
    groups: list[tuple[np.ndarray, list[str]]] = []
    for tag, link_name in targets:
        origin = urdf.get_transform(link_name, urdf.base_link)[:3, 3]
        for position, members in groups:
            if float(np.linalg.norm(origin - position)) <= tolerance_m:
                members.append(tag)
                break
        else:
            groups.append((origin, [tag]))
    for _position, members in groups:
        carrier = max(members, key=_ee_name_priority)
        text[carrier] = "+".join(sorted(members, key=_ee_name_priority, reverse=True))
    return text


def _format_position(position: np.ndarray) -> str:
    """Render a 3-vector as ``[x, y, z]`` at 0.1 mm resolution."""
    return "[" + ", ".join(f"{v:.4f}" for v in position) + "]"


def _joint_limit(urdf, joint_name: str) -> tuple[float, float] | None:
    """Best-effort joint-limit lookup; callers fall back to (-pi, pi)."""
    for joint in getattr(getattr(urdf, "robot", None), "joints", []):
        if getattr(joint, "name", None) != joint_name:
            continue
        limit = getattr(joint, "limit", None)
        if limit is None:
            return None
        lower, upper = getattr(limit, "lower", None), getattr(limit, "upper", None)
        if lower is None or upper is None or lower >= upper:
            return None
        return float(lower), float(upper)
    return None


def _initial_cfg(desc: RobotDescription, urdf, snap_home: bool) -> np.ndarray:
    """Joint vector to open with: limit midpoints, then the home pose.

    Home values are matched to joints by NAME: the registry's ``home`` covers
    the arm joints only, while the URDF also actuates the mimic pair, and a
    bimanual URDF may interleave them between arms. Every entry is clamped to
    its limits (viser asserts ``min <= value <= max``).
    """
    cfg = np.zeros(len(urdf.actuated_joint_names), dtype=float)
    for i, joint_name in enumerate(urdf.actuated_joint_names):
        limit = _joint_limit(urdf, joint_name)
        if limit is not None:
            cfg[i] = 0.5 * (limit[0] + limit[1])
    if snap_home and desc.home_pose is not None:
        home_by_name = dict(zip(desc.joint_names, desc.home_pose))
        for i, joint_name in enumerate(urdf.actuated_joint_names):
            if joint_name in home_by_name:
                cfg[i] = home_by_name[joint_name]
    for i, joint_name in enumerate(urdf.actuated_joint_names):
        limit = _joint_limit(urdf, joint_name)
        if limit is not None:
            cfg[i] = float(np.clip(cfg[i], limit[0], limit[1]))
    return cfg


def launch(
    name: str,
    *,
    port: int = 8080,
    home: bool = True,
    frames: bool = True,
    dark: bool = False,
    root: Path | None = None,
    block: bool = True,
) -> None:
    """Serve description *name* at ``http://localhost:<port>`` until Ctrl-C.

    ``home`` opens at the registry's home pose (else limit midpoints);
    ``frames`` draws the EE overlay and the TCP ruler; ``dark`` switches to
    the dark canvas. ``root`` overrides the descriptions tree (tests).
    ``block=False`` returns once the server is up, for tests.

    Raises ``ValueError`` for an MJCF-only description and ``RuntimeError``
    with the install hint when viser is missing.
    """
    viser, vtf, ViserUrdf = _load_viser()

    desc = load_robot_description(name, root)
    if desc.urdf_path is None:
        raise ValueError(
            f"description {name!r} has no URDF (MJCF only); the viewer needs one"
        )

    print(f"Loading URDF: {desc.urdf_path}")
    urdf = desc.load_urdf()
    print(
        f"  {len(urdf.actuated_joint_names)} actuated joints; "
        f"TCP link: {desc.tcp_link_name!r}"
    )

    server = viser.ViserServer(port=port)
    server.gui.configure_theme(dark_mode=dark)
    server.scene.world_axes.visible = True
    viser_urdf = ViserUrdf(server, urdf)

    cfg = _initial_cfg(desc, urdf, home)
    viser_urdf.update_cfg(cfg)

    # Fit each new client to the posed model, including standalone grippers.
    bounds = np.asarray(urdf.scene.bounds)
    center = bounds.mean(axis=0)
    extent = max(float(np.max(bounds[1] - bounds[0])), 0.1)
    server.scene.world_axes.axes_length = extent * 0.12
    server.scene.world_axes.axes_radius = extent * 0.002

    @server.on_client_connect
    def _frame_model(client) -> None:
        client.camera.position = center + extent * np.array([1.3, -1.6, 0.9])
        client.camera.look_at = center

    # ViserUrdf.update_cfg() mutates the URDF object handed to it, so
    # urdf.get_transform() below always reflects what the sliders last set.
    ee_frames: list[tuple[str, object, object]] = []
    joint_frames: list[tuple[str, object]] = []
    labels: list[object] = []
    tuned_frame = None
    refresh_tuned: Callable[[], None] | None = None

    targets = _overlay_targets(desc, urdf) if frames else []
    if frames and not targets:
        rest = "showing joint frames only" if urdf.actuated_joints else "overlay off"
        print(f"  no EE links declared for {name}; {rest}")

    label_text = _merged_label_text(targets, urdf)

    if frames and (targets or urdf.actuated_joints):
        with server.gui.add_folder("frames"):
            for tag, link_name in targets:
                # origin_radius=0: no ball at the frame origin.
                frame = server.scene.add_frame(
                    f"/overlay/{tag}",
                    axes_length=0.08,
                    axes_radius=_DEFAULT_AXES_RADIUS,
                    origin_radius=0.0,
                )
                # A child of the frame, so the label inherits the pose.
                # Labelled with the short tag; the link name is on the
                # readout's hover hint. Frames sharing a point carry one
                # merged label; the rest get none.
                if label_text.get(tag):
                    labels.append(
                        server.scene.add_label(
                            f"/overlay/{tag}/label", text=label_text[tag]
                        )
                    )
                readout = server.gui.add_text(
                    label=tag, initial_value="", disabled=True, hint=link_name
                )
                ee_frames.append((link_name, frame, readout))

            for joint in urdf.actuated_joints:
                joint_frames.append(
                    (
                        joint.child,
                        server.scene.add_frame(
                            f"/overlay/joints/{joint.name}",
                            axes_length=0.04,
                            axes_radius=_DEFAULT_AXES_RADIUS * _JOINT_AXES_RADIUS_SCALE,
                            origin_radius=0.0,
                            visible=False,
                        ),
                    )
                )
            if joint_frames:
                joints_checkbox = server.gui.add_checkbox(
                    "show all joint frames", False
                )

                @joints_checkbox.on_update
                def _on_toggle_joint_frames(_handle: object) -> None:
                    for _link, handle in joint_frames:
                        handle.visible = joints_checkbox.value

            if labels:
                labels_checkbox = server.gui.add_checkbox(
                    "frame labels",
                    True,
                    hint="Viser draws each label on an opaque pill that "
                    "paints over the geometry behind it; untick to see what "
                    "the frame is pinning down. Axes and readouts stay.",
                )

                @labels_checkbox.on_update
                def _on_toggle_labels(_handle: object) -> None:
                    for handle in labels:
                        handle.visible = labels_checkbox.value

            # axes_radius is a live prop, so retuning thickness is one write
            # per handle. Readable thickness depends on robot scale and zoom.
            thickness = server.gui.add_slider(
                "axis thickness",
                min=0.0005,
                max=0.005,
                step=0.00025,
                initial_value=_DEFAULT_AXES_RADIUS,
            )

            @thickness.on_update
            def _on_thickness(_handle: object) -> None:
                for _link, handle, _readout in ee_frames:
                    handle.axes_radius = thickness.value
                for _link, handle in joint_frames:
                    handle.axes_radius = thickness.value * _JOINT_AXES_RADIUS_SCALE
                if tuned_frame is not None:
                    tuned_frame.axes_radius = thickness.value

            if targets:
                # The nominal frames show what the registry claims; these
                # sliders work out what it SHOULD say. Nudge the tuned frame
                # onto the physical point, then paste the offset readout into
                # that frame's <origin> in the xacro.
                link_by_tag = dict(targets)
                # Ties resolve to the first target, so this picks the TCP when
                # one is declared and the leading frame otherwise.
                active_tag = max(link_by_tag, key=_ee_name_priority)
                offsets: dict[str, tuple[float, ...]] = {}
                loading_offset = False

                tuned_frame = server.scene.add_frame(
                    "/overlay/tuned",
                    axes_length=0.10,
                    axes_radius=thickness.value,
                    origin_radius=0.0,
                )
                tuned_label = server.scene.add_label(
                    "/overlay/tuned/label", text=f"{active_tag} (tuned)"
                )
                labels.append(tuned_label)

                with server.gui.add_folder("tune TCP"):
                    frame_dropdown = server.gui.add_dropdown(
                        "frame", tuple(link_by_tag), initial_value=active_tag
                    )
                    tune_sliders = [
                        server.gui.add_slider(
                            axis, min=-0.15, max=0.15, step=0.001, initial_value=0.0
                        )
                        for axis in ("x", "y", "z")
                    ] + [
                        server.gui.add_slider(
                            axis, min=-180.0, max=180.0, step=0.5, initial_value=0.0
                        )
                        for axis in ("roll", "pitch", "yaw")
                    ]
                    tuned_readout = server.gui.add_text(
                        label="tuned pos", initial_value="", disabled=True
                    )
                    offset_readout = server.gui.add_text(
                        label="offset", initial_value="", disabled=True
                    )

                    def _apply_tuning() -> None:
                        """Place the tuned frame at nominal @ offset, local axes."""
                        x, y, z, roll, pitch, yaw = (s.value for s in tune_sliders)
                        rpy = np.deg2rad([roll, pitch, yaw])
                        offset = np.eye(4)
                        # Fixed-axis rpy, matching a URDF <origin>.
                        offset[:3, :3] = vtf.SO3.from_rpy_radians(*rpy).as_matrix()
                        offset[:3, 3] = (x, y, z)
                        nominal = urdf.get_transform(
                            link_by_tag[frame_dropdown.value], urdf.base_link
                        )
                        tuned = nominal @ offset
                        tuned_frame.wxyz = vtf.SO3.from_matrix(tuned[:3, :3]).wxyz
                        tuned_frame.position = tuned[:3, 3]
                        tuned_readout.value = _format_position(tuned[:3, 3])
                        # Radians even though the sliders are degrees: this
                        # line is meant to be pasted straight into the xacro.
                        offset_readout.value = (
                            f"xyz [m]: {_format_position(np.array([x, y, z]))}  "
                            f"rpy [rad]: {_format_position(rpy)}"
                        )

                    def _load_offset(tag: str) -> None:
                        """Push a stored offset into the sliders.

                        Programmatic writes fire on_update in viser, so the
                        guard collapses six recomputes into the caller's one.
                        """
                        nonlocal loading_offset
                        loading_offset = True
                        try:
                            for slider, value in zip(
                                tune_sliders, offsets.get(tag, _ZERO_OFFSET)
                            ):
                                slider.value = value
                        finally:
                            loading_offset = False

                    for tune_slider in tune_sliders:

                        @tune_slider.on_update
                        def _on_tune(_handle: object) -> None:
                            if not loading_offset:
                                _apply_tuning()

                    @frame_dropdown.on_update
                    def _on_switch_frame(_handle: object) -> None:
                        nonlocal active_tag
                        offsets[active_tag] = tuple(s.value for s in tune_sliders)
                        active_tag = frame_dropdown.value
                        _load_offset(active_tag)
                        tuned_label.text = f"{active_tag} (tuned)"
                        _apply_tuning()

                    reset_offset_btn = server.gui.add_button("reset offset")

                    @reset_offset_btn.on_click
                    def _on_reset_offset(_event: object) -> None:
                        offsets[frame_dropdown.value] = _ZERO_OFFSET
                        _load_offset(frame_dropdown.value)
                        _apply_tuning()

                    refresh_tuned = _apply_tuning

    def _refresh_frames() -> None:
        """Push the current FK into the overlay frames and readouts."""
        for link_name, handle, readout in ee_frames:
            transform = urdf.get_transform(link_name, urdf.base_link)
            handle.wxyz = vtf.SO3.from_matrix(transform[:3, :3]).wxyz
            handle.position = transform[:3, 3]
            readout.value = _format_position(transform[:3, 3])
        # Joint frames update while hidden too.
        for link_name, handle in joint_frames:
            transform = urdf.get_transform(link_name, urdf.base_link)
            handle.wxyz = vtf.SO3.from_matrix(transform[:3, :3]).wxyz
            handle.position = transform[:3, 3]
        if refresh_tuned is not None:
            refresh_tuned()

    _refresh_frames()

    sliders: list[object] = []
    with server.gui.add_folder(f"{name} joints"):
        for joint_name, init in zip(urdf.actuated_joint_names, cfg):
            limit = _joint_limit(urdf, joint_name)
            lo, hi = limit if limit else (-np.pi, np.pi)
            slider = server.gui.add_slider(
                joint_name,
                min=lo,
                max=hi,
                step=(hi - lo) / 200.0,
                initial_value=float(init),
            )
            sliders.append(slider)

            @slider.on_update
            def _on_update(_handle: object) -> None:
                viser_urdf.update_cfg(np.array([s.value for s in sliders], dtype=float))
                _refresh_frames()

    reset_btn = server.gui.add_button("reset to home" if home else "reset")

    # Assigning slider.value fires that slider's on_update (viser skips the
    # callback only when the value is unchanged), so the reset path refreshes
    # the overlay through _on_update and needs no hook of its own.
    @reset_btn.on_click
    def _on_reset(_event: object) -> None:
        for slider, value in zip(sliders, cfg):
            slider.value = float(value)

    print(f"\nServing {name!r} on http://localhost:{port}\n  Ctrl+C to stop.")
    if not block:
        return
    hold_until_interrupt(msg=None)  # the Serving line above already said how
    print("\nShutting down.", file=sys.stderr)

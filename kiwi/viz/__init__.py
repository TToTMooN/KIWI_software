"""Visualization over existing episode artifacts, without pipeline estimation.

``kiwi viz play`` logs video, per-arm scene trajectories, and raw plus
selected tool-state output on the episode's global clock. Rerun provides
playback, scrubbing and layout. Missing artifacts and unsupported viewer
layers are named in the summary. Marker-anchor trajectories, head-marker
poses, ground results and state diagnostics have artifacts, but this player
does not draw them. Extra camera streams contribute video only; head-marker
poses do not specify ego-camera poses.

``kiwi viz render`` writes a synchronized video with a tool-state strip;
``kiwi viz trace`` writes a tool-state review PNG. Interactive tools use
viser, including :mod:`.gripper_tune` and the apps in ``kiwi.replay``.

:mod:`.proxies` supplies H.264 renditions for video the player cannot decode.
Episode renditions live at ``<episode>/<label>/proxy.mp4``. Direct standalone
proxy calls may instead use ``<stem>_proxy.mp4`` beside their source.

This package imports ``kiwi.core`` and components, not ``kiwi.episode``.
The application resolves manifests into :mod:`.sources` paths, offsets and
a ``kiwi.core.clock.GlobalClock`` before calling the visualization code.
"""

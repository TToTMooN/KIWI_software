"""Map :class:`~kiwi.core.observation_episode.FeatureColumn` -> LeRobot ``meta/info.json``.

The single owner of the LeRobot feature-schema spelling, shared across writer
versions. Matched field-for-field against a real v2.1 dataset: tabular
features carry a top-level ``fps``; video features carry a ``video_info``
block instead; the bookkeeping columns (``timestamp`` + the four index
columns) and a synthesized ``action`` round out the schema.

The stable KIWI field NAMES live in :mod:`kiwi.export.schema`; this module is
only how those names are spelled into a LeRobot meta file.
"""

from __future__ import annotations

from kiwi.core.observation_episode import FeatureColumn

from ..schema import ACTION_KEY, STATE_KEY

__all__ = [
    "action_feature",
    "build_info_features",
    "feature_to_info",
    "video_info",
]

# Bookkeeping columns every LeRobot episode parquet carries, in canonical order.
_INDEX_COLUMNS = ("frame_index", "episode_index", "index", "task_index")


def video_info(fps: float, *, codec: str = "hevc", pix_fmt: str = "yuv420p") -> dict:
    """The ``video_info`` block LeRobot stores on a video feature."""
    return {
        "video.fps": fps,
        "video.codec": codec,
        "video.pix_fmt": pix_fmt,
        "video.is_depth_map": False,
        "has_audio": False,
    }


def feature_to_info(fc: FeatureColumn, fps: float, *, codec: str = "hevc") -> dict:
    """One :class:`FeatureColumn` -> its ``meta/info.json`` feature dict."""
    entry: dict = {"dtype": fc.dtype, "shape": list(fc.shape), "names": fc.names}
    if fc.is_video:
        entry["names"] = fc.names or ["height", "width", "channels"]
        entry["video_info"] = video_info(fps, codec=codec)
    else:
        entry["fps"] = fps
    return entry


def _scalar(dtype: str, fps: float) -> dict:
    return {"dtype": dtype, "shape": [1], "names": None, "fps": fps}


def action_feature(state: FeatureColumn) -> FeatureColumn:
    """The ``action`` feature derived from ``observation.state`` (same shape/names)."""
    return FeatureColumn(
        key=ACTION_KEY, dtype="float32", shape=tuple(state.shape), names=state.names
    )


def build_info_features(
    features: dict[str, FeatureColumn],
    *,
    fps: float,
    action: FeatureColumn | None,
    codec: str = "hevc",
) -> dict:
    """Assemble the ordered ``features`` dict for ``meta/info.json``.

    Order mirrors the reference dataset: ``observation.state`` first, then
    ``action``, then the bookkeeping columns, then everything else (extra
    tabular features, then video). ``action`` is passed in (writers synthesize
    it) so the datatype itself never carries one.
    """
    tabular = {k: f for k, f in features.items() if not f.is_video}
    video = {k: f for k, f in features.items() if f.is_video}

    out: dict = {}
    if STATE_KEY in tabular:
        out[STATE_KEY] = feature_to_info(tabular.pop(STATE_KEY), fps)
    if action is not None:
        out[ACTION_KEY] = feature_to_info(action, fps)
    out["timestamp"] = _scalar("float32", fps)
    for col in _INDEX_COLUMNS:
        out[col] = _scalar("int64", fps)
    for key, f in tabular.items():
        out[key] = feature_to_info(f, fps)
    for key, f in video.items():
        out[key] = feature_to_info(f, fps, codec=codec)
    return out

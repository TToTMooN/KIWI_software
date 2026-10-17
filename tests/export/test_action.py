"""Action synthesis - especially the ``delta`` policy's rotation blocks.

Differencing quaternion components does not give a rotation delta, and a
consumer integrating one drifts off SO(3). The ``delta`` policy instead
carries the body-frame relative rotation per quaternion block, located by the
state column names, and these tests pin the property that makes it correct:
composing the deltas recovers the absolutes exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kiwi.core.observation_episode import FeatureColumn
from kiwi.export.schema import arm_state_names
from kiwi.export.writers.base import DatasetInfo, quat_spans, synthesize_action


def _bimanual_state(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = np.zeros((n, 16), dtype=np.float32)
    for arm in (0, 8):
        state[:, arm : arm + 3] = rng.normal(size=(n, 3))
        state[:, arm + 3 : arm + 7] = Rotation.random(
            n, random_state=seed + arm
        ).as_quat()
        state[:, arm + 7] = rng.uniform(0, 1, size=n)
    return state


def test_quat_spans_from_schema_names():
    names = arm_state_names("left") + arm_state_names("right")
    assert quat_spans(names) == (3, 11)
    assert quat_spans(None) == ()
    assert quat_spans(["x", "y", "z"]) == ()


def test_delta_quat_blocks_compose_back_to_the_absolutes():
    """q_t * delta_t == q_{t+1} - the property no component-wise diff has."""
    state = _bimanual_state(20)
    spans = quat_spans(arm_state_names("left") + arm_state_names("right"))
    d = synthesize_action(state, "delta", quat_spans=spans)
    for s in spans:
        q = Rotation.from_quat(state[:, s : s + 4].astype(np.float64))
        rel = Rotation.from_quat(d[:-1, s : s + 4].astype(np.float64))
        recomposed = (q[:-1] * rel).as_quat()
        want = q[1:].as_quat()
        # Quaternion double cover: q and -q are the same rotation.
        sign = np.sign(np.sum(recomposed * want, axis=1))[:, None]
        np.testing.assert_allclose(recomposed * sign, want, atol=1e-6)
        # Trailing row is the identity rotation, not zeros.
        np.testing.assert_array_equal(d[-1, s : s + 4], [0, 0, 0, 1])


def test_delta_non_quat_columns_stay_first_difference():
    state = _bimanual_state(10)
    spans = quat_spans(arm_state_names("left") + arm_state_names("right"))
    d = synthesize_action(state, "delta", quat_spans=spans)
    for col in (0, 1, 2, 7, 8, 9, 10, 15):
        np.testing.assert_allclose(d[:-1, col], np.diff(state[:, col]), atol=1e-7)
        assert d[-1, col] == 0.0


def test_next_state_is_the_shifted_state_and_copy_is_the_state():
    state = _bimanual_state(6)
    np.testing.assert_array_equal(
        synthesize_action(state, "next_state"), np.vstack([state[1:], state[-1:]])
    )
    np.testing.assert_array_equal(synthesize_action(state, "copy"), state)


def test_delta_without_names_is_refused_at_dataset_info():
    """A silent fall-through would difference quaternion components."""
    state = FeatureColumn("observation.state", "float32", (16,), None)
    with pytest.raises(ValueError, match="named observation.state columns"):
        DatasetInfo(
            repo_id="me/x",
            robot_type="r",
            fps=30.0,
            features={"observation.state": state},
            action_policy="delta",
        )


def test_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="not in"):
        DatasetInfo(
            repo_id="me/x", robot_type="r", fps=30.0, features={}, action_policy="lerp"
        )
    with pytest.raises(ValueError, match="unknown action policy"):
        synthesize_action(np.zeros((3, 4), np.float32), "lerp")

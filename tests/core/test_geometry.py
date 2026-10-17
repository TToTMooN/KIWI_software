"""Tests for kiwi.core.geometry - pure PnP-output math, verified against cv2.

Run: pixi run test
"""

from __future__ import annotations

import numpy as np
import pytest

# Guarded so the pip-only lane still collects when opencv wheels are absent;
# in CI the headless wheel is installed via pyproject dependencies.
cv2 = pytest.importorskip("cv2")

from kiwi.core.geometry import rvec_tvec_to_T  # noqa: E402


def test_rvec_tvec_to_T_matches_rodrigues():
    rvec = np.array([0.3, -0.2, 0.9])
    tvec = np.array([[0.05], [-0.02], [0.4]])  # cv2's (3, 1) column shape
    T = rvec_tvec_to_T(rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [0.05, -0.02, 0.4], atol=0)
    np.testing.assert_array_equal(T[3], [0.0, 0.0, 0.0, 1.0])


def test_rvec_tvec_to_T_roundtrips_a_point():
    # A point mapped by the matrix must agree with cv2's own projection chain:
    # p_cam = R @ p + t, homogeneously.
    rng = np.random.default_rng(3)
    rvec = rng.uniform(-1, 1, 3)
    tvec = rng.uniform(-0.5, 0.5, 3)
    p = rng.uniform(-1, 1, 3)
    T = rvec_tvec_to_T(rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    np.testing.assert_allclose((T @ [*p, 1.0])[:3], R @ p + tvec, atol=1e-12)

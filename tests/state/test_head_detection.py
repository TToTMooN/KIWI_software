"""Head-marker stage: marker pose from an image, composition with the rear pose, files."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.state import geometry, head
from kiwi.state.geometry import se3, se3_inv


def _marker_pose(z=0.8):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [180 + 20, 10, 5], degrees=True).as_matrix()
    T[:3, 3] = [0.05, -0.03, z]
    return T


def _render_marker(K, dist, T_cam_marker, size_m, marker_id, image_size):
    """A white image with the marker drawn where the KB4 camera sees its four corners."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    tile = cv2.aruco.generateImageMarker(dictionary, marker_id, 200)
    canvas = np.full((image_size, image_size), 255, np.uint8)
    rvec = Rotation.from_matrix(T_cam_marker[:3, :3]).as_rotvec().reshape(3, 1)
    uv, _ = cv2.fisheye.projectPoints(
        head.marker_object_points(size_m).reshape(-1, 1, 3),
        rvec,
        np.ascontiguousarray(T_cam_marker[:3, 3]).reshape(3, 1),
        K,
        np.asarray(dist, dtype=float),
    )
    src = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], np.float32)
    H = cv2.getPerspectiveTransform(src, uv.reshape(4, 2).astype(np.float32))
    warped = cv2.warpPerspective(tile, H, (image_size, image_size), borderValue=255)
    mask = cv2.warpPerspective(
        np.full((200, 200), 255, np.uint8), H, (image_size, image_size)
    )
    canvas[mask > 0] = warped[mask > 0]
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    return cv2.aruco.ArucoDetector(dictionary, params)


def test_detect_in_image_recovers_pose():
    K = np.array([[500.0, 0, 480.0], [0, 500.0, 480.0], [0, 0, 1]])
    dist = np.zeros(4)
    T = _marker_pose()
    image = _render_marker(K, dist, T, 0.049, 50, 960)
    T_est = head.detect_in_image(image, _detector(), 50, 0.049, K, dist)
    assert T_est is not None
    assert np.linalg.norm(T_est[:3, 3] - T[:3, 3]) < 0.01
    assert (
        head.detect_in_image(image, _detector(), 7, 0.049, K, dist) is None
    )  # wrong id


def test_compose_places_the_marker_through_the_rear_lens_of_the_front_track():
    t_front = np.array([0.0, 0.1, 0.2, 1.0])
    T_front = np.tile(np.eye(4), (4, 1, 1))
    T_front[:, 0, 3] = [0.0, 0.1, 0.2, 1.0]
    T_rear_front = se3(
        Rotation.from_euler("y", 180, degrees=True).as_matrix(), [0.0, 0.0, -0.03]
    )
    T_rear_marker = se3(np.eye(3), [0.0, 0.0, 0.5])
    t_det = np.array([-0.05, 0.05, 0.5, 1.5])
    t, T = head.compose_scene_marker(
        t_det, np.tile(T_rear_marker, (4, 1, 1)), t_front, T_front, T_rear_front
    )
    np.testing.assert_allclose(
        t, [0.05, 0.5]
    )  # detections outside the front track drop
    for row, x in zip(T, [0.05, 0.5], strict=True):
        T_scene_front = se3(np.eye(3), [x, 0.0, 0.0])
        expected = T_scene_front @ se3_inv(T_rear_front) @ T_rear_marker
        np.testing.assert_allclose(row, expected, atol=1e-12)


def _track(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.005, 0.02, n))
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(
        np.cumsum(rng.normal(0, 0.05, (n, 3)), axis=0)
    ).as_matrix()
    T[:, :3, 3] = np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0)
    return t, T


def test_interpolate_poses_at_the_samples_and_halfway_between_them():
    t_track, T_track = _track(200, seed=1)
    assert (
        np.abs(geometry.interpolate_poses(t_track, T_track, t_track) - T_track).max()
        < 1e-12
    )
    mid = 0.5 * (t_track[:-1] + t_track[1:])
    T = geometry.interpolate_poses(t_track, T_track, mid)
    assert (
        np.abs(T[:, :3, 3] - 0.5 * (T_track[:-1, :3, 3] + T_track[1:, :3, 3])).max()
        < 1e-12
    )
    R_lo = Rotation.from_matrix(T_track[:-1, :3, :3])
    R_hi = Rotation.from_matrix(T_track[1:, :3, :3])
    half = R_lo * Rotation.from_rotvec(0.5 * (R_lo.inv() * R_hi).as_rotvec())
    assert np.abs(T[:, :3, :3] - half.as_matrix()).max() < 1e-12
    assert geometry.interpolate_poses(t_track, T_track, np.zeros(0)).shape == (0, 4, 4)

"""Keypoint masks of the front-map builds: the vignette ring and the written files."""

import cv2
import numpy as np

from kiwi.track import masks

KB4 = np.array([300.0, 300.0, 480.0, 480.0, 0.0, 0.0, 0.0, 0.0])


def test_ring_mask_marks_the_outside_of_the_circle():
    ring = masks.ring_mask(50.0, 50.0, 40.0, (100, 100))
    assert ring[50, 50] == 0 and ring[50, 12] == 0
    assert ring[0, 0] == 255 and ring[50, 5] == 255


class _Masker:
    """Stands in for the segmentation model: masks the upper left quarter of a frame."""

    def segment(self, img_bgr):
        h, w = img_bgr.shape[:2]
        seg = np.zeros((h, w), np.uint8)
        seg[: h // 2, : w // 2] = 255
        return seg


def test_write_front_masks_writes_every_listed_mask(tmp_path):
    frames, out = tmp_path / "frames", tmp_path / "masks"
    frames.mkdir()
    out.mkdir()
    t_ns = [0, 33366666]
    for t in t_ns:
        cv2.imwrite(str(frames / f"{t}.jpg"), np.full((960, 960, 3), 128, np.uint8))
    (out / "0.png").write_bytes(b"left over from an earlier run")
    masks.write_front_masks(_Masker(), frames, np.array(t_ns), out, KB4)
    ring = masks.ring_mask(480.0, 480.0, masks.RING_RADIUS_FRACTION * 960, (960, 960))
    for t in t_ns:
        mask = cv2.imread(str(out / f"{t}.png"), cv2.IMREAD_UNCHANGED)
        assert mask.shape == (960, 960) and set(np.unique(mask)) <= {0, 255}
        assert np.all(mask[ring == 255] == 255)
        # Inside the ring the mask is the segmentation.
        assert mask[240, 240] == 255 and mask[600, 600] == 0

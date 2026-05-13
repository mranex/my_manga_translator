import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from detectors.base import TextRegion
from inpainting.strategy import (
    boxes_from_mask,
    composite_masked,
    crop_box,
    crop_windows_from_text_regions,
    run_inpaint_crop,
)


class TestInpaintingStrategy(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy is not available")
    def test_boxes_from_mask_detects_separate_components(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[1:3, 1:4] = 255
        mask[7:10, 8:11] = 255

        boxes = boxes_from_mask(mask)

        self.assertEqual(boxes, [(1, 1, 4, 3), (8, 7, 11, 10)])

    @unittest.skipIf(np is None, "numpy is not available")
    def test_crop_box_expands_with_margin_and_clamps(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:3, 1:3] = 255

        crop_image, crop_mask, crop_bounds = crop_box(image, mask, (1, 1, 3, 3), 4)

        self.assertEqual(crop_bounds, (0, 0, 7, 7))
        self.assertEqual(crop_image.shape[:2], (7, 7))
        self.assertEqual(crop_mask.shape, (7, 7))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_crop_windows_from_text_regions_enlarges_and_merges(self):
        text_regions = [
            TextRegion(bbox=(10, 10, 18, 18), reading_order=0),
            TextRegion(bbox=(18, 12, 26, 20), reading_order=1),
            TextRegion(bbox=(70, 70, 78, 78), reading_order=2),
        ]

        windows = crop_windows_from_text_regions(text_regions, (100, 100, 3))

        self.assertEqual(len(windows), 2)
        self.assertLessEqual(windows[0][0], 10)
        self.assertLessEqual(windows[0][1], 10)
        self.assertGreaterEqual(windows[0][2], 26)
        self.assertGreaterEqual(windows[0][3], 20)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_composite_masked_only_changes_masked_pixels(self):
        base_image = np.zeros((5, 5, 3), dtype=np.uint8)
        patch_image = np.zeros((5, 5, 3), dtype=np.uint8) + 200
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2:4, 1:3] = 255

        composite_masked(base_image, patch_image, mask, 0, 0)

        self.assertTrue(np.all(base_image[2:4, 1:3] == 200))
        self.assertTrue(np.all(base_image[0:2, :] == 0))
        self.assertTrue(np.all(base_image[:, 3:] == 0))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_supplied_crop_windows_are_used_instead_of_raw_mask_boxes(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[12:14, 12:14] = 255
        forward_shapes = []

        def fake_forward(crop_image, crop_mask, crop_bubble):
            forward_shapes.append(crop_image.shape)
            return crop_image.copy()

        run_inpaint_crop(
            fake_forward,
            image,
            mask,
            bubble_mask=None,
            crop_trigger_size=800,
            crop_margin=0,
            resize_limit=200,
            pad_mod=1,
            crop_windows=[(5, 5, 25, 25)],
        )

        self.assertEqual(forward_shapes, [(20, 20, 3)])

    @unittest.skipIf(np is None, "numpy is not available")
    def test_crop_window_with_empty_mask_is_skipped_then_residual_mask_runs(self):
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        forward_shapes = []

        def fake_forward(crop_image, crop_mask, crop_bubble):
            forward_shapes.append(crop_image.shape)
            return crop_image.copy()

        run_inpaint_crop(
            fake_forward,
            image,
            mask,
            bubble_mask=None,
            crop_trigger_size=800,
            crop_margin=0,
            resize_limit=200,
            pad_mod=1,
            crop_windows=[(20, 20, 25, 25)],
        )

        self.assertEqual(forward_shapes, [(2, 2, 3)])


if __name__ == "__main__":
    unittest.main()

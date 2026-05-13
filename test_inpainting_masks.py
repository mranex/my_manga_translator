import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from detectors.base import BubbleRegion, TextRegion
from inpainting.masks import (
    build_bubble_mask,
    build_text_block_crop_windows,
    build_text_block_removal_mask,
    build_text_removal_mask,
)


class TestInpaintingMasks(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy is not available")
    def test_build_text_block_removal_mask_uses_expanded_block_bbox_not_only_glyph_mask(self):
        glyph_mask = np.zeros((32, 32), dtype=np.uint8)
        glyph_mask[12:16, 12:16] = 255
        text_region = TextRegion(bbox=(10, 10, 20, 20), mask=glyph_mask)

        removal_mask = build_text_block_removal_mask(
            (32, 32, 3),
            [
                {
                    "kind": "outside_text",
                    "text_region": text_region,
                    "text_regions": [text_region],
                    "inpaint_bbox": (8, 8, 22, 22),
                    "render_bbox": (10, 10, 20, 20),
                }
            ],
            block_padding=0,
            dilation=0,
        )

        self.assertEqual(int(removal_mask[9, 9]), 255)
        self.assertEqual(int(removal_mask[21, 21]), 255)
        self.assertEqual(int(removal_mask[13, 13]), 255)
        self.assertEqual(int(removal_mask[4, 4]), 0)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_region_mask_merges_without_shrinking_block_bbox(self):
        glyph_mask = np.zeros((24, 24), dtype=np.uint8)
        glyph_mask[10:12, 14:16] = 255
        text_region = TextRegion(bbox=(8, 8, 18, 18), mask=glyph_mask)

        removal_mask = build_text_block_removal_mask(
            (24, 24, 3),
            [
                {
                    "kind": "outside_text",
                    "text_region": text_region,
                    "text_regions": [text_region],
                }
            ],
            block_padding=2,
            dilation=0,
        )

        self.assertEqual(int(removal_mask[7, 7]), 255)
        self.assertEqual(int(removal_mask[10, 14]), 255)
        self.assertEqual(int(removal_mask[3, 3]), 0)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_bubble_item_uses_union_text_block_not_full_bubble_mask(self):
        bubble_mask = np.zeros((30, 30), dtype=np.uint8)
        bubble_mask[2:28, 2:28] = 255
        bubble_region = BubbleRegion(bbox=(2, 2, 28, 28), mask=bubble_mask)
        text_regions = [
            TextRegion(bbox=(10, 10, 14, 14)),
            TextRegion(bbox=(18, 10, 22, 14)),
        ]

        removal_mask = build_text_block_removal_mask(
            (30, 30, 3),
            [
                {
                    "kind": "bubble",
                    "bubble_region": bubble_region,
                    "text_regions": text_regions,
                    "render_bbox": (10, 10, 22, 14),
                }
            ],
            block_padding=2,
            dilation=0,
        )

        self.assertEqual(int(removal_mask[9, 9]), 255)
        self.assertEqual(int(removal_mask[15, 20]), 255)
        self.assertEqual(int(removal_mask[4, 4]), 0)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_outside_text_item_uses_text_region_bbox(self):
        text_region = TextRegion(bbox=(10, 10, 14, 14))

        removal_mask = build_text_block_removal_mask(
            (24, 24, 3),
            [
                {
                    "kind": "outside_text",
                    "text_region": text_region,
                    "text_regions": [text_region],
                }
            ],
            block_padding=4,
            dilation=0,
        )

        self.assertEqual(int(removal_mask[7, 7]), 255)
        self.assertEqual(int(removal_mask[17, 17]), 255)
        self.assertEqual(int(removal_mask[3, 3]), 0)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_huge_region_is_not_used_as_full_page_removal_mask(self):
        huge_region = TextRegion(bbox=(0, 0, 90, 90))

        removal_mask = build_text_block_removal_mask(
            (100, 100, 3),
            [
                {
                    "kind": "outside_text",
                    "text_region": huge_region,
                    "text_regions": [huge_region],
                    "inpaint_bbox": None,
                    "huge_region_skipped": True,
                }
            ],
            block_padding=8,
            dilation=0,
        )

        self.assertEqual(int(np.count_nonzero(removal_mask)), 0)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_dilation_expands_mask(self):
        text_region = TextRegion(bbox=(10, 10, 12, 12))
        render_items = [
            {
                "kind": "outside_text",
                "text_region": text_region,
                "text_regions": [text_region],
            }
        ]

        base_mask = build_text_block_removal_mask((30, 30, 3), render_items, block_padding=0, dilation=0)
        dilated_mask = build_text_block_removal_mask((30, 30, 3), render_items, block_padding=0, dilation=2)

        self.assertGreater(int(np.count_nonzero(dilated_mask)), int(np.count_nonzero(base_mask)))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_build_text_block_crop_windows_enlarges_and_merges(self):
        render_items = [
            {
                "kind": "bubble",
                "inpaint_bbox": (10, 10, 18, 18),
            },
            {
                "kind": "bubble",
                "inpaint_bbox": (18, 12, 26, 20),
            },
            {
                "kind": "outside_text",
                "inpaint_bbox": (70, 70, 78, 78),
            },
        ]

        windows = build_text_block_crop_windows(render_items, (100, 100, 3))

        self.assertEqual(len(windows), 2)
        self.assertLessEqual(windows[0][0], 10)
        self.assertLessEqual(windows[0][1], 10)
        self.assertGreaterEqual(windows[0][2], 26)
        self.assertGreaterEqual(windows[0][3], 20)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_render_bbox_and_inpaint_bbox_can_differ(self):
        text_region = TextRegion(bbox=(20, 20, 28, 28))
        removal_mask = build_text_block_removal_mask(
            (60, 60, 3),
            [
                {
                    "kind": "outside_text",
                    "text_region": text_region,
                    "text_regions": [text_region],
                    "render_bbox": (21, 21, 27, 27),
                    "inpaint_bbox": (16, 16, 32, 32),
                }
            ],
            block_padding=0,
            dilation=0,
        )

        self.assertEqual(int(removal_mask[17, 17]), 255)
        self.assertEqual(int(removal_mask[31, 31]), 255)
        self.assertEqual(int(removal_mask[15, 15]), 0)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_build_text_removal_mask_compatibility_wrapper_uses_block_behavior(self):
        text_region = TextRegion(bbox=(10, 10, 14, 14))

        removal_mask = build_text_removal_mask(
            (24, 24, 3),
            [
                {
                    "kind": "outside_text",
                    "text_region": text_region,
                    "text_regions": [text_region],
                }
            ],
            dilation=0,
        )

        self.assertEqual(int(removal_mask[9, 9]), 255)
        self.assertEqual(int(removal_mask[15, 15]), 255)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_build_bubble_mask_uses_bubble_region_mask(self):
        bubble_mask = np.zeros((16, 16), dtype=np.uint8)
        bubble_mask[4:12, 5:11] = 255
        bubble_region = BubbleRegion(bbox=(4, 4, 12, 12), mask=bubble_mask)

        output_mask = build_bubble_mask(
            (16, 16, 3),
            [{"kind": "bubble", "bubble_region": bubble_region}],
        )

        self.assertEqual(int(output_mask[6, 6]), 255)
        self.assertEqual(int(output_mask[1, 1]), 0)


if __name__ == "__main__":
    unittest.main()

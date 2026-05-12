import unittest

from detectors import assign_text_regions_to_bubbles, bbox_iou
from detectors.base import BubbleRegion, TextRegion
from detectors.matching import bbox_center, point_in_bbox, point_in_mask


class TestDetectorMatching(unittest.TestCase):
    def test_bbox_center_returns_expected_center(self):
        self.assertEqual(bbox_center((0, 0, 10, 20)), (5.0, 10.0))

    def test_point_in_bbox_respects_padding(self):
        self.assertFalse(point_in_bbox((11, 5), (0, 0, 10, 10), padding=0))
        self.assertTrue(point_in_bbox((11, 5), (0, 0, 10, 10), padding=1))

    def test_bbox_iou_handles_overlap_and_non_overlap(self):
        self.assertEqual(bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(
            bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)),
            25.0 / 175.0,
        )

    def test_text_center_inside_bubble_bbox_assigns_bubble_id(self):
        original_text = TextRegion(bbox=(12, 12, 18, 18), text="hello")
        assigned = assign_text_regions_to_bubbles(
            [original_text],
            [BubbleRegion(bbox=(10, 10, 20, 20))],
        )

        self.assertEqual(assigned[0].bubble_id, 0)
        self.assertIsNone(original_text.bubble_id)

    def test_smaller_bubble_wins_when_multiple_contain_center(self):
        text_region = TextRegion(bbox=(9, 9, 11, 11), text="nested")
        bubbles = [
            BubbleRegion(bbox=(0, 0, 20, 20)),
            BubbleRegion(bbox=(5, 5, 15, 15)),
        ]

        assigned = assign_text_regions_to_bubbles([text_region], bubbles)

        self.assertEqual(assigned[0].bubble_id, 1)

    def test_iou_fallback_assigns_when_center_is_outside(self):
        text_region = TextRegion(bbox=(8, 8, 14, 14), text="edge")
        bubbles = [BubbleRegion(bbox=(0, 0, 10, 10))]

        assigned = assign_text_regions_to_bubbles(
            [text_region],
            bubbles,
            bbox_padding=0,
            min_iou=0.02,
        )

        self.assertEqual(assigned[0].bubble_id, 0)

    def test_unmatched_text_keeps_bubble_id_none(self):
        text_region = TextRegion(bbox=(20, 20, 30, 30), text="outside")
        bubbles = [BubbleRegion(bbox=(0, 0, 10, 10))]

        assigned = assign_text_regions_to_bubbles(
            [text_region],
            bubbles,
            bbox_padding=0,
            min_iou=0.05,
        )

        self.assertIsNone(assigned[0].bubble_id)

    def test_mutate_false_does_not_modify_original_text_region(self):
        text_region = TextRegion(bbox=(12, 12, 18, 18), text="copy")
        bubbles = [BubbleRegion(bbox=(10, 10, 20, 20))]

        assigned = assign_text_regions_to_bubbles([text_region], bubbles, mutate=False)

        self.assertIsNot(assigned[0], text_region)
        self.assertIsNone(text_region.bubble_id)
        self.assertEqual(assigned[0].bubble_id, 0)

    def test_mutate_true_modifies_original_text_region(self):
        text_region = TextRegion(bbox=(12, 12, 18, 18), text="mutate")
        bubbles = [BubbleRegion(bbox=(10, 10, 20, 20))]

        assigned = assign_text_regions_to_bubbles([text_region], bubbles, mutate=True)

        self.assertIs(assigned[0], text_region)
        self.assertEqual(text_region.bubble_id, 0)

    def test_point_in_mask_uses_positive_pixels(self):
        mask = [[0 for _ in range(5)] for _ in range(5)]
        mask[2][3] = 255

        self.assertTrue(point_in_mask((3.1, 2.8), mask))
        self.assertFalse(point_in_mask((0.0, 0.0), mask))
        self.assertFalse(point_in_mask((10.0, 10.0), mask))

    def test_mask_match_is_preferred(self):
        mask = [[0 for _ in range(30)] for _ in range(30)]
        for y in range(12, 18):
            for x in range(12, 18):
                mask[y][x] = 1
        text_region = TextRegion(bbox=(12, 12, 18, 18), text="mask")
        bubbles = [
            BubbleRegion(bbox=(0, 0, 30, 30)),
            BubbleRegion(bbox=(10, 10, 20, 20), mask=mask),
        ]

        assigned = assign_text_regions_to_bubbles([text_region], bubbles)

        self.assertEqual(assigned[0].bubble_id, 1)


if __name__ == "__main__":
    unittest.main()

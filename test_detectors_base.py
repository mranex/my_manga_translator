import unittest

from detectors.base import (
    BubbleRegion,
    LayoutRegion,
    PageDetectionResult,
    bubble_region_from_legacy_detection,
)


class TestLegacyBubbleDetections(unittest.TestCase):
    def test_six_item_detection_defaults_to_light_bubble(self):
        bubble = bubble_region_from_legacy_detection([10.9, 20.1, 30.8, 40.2, 0.95, 0])

        self.assertEqual(bubble.bbox, (10, 20, 30, 40))
        self.assertEqual(bubble.score, 0.95)
        self.assertEqual(bubble.class_id, 0)
        self.assertFalse(bubble.is_dark)

    def test_seven_item_detection_preserves_dark_flag(self):
        bubble = bubble_region_from_legacy_detection([1, 2, 3, 4, 0.7, 0, 1])

        self.assertEqual(bubble.bbox, (1, 2, 3, 4))
        self.assertTrue(bubble.is_dark)

    def test_to_legacy_detections_preserves_shape_and_order(self):
        result = PageDetectionResult(
            bubbles=[
                BubbleRegion(bbox=(10, 20, 30, 40), score=0.8, class_id=2, is_dark=False),
                BubbleRegion(bbox=(1, 2, 3, 4), score=0.9, class_id=0, is_dark=True),
            ],
            text_regions=[],
        )

        self.assertEqual(
            result.to_legacy_detections(),
            [
                [10, 20, 30, 40, 0.8, 2, 0],
                [1, 2, 3, 4, 0.9, 0, 1],
            ],
        )

    def test_layout_region_and_page_result_remain_backward_compatible(self):
        result = PageDetectionResult(
            bubbles=[BubbleRegion(bbox=(0, 0, 10, 10))],
            text_regions=[],
            layout_regions=[
                LayoutRegion(
                    bbox=(1, 2, 11, 12),
                    score=0.7,
                    class_id=3,
                    label="caption",
                    label_id=3,
                    reading_order=4,
                    polygon_points=[(1, 2), (11, 2), (11, 12), (1, 12)],
                )
            ],
            method="layout-first",
        )

        self.assertEqual(result.method, "layout-first")
        self.assertEqual(len(result.layout_regions), 1)
        self.assertEqual(result.layout_regions[0].kind, "layout")
        self.assertEqual(result.layout_regions[0].label, "caption")
        self.assertEqual(result.layout_regions[0].reading_order, 4)


if __name__ == "__main__":
    unittest.main()

import unittest

from detectors.base import BubbleRegion, LayoutRegion, TextRegion
from detectors.page_detector import (
    detect_page_regions_layout_first,
    merge_text_region_candidates,
)


class FakeImage:
    shape = (100, 120, 3)


class FakeLayoutDetector:
    def detect_layout_regions(self, image):
        return [
            LayoutRegion(bbox=(12, 12, 30, 26), label="caption", score=0.7, reading_order=0),
            LayoutRegion(bbox=(80, 8, 110, 22), label="unknown", score=0.6, reading_order=1),
            LayoutRegion(bbox=(0, 0, 120, 100), label="figure", score=0.95, reading_order=2),
        ]


class FakeBubbleDetector:
    def __init__(self):
        self.full_page_calls = 0
        self.roi_calls = 0

    def detect_segmented_bubble_regions(self, image):
        self.full_page_calls += 1
        return [
            BubbleRegion(bbox=(10, 10, 40, 40), score=0.9),
        ]

    def detect_bubble_regions_in_rois(self, image, layout_rois):
        self.roi_calls += 1
        raise AssertionError("Active page detector should not gate bubble detection on PP ROIs")


class FakeComicTextDetector:
    def __init__(self, regions):
        self._regions = list(regions)
        self.full_page_calls = 0
        self.roi_calls = 0

    def detect_text_regions(self, image):
        self.full_page_calls += 1
        return list(self._regions)

    def detect_text_regions_in_rois(self, image, layout_rois):
        self.roi_calls += 1
        raise AssertionError("Active page detector should not gate comic text detection on PP ROIs")


class TestPageDetector(unittest.TestCase):
    def test_active_pipeline_uses_full_page_bubble_and_text_detectors(self):
        image = FakeImage()
        bubble_detector = FakeBubbleDetector()
        text_detector = FakeComicTextDetector(
            [TextRegion(bbox=(14, 14, 28, 24), confidence=0.95, reading_order=3)]
        )

        result = detect_page_regions_layout_first(
            image,
            layout_detector=FakeLayoutDetector(),
            bubble_detector=bubble_detector,
            text_detector=text_detector,
        )

        self.assertEqual(
            result.method,
            "pp_doclayout_v3_text_source+yolov8_seg_bubble+comic_text_detector",
        )
        self.assertEqual(len(result.layout_regions), 3)
        self.assertEqual(len(result.bubbles), 1)
        self.assertEqual(bubble_detector.full_page_calls, 1)
        self.assertEqual(bubble_detector.roi_calls, 0)
        self.assertEqual(text_detector.full_page_calls, 1)
        self.assertEqual(text_detector.roi_calls, 0)

    def test_pp_only_text_region_is_preserved_when_comic_detector_returns_no_text(self):
        image = FakeImage()
        result = detect_page_regions_layout_first(
            image,
            layout_detector=FakeLayoutDetector(),
            bubble_detector=FakeBubbleDetector(),
            text_detector=FakeComicTextDetector([]),
        )

        self.assertEqual(len(result.text_regions), 2)
        self.assertEqual(result.text_regions[0].bbox, (8, 8, 34, 30))
        self.assertEqual(result.text_regions[0].bubble_id, 0)
        self.assertEqual(result.text_regions[0].reading_order, 0)
        self.assertEqual(result.text_regions[1].bbox, (76, 4, 114, 26))
        self.assertIsNone(result.text_regions[1].bubble_id)

    def test_comic_regions_are_preferred_over_overlapping_pp_regions(self):
        merged = merge_text_region_candidates(
            [TextRegion(bbox=(10, 10, 30, 30), confidence=0.6, reading_order=0)],
            [TextRegion(bbox=(11, 11, 29, 29), confidence=0.95, reading_order=2, text="comic")],
            image_shape=(100, 120, 3),
            iou_threshold=0.5,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "comic")
        self.assertEqual(merged[0].bbox, (11, 11, 29, 29))

    def test_unmatched_pp_text_region_remains_without_bubble_id(self):
        image = FakeImage()
        text_detector = FakeComicTextDetector(
            [TextRegion(bbox=(90, 70, 110, 90), confidence=0.9, reading_order=4)]
        )

        result = detect_page_regions_layout_first(
            image,
            layout_detector=FakeLayoutDetector(),
            bubble_detector=FakeBubbleDetector(),
            text_detector=text_detector,
        )

        unmatched = [region for region in result.text_regions if region.bubble_id is None]
        self.assertTrue(unmatched)
        self.assertIn((76, 4, 114, 26), [region.bbox for region in unmatched])
        self.assertIn((90, 70, 110, 90), [region.bbox for region in unmatched])


if __name__ == "__main__":
    unittest.main()

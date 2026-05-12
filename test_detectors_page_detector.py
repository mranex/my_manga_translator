import unittest

from detectors.base import BubbleRegion, LayoutRegion, TextRegion
from detectors.page_detector import detect_page_regions_layout_first


class FakeImage:
    shape = (100, 120, 3)


class FakeLayoutDetector:
    roi_padding = 12

    def detect_layout_regions(self, image):
        return [
            LayoutRegion(bbox=(10, 10, 40, 40), label="caption", reading_order=0),
            LayoutRegion(bbox=(50, 45, 90, 85), label="figure", reading_order=1),
        ]


class FakeBubbleDetector:
    def __init__(self):
        self.received_rois = None

    def detect_bubble_regions_in_rois(self, image, layout_rois):
        self.received_rois = list(layout_rois)
        return [
            BubbleRegion(bbox=(12, 12, 28, 28), score=0.9),
            BubbleRegion(bbox=(55, 52, 82, 78), score=0.8),
        ]


class FakeTextDetector:
    def __init__(self):
        self.received_rois = None

    def detect_text_regions_in_rois(self, image, layout_rois):
        self.received_rois = list(layout_rois)
        return [
            TextRegion(bbox=(13, 13, 24, 24), text="inside first", reading_order=0),
            TextRegion(bbox=(58, 54, 76, 74), text="inside second", reading_order=1),
            TextRegion(bbox=(95, 5, 110, 20), text="outside", reading_order=2),
        ]


class TestPageDetector(unittest.TestCase):
    def test_layout_first_orchestration_returns_layout_bubbles_and_text(self):
        image = FakeImage()
        bubble_detector = FakeBubbleDetector()
        text_detector = FakeTextDetector()

        result = detect_page_regions_layout_first(
            image,
            layout_detector=FakeLayoutDetector(),
            bubble_detector=bubble_detector,
            text_detector=text_detector,
        )

        self.assertEqual(
            result.method,
            "pp_doclayout_v3+yolov8_seg_bubble+comic_text_detector",
        )
        self.assertEqual(len(result.layout_regions), 2)
        self.assertEqual(len(result.bubbles), 2)
        self.assertEqual(len(result.text_regions), 3)
        self.assertEqual(result.text_regions[0].bubble_id, 0)
        self.assertEqual(result.text_regions[1].bubble_id, 1)
        self.assertIsNone(result.text_regions[2].bubble_id)
        self.assertEqual(len(bubble_detector.received_rois), 2)
        self.assertEqual(len(text_detector.received_rois), 2)
        self.assertEqual(bubble_detector.received_rois[0].bbox, (0, 0, 52, 52))
        self.assertEqual(text_detector.received_rois[1].bbox, (38, 33, 102, 97))


if __name__ == "__main__":
    unittest.main()

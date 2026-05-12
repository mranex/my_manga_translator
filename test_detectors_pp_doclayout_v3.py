import sys
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from detectors.base import BubbleRegion, LayoutRegion, TextRegion
from detectors.pp_doclayout_v3 import (
    PPDocLayoutV3Detector,
    build_layout_rois,
    normalize_layout_detections,
)
from detectors.runtime_utils import (
    map_bbox_from_roi_to_page,
    map_bubble_region_from_roi_to_page,
    map_mask_from_roi_to_page,
    map_text_region_from_roi_to_page,
    merge_duplicate_bubble_regions,
    merge_duplicate_text_regions,
)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        if np is None:
            raise ModuleNotFoundError("numpy is not available")
        return np.asarray(self.value)

    def to(self, device):
        return self


class FakeTorchModule:
    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    @staticmethod
    def no_grad():
        return FakeTorchModule._NoGrad()

    @staticmethod
    def tensor(value):
        return value

    class cuda:
        @staticmethod
        def is_available():
            return False


class FakeProcessor:
    def __init__(self, output):
        self.output = output

    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": FakeTensor([[1]])}

    def post_process_object_detection(self, outputs, threshold, target_sizes):
        return [self.output]


class FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(id2label={0: "text", 1: "figure"})

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True

    def __call__(self, **kwargs):
        return types.SimpleNamespace()


class TestPPDocLayoutNormalization(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy is not available")
    def test_normalize_layout_output_to_regions(self):
        regions = normalize_layout_detections(
            {
                "boxes": FakeTensor([[10, 20, 50, 60], [0, 0, 100, 100]]),
                "scores": FakeTensor([0.9, 0.8]),
                "labels": FakeTensor([0, 1]),
                "reading_order": [3, 1],
                "polygon_points": [
                    [[10, 20], [50, 20], [50, 60], [10, 60]],
                    [[0, 0], [100, 0], [100, 100], [0, 100]],
                ],
            },
            (100, 100, 3),
            id2label={0: "text", 1: "figure"},
            confidence_threshold=0.25,
            min_region_area=64,
            max_full_page_region_ratio=0.92,
        )

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].bbox, (10, 20, 50, 60))
        self.assertEqual(regions[0].label, "text")
        self.assertEqual(regions[0].reading_order, 3)
        self.assertEqual(regions[0].polygon_points[0], [10, 20])

    @unittest.skipIf(np is None, "numpy is not available")
    def test_normalize_layout_output_falls_back_to_full_page(self):
        regions = normalize_layout_detections(
            {
                "boxes": FakeTensor([]),
                "scores": FakeTensor([]),
                "labels": FakeTensor([]),
            },
            (80, 120, 3),
        )

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].bbox, (0, 0, 120, 80))
        self.assertEqual(regions[0].label, "full_page")

    def test_build_layout_rois_expands_and_clamps(self):
        rois = build_layout_rois(
            [LayoutRegion(bbox=(10, 10, 30, 30), reading_order=0)],
            (40, 50, 3),
            padding=20,
        )

        self.assertEqual(len(rois), 1)
        self.assertEqual(rois[0].bbox, (0, 0, 50, 40))

    def test_build_layout_rois_merges_overlapping_regions(self):
        rois = build_layout_rois(
            [
                LayoutRegion(bbox=(10, 10, 30, 30), reading_order=0),
                LayoutRegion(bbox=(15, 12, 35, 32), reading_order=1),
            ],
            (60, 60, 3),
            padding=4,
            merge_iou=0.2,
        )

        self.assertEqual(len(rois), 1)
        self.assertEqual(rois[0].reading_order, 0)
        self.assertEqual(rois[0].bbox, (6, 6, 39, 36))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_detector_detect_layout_regions_uses_mocked_processor_and_model(self):
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        detector = PPDocLayoutV3Detector()
        detector._image_processor = FakeProcessor(
            {
                "boxes": FakeTensor([[5, 6, 20, 25]]),
                "scores": FakeTensor([0.9]),
                "labels": FakeTensor([0]),
            }
        )
        detector._model = FakeModel()
        detector.device = "cpu"

        with patch.object(
            detector,
            "_prepare_pil_image",
            return_value=types.SimpleNamespace(size=(60, 40)),
        ):
            with patch(
                "detectors.pp_doclayout_v3.importlib.import_module",
                side_effect=lambda name: FakeTorchModule()
                if name == "torch"
                else __import__(name, fromlist=["dummy"]),
            ):
                regions = detector.detect_layout_regions(image)

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].bbox, (5, 6, 20, 25))
        self.assertEqual(regions[0].label, "text")


class TestROIMappingUtilities(unittest.TestCase):
    def test_map_bbox_from_roi_to_page(self):
        self.assertEqual(
            map_bbox_from_roi_to_page((3, 4, 13, 14), (10, 20, 50, 60)),
            (13, 24, 23, 34),
        )

    @unittest.skipIf(np is None, "numpy is not available")
    def test_map_mask_and_regions_from_roi_to_page(self):
        local_mask = np.zeros((4, 5), dtype=np.uint8)
        local_mask[1:3, 2:4] = 255
        roi_bbox = (10, 20, 15, 24)

        full_mask = map_mask_from_roi_to_page(local_mask, roi_bbox, (40, 50, 3))
        self.assertEqual(full_mask.shape, (40, 50))
        self.assertEqual(int(full_mask[21, 12]), 255)

        text_region = map_text_region_from_roi_to_page(
            TextRegion(bbox=(1, 1, 3, 3), mask=local_mask),
            roi_bbox,
            (40, 50, 3),
        )
        bubble_region = map_bubble_region_from_roi_to_page(
            BubbleRegion(bbox=(0, 0, 5, 4), mask=local_mask),
            roi_bbox,
            (40, 50, 3),
        )

        self.assertEqual(text_region.bbox, (11, 21, 13, 23))
        self.assertEqual(bubble_region.bbox, (10, 20, 15, 24))
        self.assertEqual(int(text_region.mask[21, 12]), 255)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_duplicate_region_merging_uses_iou(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:8, 2:8] = 255

        bubbles = merge_duplicate_bubble_regions(
            [
                BubbleRegion(bbox=(2, 2, 8, 8), score=0.9, mask=mask),
                BubbleRegion(bbox=(3, 3, 9, 9), score=0.7, mask=mask),
            ],
            iou_threshold=0.3,
            image_shape=(20, 20),
        )
        text_regions = merge_duplicate_text_regions(
            [
                TextRegion(bbox=(2, 2, 8, 8), confidence=0.9, text="a", mask=mask),
                TextRegion(bbox=(3, 3, 9, 9), confidence=0.7, text="b", mask=mask),
            ],
            iou_threshold=0.3,
            image_shape=(20, 20),
        )

        self.assertEqual(len(bubbles), 1)
        self.assertEqual(bubbles[0].bbox, (2, 2, 9, 9))
        self.assertEqual(len(text_regions), 1)
        self.assertEqual(text_regions[0].bbox, (2, 2, 9, 9))
        self.assertEqual(text_regions[0].text, "a")


if __name__ == "__main__":
    unittest.main()

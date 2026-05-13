import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from detectors.runtime_utils import (
    detect_dark_bubble_from_mask,
    mask_to_contour,
    normalize_binary_mask,
    process_bubble_with_mask,
    rectangular_contour,
    text_region_to_crop_data,
)
from detectors.yolov8_seg_bubble import (
    YoloSegBubbleDetector,
    ensure_yolov8_seg_bubble_weights,
    detect_bubble_regions_in_rois,
    normalize_yolov8_segmentation_result,
)
from detectors.base import BubbleRegion, TextRegion


class FakeImage:
    shape = (40, 60, 3)

    def __getitem__(self, key):
        return self


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        if np is None:
            raise ModuleNotFoundError("numpy is required for FakeTensor")
        return np.asarray(self.value)


def contour_bbox(contour):
    points = np.asarray(contour).reshape(-1, 2)
    x1 = int(points[:, 0].min())
    y1 = int(points[:, 1].min())
    x2 = int(points[:, 0].max()) + 1
    y2 = int(points[:, 1].max()) + 1
    return (x1, y1, x2 - x1, y2 - y1)


class TestYoloSegBubbleWeights(unittest.TestCase):
    def test_ensure_weights_downloads_model(self):
        download_calls = []

        def fake_hf_hub_download(repo_id, filename, local_dir, local_dir_use_symlinks):
            local_path = Path(local_dir) / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"model")
            download_calls.append((repo_id, filename, local_dir_use_symlinks))
            return str(local_path)

        fake_hf_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf_module}):
                weights = ensure_yolov8_seg_bubble_weights(model_dir=temp_dir)

        self.assertEqual(len(download_calls), 1)
        self.assertEqual(download_calls[0][0], "kitsumed/yolov8m_seg-speech-bubble")
        self.assertEqual(weights["model_path"].name, "model.pt")

    def test_ensure_weights_skips_existing_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"ready")

            with patch(
                "detectors.yolov8_seg_bubble.importlib.import_module"
            ) as mock_import_module:
                weights = ensure_yolov8_seg_bubble_weights(model_dir=temp_dir)

        self.assertEqual(weights["model_path"], model_path)
        self.assertFalse(mock_import_module.called)


class TestSegmentationMaskHelpers(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy is not available")
    def test_normalize_binary_mask_resizes_to_page_shape(self):
        raw_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        normalized = normalize_binary_mask(raw_mask, (4, 6, 3))

        self.assertEqual(normalized.shape, (4, 6))
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertTrue(set(np.unique(normalized)).issubset({0, 255}))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_mask_to_contour_returns_crop_local_contour(self):
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1:5, 2:5] = 255

        contour = mask_to_contour(mask)
        x, y, w, h = contour_bbox(contour)

        self.assertEqual((x, y, w, h), (2, 1, 3, 4))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_detect_dark_bubble_from_mask_for_dark_and_light_regions(self):
        dark_image = np.zeros((4, 4, 3), dtype=np.uint8) + 20
        light_image = np.zeros((4, 4, 3), dtype=np.uint8) + 235
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255

        self.assertTrue(detect_dark_bubble_from_mask(dark_image, mask))
        self.assertFalse(detect_dark_bubble_from_mask(light_image, mask))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_process_bubble_with_mask_fills_only_masked_area(self):
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        image[1:4, 1:4] = 240
        image[2, 2] = 10
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 255

        processed, contour, bubble_is_dark, detected_color = process_bubble_with_mask(
            image.copy(),
            mask,
        )

        self.assertFalse(bubble_is_dark)
        self.assertEqual(tuple(processed[0, 0]), (0, 0, 0))
        self.assertTrue(all(channel >= 200 for channel in detected_color))
        self.assertTrue(np.all(processed[1:4, 1:4] == np.array(detected_color)))
        self.assertEqual(contour_bbox(contour), (1, 1, 3, 3))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_text_region_crop_data_uses_bbox_with_padding(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        text_region = TextRegion(bbox=(10, 8, 14, 12))

        crop_data = text_region_to_crop_data(image, text_region, padding=2)

        self.assertEqual(crop_data["region_bbox"], (8, 6, 16, 14))
        self.assertEqual(crop_data["ocr_crop"].shape[:2], (8, 8))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_rectangular_contour_matches_crop_bounds(self):
        contour = rectangular_contour(6, 4)

        self.assertEqual(contour_bbox(contour), (0, 0, 6, 4))


class TestYoloSegBubbleNormalization(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy is not available")
    def test_normalize_yolov8_segmentation_result_returns_bubble_regions(self):
        image = np.zeros((8, 10, 3), dtype=np.uint8) + 230
        image[4:7, 6:9] = 20

        result = types.SimpleNamespace(
            boxes=types.SimpleNamespace(
                xyxy=FakeTensor(
                    [
                        [6, 4, 9, 7],
                        [1, 1, 4, 4],
                    ]
                ),
                conf=FakeTensor([0.65, 0.95]),
                cls=FakeTensor([1, 0]),
            ),
            masks=types.SimpleNamespace(
                data=FakeTensor(
                    np.array(
                        [
                            [
                                [0, 0, 0],
                                [0, 1, 1],
                                [0, 1, 1],
                            ],
                            [
                                [1, 1, 0],
                                [1, 1, 0],
                                [0, 0, 0],
                            ],
                        ],
                        dtype=np.float32,
                    )
                )
            ),
        )

        bubbles = normalize_yolov8_segmentation_result(result, image)

        self.assertEqual(len(bubbles), 2)
        self.assertEqual(bubbles[0].bbox, (1, 1, 4, 4))
        self.assertEqual(bubbles[0].class_id, 0)
        self.assertAlmostEqual(bubbles[0].score, 0.95)
        self.assertEqual(bubbles[0].mask.shape, image.shape[:2])
        self.assertEqual(bubbles[0].mask.dtype, np.uint8)
        self.assertFalse(bubbles[0].is_dark)
        self.assertEqual(bubbles[1].bbox, (6, 4, 9, 7))
        self.assertTrue(bubbles[1].is_dark)


class FakeYoloSegBubbleDetector(YoloSegBubbleDetector):
    def __init__(self):
        super().__init__()

    def detect_segmented_bubble_regions(self, image):
        return [BubbleRegion(bbox=(1, 2, 6, 8), score=0.9)]


class TestYoloSegBubbleROIs(unittest.TestCase):
    def test_detect_bubble_regions_in_rois_maps_to_page_coordinates(self):
        detector = FakeYoloSegBubbleDetector()
        bubbles = detector.detect_bubble_regions_in_rois(
            FakeImage(),
            [types.SimpleNamespace(bbox=(10, 20, 30, 40), reading_order=0)],
        )

        self.assertEqual(len(bubbles), 1)
        self.assertEqual(bubbles[0].bbox, (11, 22, 16, 28))


if __name__ == "__main__":
    unittest.main()

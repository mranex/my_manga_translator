import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from detectors.base import BubbleRegion, TextRegion
from detectors.comic_text_detector import (
    ComicTextDetector,
    ComicTextDetectorUnavailable,
    _check_runtime_dependencies,
    ensure_comic_text_detector_weights,
    normalize_text_detection,
    normalize_text_detections,
)
from detectors.runtime_utils import (
    clamp_bbox_to_image,
    convert_legacy_detections_to_bubble_regions,
    crop_bbox,
    legacy_detection_to_bubble_region,
    union_text_regions_bbox,
)


class FakeComicTextDetector(ComicTextDetector):
    def __init__(self, text_regions):
        super().__init__(lazy_load=True)
        self._text_regions = list(text_regions)

    def detect_text_regions(self, image) -> list[TextRegion]:
        return list(self._text_regions)


class FakeImage:
    shape = (40, 60, 3)

    def __getitem__(self, key):
        return self


class TestComicTextDetectorImport(unittest.TestCase):
    def test_importing_module_does_not_load_heavy_dependencies(self):
        sys.modules.pop("detectors.comic_text_detector", None)

        with patch("importlib.import_module") as mock_import_module:
            module = __import__(
                "detectors.comic_text_detector",
                fromlist=["ComicTextDetector"],
            )

        self.assertFalse(mock_import_module.called)
        self.assertTrue(hasattr(module, "ComicTextDetector"))


class TestComicTextNormalization(unittest.TestCase):
    def test_normalize_dict_bbox_and_score(self):
        region = normalize_text_detection({"bbox": [1, 2, 11, 12], "score": 0.9})

        self.assertIsNotNone(region)
        self.assertEqual(region.bbox, (1, 2, 11, 12))
        self.assertEqual(region.confidence, 0.9)
        self.assertEqual(region.score, 0.9)
        self.assertEqual(region.kind, "text")

    def test_normalize_dict_box_and_confidence(self):
        region = normalize_text_detection({"box": [3, 4, 13, 14], "confidence": 0.8})

        self.assertIsNotNone(region)
        self.assertEqual(region.bbox, (3, 4, 13, 14))
        self.assertEqual(region.confidence, 0.8)

    def test_normalize_dict_xyxy_and_conf(self):
        region = normalize_text_detection({"xyxy": [5, 6, 15, 16], "conf": 0.7})

        self.assertIsNotNone(region)
        self.assertEqual(region.bbox, (5, 6, 15, 16))
        self.assertEqual(region.confidence, 0.7)

    def test_normalize_list_with_score(self):
        region = normalize_text_detection([1, 2, 11, 12, 0.85])

        self.assertIsNotNone(region)
        self.assertEqual(region.bbox, (1, 2, 11, 12))
        self.assertEqual(region.confidence, 0.85)

    def test_normalize_list_without_score_defaults_to_one(self):
        region = normalize_text_detection([1, 2, 11, 12])

        self.assertIsNotNone(region)
        self.assertEqual(region.confidence, 1.0)
        self.assertEqual(region.score, 1.0)

    def test_low_confidence_detection_returns_none(self):
        region = normalize_text_detection({"bbox": [1, 2, 11, 12], "score": 0.2})

        self.assertIsNone(region)

    def test_invalid_bbox_returns_none(self):
        region = normalize_text_detection({"bbox": [10, 10, 10, 12], "score": 0.9})

        self.assertIsNone(region)

    def test_normalize_text_detections_preserves_order_and_filters(self):
        regions = normalize_text_detections(
            [
                {"bbox": [1, 2, 11, 12], "score": 0.9, "text": "first"},
                {"bbox": [3, 4, 3, 9], "score": 0.95, "text": "bad"},
                [5, 6, 15, 16, 0.8],
                {"xyxy": [7, 8, 17, 18], "conf": 0.1, "text": "low"},
            ],
            confidence_threshold=0.3,
        )

        self.assertEqual([region.text for region in regions], ["first", ""])
        self.assertEqual(
            [region.bbox for region in regions],
            [(1, 2, 11, 12), (5, 6, 15, 16)],
        )


class TestComicTextWeights(unittest.TestCase):
    def test_ensure_weights_downloads_missing_files(self):
        download_calls = []

        def fake_hf_hub_download(repo_id, filename, local_dir, local_dir_use_symlinks):
            local_path = Path(local_dir) / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(filename.encode("utf-8"))
            download_calls.append((repo_id, filename, local_dir_use_symlinks))
            return str(local_path)

        fake_hf_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf_module}):
                weights = ensure_comic_text_detector_weights(model_dir=temp_dir)

        self.assertEqual(len(download_calls), 3)
        self.assertTrue(weights["yolo_v5"].name.endswith("yolo-v5.safetensors"))
        self.assertTrue(weights["unet"].name.endswith("unet.safetensors"))
        self.assertTrue(weights["dbnet"].name.endswith("dbnet.safetensors"))

    def test_ensure_weights_skips_existing_files(self):
        def fake_hf_hub_download(*args, **kwargs):
            raise AssertionError("hf_hub_download should not be called when files exist")

        fake_hf_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for filename in (
                "yolo-v5.safetensors",
                "unet.safetensors",
                "dbnet.safetensors",
            ):
                (temp_path / filename).write_bytes(b"ok")

            with patch.dict(sys.modules, {"huggingface_hub": fake_hf_module}):
                weights = ensure_comic_text_detector_weights(model_dir=temp_dir)

        self.assertEqual(weights["model_dir"].name, Path(temp_dir).name)


class TestComicTextDependencies(unittest.TestCase):
    def test_runtime_dependency_check_uses_cv2_module_import(self):
        attempted_imports = []

        def fake_import_module(name):
            attempted_imports.append(name)
            if name == "torch":
                return types.SimpleNamespace(
                    cuda=types.SimpleNamespace(is_available=lambda: False)
                )
            return types.SimpleNamespace()

        with patch(
            "detectors.comic_text_detector.importlib.import_module",
            side_effect=fake_import_module,
        ):
            _check_runtime_dependencies()

        self.assertIn("cv2", attempted_imports)
        self.assertNotIn("opencv-python", attempted_imports)
        self.assertNotIn("opencv-contrib-python", attempted_imports)
        self.assertNotIn("opencv-python-headless", attempted_imports)

    def test_runtime_dependency_check_reports_exact_failure(self):
        def fake_import_module(name):
            if name == "torchvision":
                raise ModuleNotFoundError("No module named 'torchvision'")
            if name == "torch":
                return types.SimpleNamespace(
                    cuda=types.SimpleNamespace(is_available=lambda: False)
                )
            return types.SimpleNamespace()

        with patch(
            "detectors.comic_text_detector.importlib.import_module",
            side_effect=fake_import_module,
        ):
            with self.assertRaises(ComicTextDetectorUnavailable) as ctx:
                _check_runtime_dependencies()

        message = str(ctx.exception)
        self.assertIn("import torchvision failed", message)
        self.assertIn("install/check torchvision", message)
        self.assertIn("ModuleNotFoundError", message)
        self.assertIn("No module named 'torchvision'", message)


class TestComicTextRuntimeUtils(unittest.TestCase):
    def test_legacy_detection_to_bubble_region(self):
        bubble = legacy_detection_to_bubble_region([1, 2, 11, 12, 0.9, 0, 1])

        self.assertEqual(bubble.bbox, (1, 2, 11, 12))
        self.assertTrue(bubble.is_dark)

    def test_convert_legacy_detections_to_bubble_regions(self):
        bubbles = convert_legacy_detections_to_bubble_regions(
            [
                [1, 2, 11, 12, 0.9, 0, 0],
                [5, 6, 15, 16, 0.8, 1, 1],
            ]
        )

        self.assertEqual(len(bubbles), 2)
        self.assertEqual(bubbles[1].bbox, (5, 6, 15, 16))
        self.assertTrue(bubbles[1].is_dark)

    def test_clamp_bbox_to_image(self):
        self.assertEqual(
            clamp_bbox_to_image((-5, 3, 120, 80), (50, 100, 3)),
            (0, 3, 100, 50),
        )

    def test_union_text_regions_bbox(self):
        bbox = union_text_regions_bbox(
            [
                TextRegion(bbox=(10, 10, 20, 20)),
                TextRegion(bbox=(30, 25, 40, 35)),
            ],
            (60, 80, 3),
            padding=5,
        )

        self.assertEqual(bbox, (5, 5, 45, 40))

    def test_crop_bbox_shape(self):
        image = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]

        cropped = crop_bbox(image, (1, 1, 3, 3))

        self.assertEqual(cropped, [[5, 6], [9, 10]])


class TestComicTextDetectorAdapter(unittest.TestCase):
    def test_detect_page_regions_uses_fake_text_regions(self):
        detector = FakeComicTextDetector(
            [TextRegion(bbox=(12, 12, 18, 18), text="hello")]
        )

        result = detector.detect_page_regions(image=object(), bubbles=None)

        self.assertEqual(result.method, "comic_text_detector")
        self.assertEqual(result.bubbles, [])
        self.assertEqual(len(result.text_regions), 1)
        self.assertIsNone(result.text_regions[0].bubble_id)

    def test_detect_page_regions_assigns_bubble_id_when_requested(self):
        detector = FakeComicTextDetector(
            [TextRegion(bbox=(12, 12, 18, 18), text="inside")]
        )
        bubbles = [BubbleRegion(bbox=(10, 10, 20, 20))]

        result = detector.detect_page_regions(
            image=object(),
            bubbles=bubbles,
            assign_to_bubbles=True,
        )

        self.assertEqual(result.text_regions[0].bubble_id, 0)

    def test_detect_text_regions_in_rois_maps_back_to_page_coordinates(self):
        detector = FakeComicTextDetector(
            [TextRegion(bbox=(2, 3, 8, 9), text="roi-local")]
        )
        layout_rois = [
            types.SimpleNamespace(bbox=(10, 20, 20, 30), reading_order=0),
        ]

        mapped_regions = detector.detect_text_regions_in_rois(FakeImage(), layout_rois)

        self.assertEqual(len(mapped_regions), 1)
        self.assertEqual(mapped_regions[0].bbox, (12, 23, 18, 29))
        self.assertEqual(mapped_regions[0].reading_order, 0)

    def test_load_calls_weight_helper(self):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        )
        fake_backend = object()

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch(
                "detectors.comic_text_detector._check_runtime_dependencies"
            ) as mock_check:
                with patch(
                    "detectors.comic_text_detector.ensure_comic_text_detector_weights",
                    return_value={
                        "model_dir": Path("model/comic_text_detector"),
                        "yolo_v5": Path("model/comic_text_detector/yolo-v5.safetensors"),
                        "unet": Path("model/comic_text_detector/unet.safetensors"),
                        "dbnet": Path("model/comic_text_detector/dbnet.safetensors"),
                    },
                ) as mock_ensure:
                    with patch(
                        "detectors.comic_text_detector._build_comic_text_backend",
                        return_value=fake_backend,
                    ) as mock_build:
                        detector = ComicTextDetector()
                        detector.load()

        mock_check.assert_called_once()
        mock_ensure.assert_called_once()
        mock_build.assert_called_once()
        self.assertIs(detector._backend, fake_backend)
        self.assertEqual(detector.device, "cpu")

    def test_detect_text_regions_raises_when_backend_is_unavailable(self):
        detector = ComicTextDetector()

        with patch(
            "detectors.comic_text_detector._check_runtime_dependencies"
        ):
            with patch(
                "detectors.comic_text_detector.ensure_comic_text_detector_weights",
                return_value={
                    "model_dir": Path("model/comic_text_detector"),
                    "yolo_v5": Path("model/comic_text_detector/yolo-v5.safetensors"),
                    "unet": Path("model/comic_text_detector/unet.safetensors"),
                    "dbnet": Path("model/comic_text_detector/dbnet.safetensors"),
                },
            ):
                with patch.dict(
                    sys.modules,
                    {
                        "torch": types.SimpleNamespace(
                            cuda=types.SimpleNamespace(is_available=lambda: False)
                        )
                    },
                ):
                    with patch(
                        "detectors.comic_text_detector._build_comic_text_backend",
                        side_effect=ComicTextDetectorUnavailable("backend unavailable"),
                    ):
                        with self.assertRaises(ComicTextDetectorUnavailable):
                            detector.detect_text_regions(object())


if __name__ == "__main__":
    unittest.main()

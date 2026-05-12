import unittest
import importlib
import sys
import types
from unittest.mock import patch


class TestLegacyYoloBubbleAdapter(unittest.TestCase):
    def test_detect_page_regions_preserves_legacy_shape(self):
        fake_detect_bubbles_module = types.ModuleType("detect_bubbles")
        fake_detect_bubbles_module.detect_bubbles = lambda *args, **kwargs: []

        with patch.dict(sys.modules, {"detect_bubbles": fake_detect_bubbles_module}):
            sys.modules.pop("detectors.legacy_yolo_bubble", None)
            legacy_yolo_bubble = importlib.import_module("detectors.legacy_yolo_bubble")
            legacy_yolo_bubble = importlib.reload(legacy_yolo_bubble)

            image = object()
            with patch.object(
                legacy_yolo_bubble,
                "detect_bubbles",
                return_value=[
                    [10.9, 20.1, 30.8, 40.2, 0.95, 0],
                    [1, 2, 3, 4, 0.7, 1, 1],
                ],
            ) as mock_detect_bubbles:
                result = legacy_yolo_bubble.detect_page_regions(
                    "model/model.pt",
                    image,
                    enable_black_bubble=False,
                )

            mock_detect_bubbles.assert_called_once_with(
                "model/model.pt",
                image,
                enable_black_bubble=False,
            )
            self.assertEqual(result.method, "legacy_yolo")
            self.assertEqual(result.text_regions, [])
            self.assertEqual(
                result.to_legacy_detections(),
                [
                    [10, 20, 30, 40, 0.95, 0, 0],
                    [1, 2, 3, 4, 0.7, 1, 1],
                ],
            )


if __name__ == "__main__":
    unittest.main()

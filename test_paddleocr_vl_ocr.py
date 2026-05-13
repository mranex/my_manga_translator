import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from ocr.paddleocr_vl_ocr import (
    PaddleOCRVLError,
    PaddleOCRVLOCR,
    _build_llama_server_command,
    _chat_completion_content,
    _resolve_model_paths,
    clean_paddleocr_vl_output,
)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data


class TestCleanOutput(unittest.TestCase):
    def test_clean_output_removes_labels_and_fences(self):
        raw = "```text\nOCR: Hello   world\n```"

        self.assertEqual(clean_paddleocr_vl_output(raw), "Hello world")

    def test_clean_output_extracts_json_text(self):
        raw = '{"text": "Recognized text:\\nHello\\nworld"}'

        self.assertEqual(clean_paddleocr_vl_output(raw), "Hello\nworld")


class TestPathResolution(unittest.TestCase):
    def test_model_path_resolution_from_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.gguf"
            mmproj_path = Path(temp_dir) / "mmproj.gguf"
            model_path.write_bytes(b"model")
            mmproj_path.write_bytes(b"mmproj")

            with patch.dict(
                os.environ,
                {
                    "PADDLEOCR_VL_MODEL_PATH": str(model_path),
                    "PADDLEOCR_VL_MMPROJ_PATH": str(mmproj_path),
                },
                clear=False,
            ):
                resolved_model, resolved_mmproj = _resolve_model_paths()

        self.assertEqual(resolved_model, model_path)
        self.assertEqual(resolved_mmproj, mmproj_path)

    def test_missing_model_paths_raise_clear_error(self):
        with patch.dict(os.environ, {}, clear=False):
            with self.assertRaises(PaddleOCRVLError) as context:
                _resolve_model_paths("Z:/missing/model.gguf", "Z:/missing/mmproj.gguf")

        self.assertIn("PADDLEOCR_VL_MODEL_PATH", str(context.exception))
        self.assertIn("PADDLEOCR_VL_MMPROJ_PATH", str(context.exception))


class TestCommandAndResponseParsing(unittest.TestCase):
    def test_building_llama_server_command_includes_expected_flags(self):
        command = _build_llama_server_command(
            "llama-server.exe",
            model_path=Path("C:/models/model.gguf"),
            mmproj_path=Path("C:/models/mmproj.gguf"),
            host="127.0.0.1",
            port=8088,
            num_ctx=4096,
            gpu_layers=12,
        )

        self.assertEqual(command[0], "llama-server.exe")
        self.assertIn(str(Path("C:/models/model.gguf")), command)
        self.assertIn(str(Path("C:/models/mmproj.gguf")), command)
        self.assertIn("127.0.0.1", command)
        self.assertIn("8088", command)
        self.assertIn("4096", command)
        self.assertIn("12", command)

    def test_http_response_parser_extracts_chat_content(self):
        parsed = _chat_completion_content(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Recognized text: HELLO WORLD"
                        }
                    }
                ]
            }
        )

        self.assertEqual(parsed, "HELLO WORLD")


class TestRuntimeBehavior(unittest.TestCase):
    @unittest.skipIf(Image is None, "Pillow is not available")
    def test_process_batch_preserves_order(self):
        ocr = PaddleOCRVLOCR(server_url="http://127.0.0.1:8088", auto_start_server=False)
        outputs = iter(["first", "second", "third"])

        with patch.object(ocr, "recognize", side_effect=lambda image: next(outputs)):
            results = ocr.process_batch([object(), object(), object()])

        self.assertEqual(results, ["first", "second", "third"])

    @unittest.skipIf(Image is None, "Pillow is not available")
    def test_server_url_mode_does_not_start_subprocess(self):
        image = Image.new("RGB", (16, 16), color="white")
        fake_requests = types.SimpleNamespace(
            get=lambda url, timeout=3: FakeResponse(status_code=200),
            post=lambda url, json, timeout=120: FakeResponse(
                status_code=200,
                json_data={
                    "choices": [
                        {"message": {"content": "Text: HELLO"}}
                    ]
                },
            ),
        )
        ocr = PaddleOCRVLOCR(
            server_url="http://127.0.0.1:8088",
            auto_start_server=False,
        )

        with patch.dict(sys.modules, {"requests": fake_requests}):
            with patch("ocr.paddleocr_vl_ocr.subprocess.Popen") as mock_popen:
                result = ocr.recognize(image)

        self.assertEqual(result, "HELLO")
        mock_popen.assert_not_called()

    @unittest.skipIf(Image is None, "Pillow is not available")
    def test_auto_start_server_false_without_live_server_raises_clear_error(self):
        image = Image.new("RGB", (16, 16), color="white")
        fake_requests = types.SimpleNamespace(
            get=lambda url, timeout=3: (_ for _ in ()).throw(RuntimeError("down")),
            post=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not POST")),
        )
        ocr = PaddleOCRVLOCR(
            server_url="http://127.0.0.1:8088",
            auto_start_server=False,
        )

        with patch.dict(sys.modules, {"requests": fake_requests}):
            with self.assertRaises(PaddleOCRVLError) as context:
                ocr.recognize(image)

        self.assertIn("PADDLEOCR_VL_SERVER_URL", str(context.exception))


if __name__ == "__main__":
    unittest.main()

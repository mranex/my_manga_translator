import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from inpainting.lama_manga import (
    LamaMangaInpainter,
    LamaMangaUnavailable,
    _check_runtime_dependencies,
    ensure_lama_manga_weights,
)


class TestLamaMangaWeights(unittest.TestCase):
    def test_ensure_weights_downloads_missing_file(self):
        download_calls = []

        def fake_hf_hub_download(repo_id, filename, local_dir, local_dir_use_symlinks):
            local_path = Path(local_dir) / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"lama")
            download_calls.append((repo_id, filename, local_dir_use_symlinks))
            return str(local_path)

        fake_hf_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf_module}):
                model_path = ensure_lama_manga_weights(model_dir=temp_dir)

        self.assertEqual(model_path.name, "lama-manga.safetensors")
        self.assertEqual(len(download_calls), 1)
        self.assertEqual(download_calls[0][0], "mayocream/lama-manga")

    def test_ensure_weights_only_imports_huggingface_hub(self):
        attempted_imports = []

        def fake_hf_hub_download(repo_id, filename, local_dir, local_dir_use_symlinks):
            local_path = Path(local_dir) / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"lama")
            return str(local_path)

        fake_hf_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)

        def fake_import_module(name):
            attempted_imports.append(name)
            if name == "huggingface_hub":
                return fake_hf_module
            raise AssertionError(f"Unexpected import: {name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "inpainting.lama_manga.importlib.import_module",
                side_effect=fake_import_module,
            ):
                ensure_lama_manga_weights(model_dir=temp_dir)

        self.assertEqual(attempted_imports, ["huggingface_hub"])

    def test_ensure_weights_skips_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "lama-manga.safetensors"
            model_path.write_bytes(b"ready")

            with patch(
                "inpainting.lama_manga.importlib.import_module"
            ) as mock_import_module:
                loaded_path = ensure_lama_manga_weights(model_dir=temp_dir)

        self.assertEqual(loaded_path, model_path)
        self.assertFalse(mock_import_module.called)


class TestLamaMangaLoader(unittest.TestCase):
    def test_inpainter_is_lazy_until_load(self):
        inpainter = LamaMangaInpainter()

        self.assertIsNone(inpainter._model)

    def test_runtime_dependency_error_reports_exact_module(self):
        def fake_import_module(name):
            if name == "torch":
                raise ModuleNotFoundError("No module named 'torch'")
            return types.SimpleNamespace()

        with patch(
            "inpainting.lama_manga.importlib.import_module",
            side_effect=fake_import_module,
        ):
            with self.assertRaises(LamaMangaUnavailable) as ctx:
                _check_runtime_dependencies()

        message = str(ctx.exception)
        self.assertIn("import torch failed", message)
        self.assertIn("install/check torch", message)
        self.assertIn("ModuleNotFoundError", message)
        self.assertIn("No module named 'torch'", message)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock modules that might not be available or heavy
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['deep_translator'] = MagicMock()
sys.modules['translators'] = MagicMock()
sys.modules['manga_ocr'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

# Now import the class to test
# We need to make sure we can import it even if dependencies are mocked
from translator.translator import MangaTranslator

class TestMangaTranslatorBatch(unittest.TestCase):
    def setUp(self):
        self.translator = MangaTranslator()
        # Mock internal components
        self.translator._model_cache = {
            "nllb_model": MagicMock(),
            "nllb_tokenizer": MagicMock(),
            "hf_pipeline": None
        }
        self.translator._nllb_lock = MagicMock()
        self.translator._nllb_lock.__enter__ = MagicMock()
        self.translator._nllb_lock.__exit__ = MagicMock()

    def test_translate_batch_nllb(self):
        texts = ["Hello", "World"]

        # Mock tokenizer behavior
        tokenizer = self.translator._model_cache["nllb_tokenizer"]
        tokenizer.return_value = {"input_ids": "mock_ids"}
        tokenizer.convert_tokens_to_ids.return_value = 123
        tokenizer.batch_decode.return_value = ["Bonjour", "Monde"]

        # Mock model behavior
        model = self.translator._model_cache["nllb_model"]
        model.generate.return_value = "mock_tokens"

        # Call the method
        results = self.translator.translate_batch(texts, method="nllb")

        # Verify results
        self.assertEqual(results, ["Bonjour", "Monde"])

        # Verify tokenizer called with list
        tokenizer.assert_called_with(texts, return_tensors="pt", padding=True, truncation=True)

        # Verify model.generate called
        model.generate.assert_called()

    def test_translate_batch_fallback(self):
        texts = ["Hello", "World"]

        # Mock translate method to verify fallback
        with patch.object(self.translator, 'translate') as mock_translate:
            mock_translate.side_effect = ["Bonjour", "Monde"]

            # Call with a method that doesn't support batching (e.g. google)
            results = self.translator.translate_batch(texts, method="google")

            # Verify results
            self.assertEqual(results, ["Bonjour", "Monde"])

            # Verify translate called for each text
            self.assertEqual(mock_translate.call_count, 2)
            mock_translate.assert_any_call("Hello", "google")
            mock_translate.assert_any_call("World", "google")

    def test_translate_batch_nllb_exception_fallback(self):
        texts = ["Hello", "World"]

        # Force exception in batch processing
        self.translator._model_cache["nllb_tokenizer"].side_effect = Exception("Batch error")

        # Mock translate method to verify fallback (via _translate_with_nllb calling translate? No, _translate_with_nllb is separate)
        # But _translate_batch_with_nllb falls back to _translate_with_nllb loop

        with patch.object(self.translator, '_translate_with_nllb') as mock_single_nllb:
            mock_single_nllb.side_effect = ["Bonjour", "Monde"]

            results = self.translator.translate_batch(texts, method="nllb")

            self.assertEqual(results, ["Bonjour", "Monde"])
            self.assertEqual(mock_single_nllb.call_count, 2)

if __name__ == '__main__':
    unittest.main()

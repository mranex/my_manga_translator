from deep_translator import GoogleTranslator
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import translators as ts
import torch
import threading


class MangaTranslator:
    # NLLB language codes mapping
    NLLB_LANG_CODES = {
        "ja": "jpn_Jpan",  # Japanese
        "en": "eng_Latn",  # English
        "vi": "vie_Latn",  # Vietnamese
        "zh": "zho_Hans",  # Chinese Simplified
        "ko": "kor_Hang",  # Korean
        "th": "tha_Thai",  # Thai
        "id": "ind_Latn",  # Indonesian
        "fr": "fra_Latn",  # French
        "de": "deu_Latn",  # German
        "es": "spa_Latn",  # Spanish
        "ru": "rus_Cyrl",  # Russian
    }
    
    def __init__(self, source="ja", target="en", gemini_api_key=None):
        self.target = target
        self.source = source
        self.gemini_api_key = gemini_api_key
        self.translators = {
            "google": self._translate_with_google,
            "hf": self._translate_with_hf,
            "baidu": self._translate_with_baidu,
            "bing": self._translate_with_bing,
            "nllb": self._translate_with_nllb,
            "gemini": self._translate_with_gemini,
            "deepseek": self._translate_with_deepseek
        }
        # Lazy loading for heavy models
        # self._nllb_model and self._nllb_tokenizer are now cached in _model_cache
        self._gemini_translator = None
        self._deepseek_translator = None

    # Class-level cache for heavy models (shared across instances)
    _model_cache = {
        "nllb_model": None,
        "nllb_tokenizer": None,
        "hf_pipeline": None
    }
    _nllb_lock = threading.Lock()

    def set_languages(self, source=None, target=None):
        """Update source and/or target languages."""
        if source:
            self.source = source
        if target:
            self.target = target

    def translate(self, text, method="google"):
        """
        Translates the given text to the target language using the specified method.

        Args:
            text (str): The text to be translated.
            method (str):"google" for Google Translator, 
                         "hf" for Helsinki-NLP's opus-mt-ja-en model (HF pipeline)
                         "baidu" for Baidu Translate
                         "bing" for Microsoft Bing Translator
                         "nllb" for Meta's NLLB-200 model (offline, 200+ languages)

        Returns:
            str: The translated text.
        """
        translator_func = self.translators.get(method)
        
        if translator_func:
            return translator_func(self._preprocess_text(text))
        else:
            raise ValueError("Invalid translation method.")

    def translate_batch(self, texts: list, method="google") -> list:
        """
        Translates a list of texts using batch processing if available.
        Falls back to sequential translation if batching is not supported for the method.
        """
        if not texts:
            return []

        if method == "nllb":
            return self._translate_batch_with_nllb(texts)

        # Fallback to sequential translation
        return [self.translate(text, method) for text in texts]
            
    def _translate_with_google(self, text):
        translator = GoogleTranslator(source=self.source, target=self.target)
        translated_text = translator.translate(text)
        return translated_text if translated_text is not None else text

    def _translate_with_hf(self, text):
        # Lazy load HF pipeline (cache it like NLLB)
        if self._model_cache["hf_pipeline"] is None:
            print("Loading HuggingFace translation model (first time)...")
            self._model_cache["hf_pipeline"] = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
            print("HF pipeline loaded and cached!")
        
        translated_text = self._model_cache["hf_pipeline"](text)[0]["translation_text"]
        return translated_text if translated_text is not None else text

    def _translate_with_baidu(self, text):
        # Map language codes to Baidu format
        baidu_lang_map = {"ja": "jp", "zh": "zh", "ko": "kor", "en": "en", "vi": "vie"}
        src_lang = baidu_lang_map.get(self.source, self.source)
        translated_text = ts.translate_text(text, translator="baidu",
                                            from_language=src_lang, 
                                            to_language=self.target)
        return translated_text if translated_text is not None else text

    def _translate_with_bing(self, text):
        translated_text = ts.translate_text(text, translator="bing",
                                            from_language=self.source, 
                                            to_language=self.target)
        return translated_text if translated_text is not None else text

    def _load_nllb_model(self):
        """Lazy load NLLB model only when first needed (saves memory)"""
        if self._model_cache["nllb_model"] is None:
            print("Loading NLLB model (first time, may take a moment)...")
            model_name = "facebook/nllb-200-distilled-600M"
            self._model_cache["nllb_tokenizer"] = AutoTokenizer.from_pretrained(model_name)
            self._model_cache["nllb_model"] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            # Force CPU for stability
            self._model_cache["nllb_model"] = self._model_cache["nllb_model"].to("cpu")
            self._model_cache["nllb_model"].eval()
            print("NLLB model loaded successfully!")

    def _translate_with_nllb(self, text):
        """
        Translate using Meta's NLLB-200 model.
        Supports 200+ languages, works offline, optimized for CPU.
        """
        try:
            self._load_nllb_model()
            
            # Get NLLB language codes
            src_lang = self.NLLB_LANG_CODES.get(self.source, "jpn_Jpan")
            tgt_lang = self.NLLB_LANG_CODES.get(self.target, "eng_Latn")
            
            tokenizer = self._model_cache["nllb_tokenizer"]
            model = self._model_cache["nllb_model"]
            
            # Set source language and tokenize (thread-safe)
            with self._nllb_lock:
                tokenizer.src_lang = src_lang
                inputs = tokenizer(text, return_tensors="pt", padding=True)
            
            # Generate translation
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=256
                )
            
            # Decode
            translated_text = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]
            
            return translated_text if translated_text else text
            
        except Exception as e:
            print(f"NLLB translation error: {e}")
            return text

    def _translate_batch_with_nllb(self, texts):
        """
        Batch translation using NLLB model.
        Significantly faster for multiple bubbles.
        """
        try:
            self._load_nllb_model()

            src_lang = self.NLLB_LANG_CODES.get(self.source, "jpn_Jpan")
            tgt_lang = self.NLLB_LANG_CODES.get(self.target, "eng_Latn")

            tokenizer = self._model_cache["nllb_tokenizer"]
            model = self._model_cache["nllb_model"]

            # Preprocess all texts
            preprocessed_texts = [self._preprocess_text(text) for text in texts]

            # Set source language and tokenize batch (thread-safe)
            with self._nllb_lock:
                tokenizer.src_lang = src_lang
                # padding=True pads to the longest sequence in the batch
                # truncation=True ensures we don't exceed model limits
                inputs = tokenizer(preprocessed_texts, return_tensors="pt", padding=True, truncation=True)

            # Generate translations
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=256
                )

            # Decode batch
            translated_texts = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )

            return translated_texts

        except Exception as e:
            print(f"NLLB batch translation error: {e}, falling back to sequential")
            return [self._translate_with_nllb(t) for t in texts]

    def _translate_with_gemini(self, text):
        """
        Translate using Google Gemini 2.5 Flash-Lite.
        For batch processing, use GeminiTranslator directly.
        """
        try:
            if self._gemini_translator is None:
                from .gemini_translator import GeminiTranslator
                api_key = getattr(self, '_gemini_api_key', None) or self.gemini_api_key
                if not api_key:
                    raise ValueError("Gemini API key required. Please enter it in the web form.")
                custom_prompt = getattr(self, '_gemini_custom_prompt', None)
                self._gemini_translator = GeminiTranslator(
                    api_key=api_key, 
                    custom_prompt=custom_prompt
                )
                print(f"Gemini translator initialized! (source={self.source}, target={self.target})")
            
            return self._gemini_translator.translate_single(
                text, 
                source=self.source, 
                target=self.target
            )
        except Exception as e:
            print(f"Gemini translation error: {e}")
            return text

    def _preprocess_text(self, text):
        preprocessed_text = text.replace("．", ".")
        return preprocessed_text
        
    def _translate_with_deepseek(self, text):
        """
        Translate using DeepSeek.
        For batch/multi-page processing, use DeepSeekTranslator directly.
        """
        try:
            if self._deepseek_translator is None:
                from .deepseek_translator import DeepSeekTranslator

                api_key = getattr(self, "_deepseek_api_key", None)
                model = getattr(self, "_deepseek_model", "deepseek-v4-flash")
                custom_prompt = getattr(self, "_deepseek_custom_prompt", None)
                thinking = getattr(self, "_deepseek_thinking", False)

                self._deepseek_translator = DeepSeekTranslator(
                    api_key=api_key,
                    model=model,
                    custom_prompt=custom_prompt,
                    thinking=thinking,
                )

                print(f"DeepSeek translator initialized! model={model}")

            return self._deepseek_translator.translate_single(
                text,
                source=self.source,
                target=self.target,
            )

        except Exception as e:
            print(f"DeepSeek translation error: {e}")
            return text

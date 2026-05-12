"""
Gemini Translator with Batch Processing

Migrated from:
    google.generativeai

To:
    google-genai
    from google import genai
    from google.genai import types

This class keeps the same public interface as the original Manga-Translator
GeminiTranslator, so app.py should not need major changes.
"""

import json
import os
import re
import time
from typing import Dict, List, Optional, TYPE_CHECKING, Any

from google import genai
from google.genai import types

from .base import BaseTranslator

if TYPE_CHECKING:
    from .context_memory import ContextMemory


MAX_RETRIES = 3
RETRY_DELAY_BASE = 0.5


class GeminiTranslator(BaseTranslator):
    """
    Translator using Gemini via the new google-genai SDK.

    Original repo used:
        import google.generativeai as genai
        genai.configure(api_key=...)
        genai.GenerativeModel(...).generate_content(...)

    New SDK uses:
        from google import genai
        client = genai.Client(api_key=...)
        client.models.generate_content(...)
    """

    def __init__(
        self,
        api_key: str = None,
        custom_prompt: str = None,
        style: str = "default",
        model_name: str = None,
        temperature: float = 0.35,
        max_output_tokens: int = 8192,
    ):
        """
        Args:
            api_key:
                Gemini API key. If None, reads GEMINI_API_KEY or GOOGLE_API_KEY.

            custom_prompt:
                Custom instructions for translation style.

            style:
                Preset style name from STYLE_PRESETS, handled by BaseTranslator.

            model_name:
                Gemini model name. Defaults to env GEMINI_MODEL or gemini-2.5-flash-lite.

            temperature:
                Lower = more stable JSON / less creative drift.

            max_output_tokens:
                Enough for multi-page batch translation.
        """
        super().__init__(custom_prompt=custom_prompt, style=style)

        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY / GOOGLE_API_KEY or pass api_key."
            )

        self.model_name = (
            model_name
            or os.environ.get("GEMINI_MODEL")
            or "gemma-4-31b-it"
        )

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        self.client = genai.Client(api_key=self.api_key)

    # -------------------------------------------------------------------------
    # Public methods expected by Manga-Translator
    # -------------------------------------------------------------------------

    def translate_single(
        self,
        text: str,
        source: str = "ja",
        target: str = "en",
        custom_prompt: str = None,
    ) -> str:
        """
        Translate a single text string.

        Args:
            text:
                Text to translate.

            source:
                Source language code.

            target:
                Target language code.

            custom_prompt:
                Override custom prompt for this call.

        Returns:
            Translated text.
        """
        if not text or not text.strip():
            return text

        source_name = self.LANG_NAMES.get(source, source)
        target_name = self.LANG_NAMES.get(target, target)
        style_text = self._style_text(custom_prompt)

        system_instruction = self._build_system_instruction(
            source_name=source_name,
            target_name=target_name,
            style_text=style_text,
            json_mode=False,
            context_section="",
            multi_page=False,
        )

        prompt = f"""
Original text:
{text.strip()}

Return ONLY the translated text.
No explanations.
No quotes.
No markdown.
""".strip()

        try:
            response_text = self._generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                json_mode=False,
            )
            return response_text.strip() or text

        except Exception as e:
            print(f"Gemini translation error: {e}")
            return text

    def translate_batch(
        self,
        texts: List[str],
        source: str = "ja",
        target: str = "en",
        custom_prompt: str = None,
    ) -> List[str]:
        """
        Translate multiple bubble texts in one API call.
        Keeps empty strings in their original positions.
        """
        if not texts:
            return []

        indexed_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        if not indexed_texts:
            return texts

        texts_to_translate = [t for _, t in indexed_texts]

        translations = self._translate_batch_internal(
            texts_to_translate=texts_to_translate,
            source=source,
            target=target,
            custom_prompt=custom_prompt,
        )

        result = list(texts)

        for (orig_idx, _), trans in zip(indexed_texts, translations):
            result[orig_idx] = trans

        return result

    def translate_pages_batch(
        self,
        pages_texts: Dict[str, List[str]],
        source: str = "ja",
        target: str = "en",
        custom_prompt: str = None,
        context_memory: "ContextMemory" = None,
    ) -> Dict[str, List[str]]:
        """
        Translate texts from multiple pages in a single API call.

        Args:
            pages_texts:
                Dict mapping page names to list of bubble texts.

            source:
                Source language code.

            target:
                Target language code.

            custom_prompt:
                Override custom prompt for this call.

            context_memory:
                Optional ContextMemory object for consistent translation.

        Returns:
            Dict with the same page names and bubble order, but translated.
        """
        if not pages_texts:
            return {}

        source_name = self.LANG_NAMES.get(source, source)
        target_name = self.LANG_NAMES.get(target, target)
        style_text = self._style_text(custom_prompt)

        context_section = ""
        if context_memory:
            try:
                context_section = context_memory.generate_context_prompt()
            except Exception as e:
                print(f"Gemini context memory error: {e}")
                context_section = ""

        system_instruction = self._build_system_instruction(
            source_name=source_name,
            target_name=target_name,
            style_text=style_text,
            json_mode=True,
            context_section=context_section,
            multi_page=True,
        )

        prompt = f"""
Input JSON object.
Keys are page names.
Values are arrays of manga/comic bubble texts.

{json.dumps(pages_texts, ensure_ascii=False, indent=2)}

Return ONLY valid JSON with this exact structure:
{{
  "pages": {{
    "page_name": ["translation 1", "translation 2"]
  }}
}}

Rules:
- Keep exactly the same page names.
- Keep exactly the same number of bubbles per page.
- Keep bubble order unchanged.
- Do not add explanations.
- Do not use markdown.
""".strip()

        for attempt in range(MAX_RETRIES):
            try:
                response_text = self._generate_text(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    json_mode=True,
                )

                parsed = self._parse_json_object(response_text)

                if isinstance(parsed, dict) and "pages" in parsed:
                    translated_pages = parsed["pages"]
                else:
                    translated_pages = parsed

                self._validate_pages_result(
                    original=pages_texts,
                    translated=translated_pages,
                )

                return translated_pages

            except Exception as e:
                error_str = str(e)
                print(f"Gemini pages batch attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")

                if self._is_quota_error(error_str):
                    print("⚠️ Gemini quota/rate limit hit. Returning original texts.")
                    return pages_texts

                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print("Gemini pages batch failed. Falling back to page-by-page translation.")

        result = {}

        for page_name, texts in pages_texts.items():
            result[page_name] = self.translate_batch(
                texts,
                source=source,
                target=target,
                custom_prompt=custom_prompt,
            )

        return result

    # -------------------------------------------------------------------------
    # Internal batch translation
    # -------------------------------------------------------------------------

    def _translate_batch_internal(
        self,
        texts_to_translate: List[str],
        source: str,
        target: str,
        custom_prompt: str = None,
    ) -> List[str]:
        source_name = self.LANG_NAMES.get(source, source)
        target_name = self.LANG_NAMES.get(target, target)
        style_text = self._style_text(custom_prompt)

        system_instruction = self._build_system_instruction(
            source_name=source_name,
            target_name=target_name,
            style_text=style_text,
            json_mode=True,
            context_section="",
            multi_page=False,
        )

        prompt = f"""
Input JSON array.
Each item is one manga/comic speech bubble.

{json.dumps(texts_to_translate, ensure_ascii=False)}

Return ONLY valid JSON with this exact structure:
{{
  "translations": ["translation 1", "translation 2"]
}}

Rules:
- The "translations" array must have exactly {len(texts_to_translate)} items.
- Preserve the original order.
- Do not add explanations.
- Do not use markdown.
""".strip()

        for attempt in range(MAX_RETRIES):
            try:
                response_text = self._generate_text(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    json_mode=True,
                )

                parsed = self._parse_json_object(response_text)

                if isinstance(parsed, dict) and "translations" in parsed:
                    translations = parsed["translations"]
                elif isinstance(parsed, list):
                    translations = parsed
                else:
                    raise ValueError(f"Unexpected Gemini JSON shape: {parsed}")

                if len(translations) != len(texts_to_translate):
                    raise ValueError(
                        f"Expected {len(texts_to_translate)} translations, got {len(translations)}"
                    )

                return [str(item).strip() for item in translations]

            except Exception as e:
                error_str = str(e)
                print(f"Gemini batch attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")

                if self._is_quota_error(error_str):
                    print("⚠️ Gemini quota/rate limit hit. Returning original texts.")
                    return texts_to_translate

                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print("All Gemini batch retries failed. Falling back to single translations.")
                    return [
                        self.translate_single(t, source, target, custom_prompt)
                        for t in texts_to_translate
                    ]

        return texts_to_translate

    # -------------------------------------------------------------------------
    # Gemini API wrapper
    # -------------------------------------------------------------------------

    def _generate_text(
        self,
        prompt: str,
        system_instruction: str,
        json_mode: bool = False,
    ) -> str:
        """
        Wrapper around google-genai generate_content.

        The new SDK supports GenerateContentConfig, including system_instruction,
        temperature, max_output_tokens, and response_mime_type.
        """

        config_kwargs = {
            "system_instruction": system_instruction,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }

        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        text = self._response_to_text(response)

        if not text:
            raise ValueError("Gemini returned empty response text.")

        return text.strip()

    def _response_to_text(self, response: Any) -> str:
        """
        response.text is the normal path in google-genai.
        This fallback handles weird blocked/empty candidate cases more gracefully.
        """
        try:
            if getattr(response, "text", None):
                return response.text
        except Exception:
            pass

        try:
            candidates = getattr(response, "candidates", None) or []

            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content else None

                if not parts:
                    continue

                collected = []

                for part in parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        collected.append(part_text)

                if collected:
                    return "".join(collected)

        except Exception:
            pass

        return ""

    # -------------------------------------------------------------------------
    # Prompt helpers
    # -------------------------------------------------------------------------

    def _style_text(self, custom_prompt: str = None) -> str:
        style = custom_prompt or self.custom_prompt

        if not style:
            return ""

        return f"\nAdditional style instructions:\n{style.strip()}"

    def _build_system_instruction(
        self,
        source_name: str,
        target_name: str,
        style_text: str = "",
        json_mode: bool = False,
        context_section: str = "",
        multi_page: bool = False,
    ) -> str:
        consecutive_text = ""
        if multi_page:
            consecutive_text = """
These are consecutive manga/comic pages from the same story.
Keep story context, speaker relationships, pronouns, names, and character voices consistent.
""".strip()

        json_text = ""
        if json_mode:
            json_text = """
Return valid JSON only.
Do not wrap the JSON in markdown fences.
Do not add comments or explanations outside JSON.
""".strip()

        return f"""
You are an expert manga/comic translator specializing in {source_name} to {target_name} translation.

{consecutive_text}

{context_section}

Translation rules:
- Translate as spoken dialogue, not formal written prose.
- Do not translate word-by-word. Rewrite naturally in {target_name}.
- Preserve tone, emotion, personality, jokes, insults, flirting, sarcasm, and intimacy.
- Do not censor, moralize, summarize, or soften the original meaning.
- Keep short/punchy lines short and punchy.
- Keep translations concise enough to fit inside manga speech bubbles.
- Preserve character names unless there is a clear reason not to.
- Preserve honorifics or relationship terms when they matter.
- For Vietnamese, choose natural pronouns based on context and keep them consistent.
- Avoid stiff textbook wording.
- Avoid unnecessary Hán-Việt phrasing unless it fits the character/setting.
- Do not include OCR/source markup such as <b>, </b>, <i>, </i>, <ruby>, HTML tags, or markdown formatting.

Vietnamese guidance:
- Close friends/rivals: tao/mày, tớ/cậu, mình/cậu depending on tone.
- Polite/neutral: tôi/anh/chị/cậu/bạn depending on relationship.
- Romantic: anh/em, em/anh, mình/cậu depending on dynamic.
- Family: con/bố/mẹ/ông/bà/anh/chị/em.
- Keep slang natural, not forced.
- If the source is English comic dialogue, translate meaning and attitude, not literal grammar.
- If the source is Japanese/Korean/Chinese dialogue, preserve cultural relationship cues naturally.

{style_text}

{json_text}
""".strip()

    # -------------------------------------------------------------------------
    # JSON helpers
    # -------------------------------------------------------------------------

    def _parse_json_object(self, text: str) -> Any:
        cleaned = self._strip_code_fence(text)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            extracted = self._extract_first_json(cleaned)
            return json.loads(extracted)

    def _strip_code_fence(self, text: str) -> str:
        text = text.strip()

        if text.startswith("```json"):
            text = text[7:].strip()
        elif text.startswith("```"):
            text = text[3:].strip()

        if text.endswith("```"):
            text = text[:-3].strip()

        return text

    def _extract_first_json(self, text: str) -> str:
        """
        Fallback extractor if the model leaks small text around JSON.
        Supports object or array JSON.
        """
        text = text.strip()

        object_start = text.find("{")
        array_start = text.find("[")

        starts = [idx for idx in [object_start, array_start] if idx != -1]

        if not starts:
            raise ValueError(f"No JSON object/array found in Gemini response: {text[:500]}")

        start = min(starts)

        if text[start] == "{":
            end = text.rfind("}")
        else:
            end = text.rfind("]")

        if end == -1 or end <= start:
            raise ValueError(f"Could not extract valid JSON from Gemini response: {text[:500]}")

        return text[start : end + 1].strip()

    def _validate_pages_result(
        self,
        original: Dict[str, List[str]],
        translated: Dict[str, List[str]],
    ) -> None:
        if not isinstance(translated, dict):
            raise ValueError("Translated pages result is not a dict.")

        for page_name, source_lines in original.items():
            if page_name not in translated:
                raise ValueError(f"Missing page in Gemini response: {page_name}")

            translated_lines = translated[page_name]

            if not isinstance(translated_lines, list):
                raise ValueError(f"Page {page_name} is not a list.")

            if len(translated_lines) != len(source_lines):
                raise ValueError(
                    f"Page {page_name}: expected {len(source_lines)} lines, "
                    f"got {len(translated_lines)}"
                )

    # -------------------------------------------------------------------------
    # Error helpers
    # -------------------------------------------------------------------------

    def _is_quota_error(self, error_str: str) -> bool:
        error_str = error_str.lower()

        quota_markers = [
            "429",
            "quota",
            "rate limit",
            "resource_exhausted",
            "too many requests",
        ]

        return any(marker in error_str for marker in quota_markers)

    # -------------------------------------------------------------------------
    # Optional cleanup
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """
        New google-genai clients can be closed explicitly.
        Safe to call, but the app does not have to call it.
        """
        try:
            self.client.close()
        except Exception:
            pass
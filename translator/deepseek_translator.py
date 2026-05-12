"""
DeepSeek Translator with Batch Processing

Uses DeepSeek's OpenAI-compatible Chat Completions API.

Design goals:
- Mirror GeminiTranslator / LocalLLMTranslator style from this project.
- Support single text, bubble batch, and multi-page batch.
- No line-by-line fallback for page batch.
- If multi-page batch fails, fallback to per-page batch.
- If a page still returns the wrong number of lines, pad/truncate to preserve pipeline shape.
"""

import json
import os
import time
from typing import List, Dict, Optional, TYPE_CHECKING

import requests

from .base import BaseTranslator

if TYPE_CHECKING:
    from .context_memory import ContextMemory


# Constants for retry logic
MAX_RETRIES = 3
RETRY_DELAY_BASE = 0.8


class DeepSeekTranslator(BaseTranslator):
    """
    Translator using DeepSeek's OpenAI-compatible API.

    Official OpenAI-compatible base URL:
        https://api.deepseek.com

    Chat completions endpoint:
        https://api.deepseek.com/chat/completions
    """

    MODELS = [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        # Legacy compatibility names. Avoid for new config.
        "deepseek-chat",
        "deepseek-reasoner",
    ]

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        custom_prompt: str = None,
        style: str = "default",
        thinking: bool = False,
        reasoning_effort: str = "low",
    ):
        """
        Initialize DeepSeek translator.

        Args:
            api_key: DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env var.
            model: DeepSeek model name.
            base_url: DeepSeek base URL.
            custom_prompt: Custom translation style instructions.
            style: Preset style name from STYLE_PRESETS.
            thinking: Enable DeepSeek thinking mode.
            reasoning_effort: low / medium / high. Only meaningful when thinking=True.
        """
        super().__init__(custom_prompt=custom_prompt, style=style)

        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY or pass api_key."
            )

        self.model = model or "deepseek-v4-flash"
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort or "low"

    # -------------------------------------------------------------------------
    # Low-level API helpers
    # -------------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        *,
        json_mode: bool = False,
        max_tokens: int = 8192,
    ) -> Dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
        }

        # DeepSeek V4 supports thinking mode.
        # For manga translation, non-thinking is usually better:
        # faster, cheaper, more stable JSON.
        payload["thinking"] = {
            "type": "enabled" if self.thinking else "disabled"
        }

        if self.thinking:
            payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["temperature"] = 0.3

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        return payload

    def _post(
        self,
        messages: List[Dict[str, str]],
        *,
        json_mode: bool = False,
        timeout: int = 120,
        max_tokens: int = 8192,
    ) -> str:
        payload = self._build_payload(
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )

        response = requests.post(
            self.endpoint,
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"].get("content", "")
        return (content or "").strip()

    def _clean_response_text(self, text: str) -> str:
        text = (text or "").strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def _parse_json(self, text: str):
        text = self._clean_response_text(text)
        return json.loads(text)

    def _retry_delay(self, attempt: int):
        delay = RETRY_DELAY_BASE * (2 ** attempt)
        print(f"Retrying in {delay:.1f}s...")
        time.sleep(delay)

    # -------------------------------------------------------------------------
    # Shape repair helpers
    # -------------------------------------------------------------------------

    def _repair_list_length(
        self,
        translations: List[str],
        originals: List[str],
        *,
        label: str = "batch",
    ) -> List[str]:
        """
        Preserve pipeline shape.

        If DeepSeek returns fewer/more items than expected, do NOT call line-by-line.
        Instead:
        - Pad missing items with original text.
        - Truncate extra items.

        This keeps bubble count stable and avoids the app hanging on many tiny calls.
        """
        if not isinstance(translations, list):
            print(f"Warning: {label} returned non-list result. Using original texts.")
            return list(originals)

        expected = len(originals)
        got = len(translations)

        if got != expected:
            print(
                f"Warning: {label}: expected {expected} translations, got {got}. "
                "Padding/truncating to preserve shape."
            )

        fixed = list(translations[:expected])

        while len(fixed) < expected:
            fixed.append(originals[len(fixed)])

        return fixed

    def _validate_pages_shape(
        self,
        translated_pages: Dict[str, List[str]],
        original_pages: Dict[str, List[str]],
    ):
        """
        Validate that translated_pages has the same page keys and line counts.
        Raise ValueError if anything is wrong.
        """
        if not isinstance(translated_pages, dict):
            raise ValueError("Translated pages result is not a dict")

        for page_name, original_lines in original_pages.items():
            if page_name not in translated_pages:
                raise ValueError(f"Missing page: {page_name}")

            translated_lines = translated_pages[page_name]

            if not isinstance(translated_lines, list):
                raise ValueError(f"{page_name}: translated value is not a list")

            if len(translated_lines) != len(original_lines):
                raise ValueError(
                    f"{page_name}: expected {len(original_lines)} lines, "
                    f"got {len(translated_lines)}"
                )

    # -------------------------------------------------------------------------
    # Prompt builders
    # -------------------------------------------------------------------------

    def _style_text(self, custom_prompt: Optional[str] = None) -> str:
        style = custom_prompt or self.custom_prompt

        if style:
            return f"\nStyle instructions: {style}"

        # Use BaseTranslator preset style if available.
        return self._build_style_instructions()

    def _single_prompt(
        self,
        text: str,
        source: str,
        target: str,
        custom_prompt: Optional[str] = None,
    ) -> str:
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        style_text = self._style_text(custom_prompt)

        return f"""You are an expert manga/comic translator specializing in {source_name} to {target_name} translation.

Translation Guidelines:
- Translate for SPOKEN dialogue, not written prose. It should sound natural when read aloud.
- Preserve the character's tone, emotion, and personality through word choice.
- Use natural sentence structures in {target_name}. Avoid awkward literal translations.
- Preserve slang, jokes, flirting, insults, intimacy, anger, hesitation, and emotional rhythm.
- Do not censor, moralize, summarize, or explain.
- For Vietnamese: use appropriate pronouns based on context.
- Keep short/punchy lines short and punchy.
- Keep the result concise enough to fit inside manga speech bubbles.
{style_text}

IMPORTANT: Return ONLY the translated text. No explanations, no quotes, no formatting.

Original text:
{text}"""

    def _batch_prompt(
        self,
        texts_to_translate: List[str],
        source: str,
        target: str,
        custom_prompt: Optional[str] = None,
    ) -> str:
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        style_text = self._style_text(custom_prompt)

        return f"""Bạn là chuyên gia dịch manga/comic từ {source_name} sang {target_name}.

QUY TẮC DỊCH:
1. Đây là HỘI THOẠI NÓI - phải nghe tự nhiên như người thật nói chuyện.
2. Không dịch word-by-word; hãy diễn đạt lại theo cách người Việt nói.
3. Giữ cảm xúc, tính cách nhân vật, nhịp thoại, mỉa mai, đùa cợt, chửi thề, tán tỉnh, thân mật.
4. Không kiểm duyệt, không đạo đức hóa, không giải thích, không tóm tắt.
5. Câu ngắn giữ ngắn, impact mạnh.
6. Dịch mọi item, kể cả tiếng thở, tiếng rên, tiếng hét, ngập ngừng, dấu câu, SFX ngắn, tên gọi, hoặc câu cụt.

HƯỚNG DẪN CHO TIẾNG VIỆT:
- TÊN NHÂN VẬT: giữ nguyên tên gốc, không dịch nghĩa.
- Đại từ: chọn phù hợp với quan hệ và giữ tự nhiên.
  Ví dụ: tao/mày, tớ/cậu, tôi/anh/chị, anh/em, con/bố/mẹ...
- Thán từ dịch tự nhiên: hả, ủa, này, trời ơi, chết tiệt, khốn kiếp...
- Tránh văn sách giáo khoa, tránh câu dài lê thê, tránh Hán Việt quá mức.
{style_text}

Input texts as json array. Mỗi item là 1 bubble:
{json.dumps(texts_to_translate, ensure_ascii=False)}

IMPORTANT:
Trả về valid json object theo đúng format này:
{{
  "translations": ["bản dịch 1", "bản dịch 2"]
}}

Bắt buộc:
- "translations" phải có đúng {len(texts_to_translate)} items.
- Giữ đúng thứ tự.
- Không bỏ item.
- Không gộp item.
- Không thêm item.
- Không markdown.
"""

    def _pages_prompt(
        self,
        pages_texts: Dict[str, List[str]],
        source: str,
        target: str,
        custom_prompt: Optional[str] = None,
        context_memory: "ContextMemory" = None,
    ) -> str:
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        style_text = self._style_text(custom_prompt)

        context_section = ""
        if context_memory:
            context_section = context_memory.generate_context_prompt()

        return f"""Bạn là chuyên gia dịch manga/comic từ {source_name} sang {target_name}.

{context_section}

Đây là các trang LIÊN TIẾP trong cùng 1 story. Giữ mạch truyện và giọng nhân vật nhất quán.

QUY TẮC DỊCH:
1. Đây là HỘI THOẠI NÓI - phải nghe tự nhiên như người thật nói chuyện.
2. Không dịch word-by-word; hãy diễn đạt lại theo cách người Việt nói.
3. Mỗi nhân vật có giọng điệu riêng, giữ nhất quán xuyên suốt.
4. Giữ cảm xúc, nhịp thoại, mỉa mai, đùa cợt, chửi thề, tán tỉnh, thân mật.
5. Không kiểm duyệt, không đạo đức hóa, không giải thích, không tóm tắt.
6. Câu ngắn giữ ngắn, impact mạnh.
7. Dịch mọi item, kể cả tiếng thở, tiếng rên, tiếng hét, ngập ngừng, dấu câu, SFX ngắn, tên gọi, hoặc câu cụt.

HƯỚNG DẪN CHO TIẾNG VIỆT:
- TÊN NHÂN VẬT: giữ nguyên tên gốc, không dịch nghĩa.
- Đại từ: chọn phù hợp với quan hệ và giữ nhất quán:
  + Bạn bè thân: tao/mày, tớ/cậu
  + Người yêu: anh/em, mình/bạn
  + Người lạ/trang trọng: tôi/anh/chị
  + Gia đình: con/bố/mẹ/ông/bà
- Thán từ dịch tự nhiên: hả, ủa, này, trời ơi, chết tiệt, khốn kiếp...
- Tránh văn sách giáo khoa, tránh câu dài lê thê, tránh Hán Việt quá mức.
{style_text}

Input as json object. Các key là tên page, value là array bubble text:
{json.dumps(pages_texts, ensure_ascii=False, indent=2)}

IMPORTANT:
Trả về valid json object theo đúng format này:
{{
  "pages": {{
    "Page 001": ["bản dịch bubble 1", "bản dịch bubble 2"],
    "Page 002": ["bản dịch bubble 1"]
  }}
}}

Bắt buộc:
- Giữ nguyên tên page.
- Giữ nguyên số lượng bubble trong từng page.
- Giữ nguyên thứ tự bubble.
- Không bỏ bubble.
- Không gộp bubble.
- Không thêm bubble.
- Không markdown.
"""

    # -------------------------------------------------------------------------
    # Public translation methods
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

        This exists for compatibility with the app/router.
        It is NOT used as fallback by translate_pages_batch.
        """
        if not text or not text.strip():
            return text

        prompt = self._single_prompt(text, source, target, custom_prompt)

        for attempt in range(MAX_RETRIES):
            try:
                return self._post(
                    [{"role": "user", "content": prompt}],
                    json_mode=False,
                    timeout=60,
                    max_tokens=2048,
                )
            except Exception as e:
                print(f"DeepSeek single attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")

                if attempt < MAX_RETRIES - 1:
                    self._retry_delay(attempt)
                else:
                    print("DeepSeek single translation failed. Returning original text.")
                    return text

        return text

    def translate_batch(
        self,
        texts: List[str],
        source: str = "ja",
        target: str = "en",
        custom_prompt: str = None,
    ) -> List[str]:
        """
        Translate multiple texts in a single API call.

        No line-by-line fallback.
        If all retries fail, return original texts.
        If response length mismatches, pad/truncate with originals.
        """
        if not texts:
            return []

        indexed_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        if not indexed_texts:
            return texts

        texts_to_translate = [t for _, t in indexed_texts]

        translations = self._translate_batch_internal(
            texts_to_translate,
            source,
            target,
            custom_prompt,
        )

        translations = self._repair_list_length(
            translations,
            texts_to_translate,
            label="DeepSeek bubble batch",
        )

        result = list(texts)
        for (orig_idx, _), trans in zip(indexed_texts, translations):
            result[orig_idx] = trans

        return result

    def _translate_batch_internal(
        self,
        texts_to_translate: List[str],
        source: str,
        target: str,
        custom_prompt: str = None,
    ) -> List[str]:
        """
        Internal method to translate one bubble chunk with retry logic.

        Returns:
            List[str]. If all retries fail, returns original input list.
        """
        prompt = self._batch_prompt(
            texts_to_translate,
            source,
            target,
            custom_prompt,
        )

        for attempt in range(MAX_RETRIES):
            try:
                result_text = self._post(
                    [{"role": "user", "content": prompt}],
                    json_mode=True,
                    timeout=120,
                    max_tokens=8192,
                )

                data = self._parse_json(result_text)
                translations = data.get("translations")

                if not isinstance(translations, list):
                    raise ValueError("Missing or invalid 'translations' list")

                if len(translations) != len(texts_to_translate):
                    raise ValueError(
                        f"Expected {len(texts_to_translate)} translations, "
                        f"got {len(translations)}"
                    )

                return translations

            except Exception as e:
                error_str = str(e)
                print(f"DeepSeek batch attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")

                # Rate/quota style errors: do not keep hammering API.
                if (
                    "429" in error_str
                    or "quota" in error_str.lower()
                    or "rate" in error_str.lower()
                ):
                    print("⚠️ DeepSeek rate/quota issue. Returning original texts.")
                    return texts_to_translate

                if attempt < MAX_RETRIES - 1:
                    self._retry_delay(attempt)
                else:
                    print("DeepSeek batch failed after retries. Returning original texts.")
                    return texts_to_translate

        return texts_to_translate

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

        Mirrors the Gemini translator flow:
        - Try one multi-page batch request.
        - Retry if JSON/shape is invalid.
        - If still failing, fallback to translating each page as a batch.
        - No line-by-line fallback.
        """
        if not pages_texts:
            return {}

        prompt = self._pages_prompt(
            pages_texts,
            source,
            target,
            custom_prompt,
            context_memory,
        )

        for attempt in range(MAX_RETRIES):
            try:
                result_text = self._post(
                    [{"role": "user", "content": prompt}],
                    json_mode=True,
                    timeout=180,
                    max_tokens=16000,
                )

                data = self._parse_json(result_text)
                translated_pages = data.get("pages")

                self._validate_pages_shape(translated_pages, pages_texts)

                print(f"✓ DeepSeek translated {len(pages_texts)} pages in single batch")
                return translated_pages

            except Exception as e:
                error_str = str(e)
                print(
                    f"DeepSeek pages batch attempt "
                    f"{attempt + 1}/{MAX_RETRIES} failed: {e}"
                )

                if (
                    "429" in error_str
                    or "quota" in error_str.lower()
                    or "rate" in error_str.lower()
                ):
                    print("⚠️ DeepSeek rate/quota issue. Returning original page texts.")
                    return {page: list(texts) for page, texts in pages_texts.items()}

                if attempt < MAX_RETRIES - 1:
                    self._retry_delay(attempt)

        print(
            "DeepSeek pages batch failed after retries. "
            "Falling back to per-page batch translation."
        )

        # Important:
        # This is NOT line-by-line.
        # Each page is translated with one batch request.
        result = {}

        for page_name, texts in pages_texts.items():
            print(f"  DeepSeek translating page batch: {page_name} ({len(texts)} bubbles)")

            page_translations = self.translate_batch(
                texts,
                source=source,
                target=target,
                custom_prompt=custom_prompt,
            )

            page_translations = self._repair_list_length(
                page_translations,
                texts,
                label=f"DeepSeek page fallback {page_name}",
            )

            result[page_name] = page_translations

        return result

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def test_connection(self) -> bool:
        """
        Test if DeepSeek API is reachable.

        Note:
        Some providers may restrict /models; if this fails but chat works,
        the translator itself may still be fine.
        """
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        """
        Return available models from DeepSeek if possible, otherwise default list.
        """
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                return [m["id"] for m in data.get("data", [])]

        except Exception:
            pass

        return self.MODELS
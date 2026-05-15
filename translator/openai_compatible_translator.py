"""Generic OpenAI-compatible translator provider."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    requests = None

from mmt_core.translation_models import (
    OPENAI_COMPATIBLE_PRESET_BASE_URLS,
    OPENAI_COMPATIBLE_PRESET_CHOICES,
    OPENAI_COMPATIBLE_PRESET_OPENAI,
    normalize_openai_compatible_preset,
)

from .base import BaseTranslator

if TYPE_CHECKING:
    from .context_memory import ContextMemory


class OpenAICompatibleTranslator(BaseTranslator):
    """Translator using OpenAI-compatible chat completions endpoints."""

    def __init__(
        self,
        *,
        provider_preset: str = OPENAI_COMPATIBLE_PRESET_OPENAI,
        base_url: str = "",
        api_key: str | None = None,
        model: str = "",
        custom_prompt: str | None = None,
        style: str = "default",
        temperature: float = 0.3,
        max_tokens: int = 0,
        timeout: int = 120,
        json_mode: bool = True,
    ) -> None:
        super().__init__(custom_prompt=custom_prompt, style=style)
        if requests is None:
            raise RuntimeError("OpenAI-compatible translator requires the 'requests' package.")

        self.provider_preset = normalize_openai_compatible_preset(provider_preset)
        if self.provider_preset not in OPENAI_COMPATIBLE_PRESET_CHOICES:
            raise ValueError(f"Unsupported OpenAI-compatible provider preset: {provider_preset}")

        default_base_url = OPENAI_COMPATIBLE_PRESET_BASE_URLS.get(self.provider_preset, "")
        self.base_url = str(base_url or default_base_url).strip().rstrip("/")
        if not self.base_url:
            raise ValueError("OpenAI-compatible base URL is required.")

        self.model = str(model or "").strip()
        if not self.model:
            raise ValueError("OpenAI-compatible model is required.")

        self.api_key = str(api_key or "").strip()
        self.temperature = float(temperature)
        self.max_tokens = max(0, int(max_tokens or 0))
        self.timeout = max(1, int(timeout or 120))
        self.json_mode = bool(json_mode)
        self.endpoint = f"{self.base_url}/chat/completions"

    def translate_single(
        self,
        text: str,
        source: str = "ja",
        target: str = "en",
        custom_prompt: str | None = None,
    ) -> str:
        if not str(text or "").strip():
            return str(text or "")
        return self.translate_batch([text], source=source, target=target, custom_prompt=custom_prompt)[0]

    def translate_batch(
        self,
        texts: List[str],
        source: str = "ja",
        target: str = "en",
        custom_prompt: str | None = None,
    ) -> List[str]:
        if not texts:
            return []

        request_payload = {
            "items": [
                {"index": index, "text": str(text or "")}
                for index, text in enumerate(texts)
            ]
        }
        messages = self._build_messages(
            source=source,
            target=target,
            custom_prompt=custom_prompt,
            input_payload=request_payload,
            input_label="Input JSON object for one page of items.",
            output_shape=(
                'Return ONLY valid JSON with this exact shape:\n'
                '{\n'
                '  "items": [\n'
                '    {"index": 0, "translation": "..."},\n'
                '    {"index": 1, "translation": "..."}\n'
                "  ]\n"
                "}\n"
                "Keep the same item count and the same item indexes."
            ),
        )
        content = self._post(messages, json_mode=self.json_mode)
        parsed = self._parse_json(content)
        return self._extract_items_result(parsed, texts)

    def translate_pages_batch(
        self,
        pages_texts: Dict[str, List[str]],
        source: str = "ja",
        target: str = "en",
        custom_prompt: str | None = None,
        context_memory: "ContextMemory" = None,
        allow_internal_fallback: bool = True,
        repair_shape: bool = True,
    ) -> Dict[str, List[str]]:
        if not pages_texts:
            return {}

        try:
            return self._translate_pages_batch_once(
                pages_texts,
                source=source,
                target=target,
                custom_prompt=custom_prompt,
                context_memory=context_memory,
            )
        except Exception:
            if not allow_internal_fallback:
                raise

        fallback_results: dict[str, list[str]] = {}
        errors: list[str] = []
        for page_name, texts in pages_texts.items():
            try:
                fallback_results[page_name] = self.translate_batch(
                    texts,
                    source=source,
                    target=target,
                    custom_prompt=custom_prompt,
                )
            except Exception as exc:
                errors.append(f"{page_name}: {exc}")
                if not repair_shape:
                    raise
                fallback_results[page_name] = list(texts)

        if errors and not repair_shape:
            raise ValueError("; ".join(errors))
        return fallback_results

    def _translate_pages_batch_once(
        self,
        pages_texts: Dict[str, List[str]],
        *,
        source: str,
        target: str,
        custom_prompt: str | None,
        context_memory: "ContextMemory" | None,
    ) -> Dict[str, List[str]]:
        context_section = ""
        if context_memory is not None:
            try:
                generated = context_memory.generate_context_prompt()
            except Exception:
                generated = ""
            if str(generated or "").strip():
                context_section = (
                    "\n\nAdditional continuity context:\n"
                    f"{str(generated).strip()}"
                )

        request_payload = {
            "pages": {
                page_name: [
                    {"index": index, "text": str(text or "")}
                    for index, text in enumerate(texts)
                ]
                for page_name, texts in pages_texts.items()
            }
        }
        messages = self._build_messages(
            source=source,
            target=target,
            custom_prompt=f"{str(custom_prompt or '').strip()}{context_section}".strip(),
            input_payload=request_payload,
            input_label="Input JSON object for multiple pages.",
            output_shape=(
                'Return ONLY valid JSON with this exact shape:\n'
                '{\n'
                '  "pages": {\n'
                '    "page_001.png": [\n'
                '      {"index": 0, "translation": "..."},\n'
                '      {"index": 1, "translation": "..."}\n'
                "    ]\n"
                "  }\n"
                "}\n"
                "Preserve page keys exactly. Preserve item indexes exactly. Preserve item counts exactly."
            ),
        )
        content = self._post(messages, json_mode=self.json_mode)
        parsed = self._parse_json(content)
        return self._extract_pages_result(parsed, pages_texts)

    def _build_messages(
        self,
        *,
        source: str,
        target: str,
        custom_prompt: str | None,
        input_payload: dict[str, Any],
        input_label: str,
        output_shape: str,
    ) -> list[dict[str, str]]:
        source_name = self.get_lang_name(source, source or "Japanese")
        target_name = self.get_lang_name(target, target or "English")
        system_prompt = str(custom_prompt or self.custom_prompt or "").strip()
        if not system_prompt:
            system_prompt = (
                "You are a manga/comic translator. "
                f"Translate from {source_name} to {target_name}. "
                "Preserve tone, character voice, concise bubble fit, and item order. "
                "Return only the structured result requested by the user."
            )

        user_prompt = "\n\n".join(
            [
                input_label,
                json.dumps(input_payload, ensure_ascii=False, indent=2),
                output_shape,
                "Do not add markdown, code fences, or explanations.",
            ]
        ).strip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, messages: list[dict[str, str]], *, json_mode: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, messages: list[dict[str, str]], *, json_mode: bool) -> str:
        payload = self._build_payload(messages, json_mode=json_mode)
        try:
            response = requests.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError("OpenAI-compatible request timed out.") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError("OpenAI-compatible response is not valid JSON.") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ValueError("OpenAI-compatible response is missing message content.") from exc
        return self._message_content_to_text(content)

    def _message_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            collected: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text_value = part.get("text")
                    if text_value:
                        collected.append(str(text_value))
            return "".join(collected).strip()
        return str(content or "").strip()

    def _parse_json(self, text: str) -> Any:
        cleaned = self._strip_code_fence(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            extracted = self._extract_first_json(cleaned)
            try:
                return json.loads(extracted)
            except json.JSONDecodeError as exc:
                raise ValueError("OpenAI-compatible response did not contain valid JSON.") from exc

    def _strip_code_fence(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned

    def _extract_first_json(self, text: str) -> str:
        text = str(text or "").strip()
        object_start = text.find("{")
        array_start = text.find("[")
        starts = [index for index in (object_start, array_start) if index != -1]
        if not starts:
            raise ValueError("OpenAI-compatible response did not contain a JSON object or array.")
        start = min(starts)
        end = text.rfind("}" if text[start] == "{" else "]")
        if end == -1 or end <= start:
            raise ValueError("OpenAI-compatible response JSON was incomplete.")
        return text[start : end + 1]

    def _extract_items_result(self, parsed: Any, originals: list[str]) -> list[str]:
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI-compatible response must be a JSON object.")
        items = parsed.get("items")
        if not isinstance(items, list):
            raise ValueError("OpenAI-compatible response is missing an 'items' list.")
        if len(items) != len(originals):
            raise ValueError(
                f"OpenAI-compatible response returned {len(items)} items; expected {len(originals)}."
            )

        translations: list[str] = []
        for expected_index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError("OpenAI-compatible response items must be objects.")
            if int(item.get("index", -1)) != expected_index:
                raise ValueError(
                    f"OpenAI-compatible response item index mismatch at position {expected_index}."
                )
            translations.append(str(item.get("translation", "") or ""))
        return translations

    def _extract_pages_result(
        self,
        parsed: Any,
        originals: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI-compatible response must be a JSON object.")
        pages = parsed.get("pages")
        if not isinstance(pages, dict):
            raise ValueError("OpenAI-compatible response is missing a 'pages' object.")

        original_keys = list(originals.keys())
        response_keys = list(pages.keys())
        if response_keys != original_keys:
            raise ValueError(
                "OpenAI-compatible response returned the wrong page keys: "
                f"expected {original_keys}, got {response_keys}."
            )

        translated_pages: dict[str, list[str]] = {}
        for page_name, original_lines in originals.items():
            translated_items = pages.get(page_name)
            if not isinstance(translated_items, list):
                raise ValueError(f"OpenAI-compatible response page {page_name} is not a list.")
            if len(translated_items) != len(original_lines):
                raise ValueError(
                    f"OpenAI-compatible response page {page_name} returned {len(translated_items)} items; "
                    f"expected {len(original_lines)}."
                )

            translated_lines: list[str] = []
            for expected_index, item in enumerate(translated_items):
                if not isinstance(item, dict):
                    raise ValueError(f"OpenAI-compatible response page {page_name} contains a non-object item.")
                if int(item.get("index", -1)) != expected_index:
                    raise ValueError(
                        f"OpenAI-compatible response page {page_name} item index mismatch at position {expected_index}."
                    )
                translated_lines.append(str(item.get("translation", "") or ""))
            translated_pages[page_name] = translated_lines

        return translated_pages

    def test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=10)
            return response.status_code < 500
        except Exception:
            return False


__all__ = ["OpenAICompatibleTranslator"]

"""HTTP client for DeepSeek OCR GGUF served through a llama.cpp server."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "<image>\nFree OCR."
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0


class DeepSeekOCRClientError(RuntimeError):
    """Raised when DeepSeek OCR server communication fails."""


class DeepSeekOCRClient:
    """Small HTTP client for DeepSeek OCR over OpenAI-compatible llama.cpp endpoints."""

    def __init__(
        self,
        *,
        server_url: str,
        timeout: float = 120.0,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        normalized_url = str(server_url or "").strip().rstrip("/")
        if not normalized_url:
            raise ValueError("A valid DeepSeek OCR server URL is required.")

        self.server_url = normalized_url
        self.timeout = float(timeout)
        self.prompt = str(prompt or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT
        try:
            resolved_max_tokens = int(max_tokens)
        except Exception:
            resolved_max_tokens = DEFAULT_MAX_TOKENS
        self.max_tokens = max(32, min(resolved_max_tokens, DEFAULT_MAX_TOKENS))
        self.temperature = 0.0

    def check_server(self) -> str:
        """Verify that the target server exposes a health or models endpoint."""

        requests = self._requests_module()
        request_error = self._request_exception_class(requests)
        endpoints = (
            f"{self.server_url}/health",
            f"{self.server_url}/v1/models",
        )
        last_error = "DeepSeek OCR server check failed."

        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=min(self.timeout, 10.0))
            except request_error as exc:
                if self._is_timeout_exception(exc, requests):
                    last_error = f"Timed out while checking {endpoint}: {exc}"
                else:
                    last_error = f"Unable to reach {endpoint}: {exc}"
                continue

            if 200 <= int(response.status_code) < 400:
                return f"DeepSeek OCR server is ready at {self.server_url}."

            last_error = f"{endpoint} returned HTTP {response.status_code}."

        raise DeepSeekOCRClientError(
            "DeepSeek OCR server is not reachable. "
            "Start the llama.cpp server from the OCR tab first. "
            f"{last_error}"
        )

    def recognize_image(self, crop_path: Path | str) -> str:
        """Send one provider crop image to the running server and return recognized text."""

        image_file = Path(crop_path).expanduser().resolve()
        if not image_file.exists():
            raise FileNotFoundError(f"OCR crop file is missing: {image_file}")

        image_b64 = base64.b64encode(image_file.read_bytes()).decode("ascii")
        return self._chat_completion(image_b64)

    def _chat_completion(self, image_b64: str) -> str:
        requests = self._requests_module()
        request_error = self._request_exception_class(requests)
        payload = {
            "model": "deepseek-ocr",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
        }

        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
        except request_error as exc:
            if self._is_timeout_exception(exc, requests):
                raise DeepSeekOCRClientError(
                    f"OCR request timed out while contacting {self.server_url}: {exc}"
                ) from exc
            raise DeepSeekOCRClientError(
                "DeepSeek OCR server is not reachable. "
                "Start the llama.cpp server from the OCR tab first. "
                f"{exc}"
            ) from exc

        if int(response.status_code) == 404:
            return self._legacy_completion(image_b64)
        if int(response.status_code) >= 400:
            raise DeepSeekOCRClientError(
                "DeepSeek OCR server request failed: "
                f"HTTP {response.status_code}: {self._response_excerpt(response)}"
            )

        try:
            response_data = response.json()
        except Exception as exc:
            raise DeepSeekOCRClientError(
                "DeepSeek OCR server returned invalid JSON from /v1/chat/completions."
            ) from exc

        return self._extract_chat_text(response_data)

    def _legacy_completion(self, image_b64: str) -> str:
        requests = self._requests_module()
        request_error = self._request_exception_class(requests)
        payload = {
            "prompt": self.prompt,
            "image_data": image_b64,
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                timeout=self.timeout,
            )
        except request_error as exc:
            if self._is_timeout_exception(exc, requests):
                raise DeepSeekOCRClientError(
                    f"OCR request timed out while contacting {self.server_url}/completion: {exc}"
                ) from exc
            raise DeepSeekOCRClientError(
                "DeepSeek OCR legacy completion endpoint is not reachable. "
                f"{exc}"
            ) from exc

        if int(response.status_code) >= 400:
            raise DeepSeekOCRClientError(
                "DeepSeek OCR server did not expose a compatible OCR endpoint. "
                f"HTTP {response.status_code}: {self._response_excerpt(response)}"
            )

        try:
            response_data = response.json()
        except Exception as exc:
            raise DeepSeekOCRClientError(
                "DeepSeek OCR server returned invalid JSON from /completion."
            ) from exc

        content = response_data.get("content") or response_data.get("text") or ""
        if not isinstance(content, str):
            raise DeepSeekOCRClientError("Invalid legacy DeepSeek OCR response payload.")
        return self._clean_output(content)

    def _extract_chat_text(self, response_data: dict[str, Any]) -> str:
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DeepSeekOCRClientError("Invalid llama.cpp response: missing choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise DeepSeekOCRClientError("Invalid llama.cpp response: malformed choice payload.")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise DeepSeekOCRClientError("Invalid llama.cpp response: missing message payload.")

        content = message.get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)

        if not isinstance(content, str):
            raise DeepSeekOCRClientError("Invalid llama.cpp response: content is not text.")

        return self._clean_output(content)

    def _clean_output(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""

        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()

        if cleaned.startswith("```") and cleaned.endswith("```"):
            lines = cleaned.split("\n")
            if len(lines) >= 2:
                cleaned = "\n".join(lines[1:-1]).strip()

        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            for key in ("text", "ocr", "content", "result"):
                value = parsed.get(key)
                if isinstance(value, str):
                    cleaned = value
                    break
        elif isinstance(parsed, list):
            text_parts = [value for value in parsed if isinstance(value, str)]
            if text_parts:
                cleaned = "\n".join(text_parts)

        for prefix in (
            "OCR:",
            "Text:",
            "Recognized text:",
            "Recognized Text:",
            "Output:",
        ):
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].lstrip()
                break

        return cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _requests_module(self) -> Any:
        try:
            import requests
        except Exception as exc:
            raise DeepSeekOCRClientError(
                "The 'requests' package is required for OCR server communication."
            ) from exc
        return requests

    def _request_exception_class(self, requests_module: Any) -> type[Exception]:
        request_error = getattr(requests_module, "RequestException", None)
        if isinstance(request_error, type) and issubclass(request_error, Exception):
            return request_error
        return Exception

    def _is_timeout_exception(self, exc: Exception, requests_module: Any) -> bool:
        timeout_error = getattr(requests_module, "Timeout", None)
        return isinstance(timeout_error, type) and isinstance(exc, timeout_error)

    def _response_excerpt(self, response: Any) -> str:
        text = getattr(response, "text", "")
        if not isinstance(text, str):
            return "(no response body)"
        return text[:400] or "(empty response body)"


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_PROMPT",
    "DEFAULT_TEMPERATURE",
    "DeepSeekOCRClient",
    "DeepSeekOCRClientError",
]

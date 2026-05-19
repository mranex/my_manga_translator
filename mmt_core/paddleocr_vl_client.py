"""HTTP client for OCR inference against a persistent PaddleOCR-VL llama.cpp server."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from ocr.paddleocr_vl_ocr import (
    clean_paddleocr_vl_output,
    is_degenerate_ocr_output,
    trim_repeated_ocr_output,
)


DEFAULT_PROMPT = "OCR:"
DEFAULT_MAX_RETRIES = 3


class PaddleOCRVLClientError(RuntimeError):
    """Raised when OCR inference over the persistent llama.cpp server fails."""


class PaddleOCRVLClient:
    """Small HTTP client for sending OCR crop images to an already-running server."""

    def __init__(
        self,
        *,
        server_url: str,
        timeout: float = 120.0,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 512,
        temperature: float = 0.0,
        repeat_penalty: float = 1.2,
        repeat_last_n: int = -1,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        normalized_url = str(server_url or "").strip().rstrip("/")
        if not normalized_url:
            raise ValueError("A valid PaddleOCR-VL server URL is required.")

        self.server_url = normalized_url
        self.timeout = float(timeout)
        self.prompt = str(prompt or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT
        self.max_tokens = int(max_tokens)
        self.temperature = 0.0
        self.repeat_penalty = float(repeat_penalty)
        self.repeat_last_n = int(repeat_last_n)
        self.max_retries = max(1, int(max_retries))

    def check_server(self) -> str:
        """Verify that the server exposes a health or models endpoint."""

        requests = self._requests_module()
        request_error = self._request_exception_class(requests)
        endpoints = (
            f"{self.server_url}/health",
            f"{self.server_url}/v1/models",
        )
        last_error = "PaddleOCR-VL server check failed."

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
                return f"PaddleOCR-VL server is ready at {self.server_url}."

            last_error = f"{endpoint} returned HTTP {response.status_code}."

        raise PaddleOCRVLClientError(
            "PaddleOCR-VL server is not reachable. "
            "Start run_server.bat and check health first. "
            f"{last_error}"
        )

    def recognize_image(self, crop_path: Path | str) -> str:
        """Send one crop image to the running server and return recognized text."""

        image_file = Path(crop_path).expanduser().resolve()
        if not image_file.exists():
            raise FileNotFoundError(f"OCR crop file is missing: {image_file}")

        image_b64 = base64.b64encode(image_file.read_bytes()).decode("ascii")
        return self._recognize_with_retries(image_b64)

    def _recognize_with_retries(self, image_b64: str) -> str:
        best_trimmed_candidate = ""
        best_fallback_candidate = ""

        for attempt in range(1, self.max_retries + 1):
            request_max_tokens = (
                self.max_tokens
                if attempt == 1
                else min(self.max_tokens, 128)
            )

            raw_text = self._chat_completion(image_b64, max_tokens=request_max_tokens)
            cleaned = clean_paddleocr_vl_output(raw_text)
            trimmed, had_repeat = trim_repeated_ocr_output(cleaned)
            degenerate = is_degenerate_ocr_output(cleaned)

            if cleaned:
                best_fallback_candidate = _prefer_candidate(best_fallback_candidate, cleaned)
            if had_repeat and trimmed:
                best_trimmed_candidate = _prefer_candidate(best_trimmed_candidate, trimmed)

            if cleaned and not had_repeat and not degenerate:
                return cleaned

        if best_trimmed_candidate:
            return best_trimmed_candidate
        if best_fallback_candidate:
            return best_fallback_candidate
        return ""

    def _chat_completion(self, image_b64: str, *, max_tokens: int | None = None) -> str:
        requests = self._requests_module()
        request_error = self._request_exception_class(requests)
        token_limit = int(max_tokens if max_tokens is not None else self.max_tokens)
        payload = {
            "model": "paddleocr-vl",
            "temperature": 0.0,
            "max_tokens": token_limit,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "repetition_penalty": self.repeat_penalty,
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
                raise PaddleOCRVLClientError(
                    f"OCR request timed out while contacting {self.server_url}: {exc}"
                ) from exc
            raise PaddleOCRVLClientError(
                "PaddleOCR-VL server is not reachable. "
                "Start run_server.bat and check health first. "
                f"{exc}"
            ) from exc

        if int(response.status_code) == 404:
            return self._legacy_completion(image_b64, max_tokens=token_limit)

        if int(response.status_code) >= 400:
            raise PaddleOCRVLClientError(
                "PaddleOCR-VL server request failed: "
                f"HTTP {response.status_code}: {self._response_excerpt(response)}"
            )

        try:
            response_data = response.json()
        except Exception as exc:
            raise PaddleOCRVLClientError(
                "PaddleOCR-VL server returned invalid JSON from /v1/chat/completions."
            ) from exc

        return self._extract_chat_text(response_data)

    def _legacy_completion(self, image_b64: str, *, max_tokens: int | None = None) -> str:
        requests = self._requests_module()
        request_error = self._request_exception_class(requests)
        token_limit = int(max_tokens if max_tokens is not None else self.max_tokens)
        payload = {
            "prompt": self.prompt,
            "image_data": image_b64,
            "n_predict": token_limit,
            "temperature": 0.0,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
        }

        try:
            response = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                timeout=self.timeout,
            )
        except request_error as exc:
            if self._is_timeout_exception(exc, requests):
                raise PaddleOCRVLClientError(
                    f"OCR request timed out while contacting {self.server_url}/completion: {exc}"
                ) from exc
            raise PaddleOCRVLClientError(
                "PaddleOCR-VL legacy completion endpoint is not reachable. "
                f"{exc}"
            ) from exc

        if int(response.status_code) >= 400:
            raise PaddleOCRVLClientError(
                "PaddleOCR-VL server did not expose a compatible OCR endpoint. "
                f"HTTP {response.status_code}: {self._response_excerpt(response)}"
            )

        try:
            response_data = response.json()
        except Exception as exc:
            raise PaddleOCRVLClientError(
                "PaddleOCR-VL server returned invalid JSON from /completion."
            ) from exc

        content = response_data.get("content") or response_data.get("text") or ""
        if not isinstance(content, str):
            raise PaddleOCRVLClientError("Invalid legacy PaddleOCR-VL response payload.")
        return clean_paddleocr_vl_output(content)

    def _extract_chat_text(self, response_data: dict[str, Any]) -> str:
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise PaddleOCRVLClientError("Invalid llama.cpp response: missing choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise PaddleOCRVLClientError("Invalid llama.cpp response: malformed choice payload.")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise PaddleOCRVLClientError("Invalid llama.cpp response: missing message payload.")

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
            raise PaddleOCRVLClientError("Invalid llama.cpp response: content is not text.")

        return clean_paddleocr_vl_output(content)

    def _requests_module(self) -> Any:
        try:
            import requests
        except Exception as exc:
            raise PaddleOCRVLClientError(
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


def _candidate_score(text: str) -> tuple[int, int]:
    compact = "".join(char for char in str(text or "") if not char.isspace())
    return (len(compact), len(set(compact)))


def _prefer_candidate(current: str, candidate: str) -> str:
    if not candidate:
        return current
    if not current:
        return candidate
    return candidate if _candidate_score(candidate) > _candidate_score(current) else current


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_PROMPT",
    "PaddleOCRVLClient",
    "PaddleOCRVLClientError",
]

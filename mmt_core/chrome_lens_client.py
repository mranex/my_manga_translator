"""Thin Chrome Lens OCR client wrapper for the desktop OCR stage."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any


class ChromeLensClientError(RuntimeError):
    """Raised when Chrome Lens OCR is unavailable or fails."""


class ChromeLensClient:
    """Adapter around ``chrome-lens-py`` with friendly desktop-app errors."""

    def __init__(
        self,
        *,
        timeout: float = 120.0,
        language: str = "ja",
        max_retries: int = 5,
        headless: bool = False,
        chrome_path: str = "",
        user_data_dir: str = "",
    ) -> None:
        self.timeout = float(timeout) if float(timeout) > 0 else 120.0
        self.language = str(language or "ja").strip() or "ja"
        self.max_retries = max(1, int(max_retries))
        self.headless = bool(headless)
        self.chrome_path = str(chrome_path or "").strip()
        self.user_data_dir = str(user_data_dir or "").strip()

        self._api: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def check_dependency(cls) -> None:
        """Validate that ``chrome-lens-py`` is importable."""

        cls._import_lens_api_class()

    def validate(self) -> None:
        """Validate that the Chrome Lens dependency and session can be started."""

        self._ensure_api()

    def recognize_image(self, crop_path: Path | str) -> str:
        """Recognize text from one crop image using Chrome Lens."""

        image_file = Path(crop_path).expanduser().resolve()
        if not image_file.exists():
            raise FileNotFoundError(f"OCR crop file is missing: {image_file}")

        process_result = self._run_coroutine(self._recognize_with_retries(image_file))
        if isinstance(process_result, dict):
            text = process_result.get("ocr_text", "") or process_result.get("text", "")
            return str(text or "")
        return str(process_result or "")

    def close(self) -> None:
        """Attempt to clean up any provider/browser resources."""

        api = self._api
        self._api = None
        if api is not None:
            for method_name in ("close", "cleanup", "shutdown", "reset_session"):
                method = getattr(api, method_name, None)
                if not callable(method):
                    continue
                try:
                    result = method()
                    if inspect.isawaitable(result):
                        self._run_coroutine(result)
                except Exception:
                    continue
                break

        if self._loop is not None:
            try:
                if self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    async def _recognize_with_retries(self, image_file: Path) -> Any:
        api = self._ensure_api()
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._invoke_process_image(api, image_file),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError as exc:
                last_error = TimeoutError("Chrome Lens OCR timed out.")
            except Exception as exc:
                last_error = exc

            if attempt >= self.max_retries:
                break
            await asyncio.sleep(min(1.5 * attempt, 3.0))

        if isinstance(last_error, TimeoutError):
            raise last_error
        if last_error is not None:
            raise ChromeLensClientError(
                f"Chrome Lens OCR failed after {self.max_retries} attempt(s): {last_error}"
            ) from last_error
        raise ChromeLensClientError("Chrome Lens OCR returned no result.")

    async def _invoke_process_image(self, api: Any, image_file: Path) -> Any:
        process_image = getattr(api, "process_image", None)
        if not callable(process_image):
            raise ChromeLensClientError("Chrome Lens OCR is unavailable: process_image() is missing.")

        signature = inspect.signature(process_image)
        kwargs: dict[str, Any] = {}
        positional_args: list[Any] = []

        if "image_path" in signature.parameters:
            kwargs["image_path"] = str(image_file)
        else:
            positional_args.append(str(image_file))

        if "ocr_language" in signature.parameters:
            kwargs["ocr_language"] = self.language
        elif "language" in signature.parameters:
            kwargs["language"] = self.language
        elif "source_language" in signature.parameters:
            kwargs["source_language"] = self.language

        if "timeout" in signature.parameters:
            kwargs["timeout"] = self.timeout

        result = process_image(*positional_args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _ensure_api(self) -> Any:
        if self._api is not None:
            return self._api

        api_class = self._import_lens_api_class()
        init_signature = inspect.signature(api_class.__init__)
        kwargs: dict[str, Any] = {}

        # Pass through optional browser settings only when the installed client supports them.
        for parameter_name, value in (
            ("headless", self.headless),
            ("chrome_path", self.chrome_path),
            ("browser_path", self.chrome_path),
            ("executable_path", self.chrome_path),
            ("user_data_dir", self.user_data_dir),
            ("profile_dir", self.user_data_dir),
            ("timeout", self.timeout),
        ):
            if parameter_name not in init_signature.parameters:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            kwargs[parameter_name] = value

        try:
            self._api = api_class(**kwargs)
        except Exception as exc:
            raise ChromeLensClientError(
                f"Chrome Lens browser session failed to start: {exc}"
            ) from exc
        return self._api

    def _run_coroutine(self, coroutine: Any) -> Any:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coroutine)

    @staticmethod
    def _import_lens_api_class() -> Any:
        try:
            from chrome_lens_py import LensAPI
        except Exception as exc:
            raise ChromeLensClientError(
                "Chrome Lens OCR is unavailable: the 'chrome-lens-py' package is not installed."
            ) from exc
        return LensAPI


__all__ = [
    "ChromeLensClient",
    "ChromeLensClientError",
]

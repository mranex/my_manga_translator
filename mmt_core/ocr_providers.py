"""OCR provider abstraction for desktop OCR inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

from .chrome_lens_client import ChromeLensClient, ChromeLensClientError
from .ocr_models import (
    DEFAULT_OCR_PROVIDER,
    OCR_PROVIDER_CHROME_LENS,
    OCR_PROVIDER_PADDLE_VL_LLAMA,
    OCR_PROVIDER_LABELS,
    OCRConfig,
    is_known_ocr_provider,
)
from .paddleocr_vl_client import PaddleOCRVLClient, PaddleOCRVLClientError


class OCRProviderError(RuntimeError):
    """Raised when OCR provider selection or validation fails."""


class OCRProvider(Protocol):
    provider_key: str
    provider_label: str

    def validate(self) -> None: ...

    def recognize_image(self, crop_path: Path | str) -> str: ...

    def close(self) -> None: ...

    def item_metadata(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class PaddleOCRVLProvider:
    """PaddleOCR-VL OCR provider backed by a persistent llama.cpp server."""

    provider_key: ClassVar[str] = OCR_PROVIDER_PADDLE_VL_LLAMA
    provider_label: ClassVar[str] = OCR_PROVIDER_LABELS[OCR_PROVIDER_PADDLE_VL_LLAMA]
    config: OCRConfig
    _client: PaddleOCRVLClient = field(init=False)

    def __post_init__(self) -> None:
        server_url = str(self.config.server_url or "").strip()
        if not server_url:
            raise OCRProviderError("OCR provider is not configured.")
        self._client = PaddleOCRVLClient(
            server_url=server_url,
            timeout=float(self.config.timeout),
        )

    def validate(self) -> None:
        try:
            self._client.check_server()
        except PaddleOCRVLClientError as exc:
            raise OCRProviderError(str(exc)) from exc

    def recognize_image(self, crop_path: Path | str) -> str:
        try:
            return self._client.recognize_image(crop_path)
        except PaddleOCRVLClientError as exc:
            raise OCRProviderError(str(exc)) from exc

    def close(self) -> None:
        return None

    def item_metadata(self) -> dict[str, Any]:
        return {
            "ocr_engine": self.provider_key,
            "ocr_provider": self.provider_label,
            "server_url": self._client.server_url,
        }


@dataclass(slots=True)
class ChromeLensProvider:
    """Chrome Lens OCR provider backed by ``chrome-lens-py``."""

    provider_key: ClassVar[str] = OCR_PROVIDER_CHROME_LENS
    provider_label: ClassVar[str] = OCR_PROVIDER_LABELS[OCR_PROVIDER_CHROME_LENS]
    config: OCRConfig
    _client: ChromeLensClient = field(init=False)

    def __post_init__(self) -> None:
        self._client = ChromeLensClient(
            timeout=float(self.config.timeout),
            language=self.config.chrome_lens_language,
            max_retries=int(self.config.chrome_lens_max_retries),
            headless=bool(self.config.chrome_lens_headless),
            chrome_path=self.config.chrome_lens_chrome_path,
            user_data_dir=self.config.chrome_lens_user_data_dir,
        )

    def validate(self) -> None:
        try:
            self._client.validate()
        except ChromeLensClientError as exc:
            raise OCRProviderError(str(exc)) from exc

    def recognize_image(self, crop_path: Path | str) -> str:
        try:
            return self._client.recognize_image(crop_path)
        except ChromeLensClientError as exc:
            raise OCRProviderError(str(exc)) from exc

    def close(self) -> None:
        self._client.close()

    def item_metadata(self) -> dict[str, Any]:
        return {
            "ocr_engine": self.provider_key,
            "ocr_provider": self.provider_label,
        }


def validate_ocr_provider_config(config_value: OCRConfig | dict[str, Any] | None) -> OCRConfig:
    """Validate OCR provider settings without running OCR."""

    config = OCRConfig.from_value(config_value)
    provider_name = str(config.ocr_provider or "").strip()
    if not provider_name or not is_known_ocr_provider(provider_name):
        raise OCRProviderError("OCR provider is not configured.")

    if config.ocr_provider == OCR_PROVIDER_PADDLE_VL_LLAMA:
        if not str(config.server_url or "").strip():
            raise OCRProviderError("OCR provider is not configured.")
        return config

    try:
        ChromeLensClient.check_dependency()
    except ChromeLensClientError as exc:
        raise OCRProviderError(str(exc)) from exc
    return config


def create_ocr_provider(config_value: OCRConfig | dict[str, Any] | None) -> OCRProvider:
    """Instantiate one OCR provider from config."""

    config = OCRConfig.from_value(config_value)
    if not is_known_ocr_provider(config.ocr_provider):
        raise OCRProviderError(f"Unknown OCR provider '{config.ocr_provider}'.")

    if config.ocr_provider == OCR_PROVIDER_PADDLE_VL_LLAMA:
        return PaddleOCRVLProvider(config)
    if config.ocr_provider == OCR_PROVIDER_CHROME_LENS:
        return ChromeLensProvider(config)
    raise OCRProviderError(f"Unknown OCR provider '{config.ocr_provider or DEFAULT_OCR_PROVIDER}'.")


__all__ = [
    "ChromeLensProvider",
    "OCRProvider",
    "OCRProviderError",
    "PaddleOCRVLProvider",
    "create_ocr_provider",
    "validate_ocr_provider_config",
]

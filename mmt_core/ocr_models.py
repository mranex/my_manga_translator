"""Small OCR-stage models and provider config helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OCR_PROVIDER_PADDLE_VL_LLAMA = "paddleocr_vl_llama"
OCR_PROVIDER_CHROME_LENS = "chrome_lens"
DEFAULT_OCR_PROVIDER = OCR_PROVIDER_PADDLE_VL_LLAMA

OCR_PROVIDER_CHOICES = (
    (OCR_PROVIDER_PADDLE_VL_LLAMA, "PaddleOCR-VL Local"),
    (OCR_PROVIDER_CHROME_LENS, "Chrome Lens"),
)

OCR_PROVIDER_LABELS = {
    OCR_PROVIDER_PADDLE_VL_LLAMA: "PaddleOCR-VL Local",
    OCR_PROVIDER_CHROME_LENS: "Chrome Lens",
}


def normalize_ocr_provider_name(name: str, *, fallback: str = DEFAULT_OCR_PROVIDER) -> str:
    """Normalize saved/provider UI values into canonical provider keys."""

    cleaned = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "paddleocr_vl_llama": OCR_PROVIDER_PADDLE_VL_LLAMA,
        "paddleocr_vl": OCR_PROVIDER_PADDLE_VL_LLAMA,
        "paddleocr_vl_local": OCR_PROVIDER_PADDLE_VL_LLAMA,
        "paddleocr": OCR_PROVIDER_PADDLE_VL_LLAMA,
        "chrome_lens": OCR_PROVIDER_CHROME_LENS,
        "chrome_lens_ocr": OCR_PROVIDER_CHROME_LENS,
        "chrome_lens_local": OCR_PROVIDER_CHROME_LENS,
        "chrome_lens_browser": OCR_PROVIDER_CHROME_LENS,
        "chromelens": OCR_PROVIDER_CHROME_LENS,
        "chrome": OCR_PROVIDER_CHROME_LENS,
        "lens": OCR_PROVIDER_CHROME_LENS,
    }
    return mapping.get(cleaned, fallback)


def provider_label(provider_name: str) -> str:
    normalized = normalize_ocr_provider_name(provider_name)
    return OCR_PROVIDER_LABELS.get(normalized, OCR_PROVIDER_LABELS[DEFAULT_OCR_PROVIDER])


def is_known_ocr_provider(provider_name: str) -> bool:
    normalized = normalize_ocr_provider_name(provider_name, fallback="")
    return normalized in OCR_PROVIDER_LABELS


@dataclass(slots=True)
class OCRConfig:
    """Serializable configuration for OCR preparation and OCR inference."""

    ocr_provider: str = DEFAULT_OCR_PROVIDER
    timeout: float = 120.0
    server_url: str = "http://127.0.0.1:8080"
    chrome_lens_headless: bool = False
    chrome_lens_chrome_path: str = ""
    chrome_lens_user_data_dir: str = ""
    chrome_lens_language: str = "ja"
    chrome_lens_max_retries: int = 5

    @classmethod
    def from_value(cls, value: "OCRConfig | dict[str, Any] | None") -> "OCRConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls()

        raw_provider = str(value.get("ocr_provider", DEFAULT_OCR_PROVIDER) or DEFAULT_OCR_PROVIDER).strip()
        normalized_provider = normalize_ocr_provider_name(raw_provider, fallback=raw_provider or DEFAULT_OCR_PROVIDER)

        return cls(
            ocr_provider=normalized_provider,
            timeout=_coerce_positive_float(value.get("timeout"), 120.0),
            server_url=str(value.get("server_url", "http://127.0.0.1:8080") or "http://127.0.0.1:8080"),
            chrome_lens_headless=bool(value.get("chrome_lens_headless", False)),
            chrome_lens_chrome_path=str(value.get("chrome_lens_chrome_path", "") or ""),
            chrome_lens_user_data_dir=str(value.get("chrome_lens_user_data_dir", "") or ""),
            chrome_lens_language=str(value.get("chrome_lens_language", "ja") or "ja"),
            chrome_lens_max_retries=_coerce_positive_int(value.get("chrome_lens_max_retries"), 5),
        )

    @property
    def provider_label(self) -> str:
        return provider_label(self.ocr_provider)

    @property
    def requires_llama_server(self) -> bool:
        return self.ocr_provider == OCR_PROVIDER_PADDLE_VL_LLAMA

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ocr_provider": self.ocr_provider,
            "timeout": float(self.timeout),
            "server_url": self.server_url,
            "chrome_lens_headless": bool(self.chrome_lens_headless),
            "chrome_lens_chrome_path": self.chrome_lens_chrome_path,
            "chrome_lens_user_data_dir": self.chrome_lens_user_data_dir,
            "chrome_lens_language": self.chrome_lens_language,
            "chrome_lens_max_retries": int(self.chrome_lens_max_retries),
        }


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _coerce_positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return parsed if parsed > 0 else float(default)


__all__ = [
    "DEFAULT_OCR_PROVIDER",
    "OCR_PROVIDER_CHROME_LENS",
    "OCR_PROVIDER_CHOICES",
    "OCR_PROVIDER_LABELS",
    "OCR_PROVIDER_PADDLE_VL_LLAMA",
    "OCRConfig",
    "is_known_ocr_provider",
    "normalize_ocr_provider_name",
    "provider_label",
]

"""Small OCR-stage models and provider config helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

OCR_PROVIDER_PADDLE_VL_LLAMA = "paddleocr_vl_llama"
OCR_PROVIDER_DEEPSEEK_OCR_LLAMA = "deepseek_ocr_llama"
OCR_PROVIDER_CHROME_LENS = "chrome_lens"
DEFAULT_OCR_PROVIDER = OCR_PROVIDER_PADDLE_VL_LLAMA
OCR_PROVIDER_MODE_LOCAL = "local"
OCR_PROVIDER_MODE_CHROME_LENS = "chrome_lens"
DEFAULT_OCR_PROVIDER_MODE = OCR_PROVIDER_MODE_LOCAL
DEFAULT_LOCAL_OCR_PROVIDER = OCR_PROVIDER_PADDLE_VL_LLAMA

OCR_PROVIDER_CHOICES = (
    (OCR_PROVIDER_PADDLE_VL_LLAMA, "PaddleOCR-VL Local"),
    (OCR_PROVIDER_DEEPSEEK_OCR_LLAMA, "DeepSeek OCR (llama.cpp)"),
    (OCR_PROVIDER_CHROME_LENS, "Chrome Lens"),
)

OCR_PROVIDER_MODE_CHOICES = (
    (OCR_PROVIDER_MODE_LOCAL, "Local"),
    (OCR_PROVIDER_MODE_CHROME_LENS, "Chrome Lens"),
)

LOCAL_OCR_PROVIDER_CHOICES = (
    (OCR_PROVIDER_PADDLE_VL_LLAMA, "PaddleOCR-VL"),
    (OCR_PROVIDER_DEEPSEEK_OCR_LLAMA, "DeepSeek OCR"),
)

OCR_PROVIDER_LABELS = {
    OCR_PROVIDER_PADDLE_VL_LLAMA: "PaddleOCR-VL Local",
    OCR_PROVIDER_DEEPSEEK_OCR_LLAMA: "DeepSeek OCR (llama.cpp)",
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
        "deepseek_ocr_llama": OCR_PROVIDER_DEEPSEEK_OCR_LLAMA,
        "deepseek_ocr": OCR_PROVIDER_DEEPSEEK_OCR_LLAMA,
        "deepseek": OCR_PROVIDER_DEEPSEEK_OCR_LLAMA,
        "deepseek_ocr_gguf": OCR_PROVIDER_DEEPSEEK_OCR_LLAMA,
        "deepseek_ocr_llamacpp": OCR_PROVIDER_DEEPSEEK_OCR_LLAMA,
        "deepseek_ocr_llama_cpp": OCR_PROVIDER_DEEPSEEK_OCR_LLAMA,
        "chrome_lens": OCR_PROVIDER_CHROME_LENS,
        "chrome_lens_ocr": OCR_PROVIDER_CHROME_LENS,
        "chrome_lens_local": OCR_PROVIDER_CHROME_LENS,
        "chrome_lens_browser": OCR_PROVIDER_CHROME_LENS,
        "chromelens": OCR_PROVIDER_CHROME_LENS,
        "chrome": OCR_PROVIDER_CHROME_LENS,
        "lens": OCR_PROVIDER_CHROME_LENS,
    }
    return mapping.get(cleaned, fallback)


def normalize_ocr_provider_mode(
    value: str,
    *,
    fallback: str = DEFAULT_OCR_PROVIDER_MODE,
) -> str:
    cleaned = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "local": OCR_PROVIDER_MODE_LOCAL,
        "llama": OCR_PROVIDER_MODE_LOCAL,
        "llama_cpp": OCR_PROVIDER_MODE_LOCAL,
        "chrome_lens": OCR_PROVIDER_MODE_CHROME_LENS,
        "chrome": OCR_PROVIDER_MODE_CHROME_LENS,
        "lens": OCR_PROVIDER_MODE_CHROME_LENS,
    }
    return mapping.get(cleaned, fallback)


def normalize_local_ocr_provider(
    value: str,
    *,
    fallback: str = DEFAULT_LOCAL_OCR_PROVIDER,
) -> str:
    normalized = normalize_ocr_provider_name(value, fallback=fallback)
    if normalized == OCR_PROVIDER_CHROME_LENS:
        return fallback
    return normalized


def resolve_ocr_provider(
    *,
    ocr_provider: str = "",
    ocr_provider_mode: str = "",
    local_ocr_provider: str = "",
) -> str:
    normalized_provider = normalize_ocr_provider_name(ocr_provider, fallback="")
    normalized_mode = normalize_ocr_provider_mode(ocr_provider_mode, fallback="")
    normalized_local = normalize_local_ocr_provider(local_ocr_provider)

    if normalized_mode == OCR_PROVIDER_MODE_CHROME_LENS:
        return OCR_PROVIDER_CHROME_LENS
    if normalized_mode == OCR_PROVIDER_MODE_LOCAL:
        return normalized_local
    if normalized_provider:
        return normalized_provider
    return normalized_local


def selection_from_provider(provider_name: str) -> tuple[str, str]:
    normalized_provider = normalize_ocr_provider_name(provider_name)
    if normalized_provider == OCR_PROVIDER_CHROME_LENS:
        return OCR_PROVIDER_MODE_CHROME_LENS, DEFAULT_LOCAL_OCR_PROVIDER
    return OCR_PROVIDER_MODE_LOCAL, normalize_local_ocr_provider(normalized_provider)


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
    ocr_provider_mode: str = DEFAULT_OCR_PROVIDER_MODE
    local_ocr_provider: str = DEFAULT_LOCAL_OCR_PROVIDER
    timeout: float = 120.0
    server_url: str = "http://127.0.0.1:8080"
    host: str = "127.0.0.1"
    port: int = 8080
    llama_cpp_dir: str = ""
    model_path: str = ""
    mmproj_path: str = ""
    gpu_layers: int = 99
    ctx_size: int = 8192
    temperature: float = 0.0
    extra_args: str = ""
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
        raw_mode = str(value.get("ocr_provider_mode", "") or "").strip()
        raw_local_provider = str(value.get("local_ocr_provider", "") or "").strip()

        fallback_mode, fallback_local_provider = selection_from_provider(raw_provider)
        normalized_mode = normalize_ocr_provider_mode(raw_mode, fallback=fallback_mode)
        normalized_local_provider = normalize_local_ocr_provider(
            raw_local_provider,
            fallback=fallback_local_provider,
        )
        normalized_provider = resolve_ocr_provider(
            ocr_provider=raw_provider,
            ocr_provider_mode=normalized_mode,
            local_ocr_provider=normalized_local_provider,
        )

        normalized_server_url, parsed_host, parsed_port = _normalize_server_address(
            server_url=value.get("server_url", "http://127.0.0.1:8080"),
            host=value.get("host"),
            port=value.get("port"),
        )

        return cls(
            ocr_provider=normalized_provider,
            ocr_provider_mode=normalized_mode,
            local_ocr_provider=normalized_local_provider,
            timeout=_coerce_positive_float(value.get("timeout"), 120.0),
            server_url=normalized_server_url,
            host=parsed_host,
            port=parsed_port,
            llama_cpp_dir=str(value.get("llama_cpp_dir", "") or ""),
            model_path=str(value.get("model_path", "") or ""),
            mmproj_path=str(value.get("mmproj_path", "") or ""),
            gpu_layers=_coerce_int(value.get("gpu_layers"), 99),
            ctx_size=_coerce_positive_int(value.get("ctx_size"), 8192),
            temperature=_coerce_float(value.get("temperature"), 0.0),
            extra_args=str(value.get("extra_args", "") or ""),
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
        return self.ocr_provider in {
            OCR_PROVIDER_PADDLE_VL_LLAMA,
            OCR_PROVIDER_DEEPSEEK_OCR_LLAMA,
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ocr_provider": self.ocr_provider,
            "ocr_provider_mode": self.ocr_provider_mode,
            "local_ocr_provider": self.local_ocr_provider,
            "timeout": float(self.timeout),
            "server_url": self.server_url,
            "host": self.host,
            "port": int(self.port),
            "llama_cpp_dir": self.llama_cpp_dir,
            "model_path": self.model_path,
            "mmproj_path": self.mmproj_path,
            "gpu_layers": int(self.gpu_layers),
            "ctx_size": int(self.ctx_size),
            "temperature": float(self.temperature),
            "extra_args": self.extra_args,
            "chrome_lens_headless": bool(self.chrome_lens_headless),
            "chrome_lens_chrome_path": self.chrome_lens_chrome_path,
            "chrome_lens_user_data_dir": self.chrome_lens_user_data_dir,
            "chrome_lens_language": self.chrome_lens_language,
            "chrome_lens_max_retries": int(self.chrome_lens_max_retries),
        }


def _normalize_server_address(
    *,
    server_url: Any,
    host: Any,
    port: Any,
) -> tuple[str, str, int]:
    normalized_host = str(host or "").strip()
    normalized_port = _coerce_positive_int(port, 8080)
    normalized_server_url = str(server_url or "").strip()

    if normalized_host:
        return (
            f"http://{normalized_host}:{normalized_port}",
            normalized_host,
            normalized_port,
        )

    if not normalized_server_url:
        return "http://127.0.0.1:8080", "127.0.0.1", 8080

    if "://" not in normalized_server_url:
        normalized_server_url = f"http://{normalized_server_url}"

    parsed = urlparse(normalized_server_url)
    parsed_host = parsed.hostname or "127.0.0.1"
    parsed_port = int(parsed.port or normalized_port or 8080)
    scheme = parsed.scheme or "http"
    return f"{scheme}://{parsed_host}:{parsed_port}", parsed_host, parsed_port


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


__all__ = [
    "DEFAULT_OCR_PROVIDER",
    "DEFAULT_OCR_PROVIDER_MODE",
    "DEFAULT_LOCAL_OCR_PROVIDER",
    "LOCAL_OCR_PROVIDER_CHOICES",
    "OCR_PROVIDER_CHROME_LENS",
    "OCR_PROVIDER_DEEPSEEK_OCR_LLAMA",
    "OCR_PROVIDER_CHOICES",
    "OCR_PROVIDER_LABELS",
    "OCR_PROVIDER_MODE_CHOICES",
    "OCR_PROVIDER_MODE_CHROME_LENS",
    "OCR_PROVIDER_MODE_LOCAL",
    "OCR_PROVIDER_PADDLE_VL_LLAMA",
    "OCRConfig",
    "is_known_ocr_provider",
    "normalize_local_ocr_provider",
    "normalize_ocr_provider_name",
    "normalize_ocr_provider_mode",
    "provider_label",
    "resolve_ocr_provider",
    "selection_from_provider",
]

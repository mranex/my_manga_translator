"""Small translation-stage models and config helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .prompt_models import (
    PROMPT_MODE_BUILT_IN_PLUS_USER,
    PROMPT_MODE_CHOICES,
    PROMPT_MODE_FULL_CUSTOM,
)
from .prompt_studio import (
    build_prompt_preview,
    default_output_contract,
    normalize_custom_styles,
    normalize_prompt_mode,
    resolve_prompt_style,
)
from .prompt_styles import DEFAULT_PROMPT_STYLE_ID

TRANSLATOR_CHOICES = (
    "Gemini",
    "Local LLM",
    "DeepSeek",
    "OpenAI Compatible",
    "Google",
    "NLLB",
    "Baidu",
    "Bing",
)

LANGUAGE_CHOICES = (
    "ja",
    "en",
    "vi",
    "zh",
    "ko",
    "th",
    "id",
    "fr",
    "de",
    "es",
    "ru",
)

OPENAI_COMPATIBLE_PRESET_OPENAI = "OpenAI"
OPENAI_COMPATIBLE_PRESET_OPENROUTER = "OpenRouter"
OPENAI_COMPATIBLE_PRESET_LM_STUDIO = "LM Studio"
OPENAI_COMPATIBLE_PRESET_LOCALAI = "LocalAI"
OPENAI_COMPATIBLE_PRESET_CUSTOM = "Custom"

OPENAI_COMPATIBLE_PRESET_CHOICES = (
    OPENAI_COMPATIBLE_PRESET_OPENAI,
    OPENAI_COMPATIBLE_PRESET_OPENROUTER,
    OPENAI_COMPATIBLE_PRESET_LM_STUDIO,
    OPENAI_COMPATIBLE_PRESET_LOCALAI,
    OPENAI_COMPATIBLE_PRESET_CUSTOM,
)

OPENAI_COMPATIBLE_PRESET_BASE_URLS: dict[str, str] = {
    OPENAI_COMPATIBLE_PRESET_OPENAI: "https://api.openai.com/v1",
    OPENAI_COMPATIBLE_PRESET_OPENROUTER: "https://openrouter.ai/api/v1",
    OPENAI_COMPATIBLE_PRESET_LM_STUDIO: "http://localhost:1234/v1",
    OPENAI_COMPATIBLE_PRESET_LOCALAI: "http://localhost:8080/v1",
    OPENAI_COMPATIBLE_PRESET_CUSTOM: "",
}

STYLE_PROMPTS: dict[str, str] = {
    style.display_name: style.instructions
    for style in (
        resolve_prompt_style(DEFAULT_PROMPT_STYLE_ID),
        *[
            resolve_prompt_style(style_id)
            for style_id in (
                "natural_vietnamese",
                "casual_dialogue",
                "formal_polite",
                "keep_honorifics",
                "short_bubble_fit",
                "literal_faithful",
                "web_novel_style",
                "action_impact",
                "comedy_light_slang",
                "custom",
            )
        ],
    )
}


@dataclass(slots=True)
class TranslationConfig:
    """Serializable configuration for translation initialization and execution."""

    source_language: str = "ja"
    target_language: str = "en"
    translator: str = "Google"
    style: str = "Default Manga"
    custom_prompt: str = ""
    batch_size_pages: int = 3
    use_context_memory: bool = False
    local_llm_server_url: str = "http://127.0.0.1:8080"
    local_llm_model: str = "gpt-4o"
    gemini_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_thinking: bool = False
    prompt_style_id: str = DEFAULT_PROMPT_STYLE_ID
    prompt_mode: str = PROMPT_MODE_BUILT_IN_PLUS_USER
    user_instructions: str = ""
    full_custom_prompt: str = ""
    project_translation_notes: str = ""
    output_contract: str = ""
    custom_styles: list[dict[str, Any]] = field(default_factory=list)
    openai_compatible_provider_preset: str = OPENAI_COMPATIBLE_PRESET_OPENAI
    openai_compatible_base_url: str = OPENAI_COMPATIBLE_PRESET_BASE_URLS[OPENAI_COMPATIBLE_PRESET_OPENAI]
    openai_compatible_api_key: str = ""
    openai_compatible_model: str = ""
    openai_compatible_temperature: float = 0.3
    openai_compatible_max_tokens: int = 0
    openai_compatible_timeout: int = 120
    openai_compatible_json_mode: bool = True

    @classmethod
    def from_value(cls, value: "TranslationConfig | dict[str, Any] | None") -> "TranslationConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls()

        raw_style = str(value.get("style", "") or "").strip()
        raw_custom_prompt = str(value.get("custom_prompt", "") or "")
        prompt_mode = normalize_prompt_mode(value.get("prompt_mode"))
        prompt_style_id = str(value.get("prompt_style_id", "") or "").strip()
        user_instructions = str(value.get("user_instructions", "") or "")
        full_custom_prompt = str(value.get("full_custom_prompt", "") or "")

        if not str(value.get("prompt_mode", "") or "").strip():
            if raw_style.strip().lower() == "custom":
                prompt_mode = PROMPT_MODE_FULL_CUSTOM
                if not full_custom_prompt.strip():
                    full_custom_prompt = raw_custom_prompt
            elif raw_custom_prompt.strip() and not user_instructions.strip():
                user_instructions = raw_custom_prompt

        if not prompt_style_id:
            resolved_style = resolve_prompt_style(raw_style or DEFAULT_PROMPT_STYLE_ID)
            prompt_style_id = resolved_style.id

        custom_styles = normalize_custom_styles(value.get("custom_styles", []))
        resolved_style = resolve_prompt_style(prompt_style_id, custom_styles=custom_styles)
        style_display_name = str(value.get("style", "") or "").strip() or resolved_style.display_name

        openai_preset = normalize_openai_compatible_preset(value.get("openai_compatible_provider_preset"))
        default_base_url = OPENAI_COMPATIBLE_PRESET_BASE_URLS.get(openai_preset, "")

        return cls(
            source_language=str(value.get("source_language", "ja") or "ja"),
            target_language=str(value.get("target_language", "en") or "en"),
            translator=str(value.get("translator", "Google") or "Google"),
            style=style_display_name,
            custom_prompt=raw_custom_prompt,
            batch_size_pages=_coerce_positive_int(value.get("batch_size_pages"), 3),
            use_context_memory=bool(value.get("use_context_memory", False)),
            local_llm_server_url=str(value.get("local_llm_server_url", "http://127.0.0.1:8080") or "http://127.0.0.1:8080"),
            local_llm_model=str(value.get("local_llm_model", "gpt-4o") or "gpt-4o"),
            gemini_api_key=str(value.get("gemini_api_key", "") or ""),
            deepseek_api_key=str(value.get("deepseek_api_key", "") or ""),
            deepseek_model=str(value.get("deepseek_model", "deepseek-v4-flash") or "deepseek-v4-flash"),
            deepseek_thinking=bool(value.get("deepseek_thinking", False)),
            prompt_style_id=prompt_style_id,
            prompt_mode=prompt_mode,
            user_instructions=user_instructions,
            full_custom_prompt=full_custom_prompt,
            project_translation_notes=str(value.get("project_translation_notes", "") or ""),
            output_contract=str(value.get("output_contract", "") or ""),
            custom_styles=custom_styles,
            openai_compatible_provider_preset=openai_preset,
            openai_compatible_base_url=str(value.get("openai_compatible_base_url", "") or default_base_url),
            openai_compatible_api_key=str(value.get("openai_compatible_api_key", "") or ""),
            openai_compatible_model=str(value.get("openai_compatible_model", "") or ""),
            openai_compatible_temperature=_coerce_float(value.get("openai_compatible_temperature"), 0.3),
            openai_compatible_max_tokens=_coerce_non_negative_int(value.get("openai_compatible_max_tokens"), 0),
            openai_compatible_timeout=_coerce_positive_int(value.get("openai_compatible_timeout"), 120),
            openai_compatible_json_mode=bool(value.get("openai_compatible_json_mode", True)),
        )

    @property
    def translator_key(self) -> str:
        return normalize_translator_name(self.translator)

    @property
    def supports_page_batch(self) -> bool:
        return self.translator_key in {"gemini", "local_llm", "deepseek", "openai_compatible"}

    @property
    def prompt_build_result(self):
        return build_prompt_preview(
            source_language=self.source_language,
            target_language=self.target_language,
            prompt_style_id=self.prompt_style_id,
            prompt_mode=self.prompt_mode,
            user_instructions=self.user_instructions,
            full_custom_prompt=self.full_custom_prompt,
            project_notes=self.project_translation_notes,
            output_contract=self.output_contract or default_output_contract(),
            custom_styles=self.custom_styles,
        )

    def effective_prompt(self) -> str:
        return self.prompt_build_result.prompt_text

    def prompt_hash(self) -> str:
        return self.prompt_build_result.prompt_hash

    def prompt_preview(self) -> str:
        return self.prompt_build_result.prompt_preview

    def resolved_style_name(self) -> str:
        return self.prompt_build_result.style_name

    def resolved_output_contract(self) -> str:
        return self.prompt_build_result.output_contract

    def to_metadata(self) -> dict[str, Any]:
        prompt_result = self.prompt_build_result
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "translator": self.translator,
            "style": prompt_result.style_name,
            "custom_prompt": prompt_result.prompt_text,
            "batch_size_pages": int(self.batch_size_pages),
            "use_context_memory": bool(self.use_context_memory),
            "local_llm_server_url": self.local_llm_server_url,
            "local_llm_model": self.local_llm_model,
            "gemini_api_key": self.gemini_api_key,
            "deepseek_api_key": self.deepseek_api_key,
            "deepseek_model": self.deepseek_model,
            "deepseek_thinking": bool(self.deepseek_thinking),
            "prompt_style_id": prompt_result.style_id,
            "prompt_mode": prompt_result.prompt_mode,
            "user_instructions": self.user_instructions,
            "full_custom_prompt": self.full_custom_prompt,
            "project_translation_notes": self.project_translation_notes,
            "output_contract": prompt_result.output_contract,
            "prompt_preview": prompt_result.prompt_preview,
            "prompt_hash": prompt_result.prompt_hash,
            "custom_styles": normalize_custom_styles(self.custom_styles),
            "openai_compatible_provider_preset": self.openai_compatible_provider_preset,
            "openai_compatible_base_url": self.openai_compatible_base_url,
            "openai_compatible_api_key": self.openai_compatible_api_key,
            "openai_compatible_model": self.openai_compatible_model,
            "openai_compatible_temperature": float(self.openai_compatible_temperature),
            "openai_compatible_max_tokens": int(self.openai_compatible_max_tokens),
            "openai_compatible_timeout": int(self.openai_compatible_timeout),
            "openai_compatible_json_mode": bool(self.openai_compatible_json_mode),
        }


def normalize_translator_name(name: str) -> str:
    cleaned = str(name or "").strip().lower()
    mapping = {
        "gemini": "gemini",
        "local llm": "local_llm",
        "local_llm": "local_llm",
        "deepseek": "deepseek",
        "openai compatible": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "google": "google",
        "nllb": "nllb",
        "baidu": "baidu",
        "bing": "bing",
    }
    return mapping.get(cleaned, "google")


def normalize_openai_compatible_preset(name: Any) -> str:
    normalized = str(name or "").strip().lower()
    mapping = {
        "openai": OPENAI_COMPATIBLE_PRESET_OPENAI,
        "openrouter": OPENAI_COMPATIBLE_PRESET_OPENROUTER,
        "lm studio": OPENAI_COMPATIBLE_PRESET_LM_STUDIO,
        "lm_studio": OPENAI_COMPATIBLE_PRESET_LM_STUDIO,
        "localai": OPENAI_COMPATIBLE_PRESET_LOCALAI,
        "custom": OPENAI_COMPATIBLE_PRESET_CUSTOM,
    }
    return mapping.get(normalized, OPENAI_COMPATIBLE_PRESET_OPENAI)


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _coerce_non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return parsed if parsed >= 0 else int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


__all__ = [
    "LANGUAGE_CHOICES",
    "OPENAI_COMPATIBLE_PRESET_BASE_URLS",
    "OPENAI_COMPATIBLE_PRESET_CHOICES",
    "OPENAI_COMPATIBLE_PRESET_CUSTOM",
    "OPENAI_COMPATIBLE_PRESET_LM_STUDIO",
    "OPENAI_COMPATIBLE_PRESET_LOCALAI",
    "OPENAI_COMPATIBLE_PRESET_OPENAI",
    "OPENAI_COMPATIBLE_PRESET_OPENROUTER",
    "PROMPT_MODE_BUILT_IN_PLUS_USER",
    "PROMPT_MODE_CHOICES",
    "PROMPT_MODE_FULL_CUSTOM",
    "STYLE_PROMPTS",
    "TRANSLATOR_CHOICES",
    "TranslationConfig",
    "normalize_openai_compatible_preset",
    "normalize_translator_name",
]

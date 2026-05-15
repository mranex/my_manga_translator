"""Lazy translator package exports.

This keeps package import side effects small so GUI-facing code can load only the
provider modules it needs at runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["GeminiTranslator", "MangaTranslator", "OpenAICompatibleTranslator"]


def __getattr__(name: str) -> Any:
    if name == "MangaTranslator":
        return import_module(".translator", __name__).MangaTranslator
    if name == "GeminiTranslator":
        return import_module(".gemini_translator", __name__).GeminiTranslator
    if name == "OpenAICompatibleTranslator":
        return import_module(".openai_compatible_translator", __name__).OpenAICompatibleTranslator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

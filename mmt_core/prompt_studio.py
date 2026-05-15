"""Prompt Studio helpers for translation providers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any
from uuid import uuid4

from .prompt_models import (
    PROMPT_MODE_BUILT_IN_PLUS_USER,
    PROMPT_MODE_FULL_CUSTOM,
    PromptBuildResult,
    PromptStyle,
)
from .prompt_styles import (
    BUILT_IN_PROMPT_STYLES,
    DEFAULT_PROMPT_STYLE_ID,
    LEGACY_STYLE_ALIASES,
)


def list_builtin_prompt_styles() -> list[PromptStyle]:
    return [PromptStyle(**style.to_dict()) for style in BUILT_IN_PROMPT_STYLES]


def default_output_contract() -> str:
    return (
        "Return only the structured result requested by the caller.\n"
        "- Preserve every page key exactly as provided.\n"
        "- Preserve item count exactly.\n"
        "- Preserve item order exactly.\n"
        "- Do not merge, split, drop, or invent items.\n"
        "- Empty input items should map to empty output strings.\n"
        "- Do not add explanations, notes, markdown, or commentary outside the structured result."
    )


def normalize_prompt_mode(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == PROMPT_MODE_FULL_CUSTOM:
        return PROMPT_MODE_FULL_CUSTOM
    return PROMPT_MODE_BUILT_IN_PLUS_USER


def normalize_custom_styles(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized_styles: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue

        display_name = str(item.get("display_name", "") or "").strip()
        instructions = str(item.get("instructions", "") or "").strip()
        if not display_name or not instructions:
            continue

        style_id = _normalize_style_id(item.get("id"), fallback_name=display_name, suffix=index)
        if style_id in seen_ids:
            style_id = f"{style_id}_{index + 1}"
        seen_ids.add(style_id)

        normalized_styles.append(
            {
                "id": style_id,
                "display_name": display_name,
                "short_description": str(item.get("short_description", "") or item.get("description", "") or "").strip(),
                "instructions": instructions,
                "built_in": False,
                "created_at": str(item.get("created_at", "") or ""),
                "updated_at": str(item.get("updated_at", "") or ""),
            }
        )

    return normalized_styles


def create_custom_style(
    display_name: str,
    instructions: str,
    *,
    description: str = "",
    existing_styles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    style_name = str(display_name or "").strip()
    style_instructions = str(instructions or "").strip()
    if not style_name:
        raise ValueError("Custom style name is required.")
    if not style_instructions:
        raise ValueError("Custom style instructions are empty.")

    existing_ids = {
        str(item.get("id", "") or "")
        for item in normalize_custom_styles(existing_styles or [])
    }
    slug = _slugify(style_name) or "style"
    style_id = f"custom_{slug}"
    while style_id in existing_ids:
        style_id = f"custom_{slug}_{uuid4().hex[:8]}"

    timestamp = _timestamp()
    return {
        "id": style_id,
        "display_name": style_name,
        "short_description": str(description or "").strip(),
        "instructions": style_instructions,
        "built_in": False,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def delete_custom_style(custom_styles: list[dict[str, Any]] | None, style_id: str) -> list[dict[str, Any]]:
    target_id = str(style_id or "").strip()
    return [
        style
        for style in normalize_custom_styles(custom_styles or [])
        if str(style.get("id", "") or "") != target_id
    ]


def resolve_prompt_style(
    style_id: str | None,
    *,
    custom_styles: list[dict[str, Any]] | None = None,
) -> PromptStyle:
    lookup: dict[str, PromptStyle] = {}
    for style in BUILT_IN_PROMPT_STYLES:
        clone = PromptStyle(**style.to_dict())
        lookup[clone.id] = clone
        lookup[clone.display_name.strip().lower()] = clone

    for style in normalize_custom_styles(custom_styles or []):
        prompt_style = PromptStyle(
            id=str(style.get("id", "") or ""),
            display_name=str(style.get("display_name", "") or ""),
            short_description=str(style.get("short_description", "") or ""),
            instructions=str(style.get("instructions", "") or ""),
            built_in=False,
            created_at=str(style.get("created_at", "") or ""),
            updated_at=str(style.get("updated_at", "") or ""),
        )
        lookup[prompt_style.id] = prompt_style
        lookup[prompt_style.display_name.strip().lower()] = prompt_style

    normalized_key = str(style_id or "").strip()
    if not normalized_key:
        normalized_key = DEFAULT_PROMPT_STYLE_ID

    alias_key = LEGACY_STYLE_ALIASES.get(normalized_key.strip().lower())
    if alias_key and alias_key in lookup:
        return lookup[alias_key]
    if normalized_key in lookup:
        return lookup[normalized_key]
    lowered = normalized_key.lower()
    if lowered in lookup:
        return lookup[lowered]
    return lookup[DEFAULT_PROMPT_STYLE_ID]


def build_prompt_preview(
    *,
    source_language: str,
    target_language: str,
    prompt_style_id: str,
    prompt_mode: str,
    user_instructions: str = "",
    full_custom_prompt: str = "",
    project_notes: str = "",
    output_contract: str | None = None,
    custom_styles: list[dict[str, Any]] | None = None,
) -> PromptBuildResult:
    resolved_mode = normalize_prompt_mode(prompt_mode)
    style = resolve_prompt_style(prompt_style_id, custom_styles=custom_styles)
    final_output_contract = str(output_contract or "").strip() or default_output_contract()
    normalized_source = str(source_language or "").strip() or "ja"
    normalized_target = str(target_language or "").strip() or "en"
    style_instructions = str(style.instructions or "").strip()
    extra_instructions = str(user_instructions or "").strip()
    custom_override = str(full_custom_prompt or "").strip()
    notes = str(project_notes or "").strip()

    sections = [
        (
            "Role / Task",
            "\n".join(
                [
                    "You are a manga/comic translator.",
                    f"Translate from {normalized_source} to {normalized_target}.",
                    "Preserve meaning, tone, character voice, and reading flow.",
                ]
            ),
        )
    ]

    if resolved_mode == PROMPT_MODE_FULL_CUSTOM:
        custom_body = custom_override or style_instructions or (
            "Provide translation instructions for the target tone, terminology, and voice."
        )
        sections.append(("Custom Prompt Override", custom_body))
    else:
        sections.append(
            (
                f"Style Instructions ({style.display_name})",
                style_instructions or "Use clear, natural manga dialogue with concise speech-bubble fit phrasing.",
            )
        )
        if extra_instructions:
            sections.append(("User Extra Instructions", extra_instructions))

    if notes:
        sections.append(("Project Translation Notes", notes))

    sections.append(("Output Contract", final_output_contract))

    prompt_text = "\n\n".join(
        f"{title}:\n{body.strip()}"
        for title, body in sections
        if str(body or "").strip()
    ).strip()
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    return PromptBuildResult(
        prompt_mode=resolved_mode,
        prompt_text=prompt_text,
        prompt_hash=prompt_hash,
        prompt_preview=prompt_text,
        output_contract=final_output_contract,
        style_id=style.id,
        style_name=style.display_name,
        style_description=style.short_description,
        style_instructions=style_instructions,
        user_instructions=extra_instructions,
        project_notes=notes,
        full_custom_prompt=custom_override,
    )


def _normalize_style_id(value: Any, *, fallback_name: str, suffix: int) -> str:
    raw_value = str(value or "").strip()
    if raw_value:
        return raw_value
    return f"custom_{_slugify(fallback_name) or 'style'}_{suffix + 1}"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "build_prompt_preview",
    "create_custom_style",
    "default_output_contract",
    "delete_custom_style",
    "list_builtin_prompt_styles",
    "normalize_custom_styles",
    "normalize_prompt_mode",
    "resolve_prompt_style",
]

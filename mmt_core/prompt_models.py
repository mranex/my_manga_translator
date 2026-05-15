"""Prompt Studio data models and constants."""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_MODE_BUILT_IN_PLUS_USER = "built_in_plus_user_instructions"
PROMPT_MODE_FULL_CUSTOM = "full_custom_prompt"

PROMPT_MODE_CHOICES = (
    (PROMPT_MODE_BUILT_IN_PLUS_USER, "Built-in style + my instructions"),
    (PROMPT_MODE_FULL_CUSTOM, "Full custom prompt"),
)


@dataclass(slots=True)
class PromptStyle:
    """Serializable prompt style definition."""

    id: str
    display_name: str
    short_description: str
    instructions: str
    built_in: bool = True
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "short_description": self.short_description,
            "instructions": self.instructions,
            "built_in": bool(self.built_in),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class PromptBuildResult:
    """Built prompt plus resolved style metadata."""

    prompt_mode: str
    prompt_text: str
    prompt_hash: str
    prompt_preview: str
    output_contract: str
    style_id: str
    style_name: str
    style_description: str
    style_instructions: str
    user_instructions: str
    project_notes: str
    full_custom_prompt: str


__all__ = [
    "PROMPT_MODE_BUILT_IN_PLUS_USER",
    "PROMPT_MODE_CHOICES",
    "PROMPT_MODE_FULL_CUSTOM",
    "PromptBuildResult",
    "PromptStyle",
]

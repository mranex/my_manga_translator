"""Built-in Prompt Studio translation styles."""

from __future__ import annotations

from .prompt_models import PromptStyle

BUILT_IN_PROMPT_STYLES: tuple[PromptStyle, ...] = (
    PromptStyle(
        id="default_manga",
        display_name="Default Manga",
        short_description="Balanced manga dialogue with natural pacing and clean tone handling.",
        instructions=(
            "Translate as natural manga dialogue that reads smoothly inside speech bubbles. "
            "Preserve character voice, emotion, and subtext. Keep names stable. "
            "Do not add explanations, translator notes, or extra narration."
        ),
    ),
    PromptStyle(
        id="natural_vietnamese",
        display_name="Natural Vietnamese",
        short_description="Prioritizes conversational Vietnamese that sounds spoken, not textbook-like.",
        instructions=(
            "Prefer natural spoken Vietnamese over literal grammar. Choose pronouns from the relationship "
            "and keep them consistent. Avoid stiff textbook wording and avoid verbose lines that would feel "
            "unnatural in manga dialogue."
        ),
    ),
    PromptStyle(
        id="casual_dialogue",
        display_name="Casual Dialogue",
        short_description="Light, conversational, everyday speech for friendly or relaxed scenes.",
        instructions=(
            "Use casual, natural everyday dialogue. Let the lines feel spoken aloud between familiar people. "
            "Preserve teasing, hesitation, embarrassment, and slang when it fits the scene."
        ),
    ),
    PromptStyle(
        id="formal_polite",
        display_name="Formal / Polite",
        short_description="Respectful and polished wording without sounding robotic.",
        instructions=(
            "Use polite, respectful wording while keeping the lines natural and readable. "
            "Preserve hierarchy, deference, and restraint when the scene depends on it."
        ),
    ),
    PromptStyle(
        id="keep_honorifics",
        display_name="Keep Honorifics",
        short_description="Preserves honorifics and relationship cues from the source text.",
        instructions=(
            "Preserve Japanese honorifics such as -san, -kun, -chan, -sama, senpai, and sensei when they carry "
            "relationship meaning. Keep names stable and avoid replacing these cues with bland neutral wording."
        ),
    ),
    PromptStyle(
        id="short_bubble_fit",
        display_name="Short Bubble Fit",
        short_description="Keeps translations compact for tight speech bubbles.",
        instructions=(
            "Keep each line concise enough to fit comfortably inside a manga speech bubble. "
            "Prefer short, punchy phrasing without dropping meaning, emotion, or speaker intent."
        ),
    ),
    PromptStyle(
        id="literal_faithful",
        display_name="Literal / Faithful",
        short_description="Closer to source meaning while staying understandable.",
        instructions=(
            "Stay as faithful as possible to the source meaning and nuance while still sounding readable. "
            "Do not over-localize or embellish. Preserve wording choices when they matter to the scene."
        ),
    ),
    PromptStyle(
        id="web_novel_style",
        display_name="Web Novel Style",
        short_description="Dramatic and emotionally weighted wording for heightened scenes.",
        instructions=(
            "Use dramatic, emotionally vivid wording in the style of a polished web novel adaptation. "
            "Keep the emotional force strong, but avoid becoming florid or too long for speech bubbles."
        ),
    ),
    PromptStyle(
        id="action_impact",
        display_name="Action / Impact",
        short_description="Fast, punchy lines for confrontations, shouting, and impact beats.",
        instructions=(
            "Favor short, high-impact lines with strong rhythm. Preserve intensity, speed, and aggression. "
            "Do not soften threats, shouts, or battle banter."
        ),
    ),
    PromptStyle(
        id="comedy_light_slang",
        display_name="Comedy / Light Slang",
        short_description="Lighter, playful dialogue with room for natural slang.",
        instructions=(
            "Keep the dialogue playful, nimble, and funny. Use light slang where it feels natural, "
            "but avoid turning every line into a joke. Preserve timing and punchlines."
        ),
    ),
    PromptStyle(
        id="custom",
        display_name="Custom",
        short_description="Placeholder style for saved custom prompts or full custom override mode.",
        instructions="",
    ),
)

LEGACY_STYLE_ALIASES: dict[str, str] = {
    "default": "default_manga",
    "default manga": "default_manga",
    "casual": "casual_dialogue",
    "casual dialogue": "casual_dialogue",
    "formal": "formal_polite",
    "formal / polite": "formal_polite",
    "keep honorifics": "keep_honorifics",
    "short bubble fit": "short_bubble_fit",
    "literal": "literal_faithful",
    "literal / faithful": "literal_faithful",
    "web novel style": "web_novel_style",
    "action": "action_impact",
    "action / impact": "action_impact",
    "comedy / light slang": "comedy_light_slang",
    "custom": "custom",
}

DEFAULT_PROMPT_STYLE_ID = "default_manga"

__all__ = [
    "BUILT_IN_PROMPT_STYLES",
    "DEFAULT_PROMPT_STYLE_ID",
    "LEGACY_STYLE_ALIASES",
]

"""Pipeline-side OCR text filtering for provider outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re


DEEPSEEK_PROVIDER_KEY = "deepseek_ocr_llama"
STRUCTURED_TAG_MARKERS = (
    "<table",
    "</table>",
    "<thead",
    "<tbody",
    "<tr",
    "<td",
    "<svg",
    "<path",
    "<math",
    "<figure",
    "<chart",
)
COORDINATE_RE = re.compile(r"\(?\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)?")
TABLE_LIKE_WORD_RE = re.compile(r"(category|value|blue|red|total|name|score)", re.IGNORECASE)
STRUCTURAL_PUNCTUATION = set("{}[]()<>/:;|,=-_")


@dataclass(slots=True)
class OCRTextFilterResult:
    text: str
    rejected: bool
    reason: str = ""


def filter_ocr_text_for_pipeline(
    text: str,
    *,
    provider_key: str = "",
) -> OCRTextFilterResult:
    return _filter_ocr_text(str(text or ""), provider_key=str(provider_key or "").strip(), depth=0)


def _filter_ocr_text(text: str, *, provider_key: str, depth: int) -> OCRTextFilterResult:
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return OCRTextFilterResult(text="", rejected=True, reason="empty_ocr_output")

    if depth > 3:
        return OCRTextFilterResult(text="", rejected=True, reason="structured_json_non_text")

    if provider_key != DEEPSEEK_PROVIDER_KEY:
        return OCRTextFilterResult(text=cleaned, rejected=False, reason="")

    lowered = cleaned.lower()
    if any(marker in lowered for marker in STRUCTURED_TAG_MARKERS):
        return OCRTextFilterResult(text="", rejected=True, reason="structured_markup_non_text")

    json_result = _filter_json_payload(cleaned, provider_key=provider_key, depth=depth)
    if json_result is not None:
        return json_result

    coordinate_matches = COORDINATE_RE.findall(cleaned)
    if (len(coordinate_matches) >= 3 and "--" in cleaned) or len(coordinate_matches) >= 6:
        return OCRTextFilterResult(text="", rejected=True, reason="coordinate_geometry_non_text")

    if _looks_like_structured_hallucination(cleaned):
        return OCRTextFilterResult(text="", rejected=True, reason="structured_hallucination_non_text")

    if _looks_like_table_compaction(cleaned):
        return OCRTextFilterResult(text="", rejected=True, reason="table_like_non_text")

    return OCRTextFilterResult(text=cleaned, rejected=False, reason="")


def _filter_json_payload(text: str, *, provider_key: str, depth: int) -> OCRTextFilterResult | None:
    try:
        parsed = json.loads(text)
    except Exception:
        return None

    if isinstance(parsed, dict):
        for key in ("text", "ocr", "content", "result"):
            value = parsed.get(key)
            if isinstance(value, str):
                return _filter_ocr_text(value, provider_key=provider_key, depth=depth + 1)
        return OCRTextFilterResult(text="", rejected=True, reason="structured_json_non_text")

    if isinstance(parsed, list):
        if parsed and all(isinstance(value, str) for value in parsed):
            joined = "\n".join(value for value in parsed if value.strip())
            return _filter_ocr_text(joined, provider_key=provider_key, depth=depth + 1)
        return OCRTextFilterResult(text="", rejected=True, reason="structured_json_non_text")

    return OCRTextFilterResult(text="", rejected=True, reason="structured_json_non_text")


def _looks_like_structured_hallucination(text: str) -> bool:
    compact = [char for char in text if not char.isspace()]
    if len(compact) <= 120:
        return False

    structural_count = sum(1 for char in compact if char in STRUCTURAL_PUNCTUATION)
    if structural_count / max(len(compact), 1) <= 0.45:
        return False

    alpha_cjk_count = sum(1 for char in compact if _is_alpha_or_cjk(char))
    return alpha_cjk_count < structural_count


def _looks_like_table_compaction(text: str) -> bool:
    if len(text) <= 20:
        return False
    if not any(char.isdigit() for char in text):
        return False
    if len(re.findall(r"\s+", text)) > 1:
        return False

    matches = TABLE_LIKE_WORD_RE.findall(text)
    if len(matches) < 2:
        return False

    distinct_matches = {match.lower() for match in matches}
    return len(distinct_matches) >= 2


def _is_alpha_or_cjk(char: str) -> bool:
    if char.isalpha():
        return True
    codepoint = ord(char)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


__all__ = [
    "OCRTextFilterResult",
    "filter_ocr_text_for_pipeline",
]

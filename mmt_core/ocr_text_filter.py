"""Pipeline-side OCR text filtering for provider outputs."""

from __future__ import annotations

from collections import Counter
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
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+")


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

    repetition_result = _filter_repetition_hallucination(cleaned)
    if repetition_result is not None:
        return repetition_result

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


def _filter_repetition_hallucination(text: str) -> OCRTextFilterResult | None:
    repeated_line_reason = _repeated_line_hallucination_reason(text)
    if repeated_line_reason:
        return OCRTextFilterResult(text="", rejected=True, reason=repeated_line_reason)

    lines = text.split("\n")
    kept_lines: list[str] = []
    removed_reasons: list[str] = []

    for line in lines:
        normalized_line = line.strip()
        if not normalized_line:
            continue

        rejection_reason = _repetition_rejection_reason(normalized_line)
        if rejection_reason:
            removed_reasons.append(rejection_reason)
            continue
        kept_lines.append(normalized_line)

    if not removed_reasons:
        return None

    if kept_lines:
        return OCRTextFilterResult(text="\n".join(kept_lines), rejected=False, reason="")

    return OCRTextFilterResult(text="", rejected=True, reason=removed_reasons[0])


def _repeated_line_hallucination_reason(text: str) -> str:
    raw_lines = [line.strip() for line in text.split("\n")]
    non_empty_lines = [line for line in raw_lines if line]
    if len(non_empty_lines) < 5:
        return ""

    normalized_lines = [_normalize_repeated_line(line) for line in non_empty_lines]
    normalized_lines = [line for line in normalized_lines if line]
    if len(normalized_lines) < 5:
        return ""

    counts = Counter(normalized_lines)
    if len(counts) > 2:
        return ""

    most_common_count = counts.most_common(1)[0][1]
    if (most_common_count / len(normalized_lines)) < 0.70:
        return ""

    return "repeated_line_hallucination"


def _normalize_repeated_line(text: str) -> str:
    return re.sub(r"^[\W_]+|[\W_]+$", "", text.casefold()).strip()


def _repetition_rejection_reason(text: str) -> str:
    compact = "".join(char for char in text if not char.isspace())
    if _is_repeated_character_hallucination(compact):
        return "repeated_character_hallucination"
    if _is_repeated_token_hallucination(text):
        return "repeated_token_hallucination"
    if _is_repeated_pattern_hallucination(compact):
        return "repeated_pattern_hallucination"
    return ""


def _is_repeated_character_hallucination(compact_text: str) -> bool:
    if len(compact_text) < 12:
        return False

    counts = Counter(compact_text)
    if len(counts) > 2:
        return False

    most_common_count = counts.most_common(1)[0][1]
    return (most_common_count / len(compact_text)) >= 0.85


def _is_repeated_token_hallucination(text: str) -> bool:
    tokens = TOKEN_RE.findall(text)
    if len(tokens) < 6:
        return False

    normalized_tokens = [token.casefold() for token in tokens if token.strip()]
    if len(normalized_tokens) < 6:
        return False

    counts = Counter(normalized_tokens)
    if len(counts) > 2:
        return False

    most_common_count = counts.most_common(1)[0][1]
    return (most_common_count / len(normalized_tokens)) >= 0.70


def _is_repeated_pattern_hallucination(compact_text: str) -> bool:
    if len(compact_text) < 16:
        return False

    for unit_length in range(2, min(8, len(compact_text) // 4) + 1):
        unit = compact_text[:unit_length]
        repeated_chars = 0
        offset = 0
        while offset + unit_length <= len(compact_text) and compact_text[offset:offset + unit_length] == unit:
            repeated_chars += unit_length
            offset += unit_length

        if repeated_chars < unit_length * 4:
            continue
        if (repeated_chars / len(compact_text)) >= 0.80:
            return True

    return False


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

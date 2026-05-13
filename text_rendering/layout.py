from __future__ import annotations

from dataclasses import dataclass, field

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    Image = None
    ImageDraw = None
    ImageFont = None

from .fonts import get_cached_font


def _require_pillow():
    if Image is None or ImageDraw is None or ImageFont is None:
        raise ModuleNotFoundError("Pillow is required for text rendering")


@dataclass(slots=True)
class TextLayoutResult:
    font: object
    font_size: int
    wrapped_text: str
    lines: list[str]
    text_width: int
    text_height: int
    line_spacing: int
    text_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    origin_offset: tuple[int, int] = (0, 0)
    used_align: str = "center"
    overflow: bool = False
    debug: dict = field(default_factory=dict)


def _make_measure_draw():
    _require_pillow()
    canvas = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    return ImageDraw.Draw(canvas)


def measure_multiline_text(draw, text, font, *, align, spacing, stroke_width):
    bbox = draw.multiline_textbbox(
        (0, 0),
        text or " ",
        font=font,
        align=align,
        spacing=spacing,
        stroke_width=stroke_width,
    )
    width = max(0, int(bbox[2] - bbox[0]))
    height = max(0, int(bbox[3] - bbox[1]))
    return width, height, tuple(int(value) for value in bbox)


def _measure_text_block(draw, text: str, font, *, align: str, spacing: int, stroke_width: int = 0):
    width, height, _ = measure_multiline_text(
        draw,
        text,
        font,
        align=align,
        spacing=spacing,
        stroke_width=stroke_width,
    )
    return width, height


def _split_long_token(token: str, draw, font, max_width: int, *, stroke_width: int = 0) -> list[str]:
    if not token:
        return [""]
    pieces: list[str] = []
    current = ""
    for character in token:
        candidate = current + character
        width, _ = _measure_text_block(
            draw,
            candidate,
            font,
            align="left",
            spacing=0,
            stroke_width=stroke_width,
        )
        if current and width > max_width:
            pieces.append(current)
            current = character
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [token]


def _greedy_wrap_words(words: list[str], draw, font, max_width: int, *, stroke_width: int = 0) -> list[str]:
    if not words:
        return [""]

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        width, _ = _measure_text_block(
            draw,
            candidate,
            font,
            align="left",
            spacing=0,
            stroke_width=stroke_width,
        )
        if current and width > max_width:
            lines.append(current)
            current = word
            current_width, _ = _measure_text_block(
                draw,
                current,
                font,
                align="left",
                spacing=0,
                stroke_width=stroke_width,
            )
            if current_width > max_width:
                fragments = _split_long_token(current, draw, font, max_width, stroke_width=stroke_width)
                lines.extend(fragments[:-1])
                current = fragments[-1]
        else:
            current = candidate

    if current:
        lines.append(current)
    return lines or [""]


def _rebalance_lines(lines: list[str], draw, font, max_width: int, *, stroke_width: int = 0) -> list[str]:
    if len(lines) < 2:
        return lines

    balanced = list(lines)
    changed = True
    while changed:
        changed = False
        for index in range(len(balanced) - 1):
            left_words = balanced[index].split()
            right_words = balanced[index + 1].split()
            if len(left_words) < 2 or not right_words:
                continue

            left_width, _ = _measure_text_block(
                draw,
                balanced[index],
                font,
                align="left",
                spacing=0,
                stroke_width=stroke_width,
            )
            right_width, _ = _measure_text_block(
                draw,
                balanced[index + 1],
                font,
                align="left",
                spacing=0,
                stroke_width=stroke_width,
            )

            candidate_left = " ".join(left_words[:-1])
            candidate_right = f"{left_words[-1]} {' '.join(right_words)}".strip()
            candidate_left_width, _ = _measure_text_block(
                draw,
                candidate_left,
                font,
                align="left",
                spacing=0,
                stroke_width=stroke_width,
            )
            candidate_right_width, _ = _measure_text_block(
                draw,
                candidate_right,
                font,
                align="left",
                spacing=0,
                stroke_width=stroke_width,
            )

            if candidate_right_width > max_width:
                continue

            current_spread = abs(left_width - right_width)
            candidate_spread = abs(candidate_left_width - candidate_right_width)
            if candidate_spread + 2 < current_spread:
                balanced[index] = candidate_left
                balanced[index + 1] = candidate_right
                changed = True
    return balanced


def wrap_text_to_width(
    text: str,
    font,
    max_width: int,
    *,
    align: str = "center",
    stroke_width: int = 0,
) -> list[str]:
    _require_pillow()
    draw = _make_measure_draw()
    if not text:
        return [""]
    if max_width <= 0:
        return [text]

    wrapped_lines: list[str] = []
    paragraphs = text.splitlines() or [text]
    for paragraph in paragraphs:
        stripped = paragraph.strip()
        if not stripped:
            wrapped_lines.append("")
            continue
        words = stripped.split()
        paragraph_lines = _greedy_wrap_words(
            words,
            draw,
            font,
            max_width,
            stroke_width=stroke_width,
        )
        wrapped_lines.extend(
            _rebalance_lines(
                paragraph_lines,
                draw,
                font,
                max_width,
                stroke_width=stroke_width,
            )
        )

    return wrapped_lines or [text]


def optimal_wrap_text_to_width(
    text: str,
    font,
    max_width: int,
    *,
    stroke_width: int = 0,
    spacing: int = 0,
    balance_strength: float = 1.0,
) -> list[str]:
    _require_pillow()
    if not text:
        return [""]
    if max_width <= 0:
        return [text]

    draw = _make_measure_draw()
    wrapped_lines: list[str] = []

    paragraphs = text.splitlines() or [text]
    for paragraph in paragraphs:
        stripped = paragraph.strip()
        if not stripped:
            wrapped_lines.append("")
            continue
        words = stripped.split()
        if len(words) <= 2:
            wrapped_lines.extend(
                wrap_text_to_width(
                    stripped,
                    font,
                    max_width,
                    align="left",
                    stroke_width=stroke_width,
                )
            )
            continue

        tokens: list[str] = []
        for word in words:
            width, _ = _measure_text_block(
                draw,
                word,
                font,
                align="left",
                spacing=spacing,
                stroke_width=stroke_width,
            )
            if width > max_width:
                tokens.extend(_split_long_token(word, draw, font, max_width, stroke_width=stroke_width))
            else:
                tokens.append(word)

        token_count = len(tokens)
        if token_count == 0:
            wrapped_lines.append("")
            continue
        if token_count > 60:
            wrapped_lines.extend(
                wrap_text_to_width(
                    stripped,
                    font,
                    max_width,
                    align="left",
                    stroke_width=stroke_width,
                )
            )
            continue

        line_width_cache: dict[tuple[int, int], int] = {}

        def line_width(start: int, end: int) -> int:
            cache_key = (start, end)
            if cache_key not in line_width_cache:
                candidate = " ".join(tokens[start:end])
                width, _ = _measure_text_block(
                    draw,
                    candidate,
                    font,
                    align="left",
                    spacing=spacing,
                    stroke_width=stroke_width,
                )
                line_width_cache[cache_key] = width
            return line_width_cache[cache_key]

        best_cost = [float("inf")] * (token_count + 1)
        next_break = [token_count] * (token_count + 1)
        best_cost[token_count] = 0.0

        max_width_float = float(max_width)
        for start in range(token_count - 1, -1, -1):
            for end in range(start + 1, token_count + 1):
                width = line_width(start, end)
                overflow = max(0, width - max_width)
                if overflow > 0 and end > start + 1:
                    break

                if overflow > 0:
                    cost = 1_000_000.0 + (overflow / max_width_float) ** 2 * 10_000.0
                else:
                    leftover = max_width - width
                    normalized_leftover = leftover / max_width_float
                    cost = (normalized_leftover ** 2) * float(balance_strength)
                    cost += 0.035
                    if end == token_count:
                        if width < max_width * 0.45 and token_count > 3:
                            cost += 0.45
                        if len(tokens[start:end]) == 1 and start > 0:
                            cost += 0.10
                    else:
                        cost += 0.02 * (end - start == 1)

                total_cost = cost + best_cost[end]
                if total_cost < best_cost[start]:
                    best_cost[start] = total_cost
                    next_break[start] = end

        if best_cost[0] == float("inf"):
            wrapped_lines.extend(
                wrap_text_to_width(
                    stripped,
                    font,
                    max_width,
                    align="left",
                    stroke_width=stroke_width,
                )
            )
            continue

        paragraph_lines: list[str] = []
        cursor = 0
        while cursor < token_count:
            next_cursor = next_break[cursor]
            if next_cursor <= cursor:
                next_cursor = cursor + 1
            paragraph_lines.append(" ".join(tokens[cursor:next_cursor]))
            cursor = next_cursor

        wrapped_lines.extend(paragraph_lines)

    return wrapped_lines or [text]


def fit_text_layout(
    text: str,
    font_path: str,
    width: int,
    height: int,
    *,
    align: str = "center",
    padding: int = 2,
    min_font_size: int = 6,
    max_font_size: int = 160,
    stroke_width: int = 0,
    line_spacing_ratio: float = 0.18,
    min_line_spacing: int = 1,
) -> TextLayoutResult:
    _require_pillow()
    usable_width = max(1, int(width) - (int(padding) * 2))
    usable_height = max(1, int(height) - (int(padding) * 2))
    draw = _make_measure_draw()

    best_result: TextLayoutResult | None = None
    low = int(min_font_size)
    high = int(max_font_size)

    while low <= high:
        mid = (low + high) // 2
        font = get_cached_font(font_path, mid)
        base_spacing = max(int(min_line_spacing), int(round(mid * float(line_spacing_ratio))))
        spacing_candidates = [base_spacing]
        if base_spacing > min_line_spacing:
            spacing_candidates.extend(
                spacing
                for spacing in range(base_spacing - 1, int(min_line_spacing) - 1, -1)
                if spacing not in spacing_candidates and spacing >= max(int(min_line_spacing), base_spacing - 3)
            )
            if int(min_line_spacing) not in spacing_candidates:
                spacing_candidates.append(int(min_line_spacing))

        lines = optimal_wrap_text_to_width(
            text,
            font,
            usable_width,
            stroke_width=stroke_width,
            spacing=base_spacing,
        )
        wrapped_text = "\n".join(lines)
        fitted_candidate: TextLayoutResult | None = None

        for spacing in spacing_candidates:
            text_width, text_height, text_bbox = measure_multiline_text(
                draw,
                wrapped_text,
                font,
                align=align,
                spacing=spacing,
                stroke_width=stroke_width,
            )
            if text_width <= usable_width and text_height <= usable_height:
                fitted_candidate = TextLayoutResult(
                    font=font,
                    font_size=mid,
                    wrapped_text=wrapped_text,
                    lines=lines,
                    text_width=text_width,
                    text_height=text_height,
                    line_spacing=spacing,
                    text_bbox=text_bbox,
                    origin_offset=(text_bbox[0], text_bbox[1]),
                    used_align=align,
                    overflow=False,
                    debug={"spacing_candidates": spacing_candidates},
                )
                break

        if fitted_candidate is not None:
            best_result = fitted_candidate
            low = mid + 1
        else:
            high = mid - 1

    if best_result is not None:
        return best_result

    fallback_size = int(min_font_size)
    font = get_cached_font(font_path, fallback_size)
    line_spacing = max(int(min_line_spacing), int(round(fallback_size * float(line_spacing_ratio))))
    lines = optimal_wrap_text_to_width(
        text,
        font,
        usable_width,
        stroke_width=stroke_width,
        spacing=line_spacing,
    )
    wrapped_text = "\n".join(lines)
    text_width, text_height, text_bbox = measure_multiline_text(
        draw,
        wrapped_text,
        font,
        align=align,
        spacing=line_spacing,
        stroke_width=stroke_width,
    )
    return TextLayoutResult(
        font=font,
        font_size=fallback_size,
        wrapped_text=wrapped_text,
        lines=lines,
        text_width=text_width,
        text_height=text_height,
        line_spacing=line_spacing,
        text_bbox=text_bbox,
        origin_offset=(text_bbox[0], text_bbox[1]),
        used_align=align,
        overflow=text_width > usable_width or text_height > usable_height,
        debug={"spacing_candidates": [line_spacing]},
    )


__all__ = [
    "TextLayoutResult",
    "fit_text_layout",
    "get_cached_font",
    "measure_multiline_text",
    "optimal_wrap_text_to_width",
    "wrap_text_to_width",
]

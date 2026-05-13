from __future__ import annotations

from dataclasses import dataclass

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    Image = None
    ImageDraw = None
    ImageFont = None


_FONT_CACHE: dict[tuple[str, int], object] = {}


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


def get_cached_font(font_path: str, size: int):
    _require_pillow()
    cache_key = (font_path, int(size))
    if cache_key not in _FONT_CACHE:
        try:
            _FONT_CACHE[cache_key] = ImageFont.truetype(font_path, size=int(size))
        except Exception:
            _FONT_CACHE[cache_key] = ImageFont.load_default()
    return _FONT_CACHE[cache_key]


def _make_measure_draw():
    _require_pillow()
    canvas = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    return ImageDraw.Draw(canvas)


def _measure_text_block(draw, text: str, font, *, align: str, spacing: int, stroke_width: int = 0):
    bbox = draw.multiline_textbbox(
        (0, 0),
        text or " ",
        font=font,
        align=align,
        spacing=spacing,
        stroke_width=stroke_width,
    )
    return max(0, int(bbox[2] - bbox[0])), max(0, int(bbox[3] - bbox[1]))


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
        line_spacing = max(1, int(round(mid * 0.25)))
        lines = wrap_text_to_width(
            text,
            font,
            usable_width,
            align=align,
            stroke_width=stroke_width,
        )
        wrapped_text = "\n".join(lines)
        text_width, text_height = _measure_text_block(
            draw,
            wrapped_text,
            font,
            align=align,
            spacing=line_spacing,
            stroke_width=stroke_width,
        )

        if text_width <= usable_width and text_height <= usable_height:
            best_result = TextLayoutResult(
                font=font,
                font_size=mid,
                wrapped_text=wrapped_text,
                lines=lines,
                text_width=text_width,
                text_height=text_height,
                line_spacing=line_spacing,
            )
            low = mid + 1
        else:
            high = mid - 1

    if best_result is not None:
        return best_result

    fallback_size = int(min_font_size)
    font = get_cached_font(font_path, fallback_size)
    line_spacing = max(1, int(round(fallback_size * 0.25)))
    lines = wrap_text_to_width(
        text,
        font,
        usable_width,
        align=align,
        stroke_width=stroke_width,
    )
    wrapped_text = "\n".join(lines)
    text_width, text_height = _measure_text_block(
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
    )


__all__ = [
    "TextLayoutResult",
    "fit_text_layout",
    "get_cached_font",
    "wrap_text_to_width",
]

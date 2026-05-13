from __future__ import annotations

import unicodedata

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None

from .blocks import MangaRenderBlock, MangaRenderStyle, block_from_legacy_args
from .effects import alpha_composite_onto_bgr, choose_text_color_for_region
from .fonts import find_fallback_font_for_text
from .layout import fit_text_layout, measure_multiline_text


def _require_dependencies():
    if np is None or Image is None or ImageDraw is None:
        raise ModuleNotFoundError("numpy and Pillow are required for text rendering")


def contains_cjk(text: str) -> bool:
    for character in text:
        if not character.strip():
            continue
        name = unicodedata.name(character, "")
        if any(token in name for token in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            return True
    return False


def _render_vertical_text_simple(draw, block: MangaRenderBlock, font, width: int, height: int, padding: int, fill_rgba, stroke_rgba, stroke_width: int, spacing: int):
    characters = [character for character in block.text if character != "\n"]
    if not characters:
        return

    char_metrics = []
    max_char_width = 0
    max_char_height = 0
    for character in characters:
        bbox = draw.textbbox(
            (0, 0),
            character,
            font=font,
            stroke_width=stroke_width,
        )
        char_metrics.append((character, bbox))
        max_char_width = max(max_char_width, int(bbox[2] - bbox[0]))
        max_char_height = max(max_char_height, int(bbox[3] - bbox[1]))

    usable_width = max(1, width - (padding * 2))
    usable_height = max(1, height - (padding * 2))
    rows_per_column = max(1, usable_height // max(1, max_char_height + spacing))
    column_count = max(1, (len(characters) + rows_per_column - 1) // rows_per_column)
    total_width = column_count * max_char_width + max(0, column_count - 1) * spacing
    start_x = padding + max(0, (usable_width - total_width) // 2)

    for index, (character, bbox) in enumerate(char_metrics):
        column = index // rows_per_column
        row = index % rows_per_column
        text_x = start_x + (column_count - 1 - column) * (max_char_width + spacing) - int(bbox[0])
        text_y = padding + row * (max_char_height + spacing) - int(bbox[1])
        draw.text(
            (text_x, text_y),
            character,
            font=font,
            fill=fill_rgba,
            stroke_width=stroke_width,
            stroke_fill=stroke_rgba,
        )


def _clamp_position(raw_value: float, min_value: float, max_value: float) -> int:
    if max_value < min_value:
        return int(round(raw_value))
    return int(round(min(max(raw_value, min_value), max_value)))


def render_manga_text_block(
    image_bgr,
    block: MangaRenderBlock,
    style: MangaRenderStyle | None = None,
):
    _require_dependencies()
    style = MangaRenderStyle() if style is None else style
    if not block.text or not block.text.strip():
        return image_bgr

    x1, y1, x2, y2 = [int(value) for value in block.bbox]
    if x2 <= x1 or y2 <= y1:
        return image_bgr

    kind = block.kind or "bubble"
    width = x2 - x1
    height = y2 - y1
    padding = int(style.padding + (1 if kind == "outside_text" else 0))
    scale = max(1, int(style.supersample))
    scaled_width = max(1, width * scale)
    scaled_height = max(1, height * scale)
    scaled_padding = max(0, padding * scale)

    resolved_font_path = find_fallback_font_for_text(block.text, block.font_path)
    text_color = block.text_color
    stroke_color = block.stroke_color
    stroke_width = block.stroke_width
    if text_color is None:
        text_color, stroke_color, stroke_width = choose_text_color_for_region(
            image_bgr,
            block.bbox,
            prefer_stroke=(kind == "outside_text"),
            is_dark=block.is_dark,
        )
    elif stroke_width is None:
        _, default_stroke_color, default_stroke_width = choose_text_color_for_region(
            image_bgr,
            block.bbox,
            prefer_stroke=(kind == "outside_text"),
            is_dark=block.is_dark,
        )
        if stroke_color is None:
            stroke_color = default_stroke_color
        stroke_width = default_stroke_width

    if stroke_color is None and kind == "outside_text":
        stroke_color = (0, 0, 0) if text_color == (255, 255, 255) else (255, 255, 255)

    if stroke_width is None:
        stroke_width = max(1, int(round(min(width, height) / 36.0)))

    stroke_boost = style.outside_text_stroke_boost if kind == "outside_text" else style.bubble_stroke_boost
    max_stroke = max(1, int(round(min(width, height) * float(style.max_stroke_ratio))))
    effective_stroke = min(max_stroke, max(0, int(round(float(stroke_width) * float(stroke_boost)))))
    if stroke_color is None:
        effective_stroke = 0
    scaled_stroke = max(0, effective_stroke * scale)

    max_font_size = style.max_font_size if style.max_font_size is not None else max(96, min(220, max(width, height) * 2))

    overlay = Image.new("RGBA", (scaled_width, scaled_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    use_vertical = bool(
        block.source_direction == "vertical"
        and contains_cjk(block.text)
        and scaled_height > int(scaled_width * 1.1)
    )

    if use_vertical:
        from .fonts import get_cached_font

        font = get_cached_font(resolved_font_path, max(style.min_font_size * scale, min(scaled_width, scaled_height) // 6))
        fill_rgba = tuple(int(channel) for channel in text_color) + (255,)
        stroke_rgba = None if stroke_color is None else tuple(int(channel) for channel in stroke_color) + (255,)
        spacing = max(int(style.min_line_spacing), int(round((min(scaled_width, scaled_height) / 16.0))))
        _render_vertical_text_simple(
            draw,
            block,
            font,
            scaled_width,
            scaled_height,
            scaled_padding,
            fill_rgba,
            stroke_rgba,
            scaled_stroke,
            spacing,
        )
    else:
        layout = fit_text_layout(
            block.text,
            resolved_font_path,
            scaled_width,
            scaled_height,
            align=block.align,
            padding=scaled_padding,
            min_font_size=int(style.min_font_size * scale),
            max_font_size=int(max_font_size * scale),
            stroke_width=scaled_stroke,
            line_spacing_ratio=style.line_spacing_ratio,
            min_line_spacing=int(style.min_line_spacing * scale),
        )

        available_x1 = scaled_padding
        available_y1 = scaled_padding
        available_width = max(1, scaled_width - (scaled_padding * 2))
        available_height = max(1, scaled_height - (scaled_padding * 2))

        text_bbox = layout.text_bbox
        text_width = layout.text_width
        text_height = layout.text_height

        if block.align == "left":
            raw_x = available_x1 - text_bbox[0]
        elif block.align == "right":
            raw_x = available_x1 + available_width - text_width - text_bbox[0]
        else:
            raw_x = available_x1 + ((available_width - text_width) / 2.0) - text_bbox[0]
        raw_y = available_y1 + ((available_height - text_height) / 2.0) - text_bbox[1]

        min_x = available_x1 - text_bbox[0]
        max_x = scaled_width - scaled_padding - text_width - text_bbox[0]
        min_y = available_y1 - text_bbox[1]
        max_y = scaled_height - scaled_padding - text_height - text_bbox[1]

        text_x = _clamp_position(raw_x, min_x, max_x)
        text_y = _clamp_position(raw_y, min_y, max_y)

        fill_rgba = tuple(int(channel) for channel in text_color) + (255,)
        stroke_rgba = None if stroke_color is None else tuple(int(channel) for channel in stroke_color) + (255,)

        draw.multiline_text(
            (text_x, text_y),
            layout.wrapped_text,
            font=layout.font,
            fill=fill_rgba,
            align=block.align,
            spacing=layout.line_spacing,
            stroke_width=scaled_stroke,
            stroke_fill=stroke_rgba,
        )

    if scale > 1:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        overlay = overlay.resize((width, height), resample=resampling)

    alpha_composite_onto_bgr(
        image_bgr,
        np.asarray(overlay, dtype=np.uint8),
        (x1, y1, x2, y2),
    )
    return image_bgr


def render_text_block(
    image_bgr,
    text: str,
    bbox: tuple[int, int, int, int],
    font_path: str,
    text_color: tuple[int, int, int],
    stroke_color: tuple[int, int, int] | None = None,
    stroke_width: int | None = None,
    align: str = "center",
    vertical: bool = False,
    padding: int = 2,
    supersample: int = 3,
):
    block, style = block_from_legacy_args(
        text,
        bbox,
        font_path,
        text_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        align=align,
        vertical=vertical,
        padding=padding,
        supersample=supersample,
    )
    return render_manga_text_block(image_bgr, block, style)


__all__ = [
    "contains_cjk",
    "render_manga_text_block",
    "render_text_block",
]

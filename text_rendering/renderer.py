from __future__ import annotations

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None

from .effects import alpha_composite_onto_bgr
from .layout import fit_text_layout


def _require_dependencies():
    if np is None or Image is None or ImageDraw is None:
        raise ModuleNotFoundError("numpy and Pillow are required for text rendering")


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
    _require_dependencies()
    if not text or not text.strip():
        return image_bgr

    x1, y1, x2, y2 = [int(value) for value in bbox]
    if x2 <= x1 or y2 <= y1:
        return image_bgr

    width = x2 - x1
    height = y2 - y1
    scale = max(1, int(supersample))

    scaled_width = max(1, width * scale)
    scaled_height = max(1, height * scale)
    scaled_padding = max(0, int(padding) * scale)

    estimated_stroke = stroke_width
    if estimated_stroke is None:
        estimated_stroke = max(1, int(round(min(width, height) / 32.0)))
    scaled_stroke = max(0, int(estimated_stroke) * scale)

    layout = fit_text_layout(
        text,
        font_path,
        scaled_width,
        scaled_height,
        align=align,
        padding=scaled_padding,
        min_font_size=6 * scale,
        max_font_size=max(96, min(220, max(width, height) * scale)),
        stroke_width=scaled_stroke,
    )

    overlay = Image.new("RGBA", (scaled_width, scaled_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    text_bbox = draw.multiline_textbbox(
        (0, 0),
        layout.wrapped_text,
        font=layout.font,
        align=align,
        spacing=layout.line_spacing,
        stroke_width=scaled_stroke,
    )
    text_width = int(text_bbox[2] - text_bbox[0])
    text_height = int(text_bbox[3] - text_bbox[1])

    if align == "left":
        text_x = scaled_padding - int(text_bbox[0])
    elif align == "right":
        text_x = scaled_width - scaled_padding - text_width - int(text_bbox[0])
    else:
        text_x = max(scaled_padding, (scaled_width - text_width) // 2 - int(text_bbox[0]))
    text_y = max(scaled_padding, (scaled_height - text_height) // 2 - int(text_bbox[1]))

    fill_rgba = tuple(int(channel) for channel in text_color) + (255,)
    stroke_rgba = None if stroke_color is None else tuple(int(channel) for channel in stroke_color) + (255,)

    draw.multiline_text(
        (text_x, text_y),
        layout.wrapped_text,
        font=layout.font,
        fill=fill_rgba,
        align=align,
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


__all__ = [
    "render_text_block",
]

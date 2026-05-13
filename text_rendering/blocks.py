from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MangaRenderBlock:
    bbox: tuple[int, int, int, int]
    text: str
    kind: str = "bubble"
    source_direction: str | None = None
    is_dark: bool | None = None
    font_path: str | None = None
    text_color: tuple[int, int, int] | None = None
    stroke_color: tuple[int, int, int] | None = None
    stroke_width: int | None = None
    align: str = "center"


@dataclass(slots=True)
class MangaRenderStyle:
    padding: int = 2
    supersample: int = 3
    min_font_size: int = 6
    max_font_size: int | None = None
    line_spacing_ratio: float = 0.18
    min_line_spacing: int = 1
    outside_text_stroke_boost: float = 1.35
    bubble_stroke_boost: float = 1.0
    max_stroke_ratio: float = 0.07
    no_background_box: bool = True


def block_from_legacy_args(
    text: str,
    bbox: tuple[int, int, int, int],
    font_path: str,
    text_color: tuple[int, int, int],
    *,
    stroke_color: tuple[int, int, int] | None = None,
    stroke_width: int | None = None,
    align: str = "center",
    vertical: bool = False,
    padding: int = 2,
    supersample: int = 3,
):
    block = MangaRenderBlock(
        bbox=tuple(int(value) for value in bbox),
        text=text,
        kind="bubble",
        source_direction="vertical" if vertical else "horizontal",
        font_path=font_path,
        text_color=text_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        align=align,
    )
    style = MangaRenderStyle(
        padding=int(padding),
        supersample=int(supersample),
    )
    return block, style


__all__ = [
    "MangaRenderBlock",
    "MangaRenderStyle",
    "block_from_legacy_args",
]

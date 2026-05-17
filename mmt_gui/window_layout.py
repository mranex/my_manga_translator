"""Window-layout helpers for safe geometry restore."""

from __future__ import annotations

DEFAULT_MAIN_WINDOW_MIN_WIDTH = 960
DEFAULT_MAIN_WINDOW_MIN_HEIGHT = 680
DEFAULT_MAIN_WINDOW_WIDTH = 1600
DEFAULT_MAIN_WINDOW_HEIGHT = 920
DEFAULT_CONTENT_SPLITTER_SIZES = [1120, 480]
DEFAULT_WORKSPACE_SPLITTER_SIZES = [760, 220]
WINDOW_LAYOUT_VERSION = 4


def clamp_window_geometry(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    available_x: int,
    available_y: int,
    available_width: int,
    available_height: int,
    minimum_width: int = DEFAULT_MAIN_WINDOW_MIN_WIDTH,
    minimum_height: int = DEFAULT_MAIN_WINDOW_MIN_HEIGHT,
) -> tuple[int, int, int, int]:
    """Clamp saved window geometry to the current available screen."""

    safe_available_width = max(320, int(available_width))
    safe_available_height = max(240, int(available_height))
    safe_min_width = max(320, min(int(minimum_width), safe_available_width))
    safe_min_height = max(240, min(int(minimum_height), safe_available_height))

    safe_width = max(safe_min_width, int(width or 0))
    safe_height = max(safe_min_height, int(height or 0))
    safe_width = min(safe_width, safe_available_width)
    safe_height = min(safe_height, safe_available_height)

    max_x = int(available_x) + safe_available_width - safe_width
    max_y = int(available_y) + safe_available_height - safe_height
    safe_x = max(int(available_x), min(int(x), max_x))
    safe_y = max(int(available_y), min(int(y), max_y))
    return safe_x, safe_y, safe_width, safe_height


__all__ = [
    "DEFAULT_CONTENT_SPLITTER_SIZES",
    "DEFAULT_MAIN_WINDOW_HEIGHT",
    "DEFAULT_MAIN_WINDOW_MIN_HEIGHT",
    "DEFAULT_MAIN_WINDOW_MIN_WIDTH",
    "DEFAULT_MAIN_WINDOW_WIDTH",
    "DEFAULT_WORKSPACE_SPLITTER_SIZES",
    "WINDOW_LAYOUT_VERSION",
    "clamp_window_geometry",
]

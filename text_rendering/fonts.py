from __future__ import annotations

import os
from pathlib import Path

try:
    from PIL import ImageFont
except ModuleNotFoundError:
    ImageFont = None


_FONT_CACHE: dict[tuple[str, int], object] = {}
_DEFAULT_FONT_SENTINEL = "__PIL_DEFAULT__"


def _require_pillow():
    if ImageFont is None:
        raise ModuleNotFoundError("Pillow is required for text rendering fonts")


def get_cached_font(font_path: str | None, size: int):
    _require_pillow()
    normalized_path = (font_path or "").strip() or _DEFAULT_FONT_SENTINEL
    cache_key = (normalized_path, int(size))
    if cache_key not in _FONT_CACHE:
        try:
            if normalized_path == _DEFAULT_FONT_SENTINEL:
                raise OSError("using Pillow default font")
            _FONT_CACHE[cache_key] = ImageFont.truetype(normalized_path, size=int(size))
        except Exception:
            _FONT_CACHE[cache_key] = ImageFont.load_default()
    return _FONT_CACHE[cache_key]


def _sample_characters(text: str, sample_limit: int = 64) -> list[str]:
    sampled = []
    seen = set()
    for character in text:
        if character.isspace():
            continue
        if character in seen:
            continue
        seen.add(character)
        sampled.append(character)
        if len(sampled) >= sample_limit:
            break
    return sampled


def font_supports_text(font_path: str | None, text: str, sample_limit: int = 64) -> bool:
    _require_pillow()
    if not text.strip():
        return True
    sampled = _sample_characters(text, sample_limit=sample_limit)
    if not sampled:
        return True

    font = get_cached_font(font_path, 24)
    supported_count = 0
    for character in sampled:
        try:
            bbox = font.getbbox(character)
        except Exception:
            bbox = None
        if bbox is not None and (bbox[2] - bbox[0]) >= 0 and (bbox[3] - bbox[1]) >= 0:
            supported_count += 1
    return supported_count >= max(1, int(len(sampled) * 0.6))


def _candidate_font_paths(preferred_font_path: str | None, fallback_dirs=None):
    yielded = set()

    def _yield(path_candidate):
        normalized = str(path_candidate)
        if not normalized or normalized in yielded:
            return
        yielded.add(normalized)
        yield normalized

    if preferred_font_path:
        preferred_path = Path(preferred_font_path)
        if preferred_path.exists():
            yield from _yield(preferred_path.resolve())

    font_dirs = []
    if fallback_dirs:
        font_dirs.extend(Path(path) for path in fallback_dirs)

    project_fonts = Path(__file__).resolve().parents[1] / "fonts"
    font_dirs.append(project_fonts)

    windows_fonts = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
    font_dirs.append(windows_fonts)
    font_dirs.extend(
        Path(path)
        for path in (
            "/usr/share/fonts/truetype",
            "/usr/share/fonts",
            "/usr/local/share/fonts",
        )
    )

    for font_dir in font_dirs:
        if not font_dir.exists():
            continue
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for font_file in sorted(font_dir.rglob(pattern)):
                yield from _yield(font_file.resolve())


def find_fallback_font_for_text(text: str, preferred_font_path: str | None, fallback_dirs=None) -> str:
    _require_pillow()
    for candidate_path in _candidate_font_paths(preferred_font_path, fallback_dirs=fallback_dirs):
        try:
            if font_supports_text(candidate_path, text):
                return candidate_path
        except Exception:
            continue
    return (preferred_font_path or "").strip()


__all__ = [
    "find_fallback_font_for_text",
    "font_supports_text",
    "get_cached_font",
]

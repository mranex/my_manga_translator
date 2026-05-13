import os
import unittest

try:
    from PIL import ImageFont
except ModuleNotFoundError:
    ImageFont = None

from text_rendering.fonts import find_fallback_font_for_text, font_supports_text, get_cached_font


def _find_test_font():
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


FONT_PATH = _find_test_font()


class TestTextRenderingFonts(unittest.TestCase):
    @unittest.skipIf(ImageFont is None, "Pillow is required")
    def test_missing_font_path_does_not_crash(self):
        font = get_cached_font("C:/definitely_missing_font.ttf", 18)
        self.assertIsNotNone(font)

    @unittest.skipIf(ImageFont is None or FONT_PATH is None, "Pillow and a font are required")
    def test_font_supports_text_returns_true_for_ascii_text(self):
        self.assertTrue(font_supports_text(FONT_PATH, "Hello manga"))

    @unittest.skipIf(ImageFont is None, "Pillow is required")
    def test_find_fallback_font_for_text_returns_usable_path(self):
        resolved = find_fallback_font_for_text("Hello manga", "C:/definitely_missing_font.ttf")
        font = get_cached_font(resolved, 18)
        self.assertIsNotNone(font)


if __name__ == "__main__":
    unittest.main()

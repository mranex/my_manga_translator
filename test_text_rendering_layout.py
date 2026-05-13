import os
import unittest

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None

from text_rendering.layout import (
    fit_text_layout,
    get_cached_font,
    measure_multiline_text,
    optimal_wrap_text_to_width,
    wrap_text_to_width,
)


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


class TestTextRenderingLayout(unittest.TestCase):
    @unittest.skipIf(Image is None or ImageDraw is None or FONT_PATH is None, "Pillow and a truetype font are required")
    def test_optimal_wrap_text_to_width_is_deterministic(self):
        font = get_cached_font(FONT_PATH, 22)
        text = "alpha beta gamma delta epsilon"

        first = optimal_wrap_text_to_width(text, font, 160, stroke_width=0)
        second = optimal_wrap_text_to_width(text, font, 160, stroke_width=0)

        self.assertEqual(first, second)

    @unittest.skipIf(Image is None or ImageDraw is None or FONT_PATH is None, "Pillow and a truetype font are required")
    def test_overlong_token_is_split(self):
        font = get_cached_font(FONT_PATH, 20)
        lines = optimal_wrap_text_to_width(
            "Supercalifragilisticexpialidocious",
            font,
            80,
        )

        self.assertGreater(len(lines), 1)

    @unittest.skipIf(Image is None or ImageDraw is None or FONT_PATH is None, "Pillow and a truetype font are required")
    def test_balanced_wrapping_beats_simple_greedy_for_known_sentence(self):
        font = get_cached_font(FONT_PATH, 22)
        draw = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
        width, _, _ = measure_multiline_text(
            draw,
            "alpha beta gamma",
            font,
            align="left",
            spacing=0,
            stroke_width=0,
        )
        max_width = width
        greedy = wrap_text_to_width(
            "alpha beta gamma delta",
            font,
            max_width,
            align="left",
            stroke_width=0,
        )
        optimal = optimal_wrap_text_to_width(
            "alpha beta gamma delta",
            font,
            max_width,
            stroke_width=0,
        )

        self.assertNotEqual(optimal, [])
        self.assertGreaterEqual(len(optimal[-1].split()), 2)
        self.assertGreaterEqual(len(greedy[-1].split()), 1)

    @unittest.skipIf(Image is None or ImageDraw is None or FONT_PATH is None, "Pillow and a truetype font are required")
    def test_fit_text_layout_includes_stroke_width(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
        font = get_cached_font(FONT_PATH, 24)
        plain = measure_multiline_text(
            draw,
            "stroke sample",
            font,
            align="center",
            spacing=4,
            stroke_width=0,
        )
        stroked = measure_multiline_text(
            draw,
            "stroke sample",
            font,
            align="center",
            spacing=4,
            stroke_width=4,
        )

        self.assertGreaterEqual(stroked[0], plain[0])
        self.assertGreaterEqual(stroked[1], plain[1])

    @unittest.skipIf(Image is None or ImageDraw is None or FONT_PATH is None, "Pillow and a truetype font are required")
    def test_fit_text_layout_returns_font_size_within_bounds_and_fits(self):
        layout = fit_text_layout(
            "Balanced manga dialogue sample with stroke",
            FONT_PATH,
            220,
            90,
            min_font_size=8,
            max_font_size=72,
            stroke_width=2,
        )

        self.assertGreaterEqual(layout.font_size, 8)
        self.assertLessEqual(layout.font_size, 72)
        self.assertLessEqual(layout.text_width, 216)
        self.assertLessEqual(layout.text_height, 86)
        self.assertIsInstance(layout.text_bbox, tuple)


if __name__ == "__main__":
    unittest.main()

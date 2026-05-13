import os
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None

from text_rendering.layout import fit_text_layout, get_cached_font, wrap_text_to_width
from text_rendering.renderer import render_text_block


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


class TestTextRendering(unittest.TestCase):
    @unittest.skipIf(np is None or Image is None or ImageDraw is None or FONT_PATH is None, "numpy, Pillow, and a truetype font are required")
    def test_wrap_text_fits_width(self):
        font = get_cached_font(FONT_PATH, 24)
        lines = wrap_text_to_width(
            "This is a fairly long sentence for wrapping",
            font,
            140,
        )

        draw = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
        for line in lines:
            bbox = draw.textbbox((0, 0), line or " ", font=font)
            self.assertLessEqual(int(bbox[2] - bbox[0]), 140)

    @unittest.skipIf(np is None or Image is None or ImageDraw is None or FONT_PATH is None, "numpy, Pillow, and a truetype font are required")
    def test_fit_text_layout_returns_font_size_within_bounds(self):
        layout = fit_text_layout(
            "Balanced manga dialogue sample",
            FONT_PATH,
            180,
            72,
            min_font_size=8,
            max_font_size=64,
        )

        self.assertGreaterEqual(layout.font_size, 8)
        self.assertLessEqual(layout.font_size, 64)
        self.assertLessEqual(layout.text_width, 176)
        self.assertLessEqual(layout.text_height, 68)

    @unittest.skipIf(np is None or Image is None or ImageDraw is None or FONT_PATH is None, "numpy, Pillow, and a truetype font are required")
    def test_render_text_block_keeps_pixels_outside_bbox_unchanged(self):
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        original = image.copy()
        bbox = (20, 20, 80, 60)

        render_text_block(
            image,
            "Hello manga",
            bbox,
            FONT_PATH,
            (0, 0, 0),
            stroke_color=None,
            stroke_width=0,
            supersample=2,
        )

        self.assertTrue(np.array_equal(image[:20, :, :], original[:20, :, :]))
        self.assertTrue(np.array_equal(image[60:, :, :], original[60:, :, :]))
        self.assertTrue(np.array_equal(image[:, :20, :], original[:, :20, :]))
        self.assertTrue(np.array_equal(image[:, 80:, :], original[:, 80:, :]))
        self.assertFalse(np.array_equal(image[20:60, 20:80, :], original[20:60, 20:80, :]))

    @unittest.skipIf(np is None or Image is None or ImageDraw is None or FONT_PATH is None, "numpy, Pillow, and a truetype font are required")
    def test_render_text_block_uses_alpha_without_background_rectangle(self):
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        original = image.copy()
        bbox = (20, 20, 80, 60)

        render_text_block(
            image,
            "No box",
            bbox,
            FONT_PATH,
            (0, 0, 0),
            stroke_color=None,
            stroke_width=0,
            supersample=3,
        )

        changed = np.any(image != original, axis=2)
        bbox_changed = changed[20:60, 20:80]

        self.assertEqual(tuple(image[21, 21]), (255, 255, 255))
        self.assertGreater(int(np.count_nonzero(bbox_changed)), 0)
        self.assertLess(int(np.count_nonzero(bbox_changed)), int(bbox_changed.size * 0.60))


if __name__ == "__main__":
    unittest.main()

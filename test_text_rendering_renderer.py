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

from text_rendering.blocks import MangaRenderBlock
from text_rendering.effects import choose_render_style_for_item
from text_rendering.renderer import render_manga_text_block, render_text_block


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


class TestTextRenderingRenderer(unittest.TestCase):
    @unittest.skipIf(np is None or Image is None or ImageDraw is None or FONT_PATH is None, "numpy, Pillow, and a truetype font are required")
    def test_render_manga_text_block_does_not_modify_pixels_outside_bbox(self):
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        original = image.copy()
        bbox = (20, 20, 80, 60)
        block = MangaRenderBlock(
            bbox=bbox,
            text="Hello manga",
            kind="bubble",
            font_path=FONT_PATH,
            text_color=(0, 0, 0),
            stroke_color=None,
            stroke_width=0,
        )

        render_manga_text_block(image, block)

        self.assertTrue(np.array_equal(image[:20, :, :], original[:20, :, :]))
        self.assertTrue(np.array_equal(image[60:, :, :], original[60:, :, :]))
        self.assertTrue(np.array_equal(image[:, :20, :], original[:, :20, :]))
        self.assertTrue(np.array_equal(image[:, 80:, :], original[:, 80:, :]))

    @unittest.skipIf(np is None or Image is None or ImageDraw is None or FONT_PATH is None, "numpy, Pillow, and a truetype font are required")
    def test_render_manga_text_block_has_no_background_rectangle(self):
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        original = image.copy()
        bbox = (20, 20, 80, 60)
        block = MangaRenderBlock(
            bbox=bbox,
            text="No box",
            kind="outside_text",
            font_path=FONT_PATH,
            text_color=(0, 0, 0),
            stroke_color=(255, 255, 255),
            stroke_width=2,
        )

        render_manga_text_block(image, block)

        changed = np.any(image != original, axis=2)
        bbox_changed = changed[20:60, 20:80]
        self.assertEqual(tuple(image[21, 21]), (255, 255, 255))
        self.assertGreater(int(np.count_nonzero(bbox_changed)), 0)
        self.assertLess(int(np.count_nonzero(bbox_changed)), int(bbox_changed.size * 0.70))

    @unittest.skipIf(np is None or Image is None or ImageDraw is None or FONT_PATH is None, "numpy, Pillow, and a truetype font are required")
    def test_render_manga_text_block_alpha_composites_visible_pixels(self):
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        block = MangaRenderBlock(
            bbox=(20, 20, 80, 60),
            text="Visible",
            kind="bubble",
            font_path=FONT_PATH,
            text_color=(0, 0, 0),
            stroke_color=(255, 255, 255),
            stroke_width=2,
        )

        render_manga_text_block(image, block)

        self.assertGreater(int(np.count_nonzero(image != 255)), 0)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_outside_text_style_uses_stroke_width_at_least_bubble_style(self):
        image = np.full((80, 100, 3), 240, dtype=np.uint8)
        bbox = (20, 20, 80, 60)
        bubble_item = {"kind": "bubble", "text_regions": []}
        outside_item = {"kind": "outside_text", "text_regions": []}

        bubble_block = choose_render_style_for_item(image, bubble_item, bbox, FONT_PATH or "")
        outside_block = choose_render_style_for_item(image, outside_item, bbox, FONT_PATH or "")

        self.assertGreaterEqual(outside_block.stroke_width or 0, bubble_block.stroke_width or 0)

    @unittest.skipIf(np is None or Image is None or ImageDraw is None, "numpy and Pillow are required")
    def test_missing_font_path_does_not_crash(self):
        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        block = MangaRenderBlock(
            bbox=(10, 10, 70, 50),
            text="Fallback font",
            kind="outside_text",
            font_path="C:/definitely_missing_font.ttf",
        )

        render_manga_text_block(image, block)

        self.assertGreaterEqual(int(np.count_nonzero(image != 255)), 0)

    @unittest.skipIf(np is None or Image is None or ImageDraw is None or FONT_PATH is None, "numpy, Pillow, and a truetype font are required")
    def test_wrapper_render_text_block_still_works(self):
        image = np.full((60, 80, 3), 255, dtype=np.uint8)

        render_text_block(
            image,
            "Wrapper",
            (10, 10, 70, 50),
            FONT_PATH,
            (0, 0, 0),
            stroke_color=(255, 255, 255),
            stroke_width=1,
        )

        self.assertGreater(int(np.count_nonzero(image != 255)), 0)


if __name__ == "__main__":
    unittest.main()

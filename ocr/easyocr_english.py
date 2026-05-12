import re
import numpy as np
from PIL import Image, ImageOps, ImageFilter


class EnglishEasyOCR:
    def __init__(self, gpu=True, min_confidence=0.25):
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=gpu)
        self.min_confidence = min_confidence

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess bubble crop for English comic OCR.
        Keep it conservative: upscale + border + light sharpening.
        """
        image = image.convert("RGB")

        # OCR engines hate tightly cropped text.
        image = ImageOps.expand(image, border=12, fill="white")

        # Upscale small bubbles.
        w, h = image.size
        if max(w, h) < 512:
            scale = 512 / max(w, h)
            image = image.resize(
                (int(w * scale), int(h * scale)),
                Image.Resampling.LANCZOS
            )

        image = image.filter(ImageFilter.SHARPEN)
        return np.array(image)

    def __call__(self, image: Image.Image) -> str:
        arr = self._preprocess(image)

        results = self.reader.readtext(
            arr,
            detail=1,
            paragraph=False,
            text_threshold=0.4,
            low_text=0.3,
            link_threshold=0.4,
        )

        lines = []

        for bbox, text, conf in results:
            if conf < self.min_confidence:
                continue

            cleaned = self._clean_text(text)
            if not cleaned:
                continue

            # bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            center_y = sum(ys) / len(ys)
            center_x = sum(xs) / len(xs)

            lines.append((center_y, center_x, cleaned))

        # English comics are normally left-to-right, top-to-bottom.
        lines.sort(key=lambda item: (item[0], item[1]))

        return " ".join(line[2] for line in lines).strip()

    def process_batch(self, images):
        return [self(img) for img in images]

    def _clean_text(self, text: str) -> str:
        text = text.strip()

        # Common OCR cleanup for English comic bubbles.
        text = text.replace("|", "I")
        text = text.replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = re.sub(r"\s+", " ", text)

        return text
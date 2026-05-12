import html
import inspect
import re
import gc
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageFilter, ImageOps


# Tags Surya/Marker-like OCR may emit to preserve source formatting.
# We remove the tags but keep the inner text:
#   "<b>tao</b>" -> "tao"
OCR_FORMAT_TAG_RE = re.compile(
    r"</?(?:"
    r"b|strong|i|em|u|s|strike|del|ins|"
    r"sub|sup|small|big|mark|span|font|"
    r"ruby|rt|rp|br|p|div"
    r")\b[^>]*>",
    flags=re.IGNORECASE,
)

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)

# Last-resort cleanup for obvious HTML/XML-ish tags.
# This intentionally only removes tags that start with a letter,
# so it will NOT destroy things like "<3" or "3 < 5".
GENERIC_TAG_RE = re.compile(
    r"</?[a-zA-Z][a-zA-Z0-9:_-]*(?:\s+[^<>]*)?>",
    flags=re.IGNORECASE,
)


@dataclass
class OCRLine:
    text: str
    confidence: float = 1.0
    bbox: Optional[Tuple[float, float, float, float]] = None


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    """Works with both pydantic/dataclass objects and dict-like outputs."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def strip_surya_markup(text: str) -> str:
    """
    Remove OCR formatting markup emitted by Surya while preserving readable text.

    Examples:
        "<b>tao</b>"       -> "tao"
        "&lt;i&gt;hey&lt;/i&gt;" -> "hey"
        "I <3 you"         -> "I <3 you"
    """
    if not text:
        return ""

    text = str(text)

    # Decode escaped tags/entities first:
    # "&lt;b&gt;tao&lt;/b&gt;" -> "<b>tao</b>"
    text = html.unescape(text)

    # Remove HTML comments if they ever appear.
    text = HTML_COMMENT_RE.sub("", text)

    # Convert common line break tags to a space before stripping.
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

    # Remove known formatting tags.
    text = OCR_FORMAT_TAG_RE.sub("", text)

    # Last-resort: remove any remaining obvious tag-shaped fragments.
    text = GENERIC_TAG_RE.sub("", text)

    # Normalize weird whitespace.
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)

    # Remove spaces before punctuation.
    text = re.sub(r"\s+([,.!?;:…。、！？])", r"\1", text)

    return text.strip()


class SuryaOCR:
    """
    Drop-in Surya OCR wrapper for Manga-Translator.

    Interface expected by the manga pipeline:
        ocr = SuryaOCR(...)
        text = ocr(PIL.Image)
        texts = ocr.process_batch([PIL.Image, PIL.Image, ...])

    Recommended use:
        - English manga/comics: SuryaOCR(task_name="ocr_with_boxes")
        - Japanese manga:       SuryaOCR(task_name="ocr_with_boxes")

    Notes:
        - We keep ocr_with_boxes because it gives line bboxes and generally works
          nicely with manga bubble crops.
        - Surya may emit formatting markup like <b>...</b>; this wrapper strips it.
        - math_mode is disabled by default to avoid math/LaTeX-ish false positives.
    """

    def __init__(
        self,
        batch_size: int = 5,
        clear_vram_after_batch: bool = True,
        min_confidence: float = 0.15,
        min_side: int = 900,
        padding: int = 18,
        task_name: str = "ocr_with_boxes",
        preserve_line_breaks: bool = False,
        sort_lines: bool = False,
        disable_math: bool = True,
        sharpen: bool = True,
        upscale_small_images: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            min_confidence:
                Ignore lines below this confidence. Keep this low for manga;
                stylized fonts often get lower confidence but are still correct.

            min_side:
                If the bubble crop is small, upscale so max(width, height)
                is at least this value.

            padding:
                Add white padding around bubble crops. Helps when text is close
                to the crop edge.

            task_name:
                "ocr_with_boxes" is the default Surya task.
                "ocr_without_boxes" may reduce formatting in some versions,
                but can lose bbox info. This wrapper strips tags anyway.

            preserve_line_breaks:
                If True, join detected lines with "\\n".
                If False, join with spaces. For manga redraw, False is usually better.

            sort_lines:
                If True, sort lines top-to-bottom, left-to-right.
                If False, trust Surya's output order.
                For Japanese vertical text, False is usually safer.

            disable_math:
                Pass math_mode=False when Surya supports it.

            sharpen:
                Apply light sharpening after resize.

            upscale_small_images:
                Upscale small crops for better OCR.

            verbose:
                Print debug info.
        """
        self.min_confidence = min_confidence
        self.min_side = min_side
        self.padding = padding
        self.task_name = task_name
        self.preserve_line_breaks = preserve_line_breaks
        self.sort_lines = sort_lines
        self.disable_math = disable_math
        self.sharpen = sharpen
        self.upscale_small_images = upscale_small_images
        self.verbose = verbose

        self.foundation_predictor = None
        self.recognition_predictor = None
        self.detection_predictor = None
        self.batch_size = max(1, int(os.environ.get("SURYA_BATCH_SIZE", batch_size)))
        self.clear_vram_after_batch = clear_vram_after_batch
        self._load_models()

    def _load_models(self) -> None:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor

        self.foundation_predictor = FoundationPredictor()
        self.recognition_predictor = RecognitionPredictor(self.foundation_predictor)
        self.detection_predictor = DetectionPredictor()

        if self.verbose:
            print("✓ Surya OCR models loaded")

    def _preprocess(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")

        # Bubble crops are often too tight. White padding helps text detection.
        if self.padding > 0:
            image = ImageOps.expand(image, border=self.padding, fill="white")

        if self.upscale_small_images:
            w, h = image.size
            longest = max(w, h)

            if longest > 0 and longest < self.min_side:
                scale = self.min_side / longest
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

        if self.sharpen:
            image = image.filter(ImageFilter.SHARPEN)

        return image

    def _call_surya(self, images: Sequence[Image.Image]) -> List[Any]:
        """
        Call Surya in a version-tolerant way.

        Newer Surya versions support:
            recognition_predictor(
                images,
                task_names=[...],
                det_predictor=...,
                math_mode=False
            )

        If a user's installed Surya build does not accept some kwargs,
        we automatically retry with fewer args.
        """
        kwargs = {
            "det_predictor": self.detection_predictor,
        }

        # Add task_names only if the installed Surya accepts it.
        try:
            sig = inspect.signature(self.recognition_predictor.__call__)
            params = sig.parameters

            if "task_names" in params:
                kwargs["task_names"] = [self.task_name] * len(images)

            if "math_mode" in params:
                kwargs["math_mode"] = not self.disable_math

        except Exception:
            # If introspection fails, try the modern call first anyway.
            kwargs["task_names"] = [self.task_name] * len(images)
            kwargs["math_mode"] = not self.disable_math

        try:
            return self.recognition_predictor(list(images), **kwargs)

        except TypeError as e:
            if self.verbose:
                print(f"Surya modern call failed, retrying basic call: {e}")

            return self.recognition_predictor(
                list(images),
                det_predictor=self.detection_predictor,
            )

    def _extract_lines(self, prediction: Any) -> List[OCRLine]:
        raw_lines = _safe_get(prediction, "text_lines", []) or []
        lines: List[OCRLine] = []

        for raw_line in raw_lines:
            raw_text = _safe_get(raw_line, "text", "")
            cleaned = self._clean_text(raw_text)

            if not cleaned:
                continue

            confidence = _safe_get(raw_line, "confidence", 1.0)
            if confidence is None:
                confidence = 1.0

            try:
                confidence = float(confidence)
            except Exception:
                confidence = 1.0

            if confidence < self.min_confidence:
                continue

            bbox = _safe_get(raw_line, "bbox", None)
            parsed_bbox = self._parse_bbox(bbox)

            lines.append(
                OCRLine(
                    text=cleaned,
                    confidence=confidence,
                    bbox=parsed_bbox,
                )
            )

        if self.sort_lines:
            lines.sort(key=self._line_sort_key)

        return lines

    def _parse_bbox(self, bbox: Any) -> Optional[Tuple[float, float, float, float]]:
        if not bbox:
            return None

        try:
            if len(bbox) != 4:
                return None
            x1, y1, x2, y2 = bbox
            return float(x1), float(y1), float(x2), float(y2)
        except Exception:
            return None

    def _line_sort_key(self, line: OCRLine) -> Tuple[float, float]:
        if not line.bbox:
            return 0.0, 0.0

        x1, y1, x2, y2 = line.bbox
        return y1, x1

    def _clean_text(self, text: str) -> str:
        text = strip_surya_markup(text)

        # OCR cleanup that is generally safe for manga/comic text.
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("…", "...")

        # Normalize repeated spaces again after punctuation replacements.
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _join_lines(self, lines: Iterable[OCRLine]) -> str:
        texts = [line.text for line in lines if line.text]

        if self.preserve_line_breaks:
            joined = "\n".join(texts)
        else:
            joined = " ".join(texts)

        # Final cleanup pass, just in case adjacent line joins create junk spaces.
        joined = re.sub(r"\s+([,.!?;:…。、！？])", r"\1", joined)
        joined = re.sub(r"\s+", " ", joined)

        return joined.strip()

    def __call__(self, image: Image.Image) -> str:
        if image is None:
            return ""

        try:
            processed = self._preprocess(image)
            predictions = self._call_surya([processed])

            if not predictions:
                return ""

            lines = self._extract_lines(predictions[0])
            text = self._join_lines(lines)

            if self.verbose:
                print(f"Surya OCR: {len(lines)} lines -> {text!r}")

            return text

        except Exception as e:
            print(f"Surya OCR error: {e}")
            return ""

        finally:
            self._cleanup_memory()

    def process_batch(self, images) -> List[str]:
        """
        Memory-safe batch OCR.

        Important:
        We DO NOT send all images to Surya at once.
        Surya can consume a lot of VRAM, especially after preprocessing/upscaling.
        This method processes images in micro-batches, default batch_size=1.
        """
        if not images:
            return []

        results: List[str] = []
        total = len(images)

        for start_idx, chunk in self._iter_chunks(list(images), self.batch_size):
            if self.verbose:
                end_idx = min(start_idx + len(chunk), total)
                print(
                    f"Surya OCR batch: processing {start_idx + 1}-{end_idx}/{total} "
                    f"(batch_size={self.batch_size})"
                )

            processed_images = []

            try:
                # Preprocess only this small chunk, not the whole input list.
                processed_images = [self._preprocess(img) for img in chunk]

                predictions = self._call_surya(processed_images)

                for pred in predictions:
                    lines = self._extract_lines(pred)
                    results.append(self._join_lines(lines))

            except RuntimeError as e:
                # If GPU still OOMs, fall back to strict one-by-one for this chunk.
                msg = str(e).lower()

                if "out of memory" in msg or "cuda" in msg:
                    print("Surya OCR OOM detected. Falling back to image-by-image OCR for this chunk.")

                    self._cleanup_memory()

                    for img in chunk:
                        try:
                            processed = self._preprocess(img)
                            predictions = self._call_surya([processed])

                            if predictions:
                                lines = self._extract_lines(predictions[0])
                                results.append(self._join_lines(lines))
                            else:
                                results.append("")

                        except Exception as inner_e:
                            print(f"Surya OCR failed on single image: {inner_e}")
                            results.append("")

                        finally:
                            self._cleanup_memory()

                else:
                    print(f"Surya OCR runtime error: {e}")
                    results.extend([""] * len(chunk))

            except Exception as e:
                print(f"Surya OCR batch error: {e}")
                results.extend([""] * len(chunk))

            finally:
                # Drop references before clearing cache.
                processed_images = None
                self._cleanup_memory()

        return results

    def extract_lines(self, image: Image.Image) -> List[OCRLine]:
        """
        Optional helper if later you want full-page OCR + assign lines to bubbles.
        Returns cleaned text lines with bbox/confidence.
        """
        if image is None:
            return []

        processed = self._preprocess(image)
        predictions = self._call_surya([processed])

        if not predictions:
            return []

        return self._extract_lines(predictions[0])


    def _iter_chunks(self, items, size: int):
        size = max(1, int(size))
        for i in range(0, len(items), size):
            yield i, items[i:i + size]


    def _cleanup_memory(self) -> None:
        if not self.clear_vram_after_batch:
            return

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # For Apple Silicon / MPS, if available.
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        except Exception:
            pass
# Backward-compatible aliases.
# Use whichever name your app.py currently imports.
SuryaEnglishOCR = SuryaOCR
SuryaJapaneseOCR = SuryaOCR
SuryaMangaOCR = SuryaOCR
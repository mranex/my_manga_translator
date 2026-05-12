import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


_DECODER_PATCHED = False


def patch_paddlex_decoder() -> None:
    """
    Fix PaddleX/PaddleOCR decoder crash:

        IndexError: list index out of range
        self.character[text_id]

    Cause:
        The recognizer sometimes emits a class index that is outside the
        loaded character dictionary. The official decoder indexes directly
        without checking bounds.

    Fix:
        Filter invalid ids before converting ids to chars.
    """
    global _DECODER_PATCHED

    if _DECODER_PATCHED:
        return

    try:
        from paddlex.inference.models.text_recognition import processors
    except Exception as e:
        print(f"[PaddleOCR-English] decoder patch skipped: cannot import processors: {e}")
        return

    BaseRecLabelDecode = getattr(processors, "BaseRecLabelDecode", None)
    if BaseRecLabelDecode is None:
        print("[PaddleOCR-English] decoder patch skipped: BaseRecLabelDecode not found")
        return

    original_decode = BaseRecLabelDecode.decode

    def safe_decode(
        self,
        text_index,
        text_prob=None,
        is_remove_duplicate=False,
        return_word_box=False,
    ):
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            indexes = np.asarray(text_index[batch_idx])

            selection = np.ones(len(indexes), dtype=bool)

            if is_remove_duplicate and len(indexes) > 1:
                selection[1:] = indexes[1:] != indexes[:-1]

            for ignored_token in ignored_tokens:
                selection &= indexes != ignored_token

            selected_positions = np.where(selection)[0]

            char_list = []
            valid_positions = []

            for pos in selected_positions:
                try:
                    text_id = int(indexes[pos])
                except Exception:
                    continue

                # THE ACTUAL FIX:
                # skip invalid ids instead of crashing.
                if text_id < 0 or text_id >= len(self.character):
                    continue

                char_list.append(self.character[text_id])
                valid_positions.append(pos)

            if text_prob is not None and len(valid_positions) > 0:
                probs = np.asarray(text_prob[batch_idx])
                conf_list = [probs[pos] for pos in valid_positions if pos < len(probs)]
            else:
                conf_list = [1] * len(char_list)

            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            if getattr(self, "reverse", False):
                text = self.pred_reverse(text)

            if return_word_box:
                # Keep word-box mode alive. If filtering changed selection,
                # make a safe selection mask with only valid positions.
                safe_selection = np.zeros(len(indexes), dtype=bool)
                for pos in valid_positions:
                    if pos < len(safe_selection):
                        safe_selection[pos] = True

                try:
                    word_list, word_col_list, state_list = self.get_word_info(
                        text,
                        safe_selection,
                    )
                except Exception:
                    word_list, word_col_list, state_list = [], [], []

                result_list.append(
                    (
                        text,
                        np.mean(conf_list).tolist(),
                        [
                            len(indexes),
                            word_list,
                            word_col_list,
                            state_list,
                        ],
                    )
                )
            else:
                result_list.append((text, np.mean(conf_list).tolist()))

        return result_list

    BaseRecLabelDecode.decode = safe_decode
    _DECODER_PATCHED = True

    print("[PaddleOCR-English] Patched PaddleX BaseRecLabelDecode.decode successfully")


class EnglishPaddleOCR:
    """
    PaddleOCR English backend for Manga-Translator.

    Contract:
        text = ocr(PIL.Image)
        texts = ocr.process_batch(list[PIL.Image])

    This backend patches PaddleX decoder so invalid character ids do not crash OCR.
    """

    def __init__(
        self,
        device: str = "gpu:0",
        min_score: float = 0.05,
        debug: bool = True,
        debug_dir: str = "debug/paddleocr_failed",
        keep_debug_images: bool = True,
    ):
        self.device = device
        self.min_score = min_score
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        self.keep_debug_images = keep_debug_images

        patch_paddlex_decoder()

        self.ocr = None
        self._init_engine()

    def __call__(self, image: Image.Image) -> str:
        if image is None:
            self._log("received None image")
            return ""

        started = time.time()
        tmp_dir = Path(tempfile.mkdtemp(prefix="paddleocr_en_"))

        try:
            normal_path = tmp_dir / "bubble_normal.png"
            strong_path = tmp_dir / "bubble_strong.png"

            self._preprocess(image, strong=False).save(normal_path)
            self._preprocess(image, strong=True).save(strong_path)

            attempts = [
                ("normal", normal_path),
                ("strong", strong_path),
            ]

            errors = []

            for attempt_name, path in attempts:
                try:
                    items = self._ocr_path(path)
                    text = self._items_to_text(items)

                    if text:
                        self._log(
                            f"{attempt_name}: OK, {len(items)} items, "
                            f"{len(text)} chars, {time.time() - started:.2f}s"
                        )
                        return text

                    errors.append(f"{attempt_name}: no text")

                except Exception as e:
                    msg = f"{attempt_name}: {type(e).__name__}: {e}"
                    errors.append(msg)
                    self._log(msg)

            self._save_failed_debug(image, normal_path, strong_path, errors)
            return ""

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def process_batch(self, images: List[Image.Image]) -> List[str]:
        return [self(img) for img in images]

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_engine(self) -> None:
        from paddleocr import PaddleOCR

        # IMPORTANT:
        # Do not force en_PP-OCRv5_mobile_rec here.
        # The crash shows recognizer output ids and decoder dictionary do not agree.
        # Let PaddleOCR choose the correct model/dict pair for lang="en" + PP-OCRv5.
        kwargs = {
            "lang": "en",
            "ocr_version": "PP-OCRv5",
            "device": self.device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,

            # Tuning for comic bubble text.
            "text_det_limit_side_len": 1280,
            "text_det_limit_type": "min",
            "text_det_thresh": 0.18,
            "text_det_box_thresh": 0.22,
            "text_det_unclip_ratio": 1.8,
            "text_rec_score_thresh": 0.0,
        }

        try:
            self.ocr = PaddleOCR(**kwargs)
            self._log("PaddleOCR initialized with PP-OCRv5 English config")
        except TypeError as e:
            self._log(f"full constructor rejected args: {e}")
            self.ocr = PaddleOCR(
                lang="en",
                device=self.device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            self._log("PaddleOCR initialized with minimal English config")

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _ocr_path(self, image_path: Path) -> List[Dict[str, Any]]:
        result = self.ocr.predict(str(image_path))
        result_list = list(result) if result is not None else []

        if not result_list:
            return []

        return self._extract_items(result_list)

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    def _extract_items(self, obj: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        for node in self._walk_nodes(obj):
            data = self._node_to_dict(node)
            if not data:
                continue

            roots = [data]

            if isinstance(data.get("res"), dict):
                roots.append(data["res"])

            if isinstance(data.get("prunedResult"), dict):
                roots.append(data["prunedResult"])

            for root in roots:
                texts = self._as_list(root.get("rec_texts"))
                scores = self._as_list(root.get("rec_scores"))

                boxes = root.get("rec_boxes")
                if boxes is None:
                    boxes = root.get("rec_polys")
                if boxes is None:
                    boxes = root.get("dt_polys")

                boxes = self._as_list(boxes)

                if not texts:
                    continue

                for i, raw_text in enumerate(texts):
                    text = self._clean_piece(str(raw_text))
                    if not text:
                        continue

                    score = 1.0
                    if i < len(scores):
                        score = self._to_float(scores[i], default=1.0)

                    if score < self.min_score:
                        continue

                    box = boxes[i] if i < len(boxes) else None
                    cx, cy = self._box_center(box)

                    items.append(
                        {
                            "text": text,
                            "score": score,
                            "cx": cx,
                            "cy": cy,
                            "box": box,
                        }
                    )

        return self._dedupe(items)

    def _walk_nodes(self, obj: Any) -> Iterable[Any]:
        yield obj

        if isinstance(obj, dict):
            for value in obj.values():
                yield from self._walk_nodes(value)

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                yield from self._walk_nodes(item)

    def _node_to_dict(self, node: Any) -> Optional[Dict[str, Any]]:
        if isinstance(node, dict):
            return node

        if hasattr(node, "json"):
            try:
                value = node.json
                return value() if callable(value) else value
            except Exception:
                return None

        if hasattr(node, "to_dict"):
            try:
                return node.to_dict()
            except Exception:
                return None

        return None

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    def _preprocess(self, image: Image.Image, strong: bool = False) -> Image.Image:
        image = self._flatten_white(image)

        # Bubble crop padding. Too much padding can hurt detection, so keep moderate.
        image = ImageOps.expand(image, border=36, fill=(255, 255, 255))

        w, h = image.size
        if w <= 0 or h <= 0:
            return Image.new("RGB", (800, 400), "white")

        long_side = max(w, h)
        short_side = min(w, h)

        scale = 1.0

        if long_side < 1100:
            scale = max(scale, 1100 / long_side)

        if short_side < 360:
            scale = max(scale, 360 / short_side)

        scale = min(scale, 4.0)

        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        max_side = 2200
        if max(image.size) > max_side:
            down = max_side / max(image.size)
            image = image.resize(
                (int(image.size[0] * down), int(image.size[1] * down)),
                Image.Resampling.LANCZOS,
            )

        image = ImageOps.autocontrast(image)

        if strong:
            image = ImageEnhance.Contrast(image).enhance(1.7)
            image = ImageEnhance.Sharpness(image).enhance(1.8)
            image = image.filter(ImageFilter.SHARPEN)
        else:
            image = ImageEnhance.Contrast(image).enhance(1.2)
            image = ImageEnhance.Sharpness(image).enhance(1.25)

        return image.convert("RGB")

    def _flatten_white(self, image: Image.Image) -> Image.Image:
        if image.mode in ("RGBA", "LA"):
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            bg.alpha_composite(image.convert("RGBA"))
            return bg.convert("RGB")

        return image.convert("RGB")

    # ------------------------------------------------------------------
    # Text cleanup
    # ------------------------------------------------------------------

    def _items_to_text(self, items: List[Dict[str, Any]]) -> str:
        if not items:
            return ""

        items = sorted(items, key=lambda x: (x["cy"], x["cx"]))
        text = " ".join(item["text"] for item in items)
        return self._clean_joined_text(text)

    def _clean_piece(self, text: str) -> str:
        text = text.strip()

        if not text:
            return ""

        text = text.replace("’", "'").replace("‘", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _clean_joined_text(self, text: str) -> str:
        text = self._clean_piece(text)

        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"\b([A-Za-z])\s+'\s*([A-Za-z])", r"\1'\2", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _dedupe(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best = {}

        for item in items:
            key = re.sub(r"[^a-z0-9]+", "", item["text"].lower())

            if not key:
                continue

            if key not in best or item["score"] > best[key]["score"]:
                best[key] = item

        return list(best.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _as_list(self, value: Any) -> List[Any]:
        if value is None:
            return []

        if hasattr(value, "tolist"):
            try:
                value = value.tolist()
            except Exception:
                pass

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        return [value]

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _box_center(self, box: Any) -> Tuple[float, float]:
        if box is None:
            return 0.0, 0.0

        try:
            arr = np.array(box, dtype=float)
        except Exception:
            return 0.0, 0.0

        if arr.ndim == 1 and arr.shape[0] >= 4:
            x1, y1, x2, y2 = arr[:4]
            return float((x1 + x2) / 2), float((y1 + y2) / 2)

        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())

        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def _save_failed_debug(
        self,
        original: Image.Image,
        normal_path: Path,
        strong_path: Path,
        errors: List[str],
    ) -> None:
        self._log("OCR returned empty. Attempts:\n  - " + "\n  - ".join(errors))

        if not self.keep_debug_images:
            return

        self.debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)

        original_path = self.debug_dir / f"{stamp}_original.png"
        normal_debug_path = self.debug_dir / f"{stamp}_normal.png"
        strong_debug_path = self.debug_dir / f"{stamp}_strong.png"
        log_path = self.debug_dir / f"{stamp}_log.txt"

        try:
            original.convert("RGB").save(original_path)
            shutil.copyfile(normal_path, normal_debug_path)
            shutil.copyfile(strong_path, strong_debug_path)

            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(errors))

            self._log(f"debug saved: {self.debug_dir}")

        except Exception as e:
            self._log(f"failed to save debug images: {e}")

    def _log(self, message: str) -> None:
        if self.debug:
            print(f"[PaddleOCR-English] {message}")
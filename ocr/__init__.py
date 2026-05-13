# OCR modules
try:
    from .chrome_lens_ocr import ChromeLensOCR
except Exception:
    ChromeLensOCR = None

try:
    from .paddleocr_vl_ocr import PaddleOCRVLOCR
except Exception:
    PaddleOCRVLOCR = None

__all__ = ["ChromeLensOCR", "PaddleOCRVLOCR"]

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    device="gpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

print("PaddleOCR object init OK")
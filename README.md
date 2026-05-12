# Manga Translator Upgrade Handoff

> Tài liệu chuyển giao cho việc tiếp tục nâng cấp fork/biến thể cá nhân của `pedguedes090/Manga-Translator`.
>
> Dự kiến repo mới của người dùng: `mranex/my_manga_translator`  
> Ghi chú: đường dẫn này hiện là dự kiến, chưa xác nhận đã public.

---

## 1. Bối cảnh dự án

Mục tiêu của dự án là tạo một pipeline dịch manga/comic cá nhân, ưu tiên:

- Chạy local hoặc bán-local tùy provider.
- Dịch manga từ source tiếng Nhật hoặc tiếng Anh sang tiếng Việt.
- Hạn chế phụ thuộc vào các provider cloud có policy/filter không phù hợp với nhu cầu đọc cá nhân.
- Batch translate nhiều ảnh/chapter.
- OCR, dịch, xóa chữ gốc và redraw text dịch trực tiếp lên trang manga.

Pipeline hiện tại đã hoạt động ổn định sau nhiều chỉnh sửa:

```text
Upload image(s)
→ detect speech bubble / text region
→ crop vùng cần OCR
→ OCR
→ translate
→ remove/fill text cũ
→ render text dịch
→ export image/ZIP
```

---

## 2. Repo gốc: `pedguedes090/Manga-Translator`

Repo gốc: https://github.com/pedguedes090/Manga-Translator

### 2.1. Chức năng chính của repo gốc

Theo README và code gốc, repo hỗ trợ:

- Web UI bằng Flask.
- Upload nhiều ảnh manga/manhwa/manhua.
- Detect speech bubble bằng YOLOv8.
- Fallback detect bubble đen bằng OpenCV.
- OCR bằng:
  - `manga-ocr`
  - Chrome Lens OCR
- Translator:
  - Gemini
  - Local LLM qua endpoint OpenAI-compatible, ví dụ Ollama/LM Studio
  - NLLB
- Context Memory để giữ ngữ cảnh giữa nhiều trang.
- Auto font matching bằng Gemini Vision.
- Render text dịch bằng PIL.
- Download ảnh riêng lẻ hoặc ZIP.

### 2.2. Tech stack gốc

```text
Backend: Flask + Flask-SocketIO
Detection: YOLOv8 + OpenCV black bubbles
OCR: Manga-OCR, Chrome Lens
Translation: Gemini API, OpenAI-compatible endpoints, NLLB
Rendering: PIL / Pillow
```

### 2.3. Pipeline gốc

Pipeline gốc chủ yếu theo bubble-level:

```text
detect_bubbles()
→ lấy bbox speech bubble
→ crop bbox bubble
→ OCR crop
→ translate text
→ process_bubble_auto() / fill bubble
→ add_text()
```

Điểm quan trọng: trong repo gốc, OCR chỉ nhìn thấy các crop được detector trả về. Nếu detector bỏ sót bubble hoặc vùng chữ, OCR không có cơ hội đọc vùng đó.

### 2.4. Detector gốc

Repo gốc dùng:

```python
MODEL_PATH = "model/model.pt"
```

và gọi:

```python
detect_bubbles(MODEL_PATH, image, ...)
```

Trong `detect_bubbles.py`, detector trả về `results.boxes.data.tolist()`, tức là dùng bounding boxes. Đây là bubble bbox detector, chưa phải segmentation/mask detector.

Repo gốc cũng có fallback OpenCV cho bubble đen.

### 2.5. Hạn chế của repo gốc

Các hạn chế đã phát hiện trong quá trình nâng cấp:

1. **Detector là bottleneck lớn**
   - Nếu bubble/text không được detect, OCR không đọc được.
   - Bounding box bubble không đủ tốt cho text sát viền, text ngoài bubble, caption, SFX, narration.

2. **OCR gốc thiên về tiếng Nhật**
   - `manga-ocr` tốt cho Japanese manga text, nhưng không dùng tốt cho English source.
   - Chrome Lens OCR có thể tốt nhưng không local/private.

3. **Provider translator gốc hạn chế**
   - Repo gốc hỗ trợ Gemini, Local LLM, NLLB.
   - Chưa có DeepSeek.
   - Chưa có Google GenAI SDK/provider mới theo implementation riêng của người dùng.

4. **Text removal/rendering còn cơ bản**
   - Chủ yếu fill vùng bubble/crop.
   - Chưa dùng text mask chính xác.
   - Chưa có inpainting chuyên sâu theo mask.
   - Chưa xử lý HTML/format tag từ OCR.

5. **Batch OCR có nguy cơ ăn VRAM**
   - Khi OCR backend nặng như Surya xử lý toàn bộ list ảnh/crop cùng lúc, dễ crash vì thiếu VRAM.
   - Cần micro-batch hoặc xử lý tuần tự.

---

## 3. Các nâng cấp đã thực hiện

Phần này mô tả các thay đổi đã được người dùng áp dụng vào fork/local version.

---

## 3.1. Thêm EasyOCR cho English source

### Mục tiêu

Bổ sung OCR local cho manga/comic tiếng Anh, vì `manga-ocr` không phù hợp với English source.

### Lý do

Source tiếng Anh thường tương thích pipeline tốt hơn:

- Text ngang.
- Latin font dễ OCR hơn kanji/kana/furigana.
- Ít vấn đề vertical text.
- Người dùng biết tiếng Anh nên dễ phát hiện lỗi dịch/OCR.
- Text dịch sang tiếng Việt ít bị tràn viền hơn so với dịch từ tiếng Nhật.

### Thiết kế

Thêm OCR backend mới, ví dụ:

```text
ocr/easyocr_english.py
```

Interface nên giữ giống pipeline gốc:

```python
text = ocr(PIL.Image)
texts = ocr.process_batch([PIL.Image, ...])
```

### Gợi ý class/interface

```python
class EnglishEasyOCR:
    def __init__(self, gpu=True, min_confidence=0.25):
        ...

    def __call__(self, image: Image.Image) -> str:
        ...

    def process_batch(self, images):
        return [self(img) for img in images]
```

### UI/config

Thêm option OCR:

```html
<option>EasyOCR-English</option>
```

Trong `app.py`, thêm cache key:

```python
_OCR_CACHE = {
    "chrome_lens": None,
    "manga_ocr": None,
    "easyocr_en": None,
}
```

và mapping:

```python
elif selected_ocr == "easyocr-english":
    if _OCR_CACHE["easyocr_en"] is None:
        _OCR_CACHE["easyocr_en"] = EnglishEasyOCR(gpu=True)
    mocr = _OCR_CACHE["easyocr_en"]
```

### Trạng thái

Đã hoạt động ổn định sau khi người dùng xử lý xung đột NumPy.

---

## 3.2. Thêm Surya OCR

Surya OCR: https://github.com/datalab-to/surya

### Mục tiêu

Nâng cấp OCR mạnh hơn EasyOCR, hỗ trợ tốt hơn:

- English comic text.
- Japanese text.
- Text nhỏ/bị ép trong khung thoại.
- Text stylized.
- Line-level detection.

### Kết quả thực tế

Người dùng đã test và thấy Surya OCR thậm chí đọc tiếng Nhật mượt và ổn định hơn `manga-ocr` trong pipeline hiện tại.

### Vấn đề phát sinh 1: Surya giữ HTML/format tag

Surya đôi khi output markup như:

```text
Nghe này, <b>tao</b> sẽ đánh bại bọn mày.
```

Nguyên nhân: OCR cố giữ formatting/source styling, ví dụ bold/italic.

### Cách xử lý

Thêm cleanup trong Surya wrapper để strip HTML/format tag trước khi đưa sang translator/redraw.

Ví dụ helper:

```python
import html
import re

OCR_FORMAT_TAG_RE = re.compile(
    r"</?(?:"
    r"b|strong|i|em|u|s|strike|del|ins|"
    r"sub|sup|small|big|mark|span|font|"
    r"ruby|rt|rp|br|p|div"
    r")\b[^>]*>",
    flags=re.IGNORECASE,
)

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
GENERIC_TAG_RE = re.compile(
    r"</?[a-zA-Z][a-zA-Z0-9:_-]*(?:\s+[^<>]*)?>",
    flags=re.IGNORECASE,
)

def strip_surya_markup(text: str) -> str:
    if not text:
        return ""

    text = str(text)
    text = html.unescape(text)
    text = HTML_COMMENT_RE.sub("", text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = OCR_FORMAT_TAG_RE.sub("", text)
    text = GENERIC_TAG_RE.sub("", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+([,.!?;:…。、！？])", r"\1", text)

    return text.strip()
```

Sau đó gọi trong `_clean_text()`:

```python
def _clean_text(self, text: str) -> str:
    text = strip_surya_markup(text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

### Vấn đề phát sinh 2: batch OCR gây crash VRAM

Bản wrapper ban đầu xử lý:

```python
processed_images = [self._preprocess(img) for img in images]
predictions = self._call_surya(processed_images)
```

Với nhiều ảnh/crop, Surya load batch quá lớn lên GPU → thiếu VRAM → crash.

### Cách xử lý

`process_batch()` phải chạy micro-batch, mặc định `batch_size=1`.

Ý tưởng:

```python
def process_batch(self, images):
    results = []
    for chunk in iter_chunks(images, self.batch_size):
        processed = [self._preprocess(img) for img in chunk]
        predictions = self._call_surya(processed)
        ...
        cleanup_vram()
    return results
```

Nên thêm cleanup:

```python
def _cleanup_memory(self):
    import gc
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
```

### Config Surya gợi ý

English:

```python
SuryaOCR(
    min_confidence=0.15,
    min_side=900,
    padding=18,
    task_name="ocr_with_boxes",
    preserve_line_breaks=False,
    sort_lines=False,
    disable_math=True,
    batch_size=1,
    clear_vram_after_batch=True,
)
```

Japanese:

```python
SuryaOCR(
    min_confidence=0.12,
    min_side=1000,
    padding=18,
    task_name="ocr_with_boxes",
    preserve_line_breaks=False,
    sort_lines=False,
    disable_math=True,
    batch_size=1,
    clear_vram_after_batch=True,
)
```

Nếu vẫn thiếu VRAM:

```python
min_side=768
batch_size=1
```

### Trạng thái

Đã hoạt động ổn định sau khi sửa batch/micro-batch.

---

## 3.3. Thêm DeepSeek provider

DeepSeek API docs: https://api-docs.deepseek.com/

### Mục tiêu

Bổ sung provider dịch mới ngoài Gemini/Local LLM/NLLB.

### Lý do

DeepSeek API tương thích OpenAI-style API, dễ tích hợp vào translator system.

### API hiện tại

Base URL:

```text
https://api.deepseek.com
```

Endpoint direct:

```text
https://api.deepseek.com/chat/completions
```

Models hiện nên ưu tiên:

```text
deepseek-v4-flash
deepseek-v4-pro
```

Tên cũ:

```text
deepseek-chat
deepseek-reasoner
```

được giữ để tương thích nhưng theo docs sẽ bị deprecate sau 2026-07-24.

### Thiết kế đã đề xuất

Thêm file:

```text
translator/deepseek_translator.py
```

Interface nên hỗ trợ:

```python
translate_single(text, source, target)
translate_batch(texts, source, target)
translate_pages_batch(pages_texts, source, target, context_memory)
test_connection()
```

### Gợi ý behavior

- Dùng `Authorization: Bearer <DEEPSEEK_API_KEY>`.
- Dùng JSON mode khi batch translate.
- Prompt phải yêu cầu JSON rõ ràng.
- Dùng wrapper JSON object thay vì array trần:

```json
{
  "translations": ["...", "..."]
}
```

hoặc:

```json
{
  "pages": {
    "page_001.png": ["...", "..."]
  }
}
```

### Config dịch manga khuyến nghị

```text
Provider: DeepSeek
Model: deepseek-v4-flash
Thinking: OFF by default
Source: English/Japanese
Target: Vietnamese
Context Memory: ON
```

### Lý do tắt thinking mặc định

Với manga translation, output cần:

- Ngắn.
- Ổn định.
- JSON parse tốt.
- Ít verbose.
- Ít “giải thích”.

Thinking mode có thể hữu ích trong các tác vụ khó, nhưng không nên bật mặc định cho batch dịch thoại.

### Trạng thái

Người dùng đã tích hợp xong, pipeline kết nối được DeepSeek và bản dịch khá ổn.

---

## 3.4. Google GenAI provider

### Mục tiêu

Người dùng tự nâng cấp thêm provider Google GenAI.

### Ghi chú

Chi tiết implementation không nằm trong transcript đầy đủ, nhưng cần ghi nhận đây là một provider đã được thêm bởi người dùng.

Khi tiếp tục phát triển trong chat mới/Codex, cần kiểm tra lại:

```text
translator/google_genai_translator.py
hoặc provider tương ứng trong translator/
```

Các điểm cần verify:

- SDK đang dùng là `google-genai` hay `google-generativeai`.
- Model mặc định là gì.
- Có hỗ trợ batch/page batch không.
- Có dùng JSON mode / structured output không.
- Có context memory không.
- Có retry/rate limit handling không.
- API key lấy từ form hay env var.
- UI có field riêng cho Google GenAI chưa.
- Có fallback khi JSON parse lỗi chưa.

### Gợi ý chuẩn hóa

Google GenAI provider nên có cùng interface như DeepSeek/Gemini/Local LLM:

```python
translate_single(...)
translate_batch(...)
translate_pages_batch(...)
test_connection()
```

---

## 4. Ghi chú về Local LLM / LM Studio

Người dùng đã test thành công app giao tiếp với LM Studio.

Repo gốc có Local LLM/OpenAI-compatible endpoint cho Ollama/LM Studio.

Điểm cần giữ:

- Local LLM nên nhận base URL không kèm `/v1` nếu class tự nối `/v1/chat/completions`.
- Với LM Studio thường dùng:
  ```text
  http://localhost:1234
  ```
  hoặc tùy implementation:
  ```text
  http://localhost:1234/v1
  ```
- Với Ollama thường dùng:
  ```text
  http://localhost:11434
  ```

Cần kiểm tra code hiện tại để tránh lỗi double `/v1/v1`.

---

## 5. Bài học quan trọng từ quá trình nâng cấp

### 5.1. Source English tương thích pipeline tốt hơn source Japanese

Source tiếng Anh cho kết quả tốt vì:

- OCR dễ hơn.
- Text ngang.
- Ít furigana.
- Ít vertical writing.
- Người dùng có thể tự phát hiện lỗi vì biết tiếng Anh.

Nhược điểm:

- Source tiếng Anh có thể đã qua localization.
- Có thể xa bản Nhật gốc.
- Một số nuance từ Nhật sang Anh đã bị mất/sửa.

### 5.2. Source Japanese vẫn quan trọng

Nếu muốn sát bản gốc:

```text
Japanese source → Surya OCR hoặc manga-ocr → local/provider LLM → Vietnamese
```

Nhưng cần cải thiện detector và OCR validation.

### 5.3. OCR tốt không cứu được detector tệ

Trong pipeline hiện tại, detector quyết định OCR được nhìn thấy vùng nào.

Nếu detector bỏ sót text:

```text
No detection → no crop → no OCR → no translation → no redraw
```

Do đó nâng cấp detector/text-region detector là ưu tiên tiếp theo.

---

## 6. Hướng phát triển nâng cấp detector

Đây là hướng phát triển tiếp theo đã thống nhất.

---

## 6.1. Vấn đề hiện tại

Repo gốc dùng bubble-level detector:

```text
YOLO bbox speech bubble
```

Nhược điểm:

- Không bắt đầy đủ text ngoài bubble.
- Dễ bỏ sót caption/narration.
- Khó xử lý SFX.
- Bubble bbox không chính xác bằng mask.
- Không có text mask để xóa chữ chính xác.
- Nếu bubble detect thiếu, OCR không chạy trên vùng đó.

---

## 6.2. Nâng cấp 1: Speech bubble segmentation

Model đề xuất:

```text
kitsumed/yolov8m_seg-speech-bubble
```

Link: https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble

### Mục tiêu

Thay hoặc bổ sung bubble bbox detector bằng segmentation detector.

Thay vì chỉ có:

```python
bubble = {
    "bbox": (x1, y1, x2, y2),
    "score": conf,
}
```

nên có:

```python
bubble = {
    "bbox": (x1, y1, x2, y2),
    "mask": np.ndarray | polygon,
    "score": conf,
    "type": "speech_bubble",
}
```

### Lợi ích

- Mask bubble chính xác hơn bbox.
- Hỗ trợ bubble méo/tròn/nối đuôi.
- Có thể dùng mask để:
  - xác định text line có nằm trong bubble không,
  - xóa chữ theo vùng bubble tốt hơn,
  - tính vùng render text dịch.

### Tích hợp ít phá code

Giai đoạn đầu, giữ output bbox tương thích với code cũ:

```text
detected_bubbles = [bbox, bbox, ...]
```

Nhưng lưu thêm mask trong structure mới:

```python
DetectedRegion(
    kind="bubble",
    bbox=(x1, y1, x2, y2),
    mask=mask,
    score=conf,
)
```

Sau đó dần sửa downstream để dùng mask.

---

## 6.3. Nâng cấp 2: Comic text detector

Model/repo đề xuất:

```text
dmMaze/comic-text-detector
mayocream/comic-text-detector
```

Links:

- https://github.com/dmMaze/comic-text-detector
- https://huggingface.co/mayocream/comic-text-detector

### Mục tiêu

Thêm detector chuyên tìm text region/text line/text mask trong manga/comic.

Đây là nâng cấp quan trọng hơn bubble detector nếu mục tiêu là không bỏ sót chữ.

### Vì sao cần text detector?

Bubble detector trả lời:

```text
Bubble ở đâu?
```

Text detector trả lời:

```text
Chữ ở đâu?
```

Với manga translator, câu hỏi thứ hai quan trọng hơn cho OCR.

### Output mong muốn

```python
TextRegion(
    bbox=(x1, y1, x2, y2),
    mask=mask_or_none,
    score=confidence,
    kind="text",
    reading_order=optional,
)
```

### Lợi ích

- Bắt text trong bubble.
- Bắt caption/narration.
- Bắt floating text.
- Bắt SFX nếu model đủ tốt.
- Giúp OCR chạy đúng vùng có chữ thay vì toàn bộ bubble.
- Cho phép xóa chữ theo text mask thay vì fill cả bubble.

---

## 6.4. Nâng cấp 3: PP-DocLayoutV3, optional

Model docs:

```text
PP-DocLayoutV3
```

Link: https://huggingface.co/docs/transformers/model_doc/pp_doclayout_v3

### Vai trò

Có thể dùng như layout detector phụ hoặc fallback để phân tích bố cục trang.

Theo docs, PP-DocLayoutV3 tích hợp:

- instance segmentation,
- reading order prediction,
- classification labels,
- precise masks.

### Lưu ý

Người dùng không muốn dùng PaddleOCR. PP-DocLayoutV3 có thể dùng qua `transformers`/safetensors, không nhất thiết kéo cả PaddleOCR stack.

Tuy nhiên, đây không nên là bước đầu tiên. Với manga/comic, `comic-text-detector` đúng domain hơn.

---

## 6.5. Kiến trúc detector mới đề xuất

Thay pipeline hiện tại:

```text
bubble detector
→ OCR bubble crop
→ translate
→ redraw
```

bằng pipeline 2 tầng:

```text
1. Bubble detector / bubble segmentation
   → tìm speech bubbles và mask

2. Text detector
   → tìm mọi vùng chữ/text line/text mask

3. Match text region vào bubble
   → text nằm trong bubble nào?
   → text ngoài bubble là caption/SFX/narration?

4. OCR theo text region hoặc group text region
   → Surya OCR

5. Translate theo reading order/context

6. Remove original text
   → ưu tiên text mask
   → fallback bubble mask/bbox

7. Redraw translated text
   → theo bubble area hoặc text region group
```

---

## 6.6. Data structure đề xuất

### Region base

```python
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class Region:
    kind: str
    bbox: Tuple[int, int, int, int]
    score: float = 1.0
    mask: Optional[np.ndarray] = None
```

### Bubble region

```python
@dataclass
class BubbleRegion(Region):
    kind: str = "bubble"
    is_dark: bool = False
```

### Text region

```python
@dataclass
class TextRegion(Region):
    kind: str = "text"
    text: str = ""
    confidence: float = 1.0
    bubble_id: Optional[int] = None
    reading_order: Optional[int] = None
```

### Page result

```python
@dataclass
class PageDetectionResult:
    bubbles: list[BubbleRegion]
    text_regions: list[TextRegion]
```

---

## 6.7. Matching text region vào bubble

### Basic center-point matching

```python
def center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def point_in_bbox(point, bbox, padding=0):
    x, y = point
    x1, y1, x2, y2 = bbox
    return (
        x1 - padding <= x <= x2 + padding
        and y1 - padding <= y <= y2 + padding
    )
```

Dùng:

```python
for text_region in text_regions:
    c = center(text_region.bbox)
    for bubble_id, bubble in enumerate(bubbles):
        if point_in_bbox(c, bubble.bbox, padding=12):
            text_region.bubble_id = bubble_id
            break
```

### Mask-aware matching

Nếu có bubble mask:

```python
def point_in_mask(point, mask):
    x, y = map(int, point)
    h, w = mask.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return mask[y, x] > 0
```

Ưu tiên:

```text
point inside bubble mask
→ fallback point inside padded bbox
→ fallback IoU
```

---

## 6.8. Reading order

Giai đoạn đầu:

- English: sort top-to-bottom, left-to-right.
- Japanese vertical: cẩn thận, không sort bừa nếu Surya đã trả order tốt.
- Nếu dùng PP-DocLayoutV3 hoặc text detector có reading order, ưu tiên order model trả về.

Basic sort:

```python
text_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
```

Nhưng với manga Nhật vertical hoặc layout phức tạp, cần test kỹ.

---

## 6.9. OCR strategy mới

### Option A: OCR theo bubble crop

Hiện tại:

```text
crop bubble → OCR
```

Ưu:

- Dễ.
- Giữ context trong bubble.
- Redraw dễ.

Nhược:

- Detector bubble bỏ sót là chết.
- Text sát viền/caption/SFX dễ mất.
- OCR phải đọc cả vùng bubble, có thể noise.

### Option B: OCR theo text region

Đề xuất:

```text
detect text region → crop text region → OCR
```

Ưu:

- OCR tập trung vào đúng vùng chữ.
- Ít noise.
- Bắt được text ngoài bubble.
- Có text mask để xóa chữ.

Nhược:

- Cần group text lines vào bubble.
- Cần đảm bảo reading order.
- Text line quá nhỏ có thể thiếu context.

### Option C: Hybrid

Khuyến nghị:

```text
text detector để biết chữ ở đâu
bubble detector để biết vùng thoại/render ở đâu
Surya OCR trên text regions hoặc grouped regions
fallback OCR bubble crop nếu text detector fail
```

---

## 6.10. Text removal strategy mới

Hiện tại repo thiên về fill bubble/crop.

Nâng cấp:

```text
text mask → dilate nhẹ → inpaint/fill → redraw
```

Pseudo:

```python
mask = text_region.mask
mask = dilate(mask, kernel_size=3 or 5)
image = inpaint_or_fill(image, mask)
```

Fallback:

```text
không có text mask → dùng text bbox expanded
không có text bbox → dùng bubble mask/bbox
```

---

## 6.11. Roadmap triển khai detector

### Phase 1: Tách detector abstraction

Tạo module:

```text
detectors/
  __init__.py
  base.py
  legacy_yolo_bubble.py
  yolov8_seg_bubble.py
  comic_text_detector.py
  doclayout_detector.py
```

Interface:

```python
class BaseDetector:
    def detect(self, image: np.ndarray) -> PageDetectionResult:
        raise NotImplementedError
```

Mục tiêu: không để `app.py` phụ thuộc trực tiếp vào từng model.

### Phase 2: Bọc legacy detector

Đưa detector gốc vào wrapper:

```text
LegacyYoloBubbleDetector
```

để giữ app chạy ổn.

### Phase 3: Thêm YOLOv8 segmentation bubble detector

Tích hợp model:

```text
kitsumed/yolov8m_seg-speech-bubble
```

Output:

```text
bubbles with bbox + mask
```

Ban đầu vẫn crop theo bbox để không phá downstream.

### Phase 4: Thêm comic text detector

Tích hợp:

```text
dmMaze/mayocream comic-text-detector
```

Output:

```text
text regions with bbox/mask
```

### Phase 5: Match text ↔ bubble

Thêm function:

```python
assign_text_regions_to_bubbles(text_regions, bubbles)
```

### Phase 6: OCR theo text region/group

Thay:

```text
for bubble in bubbles:
    OCR(bubble_crop)
```

bằng:

```text
for bubble in bubbles:
    collect text_regions inside bubble
    OCR group/crops
```

hoặc:

```text
OCR full page via Surya
assign OCR lines to text regions/bubbles
```

### Phase 7: Remove text bằng text mask

Ưu tiên text mask từ text detector.

### Phase 8: UI debug overlay

Thêm debug mode:

```text
Show bubble boxes/masks
Show text boxes/masks
Show assigned bubble_id
Show OCR text per region
```

Đây là cực kỳ quan trọng khi tune detector.

---

## 7. File/module hiện nên kiểm tra khi tiếp tục

Khi mở dự án trong Codex/IDE, kiểm tra các file sau:

```text
app.py
detect_bubbles.py
process_bubble.py
add_text.py
requirements.txt

ocr/
  chrome_lens_ocr.py
  easyocr_english.py
  surya_ocr.py

translator/
  translator.py
  local_llm_translator.py
  gemini_translator.py
  deepseek_translator.py
  google_genai_translator.py
  context_memory.py

templates/
  index.html
```

Nếu repo đã được refactor, tìm các pattern:

```text
selected_ocr
selected_translator
translator_map
process_single_image
process_images_with_batch
process_batch
MODEL_PATH
detect_bubbles
```

---

## 8. Prompt handoff cho chat mới/Codex

Có thể paste đoạn này vào chat mới/Codex:

```text
Tôi đang phát triển một fork cá nhân của pedguedes090/Manga-Translator, dự kiến public tại mranex/my_manga_translator.

Repo gốc là Flask manga translator: YOLOv8 bbox speech-bubble detection, Manga-OCR/Chrome Lens OCR, Gemini/Local LLM/NLLB translation, Context Memory, PIL redraw.

Tôi đã nâng cấp:
- EasyOCR-English cho source manga/comic tiếng Anh.
- SuryaOCR cho English/Japanese OCR, có strip HTML tags như <b>, <i>, <span>, có micro-batch để tránh crash VRAM.
- DeepSeek provider qua OpenAI-compatible API.
- Google GenAI provider do tôi tự thêm.
- Pipeline LM Studio/local LLM đã chạy ổn.
- English source hiện chạy rất tốt, Japanese source cũng cải thiện mạnh với Surya.

Vấn đề tiếp theo:
Detector gốc còn yếu vì chỉ detect speech bubble bbox. Nếu detector bỏ sót text thì OCR không thể đọc. Tôi muốn nâng cấp detection theo hướng:
1. Speech bubble segmentation bằng kitsumed/yolov8m_seg-speech-bubble.
2. Text region/text line/text mask detection bằng dmMaze/mayocream comic-text-detector.
3. Có thể xem PP-DocLayoutV3 như optional layout/reading-order fallback, nhưng không dùng PaddleOCR stack.
4. Refactor detectors thành abstraction riêng.
5. Match text regions vào bubble bằng mask/center point/IoU.
6. OCR bằng Surya trên text regions hoặc grouped regions.
7. Remove text bằng text mask thay vì fill cả bubble.
8. Thêm debug overlay cho bubble/text regions.

Hãy đọc code hiện tại trước, không giả định giống repo gốc hoàn toàn vì tôi đã chỉnh sửa nhiều.
Ưu tiên giữ pipeline hiện tại chạy ổn, refactor từng bước, có fallback về legacy detector.
```

---

## 9. Checklist trước khi public GitHub

Dự kiến repo: `mranex/my_manga_translator`

### Code/license

- [ ] Giữ nguyên license MIT của repo gốc nếu fork từ repo MIT.
- [ ] Ghi credit rõ cho repo gốc `pedguedes090/Manga-Translator`.
- [ ] Ghi credit cho các model/repo được tích hợp:
  - `kha-white/manga-ocr`
  - `datalab-to/surya`
  - `jaidedai/easyocr`
  - `kitsumed/yolov8m_seg-speech-bubble`
  - `dmMaze/comic-text-detector`
  - `mayocream/comic-text-detector`
  - DeepSeek API docs/provider
  - Google GenAI provider nếu dùng SDK chính thức
- [ ] Kiểm tra license của từng dependency/model trước khi redistribute.
- [ ] Không commit API keys.
- [ ] Không commit manga copyrighted sample.
- [ ] Không commit model weights nếu license không cho phép.

### `.gitignore`

Đảm bảo ignore:

```text
.env
*.env
__pycache__/
*.pyc
outputs/
uploads/
translated/
cache/
models/
*.pt
*.onnx
*.safetensors
*.ckpt
```

Tùy ý: nếu muốn tự động download model từ HF thì không commit weights.

### README nên có

- [ ] Mô tả đây là fork/nâng cấp.
- [ ] Hướng dẫn cài đặt.
- [ ] Hướng dẫn dùng Local LLM/LM Studio.
- [ ] Hướng dẫn dùng DeepSeek.
- [ ] Hướng dẫn dùng Google GenAI.
- [ ] Hướng dẫn chọn OCR:
  - Manga-OCR
  - EasyOCR-English
  - Surya-English
  - Surya-Japanese
- [ ] Ghi chú VRAM/batch size cho Surya.
- [ ] Ghi chú chỉ dùng cho nội dung người dùng có quyền truy cập hợp pháp.
- [ ] Không khuyến khích chia sẻ bản dịch của manga có bản quyền.

---

## 10. Nguồn tham khảo chính

- Original repo: https://github.com/pedguedes090/Manga-Translator
- Manga OCR: https://github.com/kha-white/manga-ocr
- Surya OCR: https://github.com/datalab-to/surya
- DeepSeek API: https://api-docs.deepseek.com/
- Speech bubble segmentation model: https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble
- Comic Text Detector repo: https://github.com/dmMaze/comic-text-detector
- Comic Text Detector HF weights: https://huggingface.co/mayocream/comic-text-detector
- PP-DocLayoutV3 docs: https://huggingface.co/docs/transformers/model_doc/pp_doclayout_v3

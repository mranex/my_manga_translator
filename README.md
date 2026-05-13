# My Manga Translator

## Tổng quan

Đây là web app dịch manga/comic chạy local hoặc bán-local trên Flask. App tập trung vào pipeline hiện tại đã ổn định:

- phát hiện vùng layout và text block
- phát hiện speech bubble bằng segmentation
- gom OCR item theo container thật sự
- OCR crop bằng PaddleOCR-VL Local qua `llama.cpp` hoặc Chrome Lens fallback
- dịch bằng translator đã chọn
- xóa text gốc bằng LaMa Manga
- render lại text bằng manga renderer trong suốt, không còn box fill kiểu cũ

## Pipeline hiện tại

Luồng xử lý một trang hiện tại:

1. Upload một hoặc nhiều ảnh.
2. `PP-DocLayoutV3` tìm layout/text block trên full page.
3. `YOLOv8m seg speech bubble` tìm bubble mask trên full page.
4. `comic-text-detector` tìm text evidence để refine crop.
5. App gom các detection thành OCR item theo container:
   - `1 bubble = 1 bubble item`
   - `1 PP outside text block = 1 outside_text item`
6. OCR từng crop bằng:
   - `PaddleOCR-VL Local` qua `llama.cpp` sidecar, hoặc
   - `Chrome Lens` fallback nếu bạn vẫn muốn dùng
7. Translate theo provider đã chọn.
8. Tạo text-removal mask theo text block.
9. Inpaint bằng `LaMa Manga`.
10. Render text dịch lại bằng transparent manga renderer.
11. Export ảnh hoặc ZIP.

## Thành phần chính

- Flask UI: upload, chọn OCR/translator, trả ảnh kết quả
- `PP-DocLayoutV3`: layout-first text block planner
- `YOLOv8 segmentation`: speech bubble detector với full-page mask
- `comic-text-detector`: text evidence để refine OCR crop
- `PaddleOCR-VL Local`: OCR local qua `llama.cpp`
- `LaMa Manga`: xóa text gốc bằng inpainting
- `manga-style renderer`: render lại text bằng RGBA overlay + alpha composite

## OCR providers

Đang hỗ trợ:

- `PaddleOCR-VL Local`: OCR local/private, khuyến nghị dùng chính
- `Chrome-Lens`: fallback tùy chọn nếu đã cấu hình ổn định

Đã loại khỏi app chính:

- `MangaOCR`
- `EasyOCR`
- `SuryaOCR`

Các OCR cũ đã được gỡ khỏi UI, active app path và `requirements.txt` để giảm xung đột dependency, đặc biệt với stack `PP-DocLayoutV3 + transformers`.

## Translator providers

Các translator hiện còn trong app:

- Gemini
- Local LLM (OpenAI-compatible / Ollama / LM Studio / LocalAI)
- DeepSeek
- Google
- NLLB
- Baidu
- Bing

## PaddleOCR-VL setup

Xem thêm: [docs/paddleocr_vl_llama.md](docs/paddleocr_vl_llama.md)

Biến môi trường quan trọng:

- `PADDLEOCR_VL_MODEL_PATH`
- `PADDLEOCR_VL_MMPROJ_PATH`
- `LLAMA_CPP_DIR`

Tùy chọn:

- `PADDLEOCR_VL_SERVER_URL`
- `PADDLEOCR_VL_LLAMA_SERVER_CMD`
- `PADDLEOCR_VL_MAX_WORKERS`

Nếu không set env var, app sẽ tìm:

- `model/paddleocr_vl/model.gguf`
- `model/paddleocr_vl/mmproj.gguf`

## Cài đặt

### PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Git Bash / CMD

```bash
py -3.11 -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Chạy app

```powershell
python app.py
```

Hoặc dùng:

```powershell
run.bat
```

`run.bat` có thể pre-start `llama-server` cho PaddleOCR-VL nếu bạn đã đặt đúng model/mmproj và `LLAMA_CPP_DIR`.

## Model và cache

- Model được tải hoặc đặt local trong máy.
- Thư mục `model/` hiện đã được ignore khỏi git.
- `llama.cpp` binary cũng được xem là local tool và đã được ignore.

Nếu trước đây `model/` đã bị track trong repo local của bạn, cần tự bỏ nó khỏi index:

```powershell
git rm --cached -r model/
```

Lệnh trên chỉ gỡ khỏi git index, không xóa file local nếu dùng đúng như trên.

## Ghi chú cleanup hiện tại

- Đã bỏ active OpenCV black-bubble path khỏi UI và backend.
- `detect_bubbles.py` và legacy adapter vẫn có thể còn trong repo để tham chiếu/test legacy, nhưng không còn là runtime path chính.
- Active render path không còn dùng `add_text()` hay box-fill kiểu cũ.

## Troubleshooting

### PaddleOCR-VL model/mmproj không tìm thấy

Set:

- `PADDLEOCR_VL_MODEL_PATH`
- `PADDLEOCR_VL_MMPROJ_PATH`

hoặc đặt file vào `model/paddleocr_vl/`.

### Không tìm thấy llama-server

Set `LLAMA_CPP_DIR` trỏ tới thư mục chứa `llama-server.exe`, hoặc add binary vào `PATH`, hoặc set trực tiếp `PADDLEOCR_VL_SERVER_URL` nếu bạn đã chạy server sẵn.

### Không muốn dùng PaddleOCR-VL ngay

Bạn vẫn có thể chọn `Chrome-Lens` làm fallback OCR.

### CUDA không bắt buộc

Pipeline hiện tại vẫn có thể chạy CPU fallback cho OCR sidecar và nhiều thành phần khác. CUDA là tùy chọn, không bắt buộc để app khởi động.

## PaddleOCR-VL via llama.cpp

`PaddleOCR-VL Local` là OCR local được khuyến nghị cho app hiện tại.

Backend này chỉ OCR các crop text đã được pipeline chọn sẵn. Nó không chạy page detection và không yêu cầu cài `paddleocr`, `paddlepaddle` hay đổi phiên bản `transformers` trong Python venv chính.

Chrome Lens vẫn có thể giữ lại như fallback tùy chọn, nhưng OCR chính của app hiện tại là `paddleocr-vl`.

### Biến môi trường cần có

- `PADDLEOCR_VL_MODEL_PATH`
- `PADDLEOCR_VL_MMPROJ_PATH`
- `LLAMA_CPP_DIR`

Tùy chọn:

- `PADDLEOCR_VL_SERVER_URL`
- `PADDLEOCR_VL_LLAMA_SERVER_CMD`
- `PADDLEOCR_VL_MAX_WORKERS`

### Đường dẫn local mặc định

Nếu không set env var, app sẽ thử đọc:

- `model/paddleocr_vl/model.gguf`
- `model/paddleocr_vl/mmproj.gguf`

### Ví dụ cấu hình trên Windows

```powershell
set PADDLEOCR_VL_MODEL_PATH=C:\path\to\model.gguf
set PADDLEOCR_VL_MMPROJ_PATH=C:\path\to\mmproj.gguf
set LLAMA_CPP_DIR=C:\path\to\llama.cpp
```

### Dùng server có sẵn

Nếu bạn đã tự chạy một `llama-server` tương thích:

```powershell
set PADDLEOCR_VL_SERVER_URL=http://127.0.0.1:8088
```

### Dùng command tùy biến để start llama-server

Nếu bản `llama.cpp` của bạn cần lệnh khác:

```powershell
set PADDLEOCR_VL_LLAMA_SERVER_CMD=llama-server -m {model} --mmproj {mmproj} --host {host} --port {port} -c {ctx} -ngl {ngl}
```

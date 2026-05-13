## PaddleOCR-VL via llama.cpp

This app supports a local OCR provider named `paddleocr-vl` / `PaddleOCR-VL Local`.

It OCRs the existing text crops produced by the current detector pipeline. It does not run page detection and it does not require PaddleOCR, PaddlePaddle, or Transformers changes in the main Python environment.

### Required environment variables

- `PADDLEOCR_VL_MODEL_PATH`
- `PADDLEOCR_VL_MMPROJ_PATH`
- `LLAMA_CPP_DIR`

Optional:

- `PADDLEOCR_VL_SERVER_URL`
- `PADDLEOCR_VL_LLAMA_SERVER_CMD`
- `PADDLEOCR_VL_MAX_WORKERS`

### Default local paths

If the env vars are not set, the app looks for:

- `model/paddleocr_vl/model.gguf`
- `model/paddleocr_vl/mmproj.gguf`

### Example Windows setup

```powershell
set PADDLEOCR_VL_MODEL_PATH=C:\path\to\model.gguf
set PADDLEOCR_VL_MMPROJ_PATH=C:\path\to\mmproj.gguf
set LLAMA_CPP_DIR=C:\path\to\llama.cpp
```

### Optional server URL

If you already run a compatible llama.cpp server:

```powershell
set PADDLEOCR_VL_SERVER_URL=http://127.0.0.1:8088
```

### Optional custom server command

If your llama.cpp build needs a custom launch command:

```powershell
set PADDLEOCR_VL_LLAMA_SERVER_CMD=llama-server -m {model} --mmproj {mmproj} --host {host} --port {port} -c {ctx} -ngl {ngl}
```

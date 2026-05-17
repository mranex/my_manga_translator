# Manga Translator

Ứng dụng desktop dịch manga/comic theo quy trình nhiều bước, xây dựng bằng PyQt6. Dự án tập trung vào một workflow thực tế: phát hiện vùng chữ, OCR, dịch, xóa chữ gốc, render lại bản dịch và xuất kết quả cuối cùng theo từng trang hoặc cả project.

## Mục lục

- [Mục đích của ứng dụng](#mục-đích-của-ứng-dụng)
- [Tính năng chính](#tính-năng-chính)
- [Kiến trúc tổng thể](#kiến-trúc-tổng-thể)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Pipeline xử lý](#pipeline-xử-lý)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Chuẩn bị model và công cụ](#chuẩn-bị-model-và-công-cụ)
- [Khởi chạy ứng dụng](#khởi-chạy-ứng-dụng)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [OCR và translator đang hỗ trợ](#ocr-và-translator-đang-hỗ-trợ)
- [Cấu trúc project làm việc](#cấu-trúc-project-làm-việc)
- [Log, cache và dữ liệu sinh ra](#log-cache-và-dữ-liệu-sinh-ra)
- [Ghi chú phát triển](#ghi-chú-phát-triển)
- [Sự cố thường gặp](#sự-cố-thường-gặp)

## Mục đích của ứng dụng

`Manga Translator` được xây dựng để hỗ trợ dịch manga/comic theo ba mức độ:

- Tự động: chạy gần như toàn bộ pipeline từ ảnh nguồn đến ảnh đã render.
- Bán tự động: cho phép kiểm tra và chỉnh sửa ở từng stage quan trọng.
- Thủ công có hỗ trợ: dùng ứng dụng như một workbench để quản lý project, cache và dữ liệu trung gian.

Mục tiêu của dự án không chỉ là dịch văn bản, mà là tạo ra một môi trường xử lý truyện tranh có thể:

- làm việc theo từng trang;
- lưu cache để không phải chạy lại từ đầu;
- cho phép can thiệp thủ công khi detection, OCR hoặc render chưa đạt chất lượng mong muốn;
- hỗ trợ cả local workflow lẫn workflow dùng API.

## Tính năng chính

- Giao diện desktop PyQt6 theo workflow nhiều stage.
- Quản lý project riêng với `project.json`, `source/` và `cache/`.
- Tách rõ các bước Detection, OCR, Translation, Inpaint, Render, Export.
- Hỗ trợ chạy từng bước riêng lẻ hoặc chạy pipeline một chạm.
- Lưu kết quả trung gian theo từng trang để tiếp tục làm việc ở lần mở sau.
- Cho phép chỉnh sửa detection item, OCR item, translation item và render item.
- Hỗ trợ OCR local qua `llama.cpp`.
- Hỗ trợ nhiều translator, từ local LLM đến các backend API.
- Render lại text bằng font truyện tranh thay vì cách chèn text thô đơn giản.

## Kiến trúc tổng thể

Dự án hiện xoay quanh hai phần chính:

- `mmt_gui/`: giao diện desktop, stage panel, worker, service và thành phần hiển thị.
- `mmt_core/`: lõi pipeline, cache I/O, orchestration và xử lý nghiệp vụ.

Các backend xử lý dùng chung:

- `detectors/`: phát hiện layout và speech bubble.
- `ocr/`: tích hợp OCR engine.
- `inpainting/`: xử lý mask và LaMa Manga.
- `translator/`: các translator và helper dịch batch.
- `text_rendering/`: layout text và render kết quả cuối.

Thiết kế hiện tại theo hướng:

- GUI chịu trách nhiệm điều phối và hiển thị trạng thái.
- `mmt_core` chịu trách nhiệm xử lý pipeline và quản lý dữ liệu trung gian.
- Các package backend được tái sử dụng xuyên suốt nhiều stage.

## Cấu trúc thư mục

Các thành phần chính còn có trong repo:

```text
Manga-Translator/
├─ mmt_gui/                    # Giao diện desktop PyQt6
├─ mmt_core/                   # Lõi pipeline và quản lý cache
├─ detectors/                  # Detection backend
├─ ocr/                        # OCR backend
├─ inpainting/                 # Inpainting backend
├─ translator/                 # Translation backend
├─ text_rendering/             # Render text
├─ fonts/                      # Font dùng khi render
├─ model/                      # Model local
├─ tools/                      # Công cụ local như llama.cpp
├─ pj/                         # Project mẫu / dữ liệu làm việc mẫu
├─ logs/                       # Log runtime cấp workspace
├─ PyQt_run.bat                # Cách chạy app nhanh trên Windows
├─ requirements.txt            # Phụ thuộc Python chính
├─ requirements-merged-final.txt
├─ requirements_1.txt
└─ README.md
```

Ngoài ra ở thư mục gốc vẫn còn một số script helper hoặc mã cũ phục vụ tham chiếu kỹ thuật, ví dụ:

- `add_text.py`
- `batch_ocr_flow.py`
- `detect_bubbles.py`
- `font_analyzer.py`
- `ocr_crop_utils.py`
- `process_bubble.py`
- `render_item_utils.py`

Chúng không phải entrypoint chính của ứng dụng hiện tại. Đường chạy chính là desktop app trong `mmt_gui/` và `mmt_core/`.

## Pipeline xử lý

Pipeline hiện tại của ứng dụng gồm các bước chính sau:

1. `Detection`
   - Phát hiện layout/text region.
   - Phát hiện speech bubble bằng segmentation.

2. `OCR Prepare`
   - Chuẩn hóa detection thành OCR item nội bộ.
   - Cắt crop theo vùng OCR.

3. `OCR`
   - Gửi crop đến OCR provider đã chọn.
   - Ghi kết quả OCR vào cache.

4. `Translation Init`
   - Khởi tạo dữ liệu dịch từ OCR cache.

5. `Translation`
   - Dịch text bằng translator đã chọn.
   - Có thể chạy theo batch nếu backend hỗ trợ.

6. `Mask Prepare`
   - Tạo mask để chuẩn bị xóa text gốc.

7. `Inpaint`
   - Xóa vùng text gốc trên ảnh.

8. `Render Prepare`
   - Chuyển dữ liệu dịch thành render item.

9. `Render`
   - Dàn chữ và render bản dịch lên ảnh đã inpaint.

10. `Export`
   - Xuất ảnh đầu ra từ dữ liệu render.

Ứng dụng cũng có luồng xử lý một chạm từ Detection đến Render cho một trang hoặc toàn bộ project.

## Yêu cầu hệ thống

Khuyến nghị:

- Windows
- Python 3.11
- Môi trường ảo `.venv`

Tài nguyên phần cứng:

- GPU là tùy chọn, không bắt buộc để khởi chạy ứng dụng.
- Nếu chạy OCR/inpainting/detection nặng trên CPU, thời gian xử lý có thể đáng kể.

Các thư viện nặng thường tham gia pipeline:

- `torch`
- `torchvision`
- `ultralytics`
- `transformers`

## Cài đặt

### PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### CMD

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Sau khi cài xong, kiểm tra nhanh:

```powershell
python --version
python -m mmt_gui.main
```

## Chuẩn bị model và công cụ

### Model local

Repo hiện dùng thư mục `model/` để chứa model local và cache model. Tùy cấu hình bạn có thể cần:

- model cho detection;
- model cho inpainting;
- model OCR local;
- model/mmproj cho các backend chạy qua `llama.cpp`.

`model/` là dữ liệu local, có thể lớn và thường không phù hợp để quản lý như mã nguồn.

### llama.cpp

Nếu dùng OCR local qua server, bạn cần chuẩn bị:

- `llama-server` tương thích;
- binary trong `tools/llama.cpp/` hoặc đã thêm vào `PATH`;
- model và mmproj đúng vị trí theo cấu hình trong GUI.

GUI hiện mặc định dùng:

- thư mục `tools/llama.cpp`
- server URL `http://127.0.0.1:8080`

## Khởi chạy ứng dụng

Cách chạy nhanh trên Windows:

```bat
PyQt_run.bat
```

Hoặc chạy trực tiếp:

```powershell
.venv\Scripts\Activate.ps1
python -m mmt_gui.main
```

Đây là entrypoint chính của ứng dụng hiện tại.

## Hướng dẫn sử dụng

### 1. Tạo hoặc mở project

Khi mở ứng dụng:

- tạo project mới trong một thư mục riêng; hoặc
- mở `project.json` của project đã có.

Mỗi project được tổ chức thành:

- `source/`: ảnh gốc đã import
- `cache/`: dữ liệu trung gian theo từng stage

### 2. Import ảnh

Trong stage Project:

- import một hoặc nhiều ảnh truyện;
- ứng dụng sẽ copy ảnh vào `source/`;
- danh sách trang sẽ được lưu trong `project.json`.

### 3. Cấu hình OCR

Trong stage OCR hoặc phần cấu hình liên quan:

- chọn OCR provider;
- nếu dùng OCR local, kiểm tra server URL, model path và thư mục `llama.cpp`;
- nếu dùng Chrome Lens, cấu hình đường dẫn Chrome hoặc profile nếu cần.

### 4. Cấu hình translator

Trong stage Translation:

- chọn translator phù hợp;
- nhập API key nếu backend yêu cầu;
- nhập URL/model nếu dùng local LLM hoặc OpenAI-compatible endpoint;
- cấu hình prompt style, custom prompt hoặc ghi chú project nếu cần.

### 5. Chạy từng stage riêng lẻ

Bạn có thể chạy riêng từng bước:

- Detection
- OCR Prepare
- OCR
- Translation
- Inpaint
- Render
- Export

Cách này phù hợp khi muốn kiểm tra chất lượng và can thiệp thủ công giữa pipeline.

### 6. Chạy pipeline một chạm

Ứng dụng hỗ trợ luồng process từ:

- Detection
- OCR Prepare
- OCR
- Translation Init
- Translation
- Mask Prepare
- Inpaint
- Render Prepare
- Render

Luồng này phù hợp khi đã cấu hình OCR và translator xong, và muốn xử lý nhanh cho một trang hoặc cả project.

### 7. Chỉnh dữ liệu trung gian

Workflow desktop cho phép can thiệp trực tiếp ở các stage quan trọng:

- loại bỏ hoặc khôi phục detection item;
- sửa OCR item;
- sửa text OCR nhận diện sai;
- sửa text dịch;
- loại bỏ hoặc khôi phục render item;
- xem preview theo từng bước.

Đây là một phần quan trọng của workflow bán tự động.

### 8. Export kết quả

Sau khi render:

- mở stage Export;
- chọn phạm vi trang;
- xuất ảnh đầu ra theo cấu hình hiện tại.

## OCR và translator đang hỗ trợ

### OCR provider

Các OCR provider đang được desktop app hỗ trợ:

- `PaddleOCR-VL Local`
- `DeepSeek OCR (llama.cpp)`
- `Chrome Lens`

Khuyến nghị thực tế:

- `PaddleOCR-VL Local` là lựa chọn local chính.
- `Chrome Lens` phù hợp làm fallback.

### Translator

Các translator hiện có trong ứng dụng:

- Gemini
- Local LLM
- DeepSeek
- OpenAI Compatible
- Google
- NLLB
- Baidu
- Bing

Các lựa chọn thường hữu ích nhất cho workflow hiện tại:

- Local LLM
- OpenAI Compatible
- DeepSeek
- Gemini

## Cấu trúc project làm việc

Một project điển hình có dạng:

```text
MyProject/
├─ project.json
├─ source/
│  ├─ 001.jpg
│  ├─ 002.jpg
│  └─ ...
└─ cache/
   ├─ detection/
   ├─ ocr/
   ├─ ocr_crops/
   ├─ translation/
   ├─ inpaint/
   ├─ render/
   ├─ render_sprites/
   └─ masks/
```

Ý nghĩa:

- `project.json`: trạng thái project và danh sách trang
- `source/`: ảnh nguồn
- `cache/`: dữ liệu trung gian của từng stage

Thiết kế này giúp:

- chạy lại có chọn lọc;
- debug thuận tiện;
- tiếp tục chỉnh sửa project mà không mất kết quả trước đó.

## Log, cache và dữ liệu sinh ra

Ứng dụng sinh ra các loại dữ liệu chính:

- `logs/`: log runtime cấp workspace
- `cache/`: cache theo từng stage
- crop OCR
- mask inpaint
- sprite render
- ảnh render cuối

Cache là một phần trọng yếu của workflow. Không nên xóa toàn bộ `cache/` nếu bạn còn muốn tiếp tục làm việc trên project hiện tại.

## Ghi chú phát triển

Hiện tại:

- desktop app là đường chạy chính;
- `mmt_core` là trung tâm điều phối pipeline;
- nhiều backend trong `detectors/`, `ocr/`, `inpainting/`, `translator/`, `text_rendering/` đang được dùng trực tiếp;
- một số script ở thư mục gốc tồn tại như helper hoặc mã tham chiếu, không phải luồng chạy chính.

Nếu tiếp tục phát triển repo này, hướng nên tập trung là:

- cải tiến `mmt_gui/`;
- tiếp tục chuẩn hóa `mmt_core/`;
- giữ backend dùng chung ổn định và tách biệt khỏi GUI;
- chỉ dọn các script gốc khi đã xác nhận không còn phụ thuộc nội bộ.

## Sự cố thường gặp

### Không chạy được ứng dụng

Kiểm tra:

- đã tạo `.venv`
- đã cài `requirements.txt`
- đang dùng Python 3.11

### OCR local không hoạt động

Kiểm tra:

- `llama-server` có tồn tại không
- `server_url` có đúng không
- model và mmproj có đúng vị trí không
- cổng đang dùng có bị ứng dụng khác chiếm không

### Detection, OCR hoặc inpaint quá chậm

Nguyên nhân thường gặp:

- chạy CPU thay vì GPU
- model lớn
- project nhiều trang
- OCR server local chưa tối ưu cấu hình

### Kết quả OCR hoặc render chưa tốt

Nên thử:

- chỉnh lại detection item
- sửa OCR thủ công trước khi dịch
- đổi OCR provider
- đổi translator
- kiểm tra font đang dùng

### Cache bị lệch với trạng thái hiện tại

Khi thay đổi mạnh cấu hình hoặc chỉnh sửa detection/OCR thủ công, có thể cần chạy lại các stage phía sau để đồng bộ kết quả.

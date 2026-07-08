<p align="center">
  <img src="https://raw.githubusercontent.com/tungedng2710/tungedng2710.github.io/main/assets/images/tonai_logo.png" alt="TonAI Vision Lab" width="200" style="border-radius: 80px;">
</p>

<h1 align="center">TonAI Vision Lab - TrafficCam</h1>

<p align="center">
  <strong>Hệ thống Nhận diện Biển số xe (ALPR) và Giám sát Giao thông</strong>
</p>

## Khởi động API

```bash
uv sync                              # main deps
scripts/install_fusion_inproc.sh     # vendored mf-lpr2 + eott (needed for the fusion endpoints)
uv run main.py                       # binds 0.0.0.0:7862
```

Swagger UI: `http://localhost:7862/docs`. Xem [Bắt đầu nhanh](#bắt-đầu-nhanh) bên
dưới để biết tùy chọn Docker và cách cài đặt đầy đủ.

---

## Mục lục
- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Bắt đầu nhanh](#bắt-đầu-nhanh)
- [Kiến trúc](#kiến-trúc)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Dịch vụ API](#dịch-vụ-api)
  - [Cấu hình](#cấu-hình)
  - [Endpoint API](#endpoint-api)
  - [Ví dụ sử dụng](#ví-dụ-sử-dụng)
- [Gộp biển số đa khung hình](#gộp-biển-số-đa-khung-hình)
- [Huấn luyện Model](#huấn-luyện-model)
- [Model Zoo](#model-zoo)
- [Trọng số Model](#trọng-số-model)
- [Quản lý dữ liệu](#quản-lý-dữ-liệu)
- [Tải Dataset](#tải-dataset)
- [Giao diện WebApp](#giao-diện-webapp)
- [Triển khai Docker](#triển-khai-docker)
- [Phát triển](#phát-triển)
- [Trích dẫn](#trích-dẫn)

---

## Tổng quan

TrafficCam là một hệ thống giám sát giao thông và nhận diện biển số xe toàn
diện, được xây dựng trên các công nghệ deep learning hiện đại. Hệ thống đã
được tái cấu trúc (refactor) thành một kiến trúc module rõ ràng:

- **FastAPI Service**: REST API dựa trên JSON để phát hiện xe và biển số xe
- **Core ALPR Engine**: Logic xử lý dùng chung cho phát hiện, theo dõi
  (tracking) và OCR
- **Web Application**: Giao diện trực quan hóa thời gian thực (bản beta)
- **Modular Design**: Tách biệt rõ ràng giữa API, xử lý và trực quan hóa

**Cập nhật mới nhất:**
- **Tháng 1/2026**: Tái cấu trúc thành dịch vụ FastAPI với API JSON
- **Tháng 8/2025**: PPOCRv5 và Web App (bản beta) để kiểm thử
- **Tháng 7/2024**: PPOCRv4 kèm text detection giúp nhận diện biển số tốt hơn
- **Tháng 11/2023**: Hỗ trợ inference TensorRT và model ONNX

---

## Tính năng

### Đã hoàn thành ✅
- **Vehicle Detection**: Phát hiện xe đa lớp dựa trên YOLO (ô tô, xe tải, xe
  buýt, xe máy, xe đạp)
- **License Plate Detection**: Phát hiện biển số độ chính xác cao trong nhiều
  điều kiện
- **OCR Recognition**: PaddleOCR v4/v5 để nhận diện văn bản
- **Object Tracking**: Thuật toán theo dõi SORT và DeepSORT
- **Multiple Formats**: Hỗ trợ model PyTorch (.pt) và ONNX (.onnx)
- **TensorRT Optimization**: Tăng tốc inference với TensorRT
- **REST API**: Dịch vụ dựa trên FastAPI với phản hồi JSON
- **Web Interface**: Trực quan hóa thời gian thực với hỗ trợ WebRTC

### Đang thực hiện 🚧
- Tích hợp Vision Language Model
- AutoLabel Pipeline

---

## Yêu cầu hệ thống

- **Hệ điều hành**: Ubuntu 20.04 trở lên (khuyến nghị dùng Linux)
- **Python**: 3.9 - 3.12
- **GPU**: NVIDIA GPU hỗ trợ CUDA (khuyến nghị cho production)
- **Bộ nhớ lưu trữ**: Đủ dung lượng cho model weights (~2GB)

---

## Bắt đầu nhanh

### Lựa chọn 1: Dùng Docker (khuyến nghị)

```bash
docker compose up -d
docker compose logs -f
docker compose down
```

**Các dịch vụ sẽ chạy tại:**
- Analysis API: http://localhost:7862 (Swagger: http://localhost:7862/docs)
- WebApp: http://localhost:7863

### Lựa chọn 2: Cài đặt thủ công

### 1. Cài đặt

```bash
# Install dependencies (uv, recommended)
uv sync
scripts/install_fusion_inproc.sh   # vendored mf-lpr2 + eott, for the fusion endpoints

# Or manually
pip install -r requirements.txt
```

### 2. Khởi động dịch vụ API

```bash
uv run main.py

# Using Uvicorn directly (production)
uv run uvicorn main:app --host 0.0.0.0 --port 7862
```

Dịch vụ sẽ khởi động tại `http://localhost:7862`

### 3. Kiểm tra dịch vụ

```bash
# Health check
curl http://localhost:7862/health

# Vehicle detect + plate on an image
curl -F files=@path/to/image.jpg 'http://localhost:7862/predict/batch?format=json'

# Using example client
uv run client_example.py --health
uv run client_example.py --video path/to/video.mp4
```

### 4. Tài liệu API tương tác
- **Swagger UI**: http://localhost:7862/docs
- **ReDoc**: http://localhost:7862/redoc

---

## Kiến trúc

```
┌──────────────────────────────────────────────────────────┐
│                   Traffic AI System                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐                 ┌─────────────────┐    │
│  │   main.py    │◄────────────────┤ client_example  │    │
│  │ FastAPI API  │                 │    (Python)     │    │
│  │   Service    │                 └─────────────────┘    │
│  │              │                                        │
│  │ Endpoints:   │                                        │
│  │ /predict/    │                                        │
│  │  frame,batch │                                        │
│  │ /predict/    │                                        │
│  │  plates/*    │                                        │
│  │ /fuse        │                                        │
│  │ /reset       │                                        │
│  │ /config      │                                        │
│  └──────┬───────┘                                        │
│         │                                                │
│         │ uses                                           │
│         ▼                                                │
│  ┌────────────────┐                                      │
│  │ utils/         │                                      │
│  │ traffic_       │   Core Processing Logic              │
│  │  analysis.py   │   - Vehicle Detection (YOLO)         │
│  │                │   - Plate Detection (YOLO)           │
│  │                │   - Super-Resolution (optional SR)   │
│  │                │   - OCR (PaddleOCR)                  │
│  │                │   - Tracking (SORT/DeepSORT)         │
│  └────────────────┘                                      │
│                                                          │
│  ┌─────────────────────────────────────────┐             │
│  │        webapp/ (Visualization)          │             │
│  │  ┌─────────────────┐  ┌──────────────┐  │             │
│  │  │ backend/main.py │  │  frontend/   │  │             │
│  │  │ - Video Stream  │  │  - Web UI    │  │             │
│  │  │ - WebRTC        │  │  - Player    │  │             │
│  │  │ - RTSP Support  │  │              │  │             │
│  │  └─────────────────┘  └──────────────┘  │             │
│  └─────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────┘
```

---

## Cấu trúc dự án

```
license-plate-recognition/
├── main.py                    # FastAPI service (primary entry point)
├── client_example.py          # Example client for testing
├── api_test.py                # API testing script
├── requirements.txt           # Python dependencies
├── api/                       # API docs + Postman collections (traffic-ai-*, fusion-svc-*)
├── fusion_svc/                # Multi-frame plate fusion: vendored mf-lpr2/eott,
│                               #   optional standalone app (see Multi-Frame Plate Fusion)
├── data/                      # Data management
│   ├── config/               # Configuration files (JSON, TXT)
│   ├── logs/                 # Application logs and outputs
│   ├── recordings/           # Camera recording data
│   ├── results/              # Processing results and reports
│   ├── samples/              # Test and raw sample data
│   └── scripts/              # Data processing utilities
├── weights/                   # Model weights
│   ├── vehicle/              # Vehicle detection models
│   ├── plate/                # License plate detection models
│   ├── pretrained/           # Pre-trained YOLO base models
│   ├── tracking/             # DeepSORT tracker weights
│   └── ocr/                  # PaddleOCR models
├── utils/                     # Core utilities
│   ├── traffic_analysis.py   # Traffic Analysis Service (core logic)
│   ├── ocr.py                # OCR utilities
│   ├── utils.py              # Helper functions
│   └── ppocr_configs.yaml    # PaddleOCR configuration
├── tracking/                  # Object tracking algorithms
│   ├── sort.py               # SORT tracker
│   └── deep_sort/            # DeepSORT tracker
├── detectors/                 # Detector implementations
│   ├── yolo/                 # Ultralytics YOLO (v8, v9, v10, v11)
│   └── rf-detr/              # RF-DETR transformer detector
├── scripts/                   # Training and testing scripts
└── webapp/                    # Web visualization application (beta)
    ├── backend/              # FastAPI backend for streaming
    ├── frontend/             # Web UI
    └── utils/                # Webapp utilities
```

---

## Dịch vụ API

### Cấu hình

Dịch vụ được cấu hình thông qua biến môi trường:

```bash
# Device configuration
export ALPR_DEVICE=cuda:0      # Options: auto, cpu, cuda:0, cuda:1, etc.

# Model paths
export VEHICLE_WEIGHT=weights/vehicle/vehicle_yolov9s_640_30oct2025.pt
export PLATE_WEIGHT=weights/plate/plate_yolo12n_640_2025.pt
export DSORT_WEIGHT=weights/tracking/deepsort/ckpt.t7

# Detection thresholds
export VEHICLE_CONF=0.6        # Vehicle detection confidence (0.0-1.0)
export PLATE_CONF=0.25         # Plate detection confidence (0.0-1.0)
export PLATE_IMGSZ=1280        # Plate detect inference size (full-frame path); small/distant plates need this above the weight's native 640
export OCR_THRESHOLD=0.9       # OCR confidence threshold (0.0-1.0)

# Tracking
export USE_DEEPSORT=false      # Use DeepSORT (true) or SORT (false)

# Super-resolution (plate crop, applied before OCR)
export SR_ENGINE=none          # Options: none (default), bicubic (zero-dep baseline), realesrgan (needs `uv pip install realesrgan basicsr`), lcofl (stub, needs VN retrain). Cascade: only runs on low-confidence plates.
export SR_SCALE=2              # Upscale factor (default: 2)
export REALESRGAN_WEIGHT=      # Optional path to Real-ESRGAN weights (realesrgan engine only)

# Language
export LANG=en                 # Options: en, vi, es, fr

# Server
export HOST=0.0.0.0
export PORT=7862
```

### Endpoint API

| Endpoint | Method | Mô tả |
|----------|--------|-------------|
| `/` | GET | Thông tin dịch vụ và danh sách endpoint có sẵn |
| `/health` | GET | Health check và trạng thái cấu hình |
| `/predict/frame` | POST | Xử lý một frame video (giữ trạng thái tracking qua các lần gọi) |
| `/predict/batch` | POST | N ảnh/video/zip → media đã chú thích (1→JPEG/MP4, N→ZIP) hoặc JSON (`?format=json`) |
| `/predict/plates/batch` | POST | Phát hiện biển số trên N ảnh, dual-OCR (FAST + PPOCRv6) → JPEG/ZIP đã chú thích |
| `/predict/plates/multiframe` | POST | Gộp một chuỗi (burst) crop của một biển số → dual-OCR biển số đã phục dựng |
| `/predict/plates/video` | POST | Phát hiện+theo dõi biển số trong video, gộp burst của từng track, dual-OCR |
| `/fuse` | POST | Gộp một chuỗi crop → ảnh biển số đã phục dựng (PNG), không OCR |
| `/reset` | POST | Reset trạng thái tracker |
| `/config` | GET | Lấy cấu hình hiện tại |

> Không có endpoint `/predict/image` riêng — với một ảnh đơn lẻ, dùng
> `/predict/batch` với một file (`?format=json` để nhận JSON thay vì ảnh
> đã chú thích).
>
> Gộp đa khung hình (`/fuse`, `/predict/plates/multiframe`, `/predict/plates/video`)
> mặc định chạy **in-process** — không cần sidecar (xem
> [Gộp biển số đa khung hình](#gộp-biển-số-đa-khung-hình)).

### Định dạng phản hồi

```json
{
  "detections": [
    {
      "track_id": 1,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 350.8,
        "y2": 450.2
      },
      "vehicle_type": "car",
      "license_plate": "ABC123",
      "plate_bbox": {
        "x1": 150.0,
        "y1": 380.0,
        "x2": 300.0,
        "y2": 420.0
      },
      "confidence": 0.95
    }
  ],
  "frame_count": 42
}
```

### Ví dụ sử dụng

#### Dùng cURL

```bash
# Health check
curl http://localhost:7862/health

# Process image (vehicle type + plate, JSON)
curl -F "files=@car.jpg" "http://localhost:7862/predict/batch?format=json"

# Process video frame
curl -X POST "http://localhost:7862/predict/frame?frame_number=42" \
  -F "file=@frame.jpg"

# Reset tracker
curl -X POST http://localhost:7862/reset

# Get configuration
curl http://localhost:7862/config
```

#### Dùng Python

```python
import requests

# Process single image
with open("car.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:7862/predict/batch",
        files={"files": f},
        params={"format": "json"}
    )
    data = response.json()
    for det in data['results'][0]['detections']:
        print(f"Vehicle: {det['vehicle_type']}, Plate: {det.get('license_plate', 'N/A')}")

# Process video
import cv2

cap = cv2.VideoCapture("video.mp4")
frame_num = 0

# Reset tracker before processing
requests.post("http://localhost:7862/reset")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    
    # Send to API
    files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
    params = {"frame_number": frame_num}
    response = requests.post(
        "http://localhost:7862/predict/frame",
        files=files,
        params=params
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Frame {frame_num}: {len(data['detections'])} vehicles detected")
    
    frame_num += 1

cap.release()
```

#### Dùng Example Client

```bash
# Health check
uv run client_example.py --health

# Process video (all frames)
uv run client_example.py --video path/to/video.mp4

# Process limited frames
uv run client_example.py --video path/to/video.mp4 --max-frames 100

# Get configuration
uv run client_example.py --config
```

---

## Gộp biển số đa khung hình

Biển số độ phân giải thấp thường không đọc được trong một frame đơn lẻ. Tính
năng gộp (fusion) lấy một **chuỗi (burst) crop của cùng một biển số** (8–32
frame), gộp chúng thành một biển số đã phục dựng, rồi chạy dual-OCR (FAST +
PPOCRv6).

### Kiến trúc

Gộp biển chạy **in-process** trên API chính, một process trên một cổng —
mặc định không cần khởi động sidecar:

```
main API (7862) ──in-process──> mf-lpr2 / eott adapters
  detect/track plate    merge N crops → 1 restored plate (BGR PNG)
  dual-OCR restored plate    engines: mflpr2 (mf-lpr2), eott
```

Các engine được vendor dưới dạng git submodule tại `fusion_svc/external/` và
được cài **vào venv chính** (không phải venv riêng) qua
`scripts/install_fusion_inproc.sh` — với clone mới cần chạy
`git submodule update --init --recursive` trước. `utils/fusion_client.py`
gọi trực tiếp các adapter (không qua HTTP).

`fusion_svc/` cũng có thể chạy như một **ứng dụng standalone tùy chọn** (có
`.venv` riêng, cổng 8100) nếu bạn muốn cô lập gộp biển ở một process riêng để
tránh crash lan — xem [`api/fusion-svc-API.md`](api/fusion-svc-API.md). Trỏ
API chính về đó bằng `FUSION_URL` ở chế độ tách rời này.

### Chạy

```bash
# Fresh clone only: fetch the engine submodules
git submodule update --init --recursive

uv sync                              # main deps
scripts/install_fusion_inproc.sh     # vendors mf-lpr2 + eott into this venv
                                      # re-run after any `uv sync`

uv run main.py                       # single process, port 7862
```

### Các Endpoint

```bash
# Multiframe: send N crops of ONE plate
curl -X POST "http://localhost:7862/predict/plates/multiframe?engine=mflpr2&scale=2" \
  -F "files=@01.png" -F "files=@02.png" ...   # -> {engine, frames_used, fast, ppocr}

# Video: detect+track plates, fuse each track's burst automatically
curl -X POST "http://localhost:7862/predict/plates/video?engine=mflpr2" \
  -F "file=@clip.mp4"                          # -> [{track_id, n_frames, fast, ppocr}, ...]

# Fusion only, no OCR -> restored plate image
curl -X POST "http://localhost:7862/fuse?engine=mflpr2&scale=2" \
  -F "files=@01.png" -F "files=@02.png" -o fused.png
```

Tham số: `engine` (`mflpr2` | `eott`), `scale` (hệ số phóng to, mặc định 1),
`max_frames` (mặc định 32), `min_frames` (chỉ cho video, mặc định 8).

### Ghi chú benchmark (mẫu biển RLPR, 31 crop/biển)

- **Chất lượng bị giới hạn bởi độ rõ nét của crop đầu vào, không phải bởi
  engine gộp.** Một burst tương phản cao → PPOCRv6 gần như hoàn hảo; một
  burst ~30px không đọc được → gộp làm nét hơn nhưng OCR vẫn đọc sai.
- **PPOCRv6 ≫ FAST** trên các biển số này.
- **`mflpr2` vs `eott`:** `mflpr2` phục dựng sạch hơn; `eott` đôi khi nhỉnh
  hơn về OCR với các burst vốn đã rõ nét. Mặc định dùng `mflpr2`.
- **KHÔNG áp dụng super-resolution trước khi gộp.** Nạp frame đã qua SR làm
  clip vùng sáng và *làm giảm* chất lượng OCR (quan sát thấy đảo 8→0), và
  chậm hơn 8–40×. mf-lpr2/eott đã tự upscale sẵn — hãy gộp crop **thô
  (raw)**.

---

## Huấn luyện Model

### Huấn luyện YOLO Ultralytics

Huấn luyện một detector trực tiếp từ CLI:

```bash
# Using bash wrapper
./scripts/train_yolo.sh \
  --data data/Peru_License_Plate/data.yaml \
  --model yolov8n.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 320 \
  --device 0

# Or directly
uv run detectors/yolo/train_ultralytics.py \
  --data data/Peru_License_Plate/data.yaml \
  --model yolov8n.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 320 \
  --device 0
```

**Tham số phổ biến:**
- `--data`: Đường dẫn tới `data.yaml` (vd: `data/LP-11k/data.yaml`)
- `--model`: Config hoặc weights YOLO (vd: `yolov8n.yaml`, `yolov8s.pt`)
- `--epochs`, `--batch`, `--imgsz`: Hyperparameter huấn luyện
- `--device`: ID(s) thiết bị CUDA hoặc `cpu`
- `--project`, `--name`: Cấu hình thư mục output
- `--exist-ok`, `--workers`, `--resume`, `--patience`, `--seed`: Các tùy chọn
  khác

### Xuất sang ONNX

Chuyển model đã huấn luyện sang định dạng ONNX để triển khai:

```bash
uv run detectors/yolo/exporter.py \
  --weights weights/plate/plate_yolo11n_640_2025.pt \
  --dynamic
```

**Tham số:**
- `--weights`: Đường dẫn tới file `.pt` (tương đối hoặc tuyệt đối)
- `--dynamic`: Giữ trục batch/sequence động (tùy chọn)
- `--imgsz`, `--half`: Các tùy chọn export khác

Đặt các file `.onnx` đã export vào thư mục `weights/`.

---

## Model Zoo

Các model có hiệu năng tốt nhất hiện tại, huấn luyện trên nhiều dataset khác
nhau:

| Model | mAP50 | mAP50-95 | Precision | Recall | Dataset |
|-------|-------|----------|-----------|--------|---------|
| YOLO11m | 0.875 | 0.596 | 0.941 | 0.802 | vehicle_18sep2025 |
| YOLO11n | 0.847 | 0.493 | 0.844 | 0.765 | vehicle_18sep2025 |
| YOLO12m | 0.845 | 0.580 | 0.817 | 0.777 | vehicle_18sep2025 |
| YOLO12s | 0.818 | 0.490 | 0.800 | 0.763 | vehicle_18sep2025 |
| YOLOv9e | 0.770 | 0.535 | 0.837 | 0.718 | vehicle_30oct2025 |

Tạo báo cáo model zoo cập nhật:

```bash
uv run scripts/generate_model_zoo.py --runs-dir runs/detect --output MODEL_ZOO.md
```

---

## Trọng số Model

### Cấu trúc thư mục

```
weights/
├── vehicle/              # Vehicle detection models
│   ├── vehicle_yolov9s_640_30oct2025.pt (default)
│   ├── vehicle_yolo11m_640_18sep2025.pt
│   └── ...
├── plate/                # License plate detection models
│   ├── plate_yolo12n_640_2025.pt (default, needs PLATE_IMGSZ=1280)
│   ├── plate_yolov8n_320_2024.pt (faster, lower recall on small plates)
│   └── ...
├── pretrained/           # Pre-trained YOLO base models
│   ├── yolo11m.pt
│   ├── yolo12s.pt
│   └── ...
├── tracking/             # Tracking models
│   └── deepsort/
│       └── ckpt.t7
└── ocr/                  # OCR models (PaddleOCR)
    └── ppocrv4/
```

### Quy ước đặt tên Model

Định dạng: `{task}_{architecture}_{resolution}_{date}.{ext}`

- **task**: vehicle, plate
- **architecture**: yolo11n/s/m, yolo12n/s/m, yolov8n/s/m, yolov9s/c/e
  - n: nano (nhanh nhất)
  - s: small (cân bằng)
  - m: medium (chính xác)
  - c: compact
  - e: extra-large
- **resolution**: 320, 640 (kích thước ảnh đầu vào)
- **date**: Ngày huấn luyện/phát hành
- **ext**: pt (PyTorch), onnx (ONNX Runtime)

### Model khuyến nghị

**Dùng cho Production (mặc định hiện tại của API):**
- Vehicle: `vehicle_yolov9s_640_30oct2025.pt`
- Plate: `plate_yolo12n_640_2025.pt` ở `PLATE_IMGSZ=1280` (biển số nhỏ/ở xa
  trong khung hình rộng cần giá trị này — xem [Cấu hình](#cấu-hình)).
  `plate_yolov8n_320_2024.pt` khả dụng nếu bạn cần tốc độ thô thay vì recall
  cao với biển số nhỏ.

---

## Quản lý dữ liệu

### Cấu trúc thư mục

```
data/
├── assets/              # Demo images and visual assets
├── config/              # Configuration files (JSON, TXT)
│   ├── rtsp_url.json   # RTSP camera URLs
│   └── streams.txt     # Stream configuration
├── logs/                # Application logs and outputs
│   ├── frames/         # Extracted video frames
│   ├── objects/        # Detected objects
│   └── plates/         # License plate detections
├── recordings/          # Camera recording data
│   ├── cam_thai_nguyen/
│   └── cyhome_bike/
├── results/             # Processing results and reports
├── samples/             # Sample data for testing
│   ├── raw/            # Raw unprocessed samples
│   └── test_samples/   # Test dataset samples
└── scripts/             # Data processing utilities
    ├── crop.py
    ├── download_roboflow.py
    ├── record_rtsp_streams.py
    └── yt_downloader.py
```

### Hướng dẫn sử dụng

- Đặt bản ghi camera vào `recordings/<camera_name>/`
- Lưu file cấu hình trong `config/`
- Kết quả xử lý được lưu vào `results/`
- Log được tự động sinh ra trong `logs/`
- Dùng script trong `scripts/` cho các thao tác dữ liệu

---

## Tải Dataset

### Tích hợp Roboflow

Tự động tải dataset từ Roboflow:

1. **Cài đặt Roboflow**:
```bash
uv pip install roboflow
```

2. **Cấu hình `.env`**:
```bash
ROBOFLOW_API_KEY=your-api-key
ROBOFLOW_WORKSPACE=your-workspace
ROBOFLOW_PROJECT=your-project
ROBOFLOW_VERSION=1
ROBOFLOW_EXPORT_FORMAT=yolov8          # optional, defaults to yolov8
ROBOFLOW_POLL_INTERVAL=600             # optional, seconds between checks
```

3. **Chạy downloader**:
```bash
uv run data/scripts/download_roboflow.py
```

Script kiểm tra Roboflow mỗi 10 phút (có thể cấu hình), tải phiên bản mới, và
tự động tăng số phiên bản.

---

## Giao diện WebApp

### Khởi động WebApp

```bash
uv pip install -r webapp/requirements.txt
uv run uvicorn webapp.backend.main:app --host 0.0.0.0 --port 7863
```

Mở `http://localhost:7863` trong trình duyệt.

### Tính năng

- Trực quan hóa luồng video thời gian thực
- Hỗ trợ camera RTSP
- Streaming qua WebRTC
- Quản lý preset camera
- Giao diện chọn model
- Vẽ overlay (bounding box, nhãn, biển số)

**Lưu ý**: WebApp độc lập với dịch vụ API chính và dùng instance xử lý riêng.

---

## Triển khai Docker

### Bắt đầu nhanh với Docker Compose

Chạy cả hai dịch vụ (Analysis API + WebApp) chỉ với một lệnh:

```bash
# Production mode (detached)
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

**Các dịch vụ:**
- Analysis Service API: http://localhost:7862
  - Swagger UI: http://localhost:7862/docs
  - ReDoc: http://localhost:7862/redoc
- WebApp Interface: http://localhost:7863

### Chế độ phát triển

Để hot-reload trong quá trình phát triển:

```bash
# Start in development mode
docker compose -f docker-compose.dev.yml up

# Or detached
docker compose -f docker-compose.dev.yml up -d
```

### Build Docker thủ công

```bash
# Build image
docker build -t trafficcam .

# Run analysis service
docker run -d \
  -p 7862:7862 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/data:/app/data \
  --name traffic-analysis \
  trafficcam

# Run webapp
docker run -d \
  -p 7863:7863 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/data:/app/data \
  --name traffic-webapp \
  trafficcam \
  uvicorn webapp.backend.main:app --host 0.0.0.0 --port 7863
```

### Hỗ trợ GPU

Các file docker-compose đã tích hợp sẵn hỗ trợ GPU. Nếu không có GPU, xóa
phần `deploy` khỏi docker-compose.yml hoặc đặt `ALPR_DEVICE=cpu`.

**Yêu cầu:**
- Đã cài NVIDIA Docker runtime
- NVIDIA GPU hỗ trợ CUDA

---

## Phát triển

### Kiểm thử

```bash
# Test imports and API creation
uv run api_test.py

# Test with example client
uv run client_example.py --health
uv run client_example.py --config

# Test with sample data
uv run client_example.py --video data/recordings/test.mp4 --max-frames 100

# Quick ALPR test
uv run scripts/test_alpr.py --input_source path/to/image_or_video

# Full pytest suite
uv run pytest tests/

# Smoke test the live API (all 10 endpoints)
uv run main.py &
uv run scripts/smoke_api_all.py
```

### Thêm Model mới

1. Huấn luyện model theo hướng dẫn huấn luyện
2. Lưu theo đúng quy ước đặt tên
3. Đặt vào đúng thư mục con trong `weights/`
4. Cập nhật cấu hình nếu cần
5. Kiểm tra với API

---

## Trích dẫn

Nếu dự án này hữu ích cho nghiên cứu hoặc triển khai của bạn, vui lòng trích
dẫn:

```bibtex
@misc{trafficcam2025,
  title        = {TrafficCam: Traffic Surveillance and License Plate Recognition},
  author       = {Nguyen, Tung},
  year         = {2025},
  howpublished = {\url{https://github.com/tungedng2710/license-plate-recognition}}
}
```

---

## Giấy phép

Copyright (C) 2023-2026 TonAI Vision Lab

---

## Hỗ trợ

Nếu có câu hỏi hoặc gặp vấn đề:
- Xem [Tài liệu API tương tác](http://localhost:7862/docs) khi dịch vụ đang
  chạy
- Xem code ví dụ trong [client_example.py](client_example.py)
- Kiểm tra issue hiện có hoặc liên hệ đội phát triển

## Lời cảm ơn

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) cho các
  model phát hiện
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) cho nhận diện văn
  bản
- [DeepSORT](https://github.com/John1liu/YOLOV5-DeepSORT-Vehicle-Tracking-Master)
  cho triển khai tracking

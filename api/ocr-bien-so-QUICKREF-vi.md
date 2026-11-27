# Traffic AI API · Tra cứu nhanh

Base URL: `http://10.8.0.16:7862` (cần VPN c09) · Upload: `multipart/form-data` · Tham số query trên URL · Server CPU: 2-25 s mỗi ảnh

## Endpoint

| # | Method | Path | Đầu vào | Đầu ra |
|---|---|---|---|---|
| 1 | GET | `/health` | không | JSON `{status, device, models_loaded}` |
| 2 | GET | `/config` | không | JSON: weights, ngưỡng, engine đang dùng |
| 3 | POST | `/predict/plates/multiframe` | `files` × 1..N ảnh crop biển số; `engine`, `scale`, `max_frames` | JSON `{engine, frames_used, fast, ppocr}` |
| 4 | POST | `/predict/plates/batch` | `files` × 1..N ảnh toàn cảnh | 1 file: `image/jpeg`; N file: `application/zip` (vẽ box + kết quả OCR) |
| 5 | POST | `/predict/batch` | `files` × 1..N ảnh/video/zip; `format`, `frame_stride` | `format=json`: JSON `{results}`; `media`: jpeg/mp4/zip |
| 6 | POST | `/predict/plates/video` | `file` × 1 video; `engine`, `scale`, `min_frames`, `max_frames` | JSON `[{track_id, n_frames, engine, fast, ppocr}]` |
| 7 | POST | `/predict/frame` | `file` × 1 frame; `frame_number` | JSON `{detections, frame_count}` (giữ trạng thái tracking) |
| 8 | POST | `/reset` | không | JSON `{status, message}` (xóa trạng thái tracking) |
| 9 | POST | `/fuse` | `files` × N crop cùng kích thước; `engine`, `scale` | `image/png` biển số đã phục hồi, không OCR |

## Tham số

| Tham số | Dùng ở | Mặc định | Giá trị |
|---|---|---|---|
| `engine` | 3, 6, 9 | `mflpr2` | `mflpr2`, `eott` |
| `scale` | 3, 6, 9 | `1` | hệ số phóng to, chỉ `mflpr2`. Dùng `1` |
| `max_frames` | 3, 6 | `32` | số crop tối đa được gộp |
| `min_frames` | 6 | `8` | độ dài track tối thiểu để đọc |
| `format` | 5 | `media` | `media`, `json` |
| `frame_stride` | 5 | `1` | xử lý mỗi frame thứ N của video |
| `frame_number` | 7 | không | trả lại trong `frame_count` |

## Phản hồi

**3 · `/predict/plates/multiframe`**
```json
{"engine": "mflpr2", "frames_used": 1,
 "fast":  {"text": "30A12345",   "confidence": 0.995},
 "ppocr": {"text": "30A-123.45", "confidence": 1.0}}
```
- `fast` = fast-plate-ocr: chỉ ký tự biển số.
- `ppocr` = PP-OCRv6: mọi chữ trong ảnh (có thể lẫn nhãn như `330mm`).

**5 · `/predict/batch?format=json`**
```json
{"results": [{"source": "scene.jpg", "kind": "image",
  "detections": [{"track_id": 1, "vehicle_type": "car", "license_plate": "51A12345",
    "confidence": 0.94, "bbox": {"x1": 10, "y1": 20, "x2": 200, "y2": 180},
    "plate_bbox": {"x1": 60, "y1": 140, "x2": 120, "y2": 165}}]}]}
```
Với video, trả `tracks` thay cho `detections`.

**6 · `/predict/plates/video`**
```json
[{"track_id": 3, "n_frames": 12, "engine": "mflpr2",
  "fast": {"text": "51A12345", "confidence": 0.91},
  "ppocr": {"text": "51A12345", "confidence": 0.97}}]
```

## Luồng sử dụng

| Mục đích | Gọi |
|---|---|
| Đọc 1 ảnh biển số (crop hoặc chụp cận) | `POST /predict/plates/multiframe` với 1 file |
| Đọc 1 biển số từ nhiều crop | `POST /predict/plates/multiframe` với N file |
| Tìm + đọc biển trong ảnh đường phố (xem trực quan) | `POST /predict/plates/batch` → jpeg có chú thích |
| Xe + biển số dạng JSON | `POST /predict/batch?format=json` |
| Biển số trong video | `POST /predict/plates/video` |
| Luồng trực tiếp, giữ `track_id` ổn định | `POST /reset`, rồi `POST /predict/frame?frame_number=N` cho mỗi frame |
| Chỉ lấy ảnh biển đã phục hồi | `POST /fuse` |

## Ví dụ

```bash
B=http://10.8.0.16:7862
curl $B/health
curl -F files=@plate.jpg "$B/predict/plates/multiframe?engine=mflpr2&scale=1"
curl -F files=@a.jpg -F files=@b.jpg "$B/predict/plates/multiframe"
curl -F files=@scene.jpg $B/predict/plates/batch -o plates.jpg
curl -F files=@scene.jpg "$B/predict/batch?format=json"
curl -F file=@clip.mp4 "$B/predict/plates/video"
```

## Lỗi

| Mã | Khi nào |
|---|---|
| 400 | file không đọc được, thiếu file, `engine` / `format` sai |
| 500 | lỗi xử lý |
| 503 | engine gộp biển không khả dụng |

Body: `{"detail": "<lý do>"}`

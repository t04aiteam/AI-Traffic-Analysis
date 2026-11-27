# Traffic AI API · Tra cứu nhanh

Base URL: `http://10.8.0.16:7862` (cần VPN c09) · Upload: `multipart/form-data` · Tham số query trên URL · Server CPU: 2-25 s mỗi ảnh

## Gửi gì vào đâu

| Bạn có | Gửi tới | Trường | Nhận về |
|---|---|---|---|
| Ảnh crop biển số / ảnh chụp cận biển số | `/predict/plates/multiframe` | `files` × 1 | chữ biển số (JSON) |
| Nhiều crop của **cùng một** biển số | `/predict/plates/multiframe` | `files` × N | một kết quả biển số đã gộp (JSON) |
| Ảnh toàn cảnh đường phố / giao thông | `/predict/batch?format=json` | `files` × 1..N | xe + chữ biển số (JSON); chỉ đọc biển số trên xe phát hiện được |
| Ảnh toàn cảnh, cần xem trực quan | `/predict/plates/batch` | `files` × 1..N | ảnh có khung biển số + kết quả đọc (jpeg / zip) |
| Video | `/predict/plates/video` | `file` × 1 | chữ biển số cho từng biển được theo dõi (JSON) |
| Video, cần cả xe + biển số | `/predict/batch?format=json` | `files` × 1 | kết quả theo từng track (JSON) |
| File zip chứa ảnh / video | `/predict/batch` | `files` × 1..N | JSON hoặc media có chú thích |
| Frame camera trực tiếp, từng frame | `/predict/frame` (sau `/reset`) | `file` × 1 | xe với `track_id` ổn định (JSON) |
| Nhiều crop cùng kích thước, chỉ cần ảnh phục hồi | `/fuse` | `files` × N | ảnh biển số đã phục hồi (png), không có chữ |

**Không hỗ trợ:** zip ở mọi endpoint trừ `/predict/batch`; video ở mọi endpoint trừ `/predict/batch` và `/predict/plates/video`; ảnh toàn cảnh gửi vào `/predict/plates/multiframe` (endpoint này đọc cả ảnh như một biển số nên lẫn nhãn và chữ khác).

## Định dạng hỗ trợ

| Loại | Định dạng | Endpoint |
|---|---|---|
| Ảnh | jpg, jpeg, png, bmp, webp, tif, tiff | tất cả trừ `/predict/plates/video` |
| Video | mp4, avi, mov, mkv, webm, m4v | `/predict/batch`, `/predict/plates/video` |
| Zip | zip chứa ảnh và/hoặc video (zip lồng tối đa 2 cấp; tối đa 500 file, 2 GB sau giải nén mỗi request) | chỉ `/predict/batch` |

Có thể trộn ảnh, video và zip trong một request `/predict/batch`.

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

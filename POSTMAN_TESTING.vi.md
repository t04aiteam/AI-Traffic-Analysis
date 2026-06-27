# Kiểm thử API Traffic AI bằng Postman

Hướng dẫn từng bước cho QA/tester. Không cần lập trình. API là service FastAPI
trong [`main.py`](main.py) (xem [`API.md`](API.md) để tham khảo cho dev). Có
**11 endpoint** cho phát hiện xe/biển số, theo dõi (tracking), OCR biển số và gộp biển.

> Mọi endpoint chạy trên **một cổng** (mặc định `7862`). Gộp đa khung
> (`/fuse`, `/predict/plates/multiframe`, `/predict/plates/video`) chạy
> in-process — không cần bật sidecar.

Bạn có thể **import collection dựng sẵn** (nhanh nhất) hoặc tự tạo request bằng
tay. Cả hai đều được hướng dẫn.

---

## 1. Import collection (khuyến nghị)

1. Trong Postman: **File → Import** (hoặc nút **Import** ở góc trên bên trái).
2. Chọn file **[`traffic-ai.postman_collection.json`](traffic-ai.postman_collection.json)**
   từ repo này.
3. Một collection **"Traffic AI Service API"** xuất hiện ở thanh bên trái với đầy
   đủ request đã dựng sẵn.
4. Đặt base URL một lần cho cả collection:
   - Bấm vào tên collection → tab **Variables**.
   - Đặt **`base_url`** ở cột Current Value thành server của bạn, ví dụ
     `http://127.0.0.1:7862` (không có dấu `/` ở cuối). **Save** (Ctrl/Cmd-S).
5. Với mỗi request có file, tự đính kèm lại file (Postman không lưu nội dung
   file): mở request → **Body → form-data** → ở mỗi dòng `file`/`files` bấm
   **Select Files** và chọn ảnh/video.

Sau đó chuyển tới **§3 Các kịch bản kiểm thử**.

---

## 2. Tự tạo request bằng tay (nếu không import)

Tạo một collection mới, thêm biến `base_url = http://127.0.0.1:7862`, rồi thêm các
request bên dưới. URL dùng `{{base_url}}`.

### 2a. Kiểm tra sức khoẻ — `GET {{base_url}}/health`

Không có tham số. Bấm Send. Kỳ vọng `200` và JSON dạng:

```json
{ "status": "healthy", "device": "cuda:0", "models_loaded": true }
```

**Đây là bài test đầu tiên — nếu nó lỗi thì mọi thứ khác sẽ không chạy.**

### 2b. Cấu hình — `GET {{base_url}}/config`

Không có tham số. Trả về weights đang dùng, ngưỡng tin cậy, engine OCR + SR, ngôn ngữ.

### 2c. Dự đoán batch — `POST {{base_url}}/predict/batch`

- Phương thức **POST**, URL `{{base_url}}/predict/batch`.
- Tab **Body** → **form-data** → Key = `files`, đổi kiểu _Text_ → **File**,
  **Select Files** → chọn một hoặc nhiều ảnh/video/zip (thêm dòng `files` cho
  nhiều input hơn).
- Tab **Params** → `format` = `json` để nhận JSON, hoặc bỏ trống để nhận media
  đã chú thích.
- Bấm Send. `format=json` → `{results:[...]}` kèm loại xe + biển theo từng source.

> Tên trường là **`file`** (số ít) chỉ cho `/predict/frame` và
> `/predict/plates/video`; còn lại dùng **`files`** (số nhiều), nhận 1..N file.

---

## 3. Các kịch bản kiểm thử

Chạy theo thứ tự. "scene" = ảnh đường có xe/biển số; "crop" = ảnh cắt sát biển số.

| #   | Endpoint | Body | Query | Kỳ vọng |
| --- | -------- | ---- | ----- | ------- |
| 1   | `GET /` | — | — | `200`, thông tin service + danh sách endpoint |
| 2   | `GET /health` | — | — | `200`, `status:"healthy"` |
| 3   | `GET /config` | — | — | `200`, weights/engine |
| 4   | `POST /reset` | — | — | `200`, `{"status":"success"}` |
| 5   | `POST /predict/batch` | `files`=1 scene | `format=json` | `200`, `{results:[{kind:"image",detections:[...]}]}` |
| 6   | `POST /predict/frame` | `file`=1 scene | `frame_number=7` | `200`, detections + `frame_count` (có trạng thái) |
| 7   | `POST /predict/batch` | `files`=1 scene | — | `200`, **JPEG đã chú thích** |
| 8   | `POST /predict/batch` | `files`=3 scene | — | `200`, **ZIP** chứa `*_pred.jpg` |
| 9   | `POST /predict/batch` | `files`=`.txt`/rác | — | **`400`**, `no decodable images/videos in upload` |
| 9a  | `POST /predict/batch` | `files`=2 ảnh | `format=json` | `200`, `{results:[{source,kind:"image",detections:[...]}]}` |
| 9b  | `POST /predict/batch` | `files`=1 video mp4 | `frame_stride=2` | `200`, **MP4 đã chú thích** |
| 9c  | `POST /predict/batch` | `files`=1 zip ảnh | — | `200`, **ZIP** các `*_pred.jpg` |
| 9d  | `POST /predict/batch` | `files`=1 ảnh + 1 video | — | `200`, **ZIP** gồm 1 `.jpg` + 1 `.mp4` |
| 10  | `POST /predict/plates/batch` | `files`=1 scene | — | `200`, **JPEG** có box biển + nhãn `FAST:`/`PPO:` |
| 11  | `POST /predict/plates/multiframe` | `files`=3 crop | `engine=mflpr2&scale=2` | `200`, JSON `{engine,frames_used,fast,ppocr}` |
| 12  | `POST /predict/plates/multiframe` | `files`=3 crop | `engine=eott` | `200`, JSON (gộp nhị phân + OCR) |
| 13  | `POST /predict/plates/multiframe` | `files`=3 crop | `engine=bogus` | **`400`**, `unknown engine: 'bogus'` |
| 14  | `POST /predict/plates/video` | `file`=1 mp4 | `engine=mflpr2` | `200`, **danh sách** theo track `{track_id,...,fast,ppocr}` |
| 15  | `POST /fuse` | `files`=3 crop | `engine=mflpr2&scale=2` | `200`, **PNG** biển số phục dựng (không OCR) |
| 16  | `POST /fuse` | `files`=3 crop | `engine=eott` | `200`, **PNG** biển số nhị phân |
| 17  | `POST /fuse` | `files`=3 crop | `engine=bogus` | **`400`**, `unknown engine: 'bogus'` |

### Phản hồi `/predict/batch?format=json` (kịch bản 5) — cần kiểm tra gì

```json
{
  "results": [
    { "source": "scene.jpg", "kind": "image",
      "detections": [
        { "track_id": 1,
          "bbox": { "x1": 220.0, "y1": 200.0, "x2": 420.0, "y2": 400.0 },
          "vehicle_type": "car", "license_plate": "51A12345",
          "plate_bbox": { "x1": 290.0, "y1": 360.0, "x2": 360.0, "y2": 390.0 },
          "confidence": 0.87 }
      ] }
  ]
}
```

Với source là video, `kind` là `"video"` kèm `n_frames`, `stride`, và danh sách
`tracks` thay cho `detections`. `license_plate`/`plate_bbox`/`confidence` có thể
là `null` khi không đọc được biển.

### `batch` vs `frame`

`/predict/batch` **không trạng thái** — reset tracker theo từng file.
`/predict/frame` **giữ** trạng thái tracking, nên `track_id` duy trì qua các lần
gọi khi bạn nạp chuỗi ảnh frame-by-frame — gọi `POST /reset` trước khi bắt đầu
một clip không liên quan.

---

## 4. Xử lý phản hồi

- **JSON** (`/`, `/health`, `/config`, `/reset`, `/predict/frame`,
  `/predict/batch?format=json`, `/predict/plates/multiframe`,
  `/predict/plates/video`): đọc trực tiếp ở khung response.
- **Ảnh** (`/predict/batch`, `/predict/plates/batch` với 1 file → JPEG; `/fuse` →
  PNG): tab **Preview** hiển thị. Lưu qua **⋯ / Save Response → Save to a file**
  (`.jpg` / `.png`).
- **ZIP** (`batch`/`plates/batch` với >1 file): Postman không xem trước được.
  **Save Response → Save to a file** đặt đuôi `.zip`; header `Content-Disposition`
  mang tên gợi ý (`annotated_*.zip` / `plates_*.zip`).

---

## 5. Tham số cho từng request

| Endpoint | Tham số | Kiểu | Ý nghĩa |
| -------- | ------- | ---- | ------- |
| `/predict/frame` | `frame_number` | int | số tham chiếu, trả lại trong `frame_count` |
| `/predict/batch` | `format` | `media`\|`json` | ảnh/mp4 đã chú thích (mặc định) hoặc JSON detections/tracks |
| `/predict/batch` | `frame_stride` | int | với input video: xử lý mỗi N frame (mặc định 1) |
| `multiframe` / `video` / `fuse` | `engine` | `mflpr2`\|`eott` | engine gộp (mặc định `mflpr2`); `eott` xuất nhị phân |
| `multiframe` / `video` / `fuse` | `scale` | int | hệ số phóng to — `mflpr2` áp dụng, **`eott` bỏ qua** |
| `multiframe` / `video` | `max_frames` | int | giới hạn số crop/frame gộp (mặc định 32) |
| `video` | `min_frames` | int | độ dài track tối thiểu để gộp (mặc định 8) |

---

## 6. Danh sách kiểm tra khi báo lỗi

Khi thấy bất thường, ghi lại cho dev:

- **Request**: endpoint, URL đầy đủ kèm query params, và file đã upload.
- **Response**: mã trạng thái + body JSON (hoặc ảnh/zip đã lưu).
- Output `GET /health` + `GET /config` (device, weights, engine OCR).
- Kỳ vọng vs thực tế (ví dụ "bỏ sót xe", "đọc sai biển", "danh sách rỗng").

Các kết quả **không phải lỗi** thường gặp:

- `400 Invalid image file` / `No valid images in batch` → file không phải ảnh hợp lệ.
- `400 unknown engine '...'` → `engine` phải là `mflpr2` hoặc `eott`.
- `500 <engine> failed: ...` ở `/fuse` → các crop khác kích thước; cắt về cùng
  kích thước (`/predict/plates/multiframe` tự resize, `/fuse` thì không).
- `/predict/plates/video` trả về **`[]`** → không track biển nào đạt `min_frames`;
  thử clip dài hơn hoặc giảm `min_frames`.
- Request đầu tiên **chậm** → model đang khởi động (YOLO + OCR nạp lười); thử lại.
- `track_id` reset giữa các frame → `/predict/batch` không trạng thái (reset
  theo từng file). Dùng `/predict/frame` cho chuỗi ảnh trực tiếp.

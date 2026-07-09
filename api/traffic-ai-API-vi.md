# Computer Vision API — Traffic AI Service (`traffic-ai`)

Backend FastAPI cho phát hiện phương tiện, phát hiện/tracking biển số, và OCR
biển số song công cụ, kèm phục hồi biển số đa khung hình chạy in-process. Một
process, một port — client gửi ảnh/video/chuỗi crop biển số qua HTTP và nhận
lại JSON detections, media đã annotate, hoặc ảnh biển số đã phục hồi.

## Nguyên tắc cốt lõi — GẦN NHƯ STATELESS, 1 ROUTE CÓ STATE

Mọi endpoint đều stateless (mỗi call độc lập) **trừ** `POST /predict/frame`,
route này giữ state tracker (DeepSORT/tracker) qua các lần gọi để client có
thể đẩy từng frame của một stream và nhận `track_id` nhất quán. Gọi
`POST /reset` giữa các clip/stream không liên quan khi dùng route này. Không
auth, không hàng đợi request, không lưu trữ — backend của caller tự chịu
trách nhiệm retry, lưu trữ, và điều phối nhiều request.

## Cách hoạt động (pipeline tổng quan)

**Bước 1 — Detect:** model YOLO phương tiện + YOLO biển số xác định vị trí xe
và biển số trong 1 frame. Chạy trong mọi route `/predict/*`.

**Bước 2 — Track:** DeepSORT (hoặc tracker nhẹ tích hợp) gán/giữ `track_id`
qua các frame — dùng trong `/predict/frame`, `/predict/batch` (input video),
và `/predict/plates/video`.

**Bước 3 — Đọc (dual OCR):** mỗi crop biển số được đọc bởi 2 engine độc lập —
`fast-plate-ocr` và PaddleOCR (PP-OCRv6) — trả về cả hai kết quả để caller tự
chọn/so sánh (`/predict/plates/batch`, `/predict/plates/multiframe`,
`/predict/plates/video`).

**Bước 4 — Phục hồi (tùy chọn, fusion in-process):** với biển số độ phân giải
thấp/mờ, một chuỗi crop của cùng 1 biển số có thể được fuse thành 1 ảnh chất
lượng cao hơn trước khi OCR. Hai engine: `mflpr2` (phục hồi màu, upscale qua
`scale`) và `eott` (nhị phân hóa, lưới cố định, bỏ qua `scale`). Gọi qua
`POST /fuse` (chỉ trả ảnh) hoặc để `/predict/plates/multiframe` và
`/predict/plates/video` tự fuse-rồi-OCR trong 1 lần gọi.

### Xử lý nội bộ

Thứ tự pipeline: YOLO detect xe → YOLO detect biển số → tracker (DeepSORT hoặc
loại nhẹ) → super-resolution tùy chọn trước OCR (`SR_ENGINE`) → dual OCR
(`fast-plate-ocr` + PP-OCRv6). Fusion (`mflpr2` / `eott`, vendor trong
[`fusion_svc/`](../fusion_svc/)) chạy **in-process** qua
[`utils/fusion_client.py`](../utils/fusion_client.py) — không có HTTP hop qua
sidecar. Xem [`fusion-svc-API.md`](fusion-svc-API.md) (bản gốc, tiếng Anh) cho
chi tiết nội bộ engine.

### Ràng buộc input

- Ảnh: `.jpg .jpeg .png .bmp .webp .tif .tiff .gif`. Video: `.mp4 .avi .mov
  .mkv .webm .m4v`. `.zip` được giải nén đệ quy (độ sâu ≤ 2).
- Giới hạn chống DoS trên `/predict/batch`: ≤ 500 member sau khi giải nén mỗi
  request, ≤ 300 MB mỗi entry trong zip (chưa nén), ≤ 2 GB tổng (chưa nén) mỗi
  request.
- Giải mã video giới hạn 9000 frame (~5 phút ở 30fps) mỗi file.
- `/fuse` và `/predict/plates/multiframe` yêu cầu crop của **một** biển số;
  `/predict/plates/multiframe` tự resize chuỗi crop kích thước không đều về
  cùng kích thước trước khi fuse — `/fuse` thì không, nên phải resize crop
  đồng đều trước khi gọi `/fuse`.

## Luồng end-to-end điển hình (mô tả)

Ảnh đơn lẻ → `POST /predict/batch?format=json` (loại xe + biển số mỗi
detection, không cần quản lý state). Stream trực tiếp → `POST /predict/frame`
mỗi frame, `track_id` nhất quán qua các lần gọi, `POST /reset` khi stream bắt
đầu lại. Cần mọi biển số được đọc bởi 2 OCR engine kèm box vẽ sẵn → `POST
/predict/plates/batch`. Biển số quá nhỏ/mờ để OCR tin cậy → thu thập 1 chuỗi
crop ngắn của biển số đó và gọi `POST /predict/plates/multiframe` (fuse +
dual-OCR trong 1 lần gọi), hoặc chạy `POST /predict/plates/video` để service
tự track + gom chuỗi + fuse + OCR toàn bộ clip. Chỉ cần ảnh đã phục hồi, không
cần OCR → `POST /fuse`.

## Thông tin kết nối

- **Base URL:** `http://<host>:7862` (mặc định `0.0.0.0:7862`; đổi qua
  `HOST`/`PORT`)
- **Content-Type:** `multipart/form-data` cho mọi request upload file.
  Response là JSON, trừ: `/predict/batch` (`format=media`, mặc định) trả về
  `image/jpeg`, `video/mp4`, hoặc `application/zip`; `/predict/plates/batch`
  trả về `image/jpeg` hoặc `application/zip`; `/fuse` trả về `image/png`.
- **Network:** service nội bộ, không auth — truy cập qua network nội bộ/VPN,
  không expose ra ngoài công khai ở dạng hiện tại.
- **Quy ước route:** `/{entity}/{action}` — vd `/predict/frame`,
  `/predict/plates/video`.
- **Docs tương tác:** Swagger UI tại `GET /docs` · OpenAPI JSON tại
  `GET /openapi.json`.

## Field chung (xuất hiện ở nhiều endpoint)

| Field | Type | Required/Default | Mô tả |
|---|---|---|---|
| `track_id` | int | — | id ổn định cho mỗi xe/biển số từ tracker; nhất quán qua các frame trong 1 phiên |
| `bbox` / `plate_bbox` | object `{x1,y1,x2,y2}` | — | box pixel, từ detector xe hoặc biển số |
| `vehicle_type` | str \| null | — | tên class loại xe, `null` nếu chưa phân loại |
| `license_plate` | str \| null | — | text OCR tốt nhất cho biển số của xe (từ `/predict/frame`, `/predict/batch`) |
| `confidence` | float \| null | — | độ tin cậy OCR cho `license_plate`, `null` nếu chưa đọc |
| `engine` | str | mặc định `mflpr2` | engine fusion — `mflpr2` (phục hồi màu) \| `eott` (nhị phân hóa) |
| `scale` | int | mặc định `1` | hệ số upscale khi fusion — áp dụng bởi `mflpr2`, **bỏ qua bởi `eott`** |

---

## Endpoints

### 1. Thông tin service

**GET `/`**

Trả về tên service, version, và danh sách endpoint ngắn gọn. Không tham số.

#### Response (JSON)

```json
{"service": "Traffic AI - License Plate Recognition", "version": "1.0.0",
 "endpoints": {"/health": "Health check", "...": "..."}}
```

### 2. Health check

**GET `/health`**

| Field | Type | Mô tả |
|---|---|---|
| status | str | `"healthy"` |
| device | str | thiết bị compute đã resolve (`cuda:0` / `cpu`) |
| models_loaded | bool | luôn `true` khi process đã chạy |

**Postman:** GET `{{base_url}}/health`

### 3. Lấy cấu hình đang chạy

**GET `/config`**

Trả về weights, confidence, OCR/SR engine, và cấu hình tracker mà service
đang chạy (xem [Config](#config-env-tất-cả-tùy-chọn)).

### 4. Detect xe + biển số (batch)

**POST `/predict/batch`**

Nhận bất kỳ tổ hợp ảnh, video, và zip nào (giải nén đệ quy) → trả về media đã
annotate hoặc JSON detections/tracks. Đây là entry point stateless, đa dụng.

#### Request (form-data)

| Field | Type | Required/Default | Mô tả |
|---|---|---|---|
| files | file[] | Bắt buộc, 1..N | ảnh/video/zip, bất kỳ tổ hợp |
| format | str | `media` | `media` → output đã annotate; `json` → chỉ detections/tracks |
| frame_stride | int | `1` | xử lý mỗi N frame video |

**Ví dụ — curl**

```bash
curl -F files=@scene.jpg 'http://localhost:7862/predict/batch?format=json'
```

**Postman:** POST `{{base_url}}/predict/batch` · Body → form-data: `files` (file, lặp lại) · Params: `format`, `frame_stride`

#### Response (JSON, `format=json`)

| Field | Type | Mô tả |
|---|---|---|
| results | array | 1 entry mỗi input; `detections` cho ảnh, `tracks` cho video |

```json
{"results": [{"source": "scene.jpg", "kind": "image",
  "detections": [{"track_id": 1, "vehicle_type": "car",
    "license_plate": "51A12345", "confidence": 0.94,
    "bbox": {"x1": 10, "y1": 20, "x2": 200, "y2": 180},
    "plate_bbox": {"x1": 60, "y1": 140, "x2": 120, "y2": 165}}]}]}
```

`format=media` (mặc định): 1 output → bytes `image/jpeg`/`video/mp4` thô;
nhiều output → `application/zip` gồm `<stem>_pred.jpg` / `<stem>_pred.mp4`.

**Lỗi riêng của endpoint này:** không file nào giải mã được → `400
{"detail":"no decodable images/videos in upload"}`; `format` không phải
`media`/`json` → `400`; annotate không ra output nào → `500`.

### 5. Detect + dual-OCR biển số (batch, đã annotate)

**POST `/predict/plates/batch`**

Detect biển số trong 1 hoặc nhiều ảnh, đọc mỗi biển số bằng cả 2 OCR engine,
trả về ảnh đã annotate box + label text `FAST:`/`PPO:`.

#### Request (form-data)

| Field | Type | Required/Default | Mô tả |
|---|---|---|---|
| files | file[] | Bắt buộc, 1..N | ảnh |

**Ví dụ — curl**

```bash
curl -F files=@scene.jpg http://localhost:7862/predict/plates/batch -o plates.jpg
```

**Postman:** POST `{{base_url}}/predict/plates/batch` · Body → form-data: `files` (file, lặp lại)

#### Response

1 ảnh → `image/jpeg` (đã annotate). Nhiều ảnh → `application/zip` gồm các
jpeg đã annotate.

**Lỗi riêng của endpoint này:** mọi file không giải mã được → `400
{"detail":"No valid images in batch"}`.

### 6. Xử lý 1 frame (stateful tracking)

**POST `/predict/frame`**

Xử lý 1 ảnh như 1 frame của stream, **giữ state tracker** qua các lần gọi để
cùng 1 xe giữ nguyên `track_id` ở lần gọi tiếp theo. Gọi `POST /reset` trước
khi bắt đầu clip không liên quan.

#### Request (form-data + query)

| Field | Type | Required/Default | Mô tả |
|---|---|---|---|
| file | file | Bắt buộc | 1 ảnh, coi như frame tiếp theo |
| frame_number | int | tùy chọn | echo lại trong response, để caller tự quản lý |

**Ví dụ — curl**

```bash
curl -F file=@frame0007.jpg 'http://localhost:7862/predict/frame?frame_number=7'
```

**Postman:** POST `{{base_url}}/predict/frame` · Body → form-data: `file` (file) · Params: `frame_number`

#### Response (JSON)

| Field | Type | Mô tả |
|---|---|---|
| detections | array | các xe đang được track, cùng shape với [Field chung](#field-chung-xuất-hiện-ở-nhiều-endpoint) |
| frame_count | int \| null | echo `frame_number` nếu có truyền |

**Lỗi riêng của endpoint này:** ảnh không giải mã được → `400
{"detail":"Invalid frame"}`; lỗi khi inference → `500
{"detail":"Processing error: ..."}`.

### 7. Reset tracker

**POST `/reset`**

Xóa state tracker (dùng trước khi chuyển `/predict/frame` sang clip mới).
Không body.

```json
{"status": "success", "message": "Tracker reset successfully"}
```

### 8. Fuse chuỗi crop biển số (chỉ ảnh, không OCR)

**POST `/fuse`**

Gộp N crop cùng kích thước, có thứ tự, của 1 biển số thành 1 ảnh biển số đã
phục hồi. Không OCR — mirror route `/fuse` của fusion sidecar độc lập (xem
[`fusion-svc-API.md`](fusion-svc-API.md), bản gốc tiếng Anh), nhưng chạy
in-process ở đây.

#### Request (form-data + query)

| Field | Type | Required/Default | Mô tả |
|---|---|---|---|
| files | file[] | Bắt buộc, N crop | crop cùng kích thước, có thứ tự, của 1 biển số |
| engine | str | `mflpr2` | `mflpr2` \| `eott` |
| scale | int | `1` | hệ số upscale, chỉ áp dụng cho `mflpr2` |

**Ví dụ — curl**

```bash
curl -X POST 'http://localhost:7862/fuse?engine=mflpr2&scale=2' \
  -F files=@01.png -F files=@02.png -o fused.png
```

**Postman:** POST `{{base_url}}/fuse` · Body → form-data: `files` (file, lặp lại) · Params: `engine`, `scale`

#### Response

`image/png` — biển số đã phục hồi, mã hóa BGR.

**Lỗi riêng của endpoint này:** `engine` không phải `mflpr2`/`eott` → `400`;
không file nào giải mã được → `400 {"detail":"no decodable frames"}`; engine
lỗi (vd crop kích thước không khớp) → `500 {"detail":"<engine> failed:
..."}`.

### 9. Fuse + dual-OCR chuỗi crop của 1 biển số

**POST `/predict/plates/multiframe`**

Fusion giống `/fuse`, cộng thêm dual-OCR ảnh đã phục hồi. Tự resize chuỗi crop
kích thước không đều về cùng kích thước trước khi fuse (khác `/fuse`, crop
không cần khớp sẵn kích thước).

#### Request (form-data + query)

| Field | Type | Required/Default | Mô tả |
|---|---|---|---|
| files | file[] | Bắt buộc, N crop | crop của 1 biển số, kích thước bất kỳ |
| engine | str | `mflpr2` | `mflpr2` \| `eott` |
| scale | int | `1` | hệ số upscale, chỉ áp dụng cho `mflpr2` |
| max_frames | int | `32` | giới hạn số crop được fuse |

**Ví dụ — curl**

```bash
curl -X POST 'http://localhost:7862/predict/plates/multiframe?engine=mflpr2&scale=2' \
  -F files=@01.png -F files=@02.png -F files=@03.png
```

**Postman:** POST `{{base_url}}/predict/plates/multiframe` · Body → form-data: `files` (file, lặp lại) · Params: `engine`, `scale`, `max_frames`

#### Response (JSON)

| Field | Type | Mô tả |
|---|---|---|
| engine | str | engine đã dùng |
| frames_used | int | số crop thực tế đã fuse (sau khi áp `max_frames`) |
| fast | object `{text, confidence}` | kết quả fast-plate-ocr |
| ppocr | object `{text, confidence}` | kết quả PP-OCRv6 |

```json
{"engine": "mflpr2", "frames_used": 5,
 "fast": {"text": "51A12345", "confidence": 0.91},
 "ppocr": {"text": "51A12345", "confidence": 0.97}}
```

**Lỗi riêng của endpoint này:** `engine` không hợp lệ → `400`; không có crop
hợp lệ → `400 {"detail":"no valid plate crops"}`; fusion engine không sẵn sàng
→ `503`.

### 10. Detect + track + fuse + OCR biển số trong video

**POST `/predict/plates/video`**

Chạy detect+track toàn bộ clip, gom crop biển số theo từng track, rồi fuse +
dual-OCR chuỗi crop của mỗi track — đường tự động hoàn toàn cho video có 1
hoặc nhiều biển số.

#### Request (form-data + query)

| Field | Type | Required/Default | Mô tả |
|---|---|---|---|
| file | file | Bắt buộc | 1 video |
| engine | str | `mflpr2` | `mflpr2` \| `eott` |
| scale | int | `1` | hệ số upscale, chỉ áp dụng cho `mflpr2` |
| min_frames | int | `8` | độ dài track tối thiểu để thử fuse |
| max_frames | int | `32` | giới hạn số crop fuse mỗi track |

**Ví dụ — curl**

```bash
curl -X POST 'http://localhost:7862/predict/plates/video?engine=mflpr2' -F file=@clip.mp4
```

**Postman:** POST `{{base_url}}/predict/plates/video` · Body → form-data: `file` (file) · Params: `engine`, `scale`, `min_frames`, `max_frames`

#### Response (JSON)

Mảng, mỗi entry là 1 track biển số:

```json
[{"track_id": 3, "n_frames": 12, "engine": "mflpr2",
  "fast": {"text": "51A12345", "confidence": 0.91},
  "ppocr": {"text": "51A12345", "confidence": 0.97}}]
```

**Lỗi riêng của endpoint này:** `engine` không hợp lệ hoặc video không giải
mã được → `400`; fusion engine không sẵn sàng → `503`.

<!-- === kết thúc block endpoint lặp lại === -->

---

## Luồng tích hợp (cho backend implementer)

```
A. Ảnh đơn lẻ, loại xe + biển số:
1) POST /predict/batch?format=json (files=[image]) -> results[0].detections
```

```
B. Stream trực tiếp, track_id ổn định mỗi xe qua các frame:
1) POST /reset                                      (stream mới bắt đầu)
2) POST /predict/frame?frame_number=N (file=frame_N) -> detections   # lặp mỗi frame
```

```
C. Biển số nhỏ/mờ cần phục hồi trước khi tin OCR:
# đã có chuỗi crop của MỘT biển số (vd từ tracker riêng của bạn):
   POST /predict/plates/multiframe?engine=mflpr2&scale=2 (files=[crop1..cropN])
   -> {fast, ppocr}
# có video thô, để service tự track+gom chuỗi+fuse+OCR:
   POST /predict/plates/video?engine=mflpr2 (file=clip.mp4) -> [{track_id, fast, ppocr}, ...]
# chỉ cần ảnh đã phục hồi, không cần OCR:
   POST /fuse?engine=mflpr2&scale=2 (files=[crop1..cropN]) -> image/png
```

## Mã lỗi chung

| HTTP | Ý nghĩa | Khi nào |
|---|---|---|
| 200 | OK | thành công — JSON, ảnh, video, hoặc zip tùy endpoint |
| 400 | Bad Request | file không giải mã được, `format`/`engine` không hợp lệ, batch rỗng |
| 500 | Server Error | lỗi inference, lỗi encode, lỗi nội bộ engine |
| 503 | Unavailable | fusion engine (`mflpr2`/`eott`) không có trong venv này |

---

## Cài đặt & Chạy

```bash
uv sync                              # dependency chính (torch / paddle / YOLO / OCR)
scripts/install_fusion_inproc.sh     # mf-lpr2 + eott vendor vào venv này (cần cho /fuse, /predict/plates/multiframe, /predict/plates/video)
uv run main.py                       # bind 0.0.0.0:7862 (đổi qua env HOST/PORT)
```

`install_fusion_inproc.sh` init submodule engine và cài `--no-deps` (tránh
xung đột `opencv-contrib-python` với `opencv-python-headless`). **Chạy lại
sau mỗi lần `uv sync`** — `uv sync` prune các package không có trong
`pyproject.toml`, bao gồm 2 engine này.

Chạy nền / sống sót khi logout:

```bash
nohup uv run main.py > main.log 2>&1 &
```

Dừng:

```bash
kill $(ss -ltnp | grep ':7862' | grep -oP 'pid=\K[0-9]+')
```

## Config (env, tất cả tùy chọn)

| var | mặc định | ý nghĩa |
|---|---|---|
| `ALPR_DEVICE` | `auto` | `cuda:0` \| `cpu` |
| `VEHICLE_WEIGHT` | `weights/vehicle/vehicle_yolov9s_640_30oct2025.pt` | YOLO xe |
| `PLATE_WEIGHT` | `weights/plate/plate_yolo12n_640_2025.pt` | YOLO biển số |
| `PLATE_IMGSZ` | `1280` | kích thước inference detect biển số (đường full-frame) — biển số nhỏ/xa trong frame rộng cần giá trị này cao hơn 640 gốc của weight |
| `DSORT_WEIGHT` | `weights/tracking/deepsort/ckpt.t7` | re-id DeepSORT |
| `VEHICLE_CONF` / `PLATE_CONF` | `0.6` / `0.25` | confidence detector |
| `OCR_THRESHOLD` | `0.9` | ngưỡng chấp nhận OCR |
| `OCR_ENGINE` | `paddle` | engine OCR biển số |
| `FPO_MODEL` | `cct-s-v2-global-model` | model fast-plate-ocr |
| `SR_ENGINE` / `SR_SCALE` | `none` / `2` | super-resolution tùy chọn trước OCR |
| `USE_DEEPSORT` | `false` | dùng DeepSORT (không thì tracker nhẹ hơn) |
| `LANG` | `en` | ngôn ngữ label |
| `HOST` / `PORT` | `0.0.0.0` / `7862` | địa chỉ bind |

Fusion chạy in-process — không cần cấu hình/khởi động gì thêm.

## Smoke Test

```bash
uv run main.py &                  # start API (7862) — fusion in-process
uv run scripts/smoke_api_all.py   # test cả 10 endpoint (kể cả /fuse + đường OCR)
```

Input tổng hợp; kiểm tra status code + shape response (không kiểm tra độ
chính xác OCR). Đổi target qua `MAIN_URL`.

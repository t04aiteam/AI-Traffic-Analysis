# Computer Vision API — Traffic AI Service (`traffic-ai`)

FastAPI backend for vehicle detection, license-plate detection/tracking, and
dual-engine plate OCR, with in-process multi-frame plate restoration. One
process, one port — a caller sends images/videos/plate-crop bursts over HTTP
and gets back JSON detections, annotated media, or a restored plate image.

## Core Principle — MOSTLY STATELESS, ONE STATEFUL ROUTE

Every endpoint is stateless (each call is self-contained) **except**
`POST /predict/frame`, which keeps DeepSORT/tracker state across calls so a
caller can feed a live stream frame-by-frame and get consistent `track_id`s.
Call `POST /reset` between unrelated clips/streams when using that route. No
auth, no request queue, no persistence — the caller's backend owns retries,
storage, and any multi-request orchestration.

## How It Works (pipeline overview)

**Step 1 — Detect:** YOLO vehicle model + YOLO plate model locate vehicles and
license plates in a frame. Runs inside every `/predict/*` route.

**Step 2 — Track:** DeepSORT (or a lighter built-in tracker) assigns/keeps
`track_id`s across frames — used by `/predict/frame`, `/predict/batch` (video
inputs), and `/predict/plates/video`.

**Step 3 — Read (dual OCR):** each plate crop is read by two independent
engines — `fast-plate-ocr` and PaddleOCR (PP-OCRv6) — and both results are
returned so the caller can pick/compare (`/predict/plates/batch`,
`/predict/plates/multiframe`, `/predict/plates/video`).

**Step 4 — Restore (optional, in-process fusion):** for low-res/blurry plates,
a burst of crops of the *same* plate can be fused into one higher-quality
image before OCR. Two engines: `mflpr2` (color-restored, upscale via `scale`)
and `eott` (binarized, fixed grid, `scale` ignored). Call via
`POST /fuse` (image only) or let `/predict/plates/multiframe` and
`/predict/plates/video` fuse-then-OCR in one call.

### Internal processing

Pipeline order: YOLO vehicle detect → YOLO plate detect → tracker (DeepSORT or
lighter) → optional super-resolution pre-OCR (`SR_ENGINE`) → dual OCR
(`fast-plate-ocr` + PP-OCRv6). Fusion (`mflpr2` / `eott`, vendored under
[`fusion_svc/`](../fusion_svc/)) runs **in-process** via
[`utils/fusion_client.py`](../utils/fusion_client.py) — no sidecar HTTP hop.
See [`fusion-svc-API.md`](fusion-svc-API.md) for engine internals.

### Input constraints

- Images: `.jpg .jpeg .png .bmp .webp .tif .tiff .gif`. Videos: `.mp4 .avi
  .mov .mkv .webm .m4v`. `.zip` is expanded recursively (depth ≤ 2).
- Anti-DoS caps on `/predict/batch`: ≤ 500 expanded members/request, ≤ 300 MB
  per zip entry (uncompressed), ≤ 2 GB total (uncompressed) per request.
- Video decoding caps at 9000 frames (~5 min at 30 fps) per file.
- `/fuse` and `/predict/plates/multiframe` expect crops of **one** plate;
  `/predict/plates/multiframe` auto-resizes an unequal-size burst to a common
  size before fusing — `/fuse` does not, so pre-size crops equally there.

## Typical End-to-End Flow (prose)

One-off image → `POST /predict/batch?format=json` (vehicle type + plate per
detection, no state to manage). Live stream → `POST /predict/frame` per frame,
consistent `track_id`s across calls, `POST /reset` when the stream restarts.
Need every plate read by two OCR engines with boxes drawn in → `POST
/predict/plates/batch`. Plate too small/blurry for reliable OCR → collect a
short burst of crops of that one plate and call `POST
/predict/plates/multiframe` (fuse + dual-OCR in one call), or run
`POST /predict/plates/video` to let the service track + burst + fuse + OCR an
entire clip automatically. Need the restored image only, no OCR → `POST /fuse`.

## Connection Info

- **Base URL:** `http://<host>:7862` (default `0.0.0.0:7862`; override via
  `HOST`/`PORT`)
- **Content-Type:** `multipart/form-data` for every request that uploads
  files. Responses are JSON except: `/predict/batch` (`format=media`, default)
  returns `image/jpeg`, `video/mp4`, or `application/zip`; `/predict/plates/batch`
  returns `image/jpeg` or `application/zip`; `/fuse` returns `image/png`.
- **Network:** internal service, no auth — reach it over the internal
  network/VPN, do not expose publicly as-is.
- **Route convention:** `/{entity}/{action}` — e.g. `/predict/frame`,
  `/predict/plates/video`.
- **Interactive docs:** Swagger UI at `GET /docs` · OpenAPI JSON at
  `GET /openapi.json`.

## Common Fields (appear across multiple endpoints)

| Field | Type | Required/Default | Description |
|---|---|---|---|
| `track_id` | int | — | stable per-vehicle/per-plate id from the tracker; consistent across frames within one session |
| `bbox` / `plate_bbox` | object `{x1,y1,x2,y2}` | — | pixel box, from vehicle or plate detector |
| `vehicle_type` | str \| null | — | vehicle class name, `null` if not classified |
| `license_plate` | str \| null | — | best OCR text for the vehicle's plate (from `/predict/frame`, `/predict/batch`) |
| `confidence` | float \| null | — | OCR confidence for `license_plate`, `null` if not read |
| `engine` | str | default `mflpr2` | fusion engine — `mflpr2` (color restore) \| `eott` (binarized) |
| `scale` | int | default `1` | fusion upscale factor — applied by `mflpr2`, **ignored by `eott`** |

---

## Endpoints

### 1. Service info

**GET `/`**

Returns service name, version, and a short endpoint list. No params.

#### Response (JSON)

```json
{"service": "Traffic AI - License Plate Recognition", "version": "1.0.0",
 "endpoints": {"/health": "Health check", "...": "..."}}
```

### 2. Health check

**GET `/health`**

| Field | Type | Description |
|---|---|---|
| status | str | `"healthy"` |
| device | str | resolved compute device (`cuda:0` / `cpu`) |
| models_loaded | bool | always `true` once the process is up |

**Postman:** GET `{{base_url}}/health`

### 3. Get active configuration

**GET `/config`**

Returns the resolved weights, confidences, OCR/SR engine, and tracker
settings the service is currently running with (see [Config](#config-env-all-optional)).

### 4. Detect vehicles + plates (batch)

**POST `/predict/batch`**

Accepts any mix of images, videos, and zips (extracted recursively) → either
annotated media or JSON detections/tracks. This is the stateless, general-
purpose entry point.

#### Request (form-data)

| Field | Type | Required/Default | Description |
|---|---|---|---|
| files | file[] | Required, 1..N | images/videos/zips, any mix |
| format | str | `media` | `media` → annotated output; `json` → detections/tracks only |
| frame_stride | int | `1` | process every Nth video frame |

**Example — curl**

```bash
curl -F files=@scene.jpg 'http://localhost:7862/predict/batch?format=json'
```

**Postman:** POST `{{base_url}}/predict/batch` · Body → form-data: `files` (file, repeatable) · Params: `format`, `frame_stride`

#### Response (JSON, `format=json`)

| Field | Type | Description |
|---|---|---|
| results | array | one entry per input; `detections` for images, `tracks` for videos |

```json
{"results": [{"source": "scene.jpg", "kind": "image",
  "detections": [{"track_id": 1, "vehicle_type": "car",
    "license_plate": "51A12345", "confidence": 0.94,
    "bbox": {"x1": 10, "y1": 20, "x2": 200, "y2": 180},
    "plate_bbox": {"x1": 60, "y1": 140, "x2": 120, "y2": 165}}]}]}
```

`format=media` (default): 1 output → raw `image/jpeg`/`video/mp4` bytes; >1
output → `application/zip` of `<stem>_pred.jpg` / `<stem>_pred.mp4`.

**Errors specific to this endpoint:** no decodable file in the upload → `400
{"detail":"no decodable images/videos in upload"}`; `format` not `media`/`json`
→ `400`; annotation produced nothing → `500`.

### 5. Detect + dual-OCR plates (batch, annotated)

**POST `/predict/plates/batch`**

Detects plates in one or more images, reads each with both OCR engines, and
returns the image annotated with boxes + `FAST:`/`PPO:` text labels.

#### Request (form-data)

| Field | Type | Required/Default | Description |
|---|---|---|---|
| files | file[] | Required, 1..N | images |

**Example — curl**

```bash
curl -F files=@scene.jpg http://localhost:7862/predict/plates/batch -o plates.jpg
```

**Postman:** POST `{{base_url}}/predict/plates/batch` · Body → form-data: `files` (file, repeatable)

#### Response

1 image → `image/jpeg` (annotated). >1 → `application/zip` of annotated jpegs.

**Errors specific to this endpoint:** every file undecodable → `400
{"detail":"No valid images in batch"}`.

### 6. Process one frame (stateful tracking)

**POST `/predict/frame`**

Processes a single image as one frame of a stream, **keeping tracker state**
across calls so the same vehicle keeps the same `track_id` on the next call.
Call `POST /reset` before starting an unrelated clip.

#### Request (form-data + query)

| Field | Type | Required/Default | Description |
|---|---|---|---|
| file | file | Required | one image, treated as the next frame |
| frame_number | int | optional | echoed back in the response, for the caller's own bookkeeping |

**Example — curl**

```bash
curl -F file=@frame0007.jpg 'http://localhost:7862/predict/frame?frame_number=7'
```

**Postman:** POST `{{base_url}}/predict/frame` · Body → form-data: `file` (file) · Params: `frame_number`

#### Response (JSON)

| Field | Type | Description |
|---|---|---|
| detections | array | current tracked vehicles, same shape as [Common Fields](#common-fields-appear-across-multiple-endpoints) |
| frame_count | int \| null | echoes `frame_number` if given |

**Errors specific to this endpoint:** undecodable image → `400
{"detail":"Invalid frame"}`; inference exception → `500
{"detail":"Processing error: ..."}`.

### 7. Reset tracker

**POST `/reset`**

Clears tracker state (used before switching `/predict/frame` to a new clip).
No body.

```json
{"status": "success", "message": "Tracker reset successfully"}
```

### 8. Fuse a plate-crop burst (image only, no OCR)

**POST `/fuse`**

Merges N ordered, **same-size** crops of one plate into a single restored
plate image. No OCR — mirrors the standalone fusion sidecar's `/fuse` (see
[`fusion-svc-API.md`](fusion-svc-API.md)), served in-process here.

#### Request (form-data + query)

| Field | Type | Required/Default | Description |
|---|---|---|---|
| files | file[] | Required, N crops | same-size, ordered crops of one plate |
| engine | str | `mflpr2` | `mflpr2` \| `eott` |
| scale | int | `1` | upscale factor, `mflpr2` only |

**Example — curl**

```bash
curl -X POST 'http://localhost:7862/fuse?engine=mflpr2&scale=2' \
  -F files=@01.png -F files=@02.png -o fused.png
```

**Postman:** POST `{{base_url}}/fuse` · Body → form-data: `files` (file, repeatable) · Params: `engine`, `scale`

#### Response

`image/png` — restored plate, BGR-encoded.

**Errors specific to this endpoint:** `engine` not `mflpr2`/`eott` → `400`; no
decodable file → `400 {"detail":"no decodable frames"}`; engine failure
(e.g. mismatched crop sizes) → `500 {"detail":"<engine> failed: ..."}`.

### 9. Fuse + dual-OCR one plate's crop burst

**POST `/predict/plates/multiframe`**

Same fusion as `/fuse`, plus dual-OCR of the restored plate. Auto-resizes an
unequal-size burst to a common size before fusing (so, unlike `/fuse`, crops
don't need to already match).

#### Request (form-data + query)

| Field | Type | Required/Default | Description |
|---|---|---|---|
| files | file[] | Required, N crops | crops of one plate, any size |
| engine | str | `mflpr2` | `mflpr2` \| `eott` |
| scale | int | `1` | upscale factor, `mflpr2` only |
| max_frames | int | `32` | cap on crops fused |

**Example — curl**

```bash
curl -X POST 'http://localhost:7862/predict/plates/multiframe?engine=mflpr2&scale=2' \
  -F files=@01.png -F files=@02.png -F files=@03.png
```

**Postman:** POST `{{base_url}}/predict/plates/multiframe` · Body → form-data: `files` (file, repeatable) · Params: `engine`, `scale`, `max_frames`

#### Response (JSON)

| Field | Type | Description |
|---|---|---|
| engine | str | engine used |
| frames_used | int | crops actually fused (after `max_frames` cap) |
| fast | object `{text, confidence}` | fast-plate-ocr result |
| ppocr | object `{text, confidence}` | PP-OCRv6 result |

```json
{"engine": "mflpr2", "frames_used": 5,
 "fast": {"text": "51A12345", "confidence": 0.91},
 "ppocr": {"text": "51A12345", "confidence": 0.97}}
```

**Errors specific to this endpoint:** `engine` invalid → `400`; no valid crops
→ `400 {"detail":"no valid plate crops"}`; fusion engine unavailable → `503`.

### 10. Detect + track + fuse + OCR plates in a video

**POST `/predict/plates/video`**

Runs detect+track over the whole clip, groups plate crops per track, then
fuses + dual-OCRs each track's burst — the fully automatic path for a video
with one or more plates.

#### Request (form-data + query)

| Field | Type | Required/Default | Description |
|---|---|---|---|
| file | file | Required | one video |
| engine | str | `mflpr2` | `mflpr2` \| `eott` |
| scale | int | `1` | upscale factor, `mflpr2` only |
| min_frames | int | `8` | min track length to attempt fusion |
| max_frames | int | `32` | cap on crops fused per track |

**Example — curl**

```bash
curl -X POST 'http://localhost:7862/predict/plates/video?engine=mflpr2' -F file=@clip.mp4
```

**Postman:** POST `{{base_url}}/predict/plates/video` · Body → form-data: `file` (file) · Params: `engine`, `scale`, `min_frames`, `max_frames`

#### Response (JSON)

Array, one entry per plate track:

```json
[{"track_id": 3, "n_frames": 12, "engine": "mflpr2",
  "fast": {"text": "51A12345", "confidence": 0.91},
  "ppocr": {"text": "51A12345", "confidence": 0.97}}]
```

**Errors specific to this endpoint:** `engine` invalid or video undecodable →
`400`; fusion engine unavailable → `503`.

<!-- === end repeatable endpoint block === -->

---

## Integration Flows (for backend implementers)

```
A. One-off image, vehicle type + plate:
1) POST /predict/batch?format=json (files=[image]) -> results[0].detections
```

```
B. Live stream, stable per-vehicle track_id across frames:
1) POST /reset                                      (new stream starts)
2) POST /predict/frame?frame_number=N (file=frame_N) -> detections   # repeat per frame
```

```
C. Small/blurry plate needs restoration before trusting OCR:
# have a burst of crops of ONE plate already (e.g. from your own tracker):
   POST /predict/plates/multiframe?engine=mflpr2&scale=2 (files=[crop1..cropN])
   -> {fast, ppocr}
# have a raw video instead, let the service track+burst+fuse+OCR:
   POST /predict/plates/video?engine=mflpr2 (file=clip.mp4) -> [{track_id, fast, ppocr}, ...]
# want the restored image only, no OCR:
   POST /fuse?engine=mflpr2&scale=2 (files=[crop1..cropN]) -> image/png
```

## Common Error Codes

| HTTP | Meaning | When |
|---|---|---|
| 200 | OK | success — JSON, image, video, or zip depending on endpoint |
| 400 | Bad Request | undecodable file(s), invalid `format`/`engine`, empty batch |
| 500 | Server Error | inference exception, encode failure, engine-internal error |
| 503 | Unavailable | fusion engine (`mflpr2`/`eott`) not available in this venv |

---

## Install & Run

```bash
uv sync                              # main deps (torch / paddle / YOLO / OCR)
scripts/install_fusion_inproc.sh     # vendored mf-lpr2 + eott into this venv (needed for /fuse, /predict/plates/multiframe, /predict/plates/video)
uv run main.py                       # binds 0.0.0.0:7862 (HOST/PORT env to override)
```

`install_fusion_inproc.sh` inits the engine submodules and installs them
`--no-deps` (avoids an `opencv-contrib-python` vs `opencv-python-headless`
clash). **Re-run it after any `uv sync`** — `uv sync` prunes packages not in
`pyproject.toml`, which includes the two engines.

Background / survives logout:

```bash
nohup uv run main.py > main.log 2>&1 &
```

Stop:

```bash
kill $(ss -ltnp | grep ':7862' | grep -oP 'pid=\K[0-9]+')
```

## Config (env, all optional)

| var | default | meaning |
|---|---|---|
| `ALPR_DEVICE` | `auto` | `cuda:0` \| `cpu` |
| `VEHICLE_WEIGHT` | `weights/vehicle/vehicle_yolov9s_640_30oct2025.pt` | vehicle YOLO |
| `PLATE_WEIGHT` | `weights/plate/plate_yolo12n_640_2025.pt` | plate YOLO |
| `PLATE_IMGSZ` | `1280` | plate detect inference size (full-frame path) — small/distant plates in wide frames need this above the weight's native 640 |
| `DSORT_WEIGHT` | `weights/tracking/deepsort/ckpt.t7` | DeepSORT re-id |
| `VEHICLE_CONF` / `PLATE_CONF` | `0.6` / `0.25` | detector confidences |
| `OCR_THRESHOLD` | `0.9` | OCR accept threshold |
| `OCR_ENGINE` | `paddle` | plate OCR engine |
| `FPO_MODEL` | `cct-s-v2-global-model` | fast-plate-ocr model |
| `SR_ENGINE` / `SR_SCALE` | `none` / `2` | optional super-resolution pre-OCR |
| `USE_DEEPSORT` | `false` | use DeepSORT (else lighter tracker) |
| `LANG` | `en` | label language |
| `HOST` / `PORT` | `0.0.0.0` / `7862` | bind address |

Fusion runs in-process — nothing else to configure or start.

## Smoke Test

```bash
uv run main.py &                  # start API (7862) — fusion is in-process
uv run scripts/smoke_api_all.py   # exercises all 10 endpoints (incl. /fuse + OCR paths)
```

Synthetic inputs; checks status codes + response shape (not OCR accuracy).
Override target with `MAIN_URL`.

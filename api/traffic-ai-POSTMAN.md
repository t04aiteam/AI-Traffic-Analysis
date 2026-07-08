# Testing the Traffic AI API with Postman

A step-by-step guide for QA/testers. No coding needed. The API is the FastAPI
service in [`main.py`](../main.py) (see [`traffic-ai-API.md`](traffic-ai-API.md)
for the developer reference). It has **10 endpoints** for vehicle/plate
detection, tracking, license-plate OCR, and plate fusion.

> All endpoints run on **one port** (default `7862`). Multi-frame fusion
> (`/fuse`, `/predict/plates/multiframe`, `/predict/plates/video`) runs
> in-process — no sidecar to start.

## 0. Start the API

```bash
uv sync                              # main deps
scripts/install_fusion_inproc.sh     # vendored mf-lpr2 + eott (needed for the fusion endpoints)
uv run main.py                       # binds 0.0.0.0:7862
```

Confirm it's up: `GET http://127.0.0.1:7862/health` → `200`.

You can either **import the ready-made collection** (fastest) or build the
requests by hand. Both are covered.

---

## 1. Import the collection (recommended)

1. In Postman: **File → Import** (or the **Import** button, top-left).
2. Select **[`traffic-ai.postman_collection.json`](traffic-ai.postman_collection.json)**
   from this repo.
3. A collection **"Traffic AI Service API"** appears in the left sidebar with all
   requests pre-built.
4. Set the base URL once for the whole collection:
   - Click the collection name → **Variables** tab.
   - Set **`base_url`** Current Value to your server, e.g. `http://127.0.0.1:7862`
     (no trailing slash). **Save** (Ctrl/Cmd-S).
5. For any request with a file, re-attach your own file (Postman does not store
   file contents): open the request → **Body → form-data** → on each `file`/`files`
   row click **Select Files** and pick your image/video.

Then jump to **§3 Test scenarios**.

---

## 2. Build requests by hand (if not importing)

Create a new collection, add a variable `base_url = http://127.0.0.1:7862`, then
add the requests below. URLs use `{{base_url}}`.

### 2a. Health check — `GET {{base_url}}/health`

No params. Send. Expect `200` and JSON like:

```json
{ "status": "healthy", "device": "cuda:0", "models_loaded": true }
```

**This is your first test — if it fails, nothing else will work.**

### 2b. Config — `GET {{base_url}}/config`

No params. Returns the active weights, confidences, OCR + SR engine, language.

### 2c. Predict batch — `POST {{base_url}}/predict/batch`

- Method **POST**, URL `{{base_url}}/predict/batch`.
- **Body** tab → **form-data** → Key = `files`, change type _Text_ → **File**,
  **Select Files** → pick one or more images/videos/zips (add more `files` rows
  for more inputs).
- **Params** tab → `format` = `json` for JSON, or omit for annotated media.
- Send. `format=json` → `{results:[...]}` with vehicle type + plate per source.

> Field name is **`file`** (singular) only for `/predict/frame` and
> `/predict/plates/video`; everything else uses **`files`** (plural) and takes
> 1..N files.

---

## 3. Test scenarios

Run in order. "scene" = a road image with a vehicle/plate; "crop" = a tight
plate crop.

| #   | Endpoint | Body | Query | Expected |
| --- | -------- | ---- | ----- | -------- |
| 1   | `GET /` | — | — | `200`, service info + endpoint list |
| 2   | `GET /health` | — | — | `200`, `status:"healthy"` |
| 3   | `GET /config` | — | — | `200`, weights/engines |
| 4   | `POST /reset` | — | — | `200`, `{"status":"success"}` |
| 5   | `POST /predict/batch` | `files`=1 scene | `format=json` | `200`, `{results:[{kind:"image",detections:[...]}]}` |
| 6   | `POST /predict/frame` | `file`=1 scene | `frame_number=7` | `200`, detections + `frame_count` (stateful) |
| 7   | `POST /predict/batch` | `files`=1 scene | — | `200`, **annotated JPEG** |
| 8   | `POST /predict/batch` | `files`=3 scenes | — | `200`, **ZIP** of `*_pred.jpg` |
| 9   | `POST /predict/batch` | `files`=`.txt`/garbage | — | **`400`**, `no decodable images/videos in upload` |
| 9a  | `POST /predict/batch` | `files`=2 scenes | `format=json` | `200`, `{results:[{source,kind:"image",detections:[...]}]}` |
| 9b  | `POST /predict/batch` | `files`=1 video (mp4) | `frame_stride=2` | `200`, **annotated MP4** |
| 9c  | `POST /predict/batch` | `files`=1 zip of images | — | `200`, **ZIP** of annotated `*_pred.jpg` |
| 9d  | `POST /predict/batch` | `files`= 1 image + 1 video | — | `200`, **ZIP** with a `.jpg` + an `.mp4` |
| 10  | `POST /predict/plates/batch` | `files`=1 scene | — | `200`, **JPEG** w/ plate boxes + `FAST:`/`PPO:` labels |
| 11  | `POST /predict/plates/multiframe` | `files`=3 crops | `engine=mflpr2&scale=2` | `200`, JSON `{engine,frames_used,fast,ppocr}` |
| 12  | `POST /predict/plates/multiframe` | `files`=3 crops | `engine=eott` | `200`, JSON (binarized fuse + OCR) |
| 13  | `POST /predict/plates/multiframe` | `files`=3 crops | `engine=bogus` | **`400`**, `unknown engine: 'bogus'` |
| 14  | `POST /predict/plates/video` | `file`=1 mp4 | `engine=mflpr2` | `200`, **list** of per-track `{track_id,...,fast,ppocr}` |
| 15  | `POST /fuse` | `files`=3 crops | `engine=mflpr2&scale=2` | `200`, **PNG** restored plate (no OCR) |
| 16  | `POST /fuse` | `files`=3 crops | `engine=eott` | `200`, **PNG** binarized plate |
| 17  | `POST /fuse` | `files`=3 crops | `engine=bogus` | **`400`**, `unknown engine: 'bogus'` |

### `/predict/batch?format=json` response (scenario 5) — what to check

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

For a video source, `kind` is `"video"` with `n_frames`, `stride`, and a
`tracks` list instead of `detections`. `license_plate`/`plate_bbox`/`confidence`
may be `null` when no plate is read.

### `batch` vs `frame`

`/predict/batch` is **stateless** — it resets the tracker per file. `/predict/frame`
**keeps** tracking state, so `track_id`s persist across calls when you feed a
sequence frame-by-frame — call `POST /reset` before starting an unrelated clip.

---

## 4. Handling responses

- **JSON** (`/`, `/health`, `/config`, `/reset`, `/predict/frame`,
  `/predict/batch?format=json`, `/predict/plates/multiframe`,
  `/predict/plates/video`): read directly in the response pane.
- **Image** (`/predict/batch`, `/predict/plates/batch` with 1 file → JPEG;
  `/fuse` → PNG): the **Preview** tab renders it. Save via **⋯ / Save Response →
  Save to a file** (`.jpg` / `.png`).
- **ZIP** (`batch`/`plates/batch` with >1 file): Postman can't preview. **Save
  Response → Save to a file** as `.zip`; the `Content-Disposition` header carries
  the suggested name (`annotated_*.zip` / `plates_*.zip`).

---

## 5. Per-request parameters

| Endpoint | Param | Type | Meaning |
| -------- | ----- | ---- | ------- |
| `/predict/frame` | `frame_number` | int | reference number echoed back as `frame_count` |
| `/predict/batch` | `format` | `media`\|`json` | annotated images/mp4 (default) or JSON detections/tracks |
| `/predict/batch` | `frame_stride` | int | for video inputs: process every Nth frame (default 1) |
| `multiframe` / `video` / `fuse` | `engine` | `mflpr2`\|`eott` | fusion engine (default `mflpr2`); `eott` outputs binarized |
| `multiframe` / `video` / `fuse` | `scale` | int | upscale factor — applied by `mflpr2`, **ignored by `eott`** |
| `multiframe` / `video` | `max_frames` | int | cap on crops/frames fused (default 32) |
| `video` | `min_frames` | int | min track length to attempt fusion (default 8) |

---

## 6. Bug-report checklist

When something looks wrong, capture for the dev:

- The **request**: endpoint, full URL incl. query params, and the file(s) uploaded.
- The **response**: status code + JSON body (or the saved image/zip).
- `GET /health` + `GET /config` output (device, weights, OCR engine).
- Expected vs actual (e.g. "missed a vehicle", "wrong plate text", "empty list").

Common, **not-a-bug** results:

- `400 no decodable images/videos in upload` → no file decoded as an image/video.
- `400 unknown engine '...'` → `engine` must be `mflpr2` or `eott`.
- `500 <engine> failed: ...` on `/fuse` → crops were different sizes; crop them
  to the same size (`/predict/plates/multiframe` auto-resizes, `/fuse` does not).
- `/predict/plates/video` returns **`[]`** → no plate track met `min_frames`; try
  a longer clip or lower `min_frames`.
- First request is **slow** → model warm-up (YOLO + OCR lazy-load); retry.
- `track_id`s reset between frames → `/predict/batch` is stateless (resets per
  file). Use `/predict/frame` for a live sequence.

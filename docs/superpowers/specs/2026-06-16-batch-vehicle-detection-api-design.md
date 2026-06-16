# Batch Vehicle Detection API — Design Spec
**Date:** 2026-06-16  
**Status:** Approved

## Problem

The existing `/predict/image` endpoint returns only JSON detections. Callers need annotated output images with bounding boxes and confidence scores drawn on them. When multiple images are submitted, a single response must carry all annotated results.

## Requirements

- Accept 1 or more images in a single request
- Run vehicle detection only (no plate OCR, no tracker state)
- Return each input image with bboxes and confidence labels drawn on it
- 1 image → return `image/jpeg` directly
- N > 1 images → return `application/zip` containing one JPEG per input
- Concurrent requests must not race (stateless per-request processing)
- Existing endpoints remain unchanged

## Architecture

### Two-file change

**`utils/traffic_analysis.py`** — add `detect_vehicles_only(frame: np.ndarray) -> np.ndarray`

- Calls `self.vehicle_detector` with `conf=self.opts.vconf`
- Iterates detections: draws rectangle + label `"{class} {conf:.2f}"` on a frame copy
- Returns annotated frame
- Reads shared model weights (immutable after init); writes nothing → concurrent-safe
- No tracker mutation, no plate detector, no OCR

**`main.py`** — add `POST /predict/batch`

- Parameter: `files: List[UploadFile]`
- Per file: decode bytes → `cv2.imdecode` → `detect_vehicles_only` → `cv2.imencode(".jpg", ...)`
- Invalid file (decode fails): skip silently, log warning — do not abort batch
- 1 file: `StreamingResponse(jpeg_bytes, media_type="image/jpeg")`
- N files: pack into `io.BytesIO` zip via `zipfile.ZipFile`; filenames = original upload names with `.jpg` extension; `StreamingResponse(zip_bytes, media_type="application/zip")`, header `Content-Disposition: attachment; filename="annotated_{timestamp}.zip"`

## Data Flow

```
Client
  └── POST /predict/batch  [files: img1.jpg, img2.jpg, ...]
        │
        ├── decode img1 → np.ndarray
        │     └── detect_vehicles_only(frame)
        │           └── YOLO forward pass
        │           └── draw bboxes + conf labels
        │           └── return annotated frame
        │     └── imencode → JPEG bytes
        │
        ├── decode img2 → ... (same)
        │
        └── if 1 file  → StreamingResponse(image/jpeg)
            if N files → zip all → StreamingResponse(application/zip)
```

## Error Handling

| Scenario | Behavior |
|---|---|
| File cannot be decoded as image | Skip file, log warning; batch continues |
| All files invalid | Return HTTP 400 |
| YOLO inference error | HTTP 500 with detail |
| Zero files uploaded | HTTP 422 (FastAPI validation) |

## Files Changed

| File | Change |
|---|---|
| `utils/traffic_analysis.py` | Add `detect_vehicles_only(frame)` method |
| `main.py` | Add `POST /predict/batch` endpoint |

## What Does Not Change

- `POST /predict/image` — JSON response, unchanged
- `POST /predict/frame` — stateful tracking, unchanged
- `POST /reset`, `GET /health`, `GET /config` — unchanged
- `TrafficAnalysisService.__init__` — unchanged
- No new dependencies (zipfile, io are stdlib; cv2 already present)

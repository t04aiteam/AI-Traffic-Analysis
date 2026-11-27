# Traffic AI API · Quick reference

Base URL: `http://10.8.0.16:7862` (c09 VPN required) · Uploads: `multipart/form-data` · Query params in the URL · CPU server: 2-25 s per image

## Endpoints

| # | Method | Path | Input | Output |
|---|---|---|---|---|
| 1 | GET | `/health` | none | JSON `{status, device, models_loaded}` |
| 2 | GET | `/config` | none | JSON: active weights, thresholds, engines |
| 3 | POST | `/predict/plates/multiframe` | `files` × 1..N plate crops; `engine`, `scale`, `max_frames` | JSON `{engine, frames_used, fast, ppocr}` |
| 4 | POST | `/predict/plates/batch` | `files` × 1..N scene images | 1 file: `image/jpeg`; N: `application/zip` (boxes + OCR drawn on) |
| 5 | POST | `/predict/batch` | `files` × 1..N images/videos/zips; `format`, `frame_stride` | `format=json`: JSON `{results}`; `media`: jpeg/mp4/zip |
| 6 | POST | `/predict/plates/video` | `file` × 1 video; `engine`, `scale`, `min_frames`, `max_frames` | JSON `[{track_id, n_frames, engine, fast, ppocr}]` |
| 7 | POST | `/predict/frame` | `file` × 1 frame; `frame_number` | JSON `{detections, frame_count}` (keeps tracking state) |
| 8 | POST | `/reset` | none | JSON `{status, message}` (clears tracking state) |
| 9 | POST | `/fuse` | `files` × N same-size crops; `engine`, `scale` | `image/png` restored plate, no OCR |

## Parameters

| Param | Used by | Default | Values |
|---|---|---|---|
| `engine` | 3, 6, 9 | `mflpr2` | `mflpr2`, `eott` |
| `scale` | 3, 6, 9 | `1` | upscale factor, `mflpr2` only. Use `1` |
| `max_frames` | 3, 6 | `32` | max crops fused |
| `min_frames` | 6 | `8` | min track length to read |
| `format` | 5 | `media` | `media`, `json` |
| `frame_stride` | 5 | `1` | process every Nth video frame |
| `frame_number` | 7 | none | echoed back as `frame_count` |

## Responses

**3 · `/predict/plates/multiframe`**
```json
{"engine": "mflpr2", "frames_used": 1,
 "fast":  {"text": "30A12345",   "confidence": 0.995},
 "ppocr": {"text": "30A-123.45", "confidence": 1.0}}
```
- `fast` = fast-plate-ocr: plate characters only.
- `ppocr` = PP-OCRv6: all text in the image (may include labels such as `330mm`).

**5 · `/predict/batch?format=json`**
```json
{"results": [{"source": "scene.jpg", "kind": "image",
  "detections": [{"track_id": 1, "vehicle_type": "car", "license_plate": "51A12345",
    "confidence": 0.94, "bbox": {"x1": 10, "y1": 20, "x2": 200, "y2": 180},
    "plate_bbox": {"x1": 60, "y1": 140, "x2": 120, "y2": 165}}]}]}
```
Videos return `tracks` instead of `detections`.

**6 · `/predict/plates/video`**
```json
[{"track_id": 3, "n_frames": 12, "engine": "mflpr2",
  "fast": {"text": "51A12345", "confidence": 0.91},
  "ppocr": {"text": "51A12345", "confidence": 0.97}}]
```

## Flows

| Goal | Calls |
|---|---|
| Read one plate photo (crop or close-up) | `POST /predict/plates/multiframe` with 1 file |
| Read one plate from several crops | `POST /predict/plates/multiframe` with N files |
| Find + read plates in a street scene (visual check) | `POST /predict/plates/batch` → annotated jpeg |
| Vehicles + plates as JSON | `POST /predict/batch?format=json` |
| Plates in a video | `POST /predict/plates/video` |
| Live stream, stable `track_id` | `POST /reset`, then `POST /predict/frame?frame_number=N` per frame |
| Restored plate image only | `POST /fuse` |

## Examples

```bash
B=http://10.8.0.16:7862
curl $B/health
curl -F files=@plate.jpg "$B/predict/plates/multiframe?engine=mflpr2&scale=1"
curl -F files=@a.jpg -F files=@b.jpg "$B/predict/plates/multiframe"
curl -F files=@scene.jpg $B/predict/plates/batch -o plates.jpg
curl -F files=@scene.jpg "$B/predict/batch?format=json"
curl -F file=@clip.mp4 "$B/predict/plates/video"
```

## Errors

| Code | When |
|---|---|
| 400 | file not decodable, no files, bad `engine` / `format` |
| 500 | processing error |
| 503 | fusion engine unavailable |

Body: `{"detail": "<reason>"}`

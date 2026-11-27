# Traffic AI API · Quick reference

Base URL: `http://10.8.0.16:7862` (c09 VPN required) · Uploads: `multipart/form-data` · Query params in the URL · CPU server: 2-25 s per image

## What to send where

| You have | Send to | Field | Get back |
|---|---|---|---|
| Cropped plate / close-up plate photo | `/predict/plates/multiframe` | `files` × 1 | plate text (JSON) |
| Several crops of the **same** plate | `/predict/plates/multiframe` | `files` × N | one fused plate text (JSON) |
| Street / traffic scene image(s) | `/predict/batch?format=json` | `files` × 1..N | vehicles + plate text (JSON); plates are read only on detected vehicles |
| Street / traffic scene image(s), visual check | `/predict/plates/batch` | `files` × 1..N | image with plates boxed + read (jpeg / zip) |
| Video | `/predict/plates/video` | `file` × 1 | plate text per tracked plate (JSON) |
| Video, vehicles + plates | `/predict/batch?format=json` | `files` × 1 | per-track results (JSON) |
| Zip of images / videos | `/predict/batch` | `files` × 1..N | JSON or annotated media |
| Live camera frames, one at a time | `/predict/frame` (after `/reset`) | `file` × 1 | vehicles with stable `track_id` (JSON) |
| Several same-size crops, restored image only | `/fuse` | `files` × N | restored plate (png), no text |

**Not supported:** zips anywhere except `/predict/batch`; videos anywhere except `/predict/batch` and `/predict/plates/video`; scene images to `/predict/plates/multiframe` (it reads the whole image as one plate, so labels and other text get mixed in).

## Accepted formats

| Kind | Formats | Endpoints |
|---|---|---|
| Image | jpg, jpeg, png, bmp, webp, tif, tiff | all except `/predict/plates/video` |
| Video | mp4, avi, mov, mkv, webm, m4v | `/predict/batch`, `/predict/plates/video` |
| Zip | zip of images and/or videos (nested zips up to 2 levels; max 500 files, 2 GB unzipped per request) | `/predict/batch` only |

Mixed images, videos and zips can go in one `/predict/batch` request.

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

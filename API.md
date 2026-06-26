# Traffic AI Service — API

FastAPI backend for vehicle detection, plate detection/tracking, and license-plate
OCR. Source: [`main.py`](main.py). Pipeline: YOLO vehicle/plate detect → DeepSORT
tracking → dual-OCR (fast-plate-ocr + PP-OCRv6). Multi-frame plate restoration is
delegated to the **fusion sidecar** ([`fusion_svc/`](fusion_svc/), port `8100`)
via [`utils/fusion_client.py`](utils/fusion_client.py).

## Run

From the repo root (`/mnt/data/nblong-t04/AI-Traffic-Analysis`):

```bash
uv run main.py                       # binds 0.0.0.0:7862 (HOST/PORT env to override)
```

Swagger UI (browser upload): `http://<host>:7862/docs`.

The two `/predict/plates/{multiframe,video}` endpoints additionally need the
fusion sidecar running:

```bash
uv run --directory fusion_svc uvicorn fusion_svc.app:app --port 8100
```

Background / survives logout:

```bash
nohup uv run main.py > main.log 2>&1 &
```

Stop:

```bash
kill $(ss -ltnp | grep ':7862' | grep -oP 'pid=\K[0-9]+')
```

## Endpoints (10)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/`        | service info + endpoint list |
| `GET`  | `/health`  | `{status, device, models_loaded}` |
| `GET`  | `/config`  | active weights / confidences / OCR + SR engine |
| `POST` | `/reset`   | reset the tracker (new video/stream) |
| `POST` | `/predict/image`  | 1 image → JSON vehicle detections (**resets tracker** each call) |
| `POST` | `/predict/frame`  | 1 frame → JSON detections (**keeps** tracking state across calls) |
| `POST` | `/predict/batch`  | N images → **annotated** vehicle output (jpeg / zip) |
| `POST` | `/predict/plates/batch` | N images → **annotated** plate output + dual-OCR labels (jpeg / zip) |
| `POST` | `/predict/plates/multiframe` | fuse a burst of one plate's crops → dual-OCR (JSON) |
| `POST` | `/predict/plates/video` | detect+track plates in a video, fuse each track's burst → dual-OCR (JSON) |

### Output by endpoint

| endpoint | body field | output |
|---|---|---|
| `/predict/image` | `file` | `{detections:[{track_id,bbox,vehicle_type,license_plate,plate_bbox,confidence}]}` |
| `/predict/frame` | `file` (+ `?frame_number=`) | same shape + `frame_count` |
| `/predict/batch` | `files` (1) | annotated `image/jpeg` |
| `/predict/batch` | `files` (>1) | `application/zip` of `<name>.jpg` |
| `/predict/plates/batch` | `files` | jpeg (1) / zip (>1), boxes + `FAST:`/`PPO:` text labels |
| `/predict/plates/multiframe` | `files` (N crops) | `{engine, frames_used, fast:{text,confidence}, ppocr:{text,confidence}}` |
| `/predict/plates/video` | `file` (1 video) | `[{track_id, n_frames, engine, fast, ppocr}, ...]` |

`image` vs `frame`: `image` resets the tracker first (stateless single shots);
`frame` preserves state so `track_id`s persist across a sequence — call `/reset`
between independent clips.

## Multi-frame fusion endpoints (engine = eott / mflpr2)

Both delegate fusion to the sidecar, then dual-OCR the restored plate.

| param | endpoint | default | meaning |
|---|---|---|---|
| `engine` | both | `mflpr2` | `mflpr2` (color restore) \| `eott` (binarized) |
| `scale` | both | `1` | upscale factor — applied by `mflpr2`, **ignored by `eott`** |
| `max_frames` | both | `32` | cap on crops/frames fused per plate |
| `min_frames` | video | `8` | min track length to attempt fusion |

- `/multiframe` takes N crops of **one** plate; it auto-resizes the burst to a
  common size (`resize_burst_to_common`) so unequal crop sizes are fine.
- `/video` runs detect+track over the whole clip, groups crops per track id, and
  fuses+OCRs each track's burst.

See [`fusion_svc/API.md`](fusion_svc/API.md) for the sidecar contract and engine
differences.

## Errors

| status | where | cause |
|---|---|---|
| `400` | image/frame | `Invalid image file` / `Invalid frame` (undecodable) |
| `400` | batch / plates/batch | `No valid images in batch` (all files undecodable) |
| `400` | multiframe | `unknown engine: '<x>'` or `no valid plate crops` |
| `400` | video | `unknown engine: '<x>'` or `could not decode video` |
| `500` | image/frame/multiframe | `Processing error: ...` (inference exception) |
| `503` | multiframe / video | `FusionUnavailable` — fusion sidecar (8100) not running |

## Examples

```bash
# single image -> JSON detections
curl -F file=@scene.jpg http://localhost:7862/predict/image

# video frame, keep tracking state
curl -F file=@frame0007.jpg 'http://localhost:7862/predict/frame?frame_number=7'

# batch annotate (many -> zip)
curl -F files=@a.jpg -F files=@b.jpg http://localhost:7862/predict/batch -o annotated.zip

# plate detect + dual OCR, annotated
curl -F files=@scene.jpg http://localhost:7862/predict/plates/batch -o plates.jpg

# multi-frame fusion + OCR (needs sidecar on 8100)
curl -X POST 'http://localhost:7862/predict/plates/multiframe?engine=mflpr2&scale=2' \
  -F files=@01.png -F files=@02.png -F files=@03.png

# video: track + fuse each plate + OCR
curl -X POST 'http://localhost:7862/predict/plates/video?engine=mflpr2' -F file=@clip.mp4

# info
curl http://localhost:7862/health
curl http://localhost:7862/config
curl -X POST http://localhost:7862/reset
```

## Config (env, all optional)

| var | default | meaning |
|---|---|---|
| `ALPR_DEVICE` | `auto` | `cuda:0` \| `cpu` |
| `VEHICLE_WEIGHT` | `weights/vehicle/vehicle_yolov9s_640_30oct2025.pt` | vehicle YOLO |
| `PLATE_WEIGHT` | `weights/plate/plate_yolov8n_320_2024.pt` | plate YOLO |
| `DSORT_WEIGHT` | `weights/tracking/deepsort/ckpt.t7` | DeepSORT re-id |
| `VEHICLE_CONF` / `PLATE_CONF` | `0.6` / `0.25` | detector confidences |
| `OCR_THRESHOLD` | `0.9` | OCR accept threshold |
| `OCR_ENGINE` | `paddle` | plate OCR engine |
| `FPO_MODEL` | `cct-s-v2-global-model` | fast-plate-ocr model |
| `SR_ENGINE` / `SR_SCALE` | `none` / `2` | optional super-resolution pre-OCR |
| `USE_DEEPSORT` | `false` | use DeepSORT (else lighter tracker) |
| `LANG` | `en` | label language |
| `FUSION_URL` | `http://127.0.0.1:8100` | fusion sidecar base URL (`utils/fusion_client.py`) |
| `HOST` / `PORT` | `0.0.0.0` / `7862` | bind address |

## Smoke test

```bash
uv run main.py &                                 # start main API (7862)
uv run --directory fusion_svc uvicorn fusion_svc.app:app --port 8100 &   # sidecar
uv run scripts/smoke_api_all.py                  # exercises all 10 main + 4 sidecar checks
```

Synthetic inputs; checks status codes + response shape (not OCR accuracy).
Override targets with `MAIN_URL` / `FUSION_URL`.

# Traffic AI Service — API

FastAPI backend for vehicle detection, plate detection/tracking, and license-plate
OCR. Source: [`main.py`](main.py). Pipeline: YOLO vehicle/plate detect → DeepSORT
tracking → dual-OCR (fast-plate-ocr + PP-OCRv6). Multi-frame plate restoration
(mf-lpr2 / eott) runs **in-process** on the same port via
[`utils/fusion_client.py`](utils/fusion_client.py) — engines vendored under
[`fusion_svc/`](fusion_svc/), installed into this venv. Everything is one process
on one port; no sidecar.

## Install

```bash
uv sync                              # main deps (torch / paddle / YOLO / OCR)
scripts/install_fusion_inproc.sh     # vendored mf-lpr2 + eott into this venv (in-process fusion)
```

`install_fusion_inproc.sh` inits the engine submodules and installs them
`--no-deps` (avoids an `opencv-contrib-python` vs `opencv-python-headless`
clash). **Re-run it after any `uv sync`** — `uv sync` prunes packages not in
pyproject, which includes the two engines.

## Run

From the repo root (`/mnt/data/nblong-t04/AI-Traffic-Analysis`):

```bash
uv run main.py                       # binds 0.0.0.0:7862 (HOST/PORT env to override)
```

Swagger UI (browser upload): `http://<host>:7862/docs`. All endpoints — including
fusion — are served on this one port; no separate process to start.

Background / survives logout:

```bash
nohup uv run main.py > main.log 2>&1 &
```

Stop:

```bash
kill $(ss -ltnp | grep ':7862' | grep -oP 'pid=\K[0-9]+')
```

## Endpoints (13)

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
| `POST` | `/predict/vehicles/image` | detect+track vehicles in 1 image → per-track **vehicle type** + plate (JSON) |
| `POST` | `/predict/vehicles/video` | detect+track vehicles in a video → per-track **vehicle type** + plate (JSON) |
| `POST` | `/fuse` | fuse a burst of crops → restored plate **image** (PNG), no OCR |

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
| `/predict/vehicles/image` | `file` (1 image) | `{tracks:[{track_id, vehicle_type, license_plate, confidence, bbox, plate_bbox}]}` |
| `/predict/vehicles/video` | `file` (1 video) (+ `?frame_stride=`) | `{n_frames, stride, tracks:[{track_id, frames_seen, vehicle_type, license_plate, confidence, bbox, plate_bbox}]}` |
| `/fuse` | `files` (N crops) | `image/png` restored plate (BGR), no OCR |

`image` vs `frame`: `image` resets the tracker first (stateless single shots);
`frame` preserves state so `track_id`s persist across a sequence — call `/reset`
between independent clips.

## Fusion endpoints (engine = eott / mflpr2)

`/fuse`, `/predict/plates/multiframe`, and `/predict/plates/video` all run the
fusion engines **in-process** (no HTTP, no sidecar). `/fuse` returns the restored
plate image; the two `/predict/plates/*` variants additionally dual-OCR it.

| param | endpoint | default | meaning |
|---|---|---|---|
| `engine` | all three | `mflpr2` | `mflpr2` (color restore) \| `eott` (binarized) |
| `scale` | all three | `1` | upscale factor — applied by `mflpr2`, **ignored by `eott`** |
| `max_frames` | multiframe / video | `32` | cap on crops/frames fused per plate |
| `min_frames` | video | `8` | min track length to attempt fusion |

- `/fuse` takes N crops → one restored plate PNG (image-only). Crops should be
  the same size and ordered.
- `/multiframe` takes N crops of **one** plate; it auto-resizes the burst to a
  common size (`resize_burst_to_common`) so unequal crop sizes are fine, then OCRs.
- `/video` runs detect+track over the whole clip, groups crops per track id, and
  fuses+OCRs each track's burst.

See [`fusion_svc/API.md`](fusion_svc/API.md) for engine differences (mflpr2 vs
eott output, scale behavior).

## Errors

| status | where | cause |
|---|---|---|
| `400` | image/frame | `Invalid image file` / `Invalid frame` (undecodable) |
| `400` | batch / plates/batch | `No valid images in batch` (all files undecodable) |
| `400` | multiframe | `unknown engine: '<x>'` or `no valid plate crops` |
| `400` | video | `unknown engine: '<x>'` or `could not decode video` |
| `400` | fuse | `unknown engine: '<x>'` or `no decodable frames` |
| `500` | image/frame/multiframe | `Processing error: ...` (inference exception) |
| `500` | fuse | `<engine> failed: ...` (engine error, e.g. mismatched crop sizes) |

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

# multi-frame fusion + OCR (in-process, single port)
curl -X POST 'http://localhost:7862/predict/plates/multiframe?engine=mflpr2&scale=2' \
  -F files=@01.png -F files=@02.png -F files=@03.png

# video: track + fuse each plate + OCR
curl -X POST 'http://localhost:7862/predict/plates/video?engine=mflpr2' -F file=@clip.mp4

# image: per-track vehicle type + plate (JSON)
curl -F file=@scene.jpg http://localhost:7862/predict/vehicles/image

# video: per-track vehicle type + plate (every 2nd frame)
curl -X POST 'http://localhost:7862/predict/vehicles/video?frame_stride=2' -F file=@clip.mp4

# fusion only -> restored plate image (no OCR)
curl -X POST 'http://localhost:7862/fuse?engine=mflpr2&scale=2' \
  -F files=@01.png -F files=@02.png -o fused.png

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
| `HOST` / `PORT` | `0.0.0.0` / `7862` | bind address |

Fusion runs in-process — nothing else to configure or start.

## Smoke test

```bash
uv run main.py &                  # start API (7862) — fusion is in-process
uv run scripts/smoke_api_all.py   # exercises all 11 endpoints (incl. /fuse + OCR paths)
```

Synthetic inputs; checks status codes + response shape (not OCR accuracy).
Override target with `MAIN_URL`.

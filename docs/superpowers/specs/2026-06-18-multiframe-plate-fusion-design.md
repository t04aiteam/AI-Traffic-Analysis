# Multi-Frame Plate Fusion Integration — Design

**Date:** 2026-06-18
**Status:** Approved (brainstorming complete)
**Repo:** AI-Traffic-Analysis (host) + mf-lpr2 + eott (fusion engines)

---

## Goal

Add multi-frame license-plate restoration to AI-Traffic-Analysis. Instead of OCR-ing a
single best frame per plate, fuse a burst of ~8–32 cropped plate frames into one
super-resolved plate, then OCR the result. Two external repos already do the fusion
(`mf-lpr2`, `eott`); this design wires them in behind new endpoints.

## Problem

The current pipeline (`utils/traffic_analysis.py:process_frame`) detects a plate per frame,
runs single-frame OCR, and keeps the highest-confidence result (sticky `Vehicle.ocr_conf`).
It throws away every other frame of the same plate. Multi-frame super-resolution fuses the
whole burst into a sharper plate image, improving OCR on low-res / motion-blurred plates.

## Decisions (locked during brainstorming)

1. **Host:** AI-Traffic-Analysis is the orchestrator/front door. The fusion repos become
   backends it calls.
2. **Coupling:** localhost HTTP microservice. Steady/live call rate needs warm models;
   same-box deployment makes it loopback (cheap serialization); matches this repo's existing
   FastAPI idiom; debuggable with `curl`.
3. **Entry points:** new endpoints on this repo accepting either a multi-frame burst or a
   video clip.
4. **Engines:** both `mf-lpr2` and `eott`, selectable + comparable via an `engine` param.
5. **OCR owner:** the fusion service returns the **restored image only**. This repo runs its
   existing dual-OCR (FAST + PPOCRv6) on it — single OCR source of truth across all endpoints.

## Key simplification

Because fusion returns image-only (no OCR), neither engine needs its heavy OCR deps:

- `mf-lpr2` default flow is `AffineFlowEstimator` — pure numpy/scipy (no torch).
- `eott` reconstruction (`gsr4`) is pure OpenCV.

So the fusion service is **one light venv** (`numpy`, `scipy`, `pillow`, `opencv-contrib`)
hosting **both** engines — no torch, no paddle, no transformers. One extra process.
Fallback: split into two services only if a dependency ever conflicts.

---

## Architecture

```
┌──────────────────────────────────────────────┐
│ traffic-api  (THIS repo, existing FastAPI)     │  front door
│  • YOLO detect + SORT track                    │
│  • dual-OCR (FAST + PPOCRv6)  ← single OCR truth│
│  • NEW: POST /predict/plates/multiframe         │
│  • NEW: POST /predict/plates/video              │
└───────────────┬────────────────────────────────┘
                │ localhost HTTP (multipart crops → PNG)
                ▼
┌──────────────────────────────────────────────┐
│ fusion-svc  (ONE new light service)            │
│  POST /fuse?engine=mflpr2|eott                 │
│  • mflpr2 adapter → MFLPR2().restore().image   │
│  • eott  adapter → eott.run(...).sr_plate      │
│  returns restored plate PNG (NO OCR)           │
└──────────────────────────────────────────────┘
```

Three processes on one box:

1. **traffic-api** — this repo, existing FastAPI (`main.py`). Two new endpoints. Owns
   detect+track+dual-OCR. Calls fusion-svc over localhost HTTP.
2. **fusion-svc** — one new light FastAPI service. Hosts both restore engines behind a
   single `POST /fuse?engine=...` contract. Returns restored plate PNG. No OCR.

---

## Component contracts

### fusion-svc: `POST /fuse`

- **Query/form params:** `engine` ∈ {`mflpr2`, `eott`} (required); optional engine knobs
  (`scale`, `agg` for mflpr2; `recon`, `hr_width`, `hr_height` for eott) with sane defaults.
- **Body:** multipart, N ordered image files (`files=@crop0.png ... files=@cropN.png`),
  all same size (caller guarantees this — see resize util).
- **Response:** `image/png` — the restored plate, uint8 RGB.
- **Errors:** 400 if `< 1` frame or unknown engine; 422 on decode failure; 500 on engine error.

**mflpr2 adapter:** decode N crops → `list[np.ndarray]` (H,W,3 RGB uint8) →
`MFLPR2(flow_estimator=AffineFlowEstimator()).restore(frames)` →
`result.image` (float32 [0,1]) → `(image*255).astype(uint8)` → PNG. Center frame = middle.

**eott adapter:** `eott.run()` always runs OCR (no "none" backend exists), so the adapter
calls the reconstruction stage **directly** to avoid loading any OCR backend in fusion-svc:
decode N crops (BGR) → build per-frame `corners` list = full-image rectangle (TL,TR,BR,BL)
ordered via `order_corners` → `Config(reconstruction="gsr4", hr_size=(W*scale, H*scale),
max_frames=N)` → `eott.reconstruction.reconstruct(frames, corners_list, cfg)` → `sr_plate`
(BGR uint8) → PNG. Tracking/registration are skipped because the bursts are already cropped
and same-size; corners are the static full rectangle for every frame.

> **Verify in the plan:** `eott.stages.reconstruction.reconstruct(frames, corners, cfg)` at
> `eott/src/eott/stages/reconstruction.py:100` takes a per-frame corners list. Confirm the
> exact signature and that `gsr4` is reachable without a tracker/registration pass.

### this repo: `POST /predict/plates/multiframe`

- **Body:** multipart, N ordered plate crops (`files`). **Params:** `engine` (default
  `mflpr2`), `return_image` (bool, default false).
- **Flow:** read N crops → resize all to common size (median of burst) → POST to
  `fusion-svc /fuse?engine=...` → receive PNG → decode → run `_ocr_plates_fpo` +
  `_ocr_plates_ppocr` on it.
- **Response (JSON):**
  ```json
  {
    "engine": "mflpr2",
    "frames_used": 12,
    "fast":  {"text": "51A12345", "confidence": 0.91},
    "ppocr": {"text": "51A12345", "confidence": 0.88}
  }
  ```
  If `return_image=true`, response is multipart/related or the JSON gains a base64
  `restored_png` field (plan picks one; default JSON-only).
- **Errors:** 400 (no files / all invalid); 503 (fusion-svc unreachable).

### this repo: `POST /predict/plates/video`

- **Body:** multipart, one video file. **Params:** `engine` (default `mflpr2`),
  `min_frames` (default 8), `max_frames` (default 32), `vconf`/`pconf` overrides optional.
- **Flow:**
  1. Decode video → iterate frames.
  2. Run detect + SORT track (reuse existing tracker). For each tracked vehicle, detect its
     plate and crop it (reuse the plate-detect + `crop_expanded_plate` logic).
  3. **Collect** per-`track_id` plate-crop buffers (new method — NOT the single-frame sticky
     OCR path). Cap each buffer at `max_frames`; drop tracks with `< min_frames`.
  4. For each qualifying track: resize burst to common size → `fusion-svc /fuse` → decode →
     dual-OCR.
- **Response (JSON):** list, one entry per track:
  ```json
  [
    {"track_id": 3, "n_frames": 18, "engine": "mflpr2",
     "fast": {"text": "51A12345", "confidence": 0.9},
     "ppocr": {"text": "51A12345", "confidence": 0.87}}
  ]
  ```
- **Errors:** 400 (invalid video); 503 (fusion-svc unreachable). Tracks below `min_frames`
  are silently skipped (not an error).

---

## New code units

| Location | Unit | Responsibility |
|---|---|---|
| fusion-svc (new repo/dir) | `app.py` | FastAPI app, `/fuse` route, engine dispatch |
| fusion-svc | `adapters/mflpr2_adapter.py` | crops → `MFLPR2().restore().image` → PNG bytes |
| fusion-svc | `adapters/eott_adapter.py` | crops → eott reconstruct → PNG bytes |
| this repo | `utils/fusion_client.py` | httpx POST burst → PNG; engine select; timeout/retry; 503 on down |
| this repo | `utils/plate_burst.py` | `resize_burst_to_common(crops) -> list[np.ndarray]` |
| this repo | `utils/traffic_analysis.py` | `collect_plate_bursts(video_path) -> dict[int, list[np.ndarray]]` |
| this repo | `main.py` | `/predict/plates/multiframe`, `/predict/plates/video` |

Reuses (no change): `_ocr_plates_fpo`, `_ocr_plates_ppocr`, `_ensure_dual_ocr`,
plate detector, SORT tracker, `crop_expanded_plate`.

---

## Data flow (multiframe)

```
client ──N crops──▶ /predict/plates/multiframe
                      │ resize_burst_to_common
                      ▼
                    fusion_client.fuse(crops, engine)
                      │ POST localhost:fusion /fuse
                      ▼
                    fusion-svc restore ──PNG──▶ back
                      │ cv2.imdecode
                      ▼
                    _ocr_plates_fpo + _ocr_plates_ppocr
                      ▼
                    JSON {engine, frames_used, fast, ppocr}
```

## Data flow (video)

```
client ──video──▶ /predict/plates/video
                    │ collect_plate_bursts: detect+track+crop per frame
                    ▼
                  {track_id: [crop, crop, ...]}   (drop < min_frames, cap max_frames)
                    │ per track: resize_burst_to_common → fusion_client.fuse
                    ▼
                  per track: PNG → dual-OCR
                    ▼
                  JSON [ {track_id, n_frames, engine, fast, ppocr}, ... ]
```

---

## Error handling

| Condition | Behaviour |
|---|---|
| fusion-svc unreachable | 503 from this repo's endpoint, message names the service + port |
| burst frames differ in size | `resize_burst_to_common` normalizes before send (never an error) |
| `< 1` frame to fusion | 400 |
| unknown engine | 400 (validated before HTTP call) |
| video track `< min_frames` | track silently skipped (not an error) |
| OCR returns nothing | `text: ""`, `confidence: 0.0` |
| engine raises internally | fusion-svc returns 500; this repo surfaces as 502 with engine name |

---

## Testing strategy

**fusion-svc:**
- Unit: synthetic burst (e.g. 8 shifted copies of a text image) → each adapter returns a
  valid PNG of expected size. Assert `mflpr2` and `eott` both produce non-empty output.
- Endpoint: `POST /fuse?engine=mflpr2` and `=eott` with sample crops → 200, `image/png`.

**this repo:**
- `utils/plate_burst.py`: bursts of mismatched sizes → all returned at common size.
- `fusion_client`: mock the HTTP call → returns decoded ndarray; fusion down → raises a typed
  error mapped to 503.
- `/predict/plates/multiframe`: sample crops from `data/adnl/andan/00` → 200 JSON with both
  OCR fields. (Requires fusion-svc running — gate behind a marker or live-service fixture.)
- `/predict/plates/video`: `data/overhead/IMG_2079.mov` (or a short plate clip) → 200 JSON
  list; assert at least one track with `n_frames >= min_frames`.

**Manual smoke:** start fusion-svc, start traffic-api, `curl` both endpoints, eyeball the
restored crops and OCR text vs single-frame baseline.

---

## Out of scope (YAGNI)

- Wiring fusion into the live `process_frame` hot path (the sticky-OCR replacement). The
  video endpoint covers offline/clip fusion; live-path integration is a later phase.
- Returning annotated video from `/predict/plates/video` (JSON only for v1).
- HTTP microservice across multiple machines; GPU offload; engine auto-selection.
- Splitting fusion-svc into two services (only if deps conflict).
- Using the fusion repos' own OCR backends.

## Sequencing

1. fusion-svc (`/fuse`, both adapters, unit tests) — independently testable with curl.
2. this repo: `fusion_client` + `plate_burst` util + `/predict/plates/multiframe`.
3. this repo: `collect_plate_bursts` + `/predict/plates/video`.
4. Manual smoke across all three.

Each step yields working, testable software on its own.

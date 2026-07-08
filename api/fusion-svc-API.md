# fusion-svc API

FastAPI **plate-fusion sidecar**: takes a burst of crops of *one* license plate
and merges them into a single restored plate image. Image-only — no OCR, no
torch, no paddle. Source: [`fusion_svc/app.py`](../fusion_svc/app.py). Adapters:
[`mflpr2_adapter.py`](../fusion_svc/adapters/mflpr2_adapter.py) /
[`eott_adapter.py`](../fusion_svc/adapters/eott_adapter.py).

**This is optional** — by default the main traffic API (port `7862`) runs these
same engines in-process, no sidecar needed. Run this standalone app only if you
want fusion crash-isolated in a separate process/venv (port `8100`).

## Start the API (standalone, optional)

```bash
git submodule update --init --recursive      # fresh clone only
uv sync --directory fusion_svc
uv run --directory fusion_svc uvicorn fusion_svc.app:app --host 127.0.0.1 --port 8100
```

Swagger UI: `http://127.0.0.1:8100/docs`. See [Install](#install-one-time) / [Run](#run) below for details.

## Install (one-time)

Submodules must be present (fresh clone), then sync the subproject venv:

```bash
git submodule update --init --recursive      # clones external/mf-lpr2 + external/eott
uv sync --directory fusion_svc               # builds .venv + editable engines
```

## Run

From the repo root (`/mnt/data/nblong-t04/AI-Traffic-Analysis`):

```bash
uv run --directory fusion_svc uvicorn fusion_svc.app:app --host 127.0.0.1 --port 8100
```

Swagger UI (browser upload): `http://<host>:8100/docs`.

Background / survives logout:

```bash
nohup uv run --directory fusion_svc uvicorn fusion_svc.app:app --host 0.0.0.0 --port 8100 \
  > fusion_svc/fusion.log 2>&1 &
```

Stop:

```bash
kill $(ss -ltnp | grep ':8100' | grep -oP 'pid=\K[0-9]+')
```

## Endpoints (2)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | liveness — `{"status":"ok"}` |
| `POST` | `/fuse`   | `files=` N plate crops → one restored plate (PNG) |

### `POST /fuse`

- **Body:** multipart `files` — N ordered, **same-size** crops of one plate.
- **Query:** `engine` (`mflpr2`\|`eott`, default `mflpr2`), `scale` (int 1–8, default 1).
- **Returns:** `image/png` — the restored plate, **BGR-encoded**.

## Engines

| engine | source | output | `scale` | notes |
|---|---|---|---|---|
| `mflpr2` | `external/mf-lpr2` | color restored plate, `(H*scale, W*scale, 3)` | **applied** | training-free restore (arXiv 2508.14797); cleaner default |
| `eott`   | `external/eott`    | **binarized** plate on a fixed HR grid (native `hr_size`, default 300×98 minus crop margin) | **ignored** | reconstruction-only; fast; native eott has no scale knob |

Both reproduce their engine's **native pre-OCR** result exactly — only the OCR
stage is omitted. `scale` is accepted by both for interface parity, but **eott
ignores it** (reconstructs onto a fixed grid). Set `scale=2` to upscale only on
`mflpr2`.

## Contract

- Wire format is **BGR** end-to-end (PNG-encoded). RGB conversion happens only
  inside the mf-lpr2 adapter.
- Fuse **raw** low-res crops — do **not** pre-apply super-resolution (clips
  highlights, degrades downstream OCR).
- Crops must be the **same size** and ordered (the engines register frames
  against a center frame, `len//2`).

## Errors

| status | body | cause |
|---|---|---|
| `400` | `unknown engine: '<x>'` | `engine` not in `{mflpr2, eott}` |
| `400` | `no decodable frames` | every uploaded file failed `cv2.imdecode` (not a valid image) |
| `422` | FastAPI validation | the `files` field was missing entirely |
| `500` | `<engine> failed: <err>` | engine-internal error (e.g. mismatched crop sizes) |

## Examples

```bash
# Merge N plate crops -> one restored plate (PNG)
curl -X POST "http://127.0.0.1:8100/fuse?engine=mflpr2&scale=2" \
  -F files=@01.png -F files=@02.png -F files=@03.png -o fused.png

# eott engine (scale ignored — fixed grid, binarized output)
curl -X POST "http://127.0.0.1:8100/fuse?engine=eott" \
  -F files=@01.png -F files=@02.png -o fused_eott.png

# Health
curl http://127.0.0.1:8100/health        # {"status":"ok"}
```

## Relationship to the main API (port 7862)

By default the engines here are **not run as a sidecar** — the main traffic API
(`main.py`, port `7862`) installs the vendored `mf-lpr2` / `eott` packages into
its own venv and calls these same adapters **in-process** through
[`utils/fusion_client.py`](../utils/fusion_client.py). So `/fuse` and the
OCR-wrapping endpoints all live on port 7862:

| main endpoint | does | params |
|---|---|---|
| `POST /fuse` | fuse a burst of crops → restored plate **image** (same as this app) | `engine`, `scale` |
| `POST /predict/plates/multiframe` | fuse a burst of one plate's crops → **dual-OCR** | `engine`, `scale`, `max_frames` (32) |
| `POST /predict/plates/video` | detect+track plates in a video, fuse each track's burst → dual-OCR | `engine`, `scale`, `min_frames` (8), `max_frames` (32) |

```
client ──HTTP──> main API (7862) ──in-process──> mf-lpr2 / eott   (default)
```

The `/predict/plates/*` variants also **resize the burst to a common size**
(`resize_burst_to_common`) before fusing, then return **JSON OCR text**
(`{engine, frames_used, fast, ppocr}`) instead of a PNG.

**This standalone app is optional** — run it on 8100 only if you want fusion in a
separate process/venv (crash isolation, independent lifecycle). In that split
mode, point the main API at it with `FUSION_URL` and have `fusion_client` POST
over HTTP instead of importing the adapters.

## Test

```bash
uv run --directory fusion_svc pytest      # 10 tests: both adapters + /fuse endpoint
```

Covers `/health`, `mflpr2` (scale applied → `(64,192,3)` for 32×96 in), `eott`
(fixed-grid binarized shape), unknown-engine `400`, no-files `400/422`.
The endpoint suite uses FastAPI `TestClient` (in-process, no live server needed).

# fusion_svc

Multi-frame plate **fusion sidecar** for AI-Traffic-Analysis. Takes a burst of
crops of one license plate and merges them into a single restored plate image.
Image-only — no OCR, no torch, no paddle (deps: numpy/scipy/pillow/opencv +
`mf-lpr2`/`eott`).

Self-contained `uv` subproject with its own `.venv`; the parent app calls it
over localhost HTTP (`utils/fusion_client.py`), so its heavy deps stay isolated.

## Engines

| engine | source repo | notes |
|--------|-------------|-------|
| `mflpr2` | `../../mf-lpr2` | training-free restore (arXiv 2508.14797); cleaner default |
| `eott`   | `../../eott`    | reconstruction-only path; fast |

## Run

```bash
uv run --directory fusion_svc uvicorn fusion_svc.app:app --port 8100
```

## API

```bash
# Merge N plate crops -> one restored plate (PNG, BGR)
curl -X POST "http://127.0.0.1:8100/fuse?engine=mflpr2&scale=2" \
  -F "files=@01.png" -F "files=@02.png" ... -o fused.png

# Health
curl http://127.0.0.1:8100/health        # {"status":"ok"}
```

`engine`: `mflpr2` | `eott`. `scale`: upscale factor (default 2).

## Test

```bash
uv run --directory fusion_svc pytest
```

## Contract

- Wire format is **BGR** end-to-end (PNG-encoded). RGB conversion happens only
  inside the mf-lpr2 adapter.
- Fuse **raw** low-res crops — do not pre-apply super-resolution (clips
  highlights, degrades downstream OCR).

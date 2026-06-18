# Multi-Frame Plate Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse a burst of ~8–32 cropped license-plate frames into one super-resolved plate, then OCR it — exposed via two new endpoints on AI-Traffic-Analysis, backed by a new localhost fusion microservice wrapping mf-lpr2 and eott.

**Architecture:** A new light FastAPI service (`fusion-svc`) hosts both restore engines (mf-lpr2 affine-flow + GSR4, eott classical GSR4) behind `POST /fuse?engine=mflpr2|eott`, returning a restored plate PNG and running **no OCR**. AI-Traffic-Analysis gains a fusion HTTP client, a burst-resize util, a per-track burst collector, and two endpoints that call fusion-svc then run this repo's existing dual-OCR (FAST + PPOCRv6) on the restored image.

**Tech Stack:** Python 3.13 (`requires-python>=3.10`), FastAPI, uvicorn, numpy, scipy, opencv, httpx, pytest. mf-lpr2 (`mf_lpr2`) and eott (`eott`) installed editable into the fusion-svc venv only.

## Global Constraints

- Use `uv` for all Python invocation and dependency management — never bare `python`/`python3`, never `--break-system-packages`, never system-wide installs.
- Run multi-line Python as a script file, never `uv run python -c`.
- fusion-svc returns the restored plate as a **PNG encoded from a BGR uint8 ndarray** (`cv2.imencode(".png", bgr)`); clients decode with `cv2.imdecode` → BGR. This is the single color contract across the wire.
- This repo's OCR functions (`_ocr_plates_fpo`, `_ocr_plates_ppocr`) expect **BGR uint8**. Keep everything BGR end-to-end except inside the mf-lpr2 adapter (which converts to/from RGB internally).
- fusion-svc lives at `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc` (sibling of the three repos), with its **own** venv. Never install mf-lpr2/eott into the AI-Traffic-Analysis venv.
- Engine values are exactly `mflpr2` and `eott` (lowercase). Default engine is `mflpr2`.
- Default super-resolution scale = `2` for both engines.
- mf-lpr2 restore-only path must never import torch/transformers: construct `MFLPR2(flow_estimator=AffineFlowEstimator(), ...)` and never call `.run()`/`.recognize()`.
- All bursts sent to fusion-svc must be same-size (caller resizes via `resize_burst_to_common`). Adapters assume same-size input.
- Default fusion-svc base URL in this repo: `http://127.0.0.1:8100`, overridable via `FUSION_URL` env var.

---

## File Structure

**New repo: `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/`**
- `pyproject.toml` — deps + editable mf-lpr2/eott
- `fusion_svc/__init__.py`
- `fusion_svc/adapters/__init__.py`
- `fusion_svc/adapters/mflpr2_adapter.py` — `fuse_mflpr2(frames, scale) -> np.ndarray` (BGR uint8)
- `fusion_svc/adapters/eott_adapter.py` — `fuse_eott(frames, scale) -> np.ndarray` (BGR uint8)
- `fusion_svc/app.py` — FastAPI `POST /fuse`, engine dispatch, PNG encode
- `tests/test_mflpr2_adapter.py`
- `tests/test_eott_adapter.py`
- `tests/test_fuse_endpoint.py`

**Modified repo: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/`**
- Create: `utils/plate_burst.py` — `resize_burst_to_common`, `select_burst_window`
- Create: `utils/fusion_client.py` — `FusionUnavailable`, `fuse(crops, engine, base_url) -> np.ndarray`
- Modify: `utils/traffic_analysis.py` — add `collect_plate_bursts(frames, min_frames, max_frames)`
- Modify: `main.py` — add `/predict/plates/multiframe`, `/predict/plates/video`
- Modify: `pyproject.toml` — add `httpx`
- Create: `tests/test_plate_burst.py`
- Create: `tests/test_fusion_client.py`
- Create: `tests/test_multiframe_endpoint.py`
- Create: `tests/test_collect_bursts.py`
- Create: `tests/test_video_endpoint.py`

mf-lpr2 and eott repos are **not modified** — installed as libraries only.

---

### Task 1: fusion-svc scaffold + mf-lpr2 adapter

**Files:**
- Create: `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/pyproject.toml`
- Create: `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/fusion_svc/__init__.py`
- Create: `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/fusion_svc/adapters/__init__.py`
- Create: `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/fusion_svc/adapters/mflpr2_adapter.py`
- Test: `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/tests/test_mflpr2_adapter.py`

**Interfaces:**
- Produces: `fuse_mflpr2(frames: list[np.ndarray], scale: int = 2) -> np.ndarray` — input BGR uint8 same-size crops, returns one BGR uint8 restored plate of shape `(H*scale, W*scale, 3)`.

- [ ] **Step 1: Create the project skeleton and venv**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/pyproject.toml`:

```toml
[project]
name = "fusion-svc"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "python-multipart>=0.0.9",
    "numpy>=1.24",
    "scipy>=1.10",
    "pillow>=9.0",
    "opencv-python>=4.8",
    "mf-lpr2",
    "eott",
]

[tool.uv.sources]
mf-lpr2 = { path = "../mf-lpr2", editable = true }
eott = { path = "../eott", editable = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["fusion_svc"]

[tool.uv]
dev-dependencies = ["pytest>=7.4", "httpx>=0.27"]

[tool.pytest.ini_options]
pythonpath = ["."]
```

Create empty `fusion_svc/__init__.py` and `fusion_svc/adapters/__init__.py`.

Then create the venv and install:

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && uv sync
```
Expected: resolves and installs fastapi, opencv, numpy, scipy, pillow, plus editable `mf-lpr2` and `eott`. No torch/paddle should be pulled by mf-lpr2 (base deps only). eott pulls `opencv-contrib-python` + `pytesseract` (installed, unused).

- [ ] **Step 2: Write the failing test**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/tests/test_mflpr2_adapter.py`:

```python
import numpy as np
from fusion_svc.adapters.mflpr2_adapter import fuse_mflpr2


def _synthetic_burst(n=8, h=32, w=96):
    """N near-identical BGR crops with tiny sub-pixel-ish shifts."""
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[8:24, 10:86] = 200  # a bright bar standing in for plate glyphs
    frames = []
    for i in range(n):
        shifted = np.roll(base, shift=i % 2, axis=1)
        frames.append(shifted.copy())
    return frames


def test_fuse_mflpr2_returns_bgr_uint8_scaled():
    frames = _synthetic_burst(n=8, h=32, w=96)
    out = fuse_mflpr2(frames, scale=2)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape[0] == 64 and out.shape[1] == 192  # (H*2, W*2)


def test_fuse_mflpr2_single_frame_ok():
    frames = _synthetic_burst(n=1, h=32, w=96)
    out = fuse_mflpr2(frames, scale=1)
    assert out.shape == (32, 96, 3)
    assert out.dtype == np.uint8
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && uv run pytest tests/test_mflpr2_adapter.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'fusion_svc.adapters.mflpr2_adapter'`.

- [ ] **Step 4: Write the adapter**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/fusion_svc/adapters/mflpr2_adapter.py`:

```python
"""Adapter: fuse N BGR plate crops into one restored BGR image via mf-lpr2.

Restore-only path — never constructs a recognizer, so torch/transformers are
never imported. Uses AffineFlowEstimator (pure numpy/scipy).
"""
import cv2
import numpy as np
from mf_lpr2 import MFLPR2, AffineFlowEstimator


def fuse_mflpr2(frames: list[np.ndarray], scale: int = 2) -> np.ndarray:
    """Fuse same-size BGR uint8 crops into one BGR uint8 restored plate.

    Returns shape (H*scale, W*scale, 3), uint8 BGR.
    """
    if not frames:
        raise ValueError("fuse_mflpr2 requires at least one frame")
    # mf-lpr2 works in RGB; convert in, convert out.
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    pipe = MFLPR2(flow_estimator=AffineFlowEstimator(), scale=int(scale))
    result = pipe.restore(rgb)            # center_idx defaults to len//2
    img = np.clip(result.image, 0.0, 1.0)  # float32 [0,1] RGB
    img_u8 = (img * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && uv run pytest tests/test_mflpr2_adapter.py -v
```
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && git init -q 2>/dev/null; git add -A && git commit -m "feat(fusion): scaffold fusion-svc + mf-lpr2 restore adapter"
```

---

### Task 2: eott reconstruct adapter

**Files:**
- Create: `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/fusion_svc/adapters/eott_adapter.py`
- Test: `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/tests/test_eott_adapter.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `fuse_eott(frames: list[np.ndarray], scale: int = 2) -> np.ndarray` — input BGR uint8 same-size crops, returns one BGR uint8 restored plate of shape `(H*scale, W*scale, 3)`.

- [ ] **Step 1: Write the failing test**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/tests/test_eott_adapter.py`:

```python
import numpy as np
from fusion_svc.adapters.eott_adapter import fuse_eott


def _synthetic_burst(n=8, h=32, w=96):
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[8:24, 10:86] = 200
    return [base.copy() for _ in range(n)]


def test_fuse_eott_returns_bgr_uint8_scaled():
    frames = _synthetic_burst(n=8, h=32, w=96)
    out = fuse_eott(frames, scale=2)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape[0] == 64 and out.shape[1] == 192  # (H*2, W*2)


def test_fuse_eott_single_frame_ok():
    frames = _synthetic_burst(n=1, h=32, w=96)
    out = fuse_eott(frames, scale=1)
    assert out.shape == (32, 96, 3)
    assert out.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && uv run pytest tests/test_eott_adapter.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'fusion_svc.adapters.eott_adapter'`.

- [ ] **Step 3: Write the adapter**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/fusion_svc/adapters/eott_adapter.py`:

```python
"""Adapter: fuse N BGR plate crops into one restored BGR image via eott.

Calls the reconstruction stage directly (skips tracking/registration/OCR).
Each frame's ROI is the full-image rectangle (frames are pre-cropped plates).
"""
import numpy as np
from eott import Config
from eott.geometry import order_corners
from eott.stages.reconstruction import reconstruct


def _full_rect(h: int, w: int) -> np.ndarray:
    # TL, TR, BR, BL
    return order_corners(
        np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    )


def fuse_eott(frames: list[np.ndarray], scale: int = 2) -> np.ndarray:
    """Fuse same-size BGR uint8 crops into one BGR uint8 restored plate.

    Returns shape (H*scale, W*scale, 3), uint8 BGR.
    """
    if not frames:
        raise ValueError("fuse_eott requires at least one frame")
    h, w = frames[0].shape[:2]
    corners = [_full_rect(h, w) for _ in frames]
    cfg = Config(
        reconstruction="gsr4",
        hr_size=(w * int(scale), h * int(scale)),  # eott hr_size is (width, height)
        max_frames=len(frames),
    )
    out = reconstruct(frames, corners, cfg)  # BGR uint8, shape (H*scale, W*scale, 3)
    return np.ascontiguousarray(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && uv run pytest tests/test_eott_adapter.py -v
```
Expected: PASS (2 passed). If `reconstruct` raises on the all-identical burst, switch the synthetic burst to use `np.roll(base, i % 2, axis=1)` like Task 1 (varied frames). Re-run.

- [ ] **Step 5: Commit**

```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && git add -A && git commit -m "feat(fusion): add eott reconstruct adapter"
```

---

### Task 3: fusion-svc `/fuse` FastAPI endpoint

**Files:**
- Create: `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/fusion_svc/app.py`
- Test: `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/tests/test_fuse_endpoint.py`

**Interfaces:**
- Consumes: `fuse_mflpr2(frames, scale)`, `fuse_eott(frames, scale)` from Tasks 1–2.
- Produces: `app` (FastAPI), `POST /fuse?engine={mflpr2|eott}&scale=N` accepting multipart `files`, returning `image/png` (BGR-encoded). `GET /health` → `{"status": "ok"}`.

- [ ] **Step 1: Write the failing test**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/tests/test_fuse_endpoint.py`:

```python
import cv2
import numpy as np
from fastapi.testclient import TestClient
from fusion_svc.app import app

client = TestClient(app)


def _png_bytes(h=32, w=96, val=200):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[8:24, 10:86] = val
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _files(n=6):
    return [("files", (f"c{i}.png", _png_bytes(val=180 + i), "image/png")) for i in range(n)]


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_fuse_mflpr2_returns_png():
    r = client.post("/fuse?engine=mflpr2&scale=2", files=_files())
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    arr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None and arr.shape == (64, 192, 3)


def test_fuse_eott_returns_png():
    r = client.post("/fuse?engine=eott&scale=2", files=_files())
    assert r.status_code == 200
    arr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None and arr.shape == (64, 192, 3)


def test_fuse_unknown_engine_400():
    r = client.post("/fuse?engine=bogus", files=_files())
    assert r.status_code == 400


def test_fuse_no_files_400():
    r = client.post("/fuse?engine=mflpr2", files=[])
    assert r.status_code in (400, 422)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && uv run pytest tests/test_fuse_endpoint.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'fusion_svc.app'`.

- [ ] **Step 3: Write the app**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc/fusion_svc/app.py`:

```python
"""fusion-svc: multi-frame plate restoration microservice (no OCR).

POST /fuse?engine=mflpr2|eott&scale=N
  body: multipart `files` — N ordered same-size plate crops
  returns: image/png (restored plate, BGR-encoded)
"""
import io

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from fusion_svc.adapters.eott_adapter import fuse_eott
from fusion_svc.adapters.mflpr2_adapter import fuse_mflpr2

app = FastAPI(title="fusion-svc", version="0.1.0")

_ENGINES = {"mflpr2": fuse_mflpr2, "eott": fuse_eott}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/fuse")
def fuse(
    files: list[UploadFile] = File(...),
    engine: str = Query("mflpr2"),
    scale: int = Query(2, ge=1, le=8),
):
    if engine not in _ENGINES:
        raise HTTPException(status_code=400, detail=f"unknown engine: {engine!r}")
    frames: list[np.ndarray] = []
    for up in files:
        data = up.file.read()
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            frames.append(arr)
    if not frames:
        raise HTTPException(status_code=400, detail="no decodable frames")
    try:
        out = _ENGINES[engine](frames, scale=scale)
    except Exception as e:  # engine-internal failure
        raise HTTPException(status_code=500, detail=f"{engine} failed: {e}")
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(status_code=500, detail="png encode failed")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && uv run pytest tests/test_fuse_endpoint.py -v
```
Expected: PASS (5 passed).

- [ ] **Step 5: Run the whole fusion-svc suite**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && uv run pytest -v
```
Expected: PASS (9 passed total across the three test files).

- [ ] **Step 6: Commit**

```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && git add -A && git commit -m "feat(fusion): add POST /fuse endpoint with engine dispatch"
```

---

### Task 4: `resize_burst_to_common` + `select_burst_window` util (this repo)

**Files:**
- Create: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/utils/plate_burst.py`
- Test: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/tests/test_plate_burst.py`

**Interfaces:**
- Produces:
  - `resize_burst_to_common(crops: list[np.ndarray], target: tuple[int, int] | None = None) -> list[np.ndarray]` — returns all crops resized to a common `(H, W)`; when `target` is None, uses the per-axis median of input sizes. BGR uint8 in, BGR uint8 out.
  - `select_burst_window(crops: list[np.ndarray], max_frames: int) -> list[np.ndarray]` — if `len(crops) <= max_frames` returns as-is, else returns `max_frames` evenly-spaced crops (always including first and last).

- [ ] **Step 1: Write the failing test**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/tests/test_plate_burst.py`:

```python
import numpy as np
from utils.plate_burst import resize_burst_to_common, select_burst_window


def _crop(h, w, val=100):
    return np.full((h, w, 3), val, dtype=np.uint8)


def test_resize_to_common_median():
    crops = [_crop(30, 90), _crop(32, 96), _crop(34, 102)]
    out = resize_burst_to_common(crops)
    sizes = {c.shape[:2] for c in out}
    assert len(sizes) == 1                      # all identical now
    assert out[0].shape[:2] == (32, 96)         # per-axis median
    assert all(c.dtype == np.uint8 for c in out)


def test_resize_to_explicit_target():
    crops = [_crop(30, 90), _crop(50, 50)]
    out = resize_burst_to_common(crops, target=(40, 80))
    assert all(c.shape[:2] == (40, 80) for c in out)


def test_resize_empty_returns_empty():
    assert resize_burst_to_common([]) == []


def test_select_window_under_cap_returns_all():
    crops = [_crop(10, 10) for _ in range(5)]
    assert len(select_burst_window(crops, max_frames=32)) == 5


def test_select_window_over_cap_downsamples_inclusive():
    crops = [_crop(10, 10, val=i) for i in range(40)]
    out = select_burst_window(crops, max_frames=8)
    assert len(out) == 8
    assert out[0][0, 0, 0] == 0     # first kept
    assert out[-1][0, 0, 0] == 39   # last kept
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv run pytest tests/test_plate_burst.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.plate_burst'`.

- [ ] **Step 3: Write the util**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/utils/plate_burst.py`:

```python
"""Helpers for preparing a multi-frame plate burst before fusion."""
import cv2
import numpy as np


def resize_burst_to_common(crops, target=None):
    """Resize all BGR crops to a common (H, W).

    target: (H, W) to force; if None, uses the per-axis median of input sizes.
    Returns a new list; input list is not mutated. Empty in -> empty out.
    """
    if not crops:
        return []
    if target is None:
        heights = [c.shape[0] for c in crops]
        widths = [c.shape[1] for c in crops]
        th = int(np.median(heights))
        tw = int(np.median(widths))
    else:
        th, tw = int(target[0]), int(target[1])
    out = []
    for c in crops:
        if c.shape[:2] == (th, tw):
            out.append(c)
        else:
            out.append(cv2.resize(c, (tw, th), interpolation=cv2.INTER_CUBIC))
    return out


def select_burst_window(crops, max_frames):
    """Cap a burst to max_frames evenly-spaced crops (keeps first and last)."""
    n = len(crops)
    if n <= max_frames:
        return list(crops)
    idx = np.linspace(0, n - 1, max_frames).round().astype(int)
    return [crops[i] for i in idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv run pytest tests/test_plate_burst.py -v
```
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && git add utils/plate_burst.py tests/test_plate_burst.py && git commit -m "feat(plate): add burst resize + window helpers"
```

---

### Task 5: `fusion_client` + httpx dependency (this repo)

**Files:**
- Create: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/utils/fusion_client.py`
- Modify: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/pyproject.toml`
- Test: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/tests/test_fusion_client.py`

**Interfaces:**
- Consumes: nothing from prior tasks (pure HTTP).
- Produces:
  - `class FusionUnavailable(Exception)` — raised when fusion-svc is unreachable.
  - `fuse(crops: list[np.ndarray], engine: str = "mflpr2", scale: int = 2, base_url: str | None = None, timeout: float = 60.0) -> np.ndarray` — POSTs PNG-encoded BGR crops to `{base_url}/fuse`, returns the decoded BGR uint8 restored plate. `base_url` defaults to `FUSION_URL` env or `http://127.0.0.1:8100`.

- [ ] **Step 1: Add httpx dependency**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv add httpx
```
Expected: adds `httpx` to `[project.dependencies]` in pyproject.toml and installs it. (Distinct from the existing `httpx2` dev dep; both can coexist.)

- [ ] **Step 2: Write the failing test**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/tests/test_fusion_client.py`:

```python
import cv2
import numpy as np
import pytest
import httpx

from utils import fusion_client
from utils.fusion_client import fuse, FusionUnavailable


def _crop(h=32, w=96, val=120):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _png(arr):
    ok, buf = cv2.imencode(".png", arr)
    return buf.tobytes()


def test_fuse_posts_and_decodes(monkeypatch):
    restored = np.full((64, 192, 3), 77, dtype=np.uint8)

    class FakeResp:
        status_code = 200
        content = _png(restored)
        def raise_for_status(self): pass

    captured = {}

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, files=None, params=None):
            captured["url"] = url
            captured["params"] = params
            captured["nfiles"] = len(files)
            return FakeResp()

    monkeypatch.setattr(fusion_client.httpx, "Client", FakeClient)
    out = fuse([_crop(), _crop(), _crop()], engine="eott", scale=2,
               base_url="http://x:8100")
    assert out.shape == (64, 192, 3)
    assert captured["url"] == "http://x:8100/fuse"
    assert captured["params"] == {"engine": "eott", "scale": 2}
    assert captured["nfiles"] == 3


def test_fuse_unreachable_raises_fusion_unavailable(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **k):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(fusion_client.httpx, "Client", FakeClient)
    with pytest.raises(FusionUnavailable):
        fuse([_crop()], base_url="http://x:8100")


def test_fuse_empty_crops_raises_value_error():
    with pytest.raises(ValueError):
        fuse([], base_url="http://x:8100")
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv run pytest tests/test_fusion_client.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.fusion_client'`.

- [ ] **Step 4: Write the client**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/utils/fusion_client.py`:

```python
"""HTTP client for the fusion-svc multi-frame restoration microservice."""
import os

import cv2
import httpx
import numpy as np

DEFAULT_BASE_URL = "http://127.0.0.1:8100"


class FusionUnavailable(Exception):
    """Raised when fusion-svc cannot be reached."""


def _png_bytes(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("failed to PNG-encode crop")
    return buf.tobytes()


def fuse(crops, engine="mflpr2", scale=2, base_url=None, timeout=60.0):
    """POST PNG-encoded BGR crops to fusion-svc; return decoded BGR restored plate."""
    if not crops:
        raise ValueError("fuse requires at least one crop")
    url_base = base_url or os.environ.get("FUSION_URL", DEFAULT_BASE_URL)
    files = [
        ("files", (f"c{i}.png", _png_bytes(c), "image/png"))
        for i, c in enumerate(crops)
    ]
    params = {"engine": engine, "scale": int(scale)}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{url_base}/fuse", files=files, params=params)
            resp.raise_for_status()
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise FusionUnavailable(f"fusion-svc unreachable at {url_base}: {e}") from e
    arr = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("fusion-svc returned undecodable image")
    return arr
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv run pytest tests/test_fusion_client.py -v
```
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && git add utils/fusion_client.py tests/test_fusion_client.py pyproject.toml uv.lock && git commit -m "feat(fusion): add fusion-svc HTTP client"
```

---

### Task 6: `POST /predict/plates/multiframe` endpoint (this repo)

**Files:**
- Modify: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/main.py`
- Test: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/tests/test_multiframe_endpoint.py`

**Interfaces:**
- Consumes: `resize_burst_to_common`, `select_burst_window` (Task 4); `fuse`, `FusionUnavailable` (Task 5); `traffic_service._ensure_dual_ocr`, `traffic_service._ocr_plates_fpo`, `traffic_service._ocr_plates_ppocr` (existing).
- Produces: `POST /predict/plates/multiframe` accepting multipart `files` + query `engine` (default `mflpr2`), `scale` (default 2), `max_frames` (default 32). Returns JSON `{"engine", "frames_used", "fast": {"text","confidence"}, "ppocr": {"text","confidence"}}`. Errors: 400 (no valid crops), 503 (fusion-svc down).

- [ ] **Step 1: Write the failing test**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/tests/test_multiframe_endpoint.py`:

```python
import sys
from unittest import mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


def _png(h=32, w=96, val=150):
    img = np.full((h, w, 3), val, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@pytest.fixture()
def client():
    sys.modules.pop("main", None)
    with mock.patch("utils.traffic_analysis.YOLO"), \
         mock.patch("utils.traffic_analysis.Sort"), \
         mock.patch("utils.traffic_analysis.DeepSort"), \
         mock.patch.object(
             __import__("utils.traffic_analysis", fromlist=["TrafficAnalysisService"]).TrafficAnalysisService,
             "_init_ocr_engine", lambda self: None), \
         mock.patch.object(
             __import__("utils.traffic_analysis", fromlist=["TrafficAnalysisService"]).TrafficAnalysisService,
             "_init_sr_engine", lambda self: None):
        import main
        # restored image returned by the (mocked) fusion client
        restored = np.full((64, 192, 3), 90, dtype=np.uint8)
        main.traffic_service._ensure_dual_ocr = lambda: None
        main.traffic_service._ocr_plates_fpo = lambda crop: ("51A12345", 0.91)
        main.traffic_service._ocr_plates_ppocr = lambda crop: ("51A12345", 0.88)
        with mock.patch("main.fusion_fuse", return_value=restored) as _:
            yield TestClient(main.app)
    sys.modules.pop("main", None)


def test_multiframe_returns_dual_ocr(client):
    files = [("files", (f"c{i}.png", _png(val=140 + i), "image/png")) for i in range(6)]
    r = client.post("/predict/plates/multiframe?engine=mflpr2", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "mflpr2"
    assert body["frames_used"] == 6
    assert body["fast"] == {"text": "51A12345", "confidence": 0.91}
    assert body["ppocr"] == {"text": "51A12345", "confidence": 0.88}


def test_multiframe_no_files_400(client):
    r = client.post("/predict/plates/multiframe", files=[])
    assert r.status_code in (400, 422)


def test_multiframe_fusion_down_503(client):
    from utils.fusion_client import FusionUnavailable
    files = [("files", ("c0.png", _png(), "image/png"))]
    with mock.patch("main.fusion_fuse", side_effect=FusionUnavailable("down")):
        r = client.post("/predict/plates/multiframe", files=files)
    assert r.status_code == 503
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv run pytest tests/test_multiframe_endpoint.py -v
```
Expected: FAIL — `main` has no attribute `fusion_fuse` / route 404.

- [ ] **Step 3: Add imports to main.py**

In `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/main.py`, after the existing `from utils.traffic_analysis import TrafficAnalysisService` (line 22), add:

```python
from utils.fusion_client import fuse as fusion_fuse, FusionUnavailable
from utils.plate_burst import resize_burst_to_common, select_burst_window
```

- [ ] **Step 4: Add the endpoint to main.py**

In `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/main.py`, add after the `/predict/plates/batch` handler (after line 429):

```python
def _decode_uploads(files):
    """Decode a list of UploadFile into BGR ndarrays (skips undecodable)."""
    crops = []
    for up in files:
        data = up.file.read()
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            crops.append(arr)
    return crops


def _dual_ocr(crop):
    """Run this repo's FAST + PPOCRv6 OCR on a BGR crop -> two OCRResult dicts."""
    traffic_service._ensure_dual_ocr()
    fpo_text, fpo_conf = traffic_service._ocr_plates_fpo(crop)
    ppo_text, ppo_conf = traffic_service._ocr_plates_ppocr(crop)
    return (
        {"text": fpo_text, "confidence": float(fpo_conf)},
        {"text": ppo_text, "confidence": float(ppo_conf)},
    )


@app.post("/predict/plates/multiframe")
def predict_plates_multiframe(
    files: List[UploadFile] = File(...),
    engine: str = "mflpr2",
    scale: int = 2,
    max_frames: int = 32,
):
    """Fuse a burst of one plate's crops, then dual-OCR the restored plate."""
    if engine not in ("mflpr2", "eott"):
        raise HTTPException(status_code=400, detail=f"unknown engine: {engine!r}")
    crops = _decode_uploads(files)
    if not crops:
        raise HTTPException(status_code=400, detail="no valid plate crops")
    crops = select_burst_window(crops, max_frames)
    crops = resize_burst_to_common(crops)
    try:
        restored = fusion_fuse(crops, engine=engine, scale=scale)
    except FusionUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    fast, ppocr = _dual_ocr(restored)
    return {
        "engine": engine,
        "frames_used": len(crops),
        "fast": fast,
        "ppocr": ppocr,
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv run pytest tests/test_multiframe_endpoint.py -v
```
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && git add main.py tests/test_multiframe_endpoint.py && git commit -m "feat(api): add POST /predict/plates/multiframe"
```

---

### Task 7: `collect_plate_bursts` on TrafficAnalysisService (this repo)

**Files:**
- Modify: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/utils/traffic_analysis.py`
- Test: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/tests/test_collect_bursts.py`

**Interfaces:**
- Consumes: existing `self.vehicle_detector`, `self.tracker`, `self.plate_detector`, `crop_expanded_plate` (already imported at `utils/traffic_analysis.py:20`).
- Produces: `collect_plate_bursts(self, frames: list[np.ndarray], min_frames: int = 8, max_frames: int = 32) -> dict[int, list[np.ndarray]]` — runs detect+track over the given frames, crops each tracked vehicle's plate per frame, groups crops by `track_id`, drops tracks with fewer than `min_frames` crops, caps each kept track to `max_frames` (evenly spaced via `select_burst_window`).

**Note on design:** this method takes already-decoded frames (not a video path) so it is unit-testable without a video file or real weights. The video endpoint (Task 8) reads frames then calls this.

- [ ] **Step 1: Write the failing test**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/tests/test_collect_bursts.py`:

```python
import sys
from unittest import mock

import numpy as np
import pytest


@pytest.fixture()
def service():
    """A TrafficAnalysisService with detectors/tracker/OCR stubbed out."""
    sys.modules.pop("utils.traffic_analysis", None)
    with mock.patch("utils.traffic_analysis.YOLO"), \
         mock.patch("utils.traffic_analysis.Sort"), \
         mock.patch("utils.traffic_analysis.DeepSort"):
        from utils.traffic_analysis import TrafficAnalysisService
        svc = TrafficAnalysisService.__new__(TrafficAnalysisService)
        # minimal attributes the method touches
        svc.opts = mock.SimpleNamespace(device="cpu", vconf=0.6, pconf=0.25)
        svc.vehicles = {}
        return svc


def _frame(val=50):
    return np.full((200, 200, 3), val, dtype=np.uint8)


def _fake_vehicle_detection(box):
    """Mimic ultralytics result[0] with .boxes.cpu().xyxy."""
    boxes = mock.MagicMock()
    xyxy = np.array([box], dtype=float)
    boxes.cpu.return_value.xyxy = xyxy
    boxes.__len__ = lambda self=boxes: 1
    res = mock.MagicMock()
    res.boxes = boxes
    return [res]


def _fake_plate_result(pbox):
    """Mimic plate detector result with .boxes.xyxy[0]."""
    res = mock.MagicMock()
    t = mock.MagicMock()
    t.cpu.return_value.numpy.return_value = np.array(pbox, dtype=float)
    res.boxes.xyxy = [t]
    res.boxes.__len__ = lambda self=res.boxes: 1
    return res


def test_collect_groups_by_track_id_and_filters(service):
    frames = [_frame(40 + i) for i in range(10)]

    # vehicle detector returns a single fixed vehicle box per frame
    service.vehicle_detector = mock.MagicMock(
        side_effect=lambda f, **k: _fake_vehicle_detection([20, 20, 120, 120]))
    # tracker assigns the same track_id=7 every frame
    service.tracker = mock.MagicMock()
    service.tracker.update.return_value = np.array([[20, 20, 120, 120, 7]])
    # plate detector finds a plate inside the vehicle crop every frame
    service.plate_detector = mock.MagicMock(
        side_effect=lambda crops, **k: [_fake_plate_result([10, 10, 60, 30])
                                        for _ in crops])

    bursts = service.collect_plate_bursts(frames, min_frames=8, max_frames=32)
    assert set(bursts.keys()) == {7}
    assert len(bursts[7]) == 10
    assert all(c.ndim == 3 for c in bursts[7])


def test_collect_drops_short_tracks(service):
    frames = [_frame(40 + i) for i in range(3)]  # only 3 frames < min_frames=8
    service.vehicle_detector = mock.MagicMock(
        side_effect=lambda f, **k: _fake_vehicle_detection([20, 20, 120, 120]))
    service.tracker = mock.MagicMock()
    service.tracker.update.return_value = np.array([[20, 20, 120, 120, 7]])
    service.plate_detector = mock.MagicMock(
        side_effect=lambda crops, **k: [_fake_plate_result([10, 10, 60, 30])
                                        for _ in crops])
    bursts = service.collect_plate_bursts(frames, min_frames=8, max_frames=32)
    assert bursts == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv run pytest tests/test_collect_bursts.py -v
```
Expected: FAIL with `AttributeError: 'TrafficAnalysisService' object has no attribute 'collect_plate_bursts'`.

- [ ] **Step 3: Write the method**

In `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/utils/traffic_analysis.py`, add this method to `TrafficAnalysisService` (place it after `detect_plates_dual_ocr`). It mirrors the detect→track→plate-crop logic from `process_frame` but collects crops instead of OCR-ing:

```python
    def collect_plate_bursts(self, frames, min_frames=8, max_frames=32):
        """Group per-track plate crops across frames.

        frames: list of BGR ndarrays (already decoded video frames).
        Returns {track_id: [plate_crop_bgr, ...]} keeping only tracks with
        >= min_frames crops, each capped to max_frames evenly-spaced crops.
        """
        from utils.plate_burst import select_burst_window

        buckets = {}
        for frame in frames:
            if frame is None or frame.size == 0:
                continue
            det = self.vehicle_detector(
                frame, verbose=False, imgsz=640,
                device=self.opts.device, conf=self.opts.vconf,
            )[0]
            boxes = det.boxes
            if len(boxes) == 0:
                continue
            try:
                outputs = self.tracker.update(boxes.cpu().xyxy).astype(int)
            except Exception:
                continue
            if len(outputs) == 0:
                continue

            # crop each tracked vehicle, batch-detect plates
            tids, vcrops = [], []
            for i in range(len(outputs)):
                tid = int(outputs[i, -1])
                x1, y1, x2, y2 = outputs[i, :4]
                vcrop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0), :]
                if vcrop.size == 0:
                    continue
                tids.append(tid)
                vcrops.append(vcrop)
            if not vcrops:
                continue

            pdets = self.plate_detector(
                vcrops, verbose=False, imgsz=640,
                device=self.opts.device, conf=self.opts.pconf,
            )
            for idx, pdet in enumerate(pdets):
                plate_xyxy = pdet.boxes.xyxy
                if len(plate_xyxy) < 1:
                    continue
                pxyxy = plate_xyxy[0].cpu().numpy().astype(int)
                try:
                    plate_crop = crop_expanded_plate(pxyxy, vcrops[idx], 0.15)
                except Exception:
                    continue
                if plate_crop is None or plate_crop.size == 0:
                    continue
                buckets.setdefault(tids[idx], []).append(plate_crop)

        return {
            tid: select_burst_window(crops, max_frames)
            for tid, crops in buckets.items()
            if len(crops) >= min_frames
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv run pytest tests/test_collect_bursts.py -v
```
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && git add utils/traffic_analysis.py tests/test_collect_bursts.py && git commit -m "feat(plate): collect per-track plate crop bursts from frames"
```

---

### Task 8: `POST /predict/plates/video` endpoint (this repo)

**Files:**
- Modify: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/main.py`
- Test: `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/tests/test_video_endpoint.py`

**Interfaces:**
- Consumes: `traffic_service.collect_plate_bursts` (Task 7); `resize_burst_to_common` (Task 4); `fusion_fuse`, `FusionUnavailable` (Task 5); `_dual_ocr` (Task 6, already in main.py).
- Produces: `POST /predict/plates/video` accepting one multipart `file` (video) + query `engine` (default `mflpr2`), `scale` (default 2), `min_frames` (default 8), `max_frames` (default 32). Returns JSON list, one entry per qualifying track: `{"track_id","n_frames","engine","fast","ppocr"}`. Errors: 400 (invalid/empty video), 503 (fusion-svc down).

- [ ] **Step 1: Write the failing test**

Create `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/tests/test_video_endpoint.py`:

```python
import sys
from unittest import mock

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    sys.modules.pop("main", None)
    with mock.patch("utils.traffic_analysis.YOLO"), \
         mock.patch("utils.traffic_analysis.Sort"), \
         mock.patch("utils.traffic_analysis.DeepSort"), \
         mock.patch.object(
             __import__("utils.traffic_analysis", fromlist=["TrafficAnalysisService"]).TrafficAnalysisService,
             "_init_ocr_engine", lambda self: None), \
         mock.patch.object(
             __import__("utils.traffic_analysis", fromlist=["TrafficAnalysisService"]).TrafficAnalysisService,
             "_init_sr_engine", lambda self: None):
        import main
        # one track (id=3) with 12 crops
        crops = [np.full((32, 96, 3), 100, dtype=np.uint8) for _ in range(12)]
        main.traffic_service.collect_plate_bursts = lambda frames, min_frames, max_frames: {3: crops}
        main.traffic_service._ensure_dual_ocr = lambda: None
        main.traffic_service._ocr_plates_fpo = lambda crop: ("51A12345", 0.9)
        main.traffic_service._ocr_plates_ppocr = lambda crop: ("51A12345", 0.87)
        # decode the uploaded video into 12 dummy frames
        main._read_video_frames = lambda data: [np.zeros((200, 200, 3), np.uint8) for _ in range(12)]
        restored = np.full((64, 192, 3), 90, dtype=np.uint8)
        with mock.patch("main.fusion_fuse", return_value=restored):
            yield TestClient(main.app)
    sys.modules.pop("main", None)


def test_video_returns_per_track_results(client):
    r = client.post("/predict/plates/video?engine=mflpr2",
                    files=[("file", ("clip.mp4", b"FAKEBYTES", "video/mp4"))])
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) == 1
    entry = body[0]
    assert entry["track_id"] == 3
    assert entry["n_frames"] == 12
    assert entry["engine"] == "mflpr2"
    assert entry["fast"] == {"text": "51A12345", "confidence": 0.9}
    assert entry["ppocr"] == {"text": "51A12345", "confidence": 0.87}


def test_video_empty_decode_400(client):
    import main
    main._read_video_frames = lambda data: []
    r = client.post("/predict/plates/video",
                    files=[("file", ("clip.mp4", b"X", "video/mp4"))])
    assert r.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv run pytest tests/test_video_endpoint.py -v
```
Expected: FAIL — route 404 / `main` has no `_read_video_frames`.

- [ ] **Step 3: Add the video reader + endpoint to main.py**

In `/Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis/main.py`, add `import tempfile` near the top imports (after `import pathlib`, line 15). Then add after the multiframe endpoint (from Task 6):

```python
def _read_video_frames(data: bytes):
    """Decode video bytes into a list of BGR frames via a temp file."""
    frames = []
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    return frames


@app.post("/predict/plates/video")
def predict_plates_video(
    file: UploadFile = File(...),
    engine: str = "mflpr2",
    scale: int = 2,
    min_frames: int = 8,
    max_frames: int = 32,
):
    """Detect+track plates across a video, fuse each track's burst, dual-OCR."""
    if engine not in ("mflpr2", "eott"):
        raise HTTPException(status_code=400, detail=f"unknown engine: {engine!r}")
    data = file.file.read()
    frames = _read_video_frames(data)
    if not frames:
        raise HTTPException(status_code=400, detail="could not decode video")

    bursts = traffic_service.collect_plate_bursts(
        frames, min_frames=min_frames, max_frames=max_frames)

    results = []
    for track_id, crops in bursts.items():
        crops = resize_burst_to_common(crops)
        try:
            restored = fusion_fuse(crops, engine=engine, scale=scale)
        except FusionUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))
        fast, ppocr = _dual_ocr(restored)
        results.append({
            "track_id": int(track_id),
            "n_frames": len(crops),
            "engine": engine,
            "fast": fast,
            "ppocr": ppocr,
        })
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv run pytest tests/test_video_endpoint.py -v
```
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full this-repo suite**

Run:
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && uv run pytest -v
```
Expected: PASS — all new tests plus the pre-existing `tests/test_batch_endpoint.py` green.

- [ ] **Step 6: Commit**

```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && git add main.py tests/test_video_endpoint.py && git commit -m "feat(api): add POST /predict/plates/video"
```

---

### Task 9: End-to-end manual smoke (live services)

**Files:** none (verification only).

**Interfaces:** exercises Tasks 1–8 against running services with real weights.

- [ ] **Step 1: Start fusion-svc**

Run (background terminal A):
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/fusion-svc && uv run uvicorn fusion_svc.app:app --host 127.0.0.1 --port 8100
```
Expected: `Uvicorn running on http://127.0.0.1:8100`. Confirm health:
```bash
curl -s http://127.0.0.1:8100/health
```
Expected: `{"status":"ok"}`.

- [ ] **Step 2: Start traffic-api**

Run (background terminal B):
```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis && OCR_ENGINE=none uv run uvicorn main:app --host 127.0.0.1 --port 7862
```
Expected: startup logs, no crash. (Dual-OCR is lazy; `OCR_ENGINE=none` only skips the legacy single-OCR engine at boot.)

- [ ] **Step 2b: Write the multiframe smoke script**

Create `/Users/logan/.claude/jobs/9869af27/tmp/smoke_multiframe.py`:

```python
import subprocess, sys
from pathlib import Path

DIR = Path("/Users/logan/Developer/vibes/WORK/LIPLA/data/adnl/andan/00")
engine = sys.argv[1] if len(sys.argv) > 1 else "mflpr2"
imgs = sorted(p for p in DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:10]
args = ["curl", "-s", "-X", "POST",
        f"http://127.0.0.1:7862/predict/plates/multiframe?engine={engine}&scale=2"]
for p in imgs:
    args += ["-F", f"files=@{p};type=image/jpeg"]
print(f"engine={engine}, {len(imgs)} crops")
subprocess.run(args)
print()
```

> Note: `data/adnl/andan/00` holds full plate-region images; they stand in as a burst for smoke purposes (the endpoint fuses whatever crops it is given). Real bursts are same-plate sequences.

- [ ] **Step 3: Smoke the multiframe endpoint (both engines)**

Run:
```bash
uv run /Users/logan/.claude/jobs/9869af27/tmp/smoke_multiframe.py mflpr2
uv run /Users/logan/.claude/jobs/9869af27/tmp/smoke_multiframe.py eott
```
Expected: each prints JSON with `engine`, `frames_used: 10`, and `fast`/`ppocr` objects containing `text` + `confidence`. Confirm no 500/503.

- [ ] **Step 4: Smoke the video endpoint**

Run:
```bash
curl -s -X POST "http://127.0.0.1:7862/predict/plates/video?engine=mflpr2&min_frames=8" \
  -F "file=@/Users/logan/Developer/vibes/WORK/LIPLA/data/overhead/IMG_2079.mov;type=video/mp4" | head -c 2000
```
Expected: a JSON list. May be `[]` if no plate track reaches `min_frames=8` in that overhead clip — that is a valid result, not an error. If `[]`, re-run with a plate-centric clip or lower `--min_frames` via `?min_frames=3` to confirm the path produces per-track entries.

- [ ] **Step 5: Record outcome**

Confirm: both engines return OCR JSON on the multiframe path, video path returns a list (entries or empty), no unhandled 500s. Stop both uvicorn processes. No commit (verification task).

---

## Self-Review

**1. Spec coverage:**
- 3-process architecture (traffic-api → fusion-svc) → Tasks 1–3 (fusion-svc), 5–8 (this repo). ✓
- Unified light fusion service, both engines, image-only → Tasks 1–3; no torch/paddle pulled (mf-lpr2 restore-only, eott reconstruct-only). ✓
- `POST /fuse?engine=` contract returning PNG → Task 3. ✓
- `/predict/plates/multiframe` (N crops → fuse → dual-OCR) → Task 6. ✓
- `/predict/plates/video` (video → track bursts → fuse → dual-OCR) → Tasks 7–8. ✓
- Single OCR source of truth (reuse `_ocr_plates_fpo`/`_ocr_plates_ppocr`) → `_dual_ocr` in Task 6, reused in Task 8. ✓
- Both engines selectable + comparable → `engine` param on both endpoints + fusion-svc. ✓
- Error handling: fusion down→503 (Tasks 6, 8), unknown engine→400 (Tasks 3, 6, 8), size mismatch→resize (Task 4), short track→skipped (Task 7), OCR empty→`("",0.0)` (existing). ✓
- Testing strategy: adapter unit tests (1–2), endpoint tests (3, 6, 8), util tests (4), client mock (5), collector mock (7), manual smoke (9). ✓
- Color contract BGR end-to-end, RGB only inside mf-lpr2 adapter → Task 1. ✓

**2. Placeholder scan:** No TBD/TODO. Every code step shows full code. Every test shows real assertions. No "similar to Task N". ✓

**3. Type consistency:**
- `fuse_mflpr2(frames, scale) -> np.ndarray` / `fuse_eott(frames, scale) -> np.ndarray` — defined Tasks 1–2, dispatched in Task 3's `_ENGINES`. ✓
- `fuse(crops, engine, scale, base_url, timeout) -> np.ndarray` + `FusionUnavailable` — defined Task 5, imported as `fusion_fuse`/`FusionUnavailable` in Tasks 6, 8. ✓
- `resize_burst_to_common(crops, target=None)` / `select_burst_window(crops, max_frames)` — defined Task 4, used Tasks 6, 7, 8. ✓
- `collect_plate_bursts(frames, min_frames, max_frames) -> dict[int, list]` — defined Task 7, called Task 8 with keyword args matching. ✓
- `_dual_ocr(crop) -> (fast_dict, ppocr_dict)` — defined Task 6, reused Task 8. ✓
- OCR helpers return `(str, float)` (verified) → wrapped into `{"text","confidence"}` consistently. ✓

No gaps found.

## Sequencing notes

- Tasks 1–3 (fusion-svc) are independent of Tasks 4–8 and can be built/tested first with no AI-Traffic-Analysis changes.
- Tasks 4–8 only need fusion-svc's HTTP contract (mocked in unit tests), so they do not require fusion-svc running until Task 9.
- Task 9 is the only step needing both services live with real weights.

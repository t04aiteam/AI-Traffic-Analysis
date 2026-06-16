# Batch Vehicle Detection API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /predict/batch` endpoint that accepts multiple images and returns annotated
JPEGs (bboxes + confidence) — single image as `image/jpeg`, multiple as `application/zip`.

**Architecture:** Add `detect_vehicles_only(frame)` to `TrafficAnalysisService` — runs only the
vehicle YOLO detector, draws bboxes and confidence labels, returns annotated frame with no
tracker/OCR mutation (concurrent-safe). The new endpoint decodes each uploaded file, calls this
method, re-encodes to JPEG, then responds with either a raw JPEG or an in-memory zip.

**Tech Stack:** FastAPI, OpenCV, NumPy, ultralytics YOLO, Python stdlib (`io`, `zipfile`,
`datetime`), pytest + starlette TestClient

---

## File Map

| Action | Path | Change |
|---|---|---|
| Modify | `utils/traffic_analysis.py` | Add `detect_vehicles_only(frame)` after line 392 |
| Modify | `main.py` | Add 3 stdlib imports (line 12); add `POST /predict/batch` before `if __name__` |
| Create | `tests/test_batch_endpoint.py` | All tests for the new method and endpoint |

---

## Task 1: `detect_vehicles_only` method (TDD)

**Files:**
- Modify: `utils/traffic_analysis.py:390-392` (append after `process_image`)
- Create: `tests/test_batch_endpoint.py`

- [ ] **Step 1: Create the test file with failing unit tests**

```python
# tests/test_batch_endpoint.py
import io
import zipfile

import cv2
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _black_jpeg(h: int = 64, w: int = 64) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _make_service():
    """TrafficAnalysisService with all heavy init (YOLO/OCR/SR) mocked out."""
    from utils.traffic_analysis import TrafficAnalysisService

    with (
        patch("utils.traffic_analysis.YOLO"),
        patch("utils.traffic_analysis.Sort"),
        patch("utils.traffic_analysis.DeepSort"),
        patch.object(TrafficAnalysisService, "_init_ocr_engine"),
        patch.object(TrafficAnalysisService, "_init_sr_engine"),
    ):
        opts = SimpleNamespace(
            vehicle_weight="v.pt",
            plate_weight="p.pt",
            dsort_weight="d.t7",
            vconf=0.6,
            pconf=0.25,
            ocr_thres=0.9,
            ocr_engine="none",
            fpo_model="x",
            sr_engine="none",
            sr_scale=2,
            realesrgan_weight=None,
            device="cpu",
            deepsort=False,
            read_plate=False,
            lang="en",
        )
        svc = TrafficAnalysisService(opts)
    return svc


# ---------------------------------------------------------------------------
# Unit: detect_vehicles_only
# ---------------------------------------------------------------------------

class TestDetectVehiclesOnly:
    def test_empty_frame_returns_same_object(self):
        svc = _make_service()
        frame = np.zeros((0, 0, 3), dtype=np.uint8)
        out = svc.detect_vehicles_only(frame)
        assert out is frame

    def test_no_detections_returns_same_shape(self):
        svc = _make_service()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_result = MagicMock()
        mock_result.boxes = []
        svc.vehicle_detector = MagicMock(return_value=[mock_result])
        out = svc.detect_vehicles_only(frame)
        assert isinstance(out, np.ndarray)
        assert out.shape == frame.shape

    def test_detection_draws_onto_frame(self):
        svc = _make_service()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        mock_box = MagicMock()
        mock_box.xyxy.__getitem__.return_value.cpu.return_value.numpy.return_value = (
            np.array([10.0, 10.0, 100.0, 100.0])
        )
        mock_box.conf.__getitem__.return_value = np.float32(0.85)
        mock_box.cls.__getitem__.return_value = np.int64(0)

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        svc.vehicle_detector = MagicMock(return_value=[mock_result])

        out = svc.detect_vehicles_only(frame)
        # Pixels drawn by cv2.rectangle differ from the black input
        assert not np.array_equal(out, frame)
```

- [ ] **Step 2: Run tests — expect failure (method doesn't exist yet)**

```bash
cd /Users/logan/Developer/vibes/WORK/LIPLA/AI-Traffic-Analysis
uv run pytest tests/test_batch_endpoint.py::TestDetectVehiclesOnly -v
```

Expected: `AttributeError: 'TrafficAnalysisService' object has no attribute 'detect_vehicles_only'`

- [ ] **Step 3: Implement `detect_vehicles_only` — append at end of `utils/traffic_analysis.py`**

Open `utils/traffic_analysis.py`. After `process_image` (line 390), append:

```python
    def detect_vehicles_only(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        annotated = frame.copy()
        detection = self.vehicle_detector(
            frame,
            verbose=False,
            imgsz=640,
            device=self.opts.device,
            conf=self.opts.vconf,
        )[0]

        label_lookup = VEHICLES.get(self.lang, VEHICLES.get("en", []))

        for box in detection.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            try:
                label = map_label(cls_idx, label_lookup)
            except Exception:
                label = str(cls_idx)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return annotated
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/test_batch_endpoint.py::TestDetectVehiclesOnly -v
```

Expected output:
```
PASSED tests/test_batch_endpoint.py::TestDetectVehiclesOnly::test_empty_frame_returns_same_object
PASSED tests/test_batch_endpoint.py::TestDetectVehiclesOnly::test_no_detections_returns_same_shape
PASSED tests/test_batch_endpoint.py::TestDetectVehiclesOnly::test_detection_draws_onto_frame
```

- [ ] **Step 5: Commit**

```bash
git add utils/traffic_analysis.py tests/test_batch_endpoint.py
git commit -m "feat(detection): add detect_vehicles_only method to TrafficAnalysisService"
```

---

## Task 2: `POST /predict/batch` endpoint (TDD)

**Files:**
- Modify: `main.py:5-12` (add imports), `main.py:270` (add endpoint before `if __name__`)
- Modify: `tests/test_batch_endpoint.py` (add endpoint tests)

- [ ] **Step 1: Add endpoint tests to test file**

Append to `tests/test_batch_endpoint.py`:

```python
# ---------------------------------------------------------------------------
# Integration: POST /predict/batch
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    import main
    # Replace the real service method with a no-op that returns the frame unchanged
    main.traffic_service.detect_vehicles_only = MagicMock(side_effect=lambda f: f)
    from fastapi.testclient import TestClient
    return TestClient(main.app)


class TestPredictBatch:
    def test_single_valid_image_returns_jpeg(self, client):
        jpeg = _black_jpeg()
        resp = client.post(
            "/predict/batch",
            files=[("files", ("img.jpg", jpeg, "image/jpeg"))],
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        # Response must be a valid JPEG
        arr = np.frombuffer(resp.content, np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert decoded is not None

    def test_multiple_valid_images_returns_zip(self, client):
        jpeg = _black_jpeg()
        files = [
            ("files", ("a.jpg", jpeg, "image/jpeg")),
            ("files", ("b.jpg", jpeg, "image/jpeg")),
        ]
        resp = client.post("/predict/batch", files=files)
        assert resp.status_code == 200
        assert "zip" in resp.headers["content-type"]
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            assert len(zf.namelist()) == 2
            assert "a.jpg" in zf.namelist()
            assert "b.jpg" in zf.namelist()

    def test_all_invalid_files_returns_400(self, client):
        resp = client.post(
            "/predict/batch",
            files=[("files", ("bad.jpg", b"not_an_image", "image/jpeg"))],
        )
        assert resp.status_code == 400

    def test_mixed_valid_and_invalid_skips_bad_file(self, client):
        jpeg = _black_jpeg()
        files = [
            ("files", ("good.jpg", jpeg, "image/jpeg")),
            ("files", ("bad.jpg", b"not_an_image", "image/jpeg")),
        ]
        resp = client.post("/predict/batch", files=files)
        # 1 valid file → jpeg, not zip
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

    def test_zip_filename_contains_annotated(self, client):
        jpeg = _black_jpeg()
        files = [
            ("files", ("x.jpg", jpeg, "image/jpeg")),
            ("files", ("y.jpg", jpeg, "image/jpeg")),
        ]
        resp = client.post("/predict/batch", files=files)
        disposition = resp.headers.get("content-disposition", "")
        assert "annotated_" in disposition
```

- [ ] **Step 2: Run tests — expect failure (endpoint missing)**

```bash
uv run pytest tests/test_batch_endpoint.py::TestPredictBatch -v
```

Expected: `404 Not Found` on all endpoint tests.

- [ ] **Step 3: Add stdlib imports to `main.py` (lines 5-12 block)**

In `main.py`, change the imports block from:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from types import SimpleNamespace
import os
```

to:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import io
import numpy as np
from datetime import datetime
from types import SimpleNamespace
import os
import zipfile
```

- [ ] **Step 4: Add `POST /predict/batch` endpoint to `main.py`**

In `main.py`, before the `if __name__ == "__main__":` block (around line 270), add:

```python
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Detect vehicles in one or more images and return annotated output.

    Single image: returns image/jpeg.
    Multiple images: returns application/zip with one annotated JPEG per input file.
    Invalid files are skipped; if all files are invalid, returns HTTP 400.
    """
    results: list[tuple[str, bytes]] = []

    for upload in files:
        contents = await upload.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        try:
            annotated = traffic_service.detect_vehicles_only(frame)
            ok, buf = cv2.imencode(".jpg", annotated)
            if not ok:
                continue
            name = upload.filename or f"image_{len(results)}.jpg"
            if not name.lower().endswith((".jpg", ".jpeg")):
                base = name.rsplit(".", 1)[0] if "." in name else name
                name = base + ".jpg"
            results.append((name, buf.tobytes()))
        except Exception:
            continue

    if not results:
        raise HTTPException(status_code=400, detail="No valid images in batch")

    if len(results) == 1:
        return StreamingResponse(io.BytesIO(results[0][1]), media_type="image/jpeg")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in results:
            zf.writestr(name, data)
    zip_buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="annotated_{ts}.zip"'},
    )
```

- [ ] **Step 5: Run all tests — expect pass**

```bash
uv run pytest tests/test_batch_endpoint.py -v
```

Expected:
```
PASSED tests/test_batch_endpoint.py::TestDetectVehiclesOnly::test_empty_frame_returns_same_object
PASSED tests/test_batch_endpoint.py::TestDetectVehiclesOnly::test_no_detections_returns_same_shape
PASSED tests/test_batch_endpoint.py::TestDetectVehiclesOnly::test_detection_draws_onto_frame
PASSED tests/test_batch_endpoint.py::TestPredictBatch::test_single_valid_image_returns_jpeg
PASSED tests/test_batch_endpoint.py::TestPredictBatch::test_multiple_valid_images_returns_zip
PASSED tests/test_batch_endpoint.py::TestPredictBatch::test_all_invalid_files_returns_400
PASSED tests/test_batch_endpoint.py::TestPredictBatch::test_mixed_valid_and_invalid_skips_bad_file
PASSED tests/test_batch_endpoint.py::TestPredictBatch::test_zip_filename_contains_annotated
```

- [ ] **Step 6: Verify `/predict/batch` appears in route list**

```bash
uv run python -c "import main; print([r.path for r in main.app.routes])"
```

Expected: output includes `'/predict/batch'`

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_batch_endpoint.py
git commit -m "feat(api): add POST /predict/batch — annotated JPEG or ZIP response"
```

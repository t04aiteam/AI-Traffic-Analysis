"""
Copyright (C) 2023 TonAI
Plate OCR API
=============

Standalone FastAPI service that combines:

  * Plate detector  : ``weights/plate/plate_yolo12n_640_2025.pt`` (Ultralytics YOLO)
  * Plate OCR        : fast-plate-ocr ``cct-s-v2-global-model`` (CCT-S v2, ONNX)

Given an image it detects license plates, crops each one (with a small
expansion margin) and reads the plate text with the CCT-S OCR model.

This service is intentionally separate from ``main.py`` (the full
vehicle-detection + tracking + PaddleOCR pipeline). It only does
plate-detect -> crop -> OCR, which is faster and uses the more accurate
fast-plate-ocr recognizer.

Run:
    uv run uvicorn plate_ocr_api:app --host 0.0.0.0 --port 7863
or:
    uv run python plate_ocr_api.py
"""

import io
import os
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

from fast_plate_ocr import LicensePlateRecognizer

from utils.utils import check_legit_plate, crop_expanded_plate

# --------------------------------------------------------------------------- #
# Configuration (all overridable via environment variables)
# --------------------------------------------------------------------------- #
CFG = SimpleNamespace(
    plate_weight=os.environ.get("PLATE_WEIGHT", "weights/plate/plate_yolo12n_640_2025.pt"),
    ocr_model=os.environ.get("OCR_MODEL", "cct-s-v2-global-model"),
    # Ultralytics device string: "cpu", "mps", "cuda", "0", ...
    yolo_device=os.environ.get("YOLO_DEVICE", "cpu"),
    # fast-plate-ocr device: "cpu", "cuda" or "auto" (picks onnxruntime providers)
    ocr_device=os.environ.get("OCR_DEVICE", "auto"),
    imgsz=int(os.environ.get("PLATE_IMGSZ", "640")),
    pconf=float(os.environ.get("PLATE_CONF", "0.25")),
    expand_ratio=float(os.environ.get("PLATE_EXPAND", "0.15")),
)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _load_plate_detector() -> YOLO:
    model = YOLO(CFG.plate_weight, task="detect")
    try:
        model.to(CFG.yolo_device)
    except Exception:
        # Fall back silently to whatever device YOLO defaults to.
        pass
    return model


plate_detector = _load_plate_detector()
ocr = LicensePlateRecognizer(hub_ocr_model=CFG.ocr_model, device=CFG.ocr_device)
# Colour mode the OCR model expects for raw numpy crops ("grayscale" or "rgb").
OCR_COLOR_MODE = ocr.config.image_color_mode


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Plate OCR Service",
    description="License plate detection (YOLO12n) + OCR (fast-plate-ocr CCT-S v2)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #
class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PlateResult(BaseModel):
    bbox: BBox
    det_conf: float
    text: str
    ocr_conf: Optional[float] = None
    region: Optional[str] = None
    legit: bool = False


class PredictResponse(BaseModel):
    plates: List[PlateResult]
    count: int


class HealthResponse(BaseModel):
    status: str
    plate_weight: str
    ocr_model: str
    ocr_color_mode: str
    models_loaded: bool


# --------------------------------------------------------------------------- #
# Core inference helpers
# --------------------------------------------------------------------------- #
def _decode_image(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def _to_ocr_color(crop_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR crop into the colour mode the OCR model expects.

    For numpy inputs fast-plate-ocr does NOT convert channels itself, so we do
    it here based on the loaded model's config.
    """
    if OCR_COLOR_MODE == "rgb":
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    # grayscale -> (H, W); resize_image adds the channel axis later.
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)


def _detect_plates(img_bgr: np.ndarray):
    """Run the YOLO plate detector. Returns (xyxy_int_list, conf_list)."""
    results = plate_detector(
        img_bgr,
        imgsz=CFG.imgsz,
        conf=CFG.pconf,
        device=CFG.yolo_device,
        verbose=False,
    )
    boxes_xyxy: List[np.ndarray] = []
    confs: List[float] = []
    for det in results:
        if det.boxes is None or len(det.boxes) == 0:
            continue
        xyxy = det.boxes.xyxy.cpu().numpy()
        cf = det.boxes.conf.cpu().numpy()
        for box, c in zip(xyxy, cf):
            boxes_xyxy.append(box.astype(int))
            confs.append(float(c))
    return boxes_xyxy, confs


def _recognize(img_bgr: np.ndarray) -> List[PlateResult]:
    boxes_xyxy, confs = _detect_plates(img_bgr)
    if not boxes_xyxy:
        return []

    crops = []
    valid_idx = []
    for i, box in enumerate(boxes_xyxy):
        crop = crop_expanded_plate(box, img_bgr, CFG.expand_ratio)
        if crop is None or crop.size == 0:
            continue
        crops.append(_to_ocr_color(crop))
        valid_idx.append(i)

    if not crops:
        return []

    preds = ocr.run(crops, return_confidence=True)

    plates: List[PlateResult] = []
    for pred, i in zip(preds, valid_idx):
        box = boxes_xyxy[i]
        ocr_conf = (
            float(np.mean(pred.char_probs)) if pred.char_probs is not None else None
        )
        text = pred.plate.strip() if pred.plate else ""
        plates.append(
            PlateResult(
                bbox=BBox(x1=float(box[0]), y1=float(box[1]),
                          x2=float(box[2]), y2=float(box[3])),
                det_conf=confs[i],
                text=text,
                ocr_conf=ocr_conf,
                region=pred.region,
                legit=check_legit_plate(text) if text else False,
            )
        )
    return plates


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
async def root():
    return {
        "service": "Plate OCR - YOLO12n detector + fast-plate-ocr CCT-S v2",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check (GET)",
            "/config": "Current configuration (GET)",
            "/predict": "Detect + OCR plates, JSON result (POST image)",
            "/predict/annotated": "Detect + OCR, return annotated JPEG (POST image)",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        plate_weight=CFG.plate_weight,
        ocr_model=CFG.ocr_model,
        ocr_color_mode=OCR_COLOR_MODE,
        models_loaded=True,
    )


@app.get("/config")
async def get_config():
    return {
        "plate_weight": CFG.plate_weight,
        "ocr_model": CFG.ocr_model,
        "ocr_color_mode": OCR_COLOR_MODE,
        "yolo_device": CFG.yolo_device,
        "ocr_device": CFG.ocr_device,
        "imgsz": CFG.imgsz,
        "plate_conf": CFG.pconf,
        "expand_ratio": CFG.expand_ratio,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """Detect plates in an image and OCR each one. Returns JSON."""
    try:
        img = _decode_image(await file.read())
        plates = _recognize(img)
        return PredictResponse(plates=plates, count=len(plates))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)):
    """Detect plates + OCR, return the image with boxes and text drawn (JPEG)."""
    try:
        img = _decode_image(await file.read())
        plates = _recognize(img)
        for p in plates:
            x1, y1, x2, y2 = int(p.bbox.x1), int(p.bbox.y1), int(p.bbox.x2), int(p.bbox.y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = p.text if p.text else "?"
            cv2.putText(img, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "7863")),
        log_level="info",
    )

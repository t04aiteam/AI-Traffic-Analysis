"""
Copyright (C) 2023 TonAI
Traffic AI Service - FastAPI Backend for License Plate Recognition
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import io
import logging
import numpy as np
import os
import pathlib
import zipfile
from datetime import datetime, timezone
from types import SimpleNamespace

logger = logging.getLogger(__name__)

from utils.traffic_analysis import TrafficAnalysisService
from utils.fusion_client import fuse as fusion_fuse, FusionUnavailable
from utils.plate_burst import resize_burst_to_common, select_burst_window

# Initialize FastAPI
app = FastAPI(
    title="Traffic AI Service",
    description="License Plate Recognition API for vehicle detection and tracking",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default device configuration
DEFAULT_DEVICE = os.environ.get("ALPR_DEVICE", "auto")

# Initialize ALPR Core
opts = SimpleNamespace(
    vehicle_weight=os.environ.get("VEHICLE_WEIGHT", "weights/vehicle/vehicle_yolov9s_640_30oct2025.pt"),
    plate_weight=os.environ.get("PLATE_WEIGHT", "weights/plate/plate_yolov8n_320_2024.pt"),
    dsort_weight=os.environ.get("DSORT_WEIGHT", "weights/tracking/deepsort/ckpt.t7"),
    vconf=float(os.environ.get("VEHICLE_CONF", "0.6")),
    pconf=float(os.environ.get("PLATE_CONF", "0.25")),
    ocr_thres=float(os.environ.get("OCR_THRESHOLD", "0.9")),
    ocr_engine=os.environ.get("OCR_ENGINE", "paddle").strip().lower(),
    fpo_model=os.environ.get("FPO_MODEL", "cct-s-v2-global-model"),
    sr_engine=os.environ.get("SR_ENGINE", "none").strip().lower(),
    sr_scale=int(os.environ.get("SR_SCALE", "2")),
    realesrgan_weight=os.environ.get("REALESRGAN_WEIGHT", None),
    device=DEFAULT_DEVICE,
    deepsort=os.environ.get("USE_DEEPSORT", "false").lower() == "true",
    read_plate=True,
    lang=os.environ.get("LANG", "en"),
)

traffic_service = TrafficAnalysisService(opts)


# Pydantic models for API responses
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class VehicleDetection(BaseModel):
    track_id: int
    bbox: BoundingBox
    vehicle_type: Optional[str] = None
    license_plate: Optional[str] = None
    plate_bbox: Optional[BoundingBox] = None
    confidence: Optional[float] = None


class PredictionResponse(BaseModel):
    detections: List[VehicleDetection]
    frame_count: Optional[int] = None


class OCRResult(BaseModel):
    text: str
    confidence: float


class PlateDetectionResult(BaseModel):
    bbox: BoundingBox
    confidence: float
    fpo: OCRResult
    ppocr: OCRResult


class PlatesImageResult(BaseModel):
    filename: str
    plates: List[PlateDetectionResult]


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Traffic AI - License Plate Recognition",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict/image": "Process single image (POST)",
            "/predict/frame": "Process video frame (POST)",
            "/reset": "Reset tracker (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        device=str(traffic_service.opts.device),
        models_loaded=True
    )


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Process a single image and return vehicle detections with license plates
    
    Args:
        file: Image file (jpg, png, etc.)
        
    Returns:
        JSON with list of detections containing track_id, bbox, vehicle_type, license_plate
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Reset tracker for single image processing
        traffic_service.reset()
        
        # Process frame (without drawing)
        _ = traffic_service.process_frame(img)
        
        # Extract detections
        detections = []
        for track_id, vehicle in traffic_service.vehicles.items():
            bbox_xyxy = vehicle.bbox_xyxy
            
            detection = VehicleDetection(
                track_id=track_id,
                bbox=BoundingBox(
                    x1=float(bbox_xyxy[0]),
                    y1=float(bbox_xyxy[1]),
                    x2=float(bbox_xyxy[2]),
                    y2=float(bbox_xyxy[3])
                ),
                vehicle_type=vehicle.vehicle_type if vehicle.vehicle_type else None,
                license_plate=vehicle.plate_number if vehicle.plate_number else None,
                confidence=float(vehicle.ocr_conf) if vehicle.ocr_conf > 0 else None
            )
            
            # Add plate bbox if available
            if vehicle.license_plate_bbox is not None:
                plate_bbox = vehicle.license_plate_bbox
                detection.plate_bbox = BoundingBox(
                    x1=float(plate_bbox[0]),
                    y1=float(plate_bbox[1]),
                    x2=float(plate_bbox[2]),
                    y2=float(plate_bbox[3])
                )
            
            detections.append(detection)
        
        return PredictionResponse(detections=detections)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/predict/frame", response_model=PredictionResponse)
async def predict_frame(file: UploadFile = File(...), frame_number: Optional[int] = None):
    """
    Process a video frame and return vehicle detections with license plates.
    Maintains tracking state across frames.
    
    Args:
        file: Image file representing a video frame
        frame_number: Optional frame number for reference
        
    Returns:
        JSON with list of detections containing track_id, bbox, vehicle_type, license_plate
    """
    try:
        # Read frame
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame")
        
        # Process frame (maintains tracking state)
        _ = traffic_service.process_frame(frame)
        
        # Extract detections
        detections = []
        for track_id, vehicle in traffic_service.vehicles.items():
            bbox_xyxy = vehicle.bbox_xyxy
            
            detection = VehicleDetection(
                track_id=track_id,
                bbox=BoundingBox(
                    x1=float(bbox_xyxy[0]),
                    y1=float(bbox_xyxy[1]),
                    x2=float(bbox_xyxy[2]),
                    y2=float(bbox_xyxy[3])
                ),
                vehicle_type=vehicle.vehicle_type if vehicle.vehicle_type else None,
                license_plate=vehicle.plate_number if vehicle.plate_number else None,
                confidence=float(vehicle.ocr_conf) if vehicle.ocr_conf > 0 else None
            )
            
            # Add plate bbox if available
            if vehicle.license_plate_bbox is not None:
                plate_bbox = vehicle.license_plate_bbox
                detection.plate_bbox = BoundingBox(
                    x1=float(plate_bbox[0]),
                    y1=float(plate_bbox[1]),
                    x2=float(plate_bbox[2]),
                    y2=float(plate_bbox[3])
                )
            
            detections.append(detection)
        
        return PredictionResponse(
            detections=detections,
            frame_count=frame_number
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/reset")
async def reset_tracker():
    """
    Reset the tracking state. Useful when starting a new video or stream.
    """
    try:
        traffic_service.reset()
        return {"status": "success", "message": "Tracker reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "vehicle_weight": traffic_service.opts.vehicle_weight,
        "plate_weight": traffic_service.opts.plate_weight,
        "device": str(traffic_service.opts.device),
        "vehicle_confidence": traffic_service.opts.vconf,
        "plate_confidence": traffic_service.opts.pconf,
        "ocr_threshold": traffic_service.ocr_thres,
        "ocr_engine": getattr(traffic_service, "ocr_engine_name", "paddle"),
        "sr_engine": getattr(traffic_service, "sr_engine_name", "none"),
        "sr_scale": getattr(traffic_service, "sr_scale", 2),
        "use_deepsort": traffic_service.deepsort,
        "read_plate": traffic_service.read_plate,
        "language": traffic_service.lang
    }


@app.post("/predict/batch")
def predict_batch(files: List[UploadFile] = File(...)):
    """
    Detect vehicles in one or more images and return annotated output.

    Single image: returns image/jpeg.
    Multiple images: returns application/zip with one annotated JPEG per input file.
    Invalid files are skipped; if all files are invalid, returns HTTP 400.
    FastAPI runs plain `def` handlers in a threadpool, keeping the event loop free
    during blocking YOLO inference.
    """
    results: list[tuple[str, bytes]] = []

    for upload in files:
        contents = upload.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        try:
            annotated = traffic_service.detect_vehicles_only(frame)
            ok, buf = cv2.imencode(".jpg", annotated)
            if not ok:
                continue
            # Sanitise filename: PurePosixPath strips both / and \ traversal sequences
            raw_name = (
                pathlib.PurePosixPath(
                    (upload.filename or "").replace("\\", "/")
                ).name
                or f"image_{len(results)}.jpg"
            )
            if not raw_name.lower().endswith((".jpg", ".jpeg")):
                base = raw_name.rsplit(".", 1)[0] if "." in raw_name else raw_name
                raw_name = base + ".jpg"
            results.append((raw_name, buf.tobytes()))
        except Exception as e:
            logger.warning("Skipping %s: %s", upload.filename, e)
            continue

    if not results:
        raise HTTPException(status_code=400, detail="No valid images in batch")

    if len(results) == 1:
        return StreamingResponse(io.BytesIO(results[0][1]), media_type="image/jpeg")

    zip_buf = io.BytesIO()
    seen: dict[str, int] = {}
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for raw_name, data in results:
            n = seen.get(raw_name, 0)
            seen[raw_name] = n + 1
            entry = raw_name if n == 0 else f"{raw_name.rsplit('.', 1)[0]}_{n}.jpg"
            zf.writestr(entry, data)
    zip_buf.seek(0)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="annotated_{ts}.zip"'},
    )


@app.post("/predict/plates/batch")
def predict_plates_batch(files: List[UploadFile] = File(...)):
    """
    Detect license plates in one or more images and return annotated output.

    Each plate crop is run through fast-plate-ocr and PPOCRv6-medium. The
    full image is returned annotated with plate bboxes and both OCR results
    as text labels.

    Single image → image/jpeg. Multiple images → application/zip.
    Invalid images are skipped; all invalid → HTTP 400.
    Both OCR engines are lazily initialised on the first call.
    """
    results: list[tuple[str, bytes]] = []

    for upload in files:
        contents = upload.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        raw_name = (
            pathlib.PurePosixPath(
                (upload.filename or "").replace("\\", "/")
            ).name or f"image_{len(results)}.jpg"
        )
        if not raw_name.lower().endswith((".jpg", ".jpeg")):
            base = raw_name.rsplit(".", 1)[0] if "." in raw_name else raw_name
            raw_name = base + ".jpg"
        if frame is None:
            continue
        try:
            plates = traffic_service.detect_plates_dual_ocr(frame)
            annotated = frame.copy()
            for p in plates:
                b = p["bbox"]
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 100, 0), 2)
                fpo_text = p["fpo"]["text"] or "-"
                fpo_conf = p["fpo"]["confidence"]
                ppo_text = p["ppocr"]["text"] or "-"
                ppo_conf = p["ppocr"]["confidence"]
                label1 = f"FAST: {fpo_text} ({fpo_conf:.2f})"
                label2 = f"PPO: {ppo_text} ({ppo_conf:.2f})"
                cy = max(y1 - 22, 0)
                cv2.putText(annotated, label1, (x1, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)
                cv2.putText(annotated, label2, (x1, max(cy - 18, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)
            ok, buf = cv2.imencode(".jpg", annotated)
            if not ok:
                continue
            results.append((raw_name, buf.tobytes()))
        except Exception as e:
            logger.warning("Skipping %s: %s", upload.filename, e)
            continue

    if not results:
        raise HTTPException(status_code=400, detail="No valid images in batch")

    if len(results) == 1:
        return StreamingResponse(io.BytesIO(results[0][1]), media_type="image/jpeg")

    zip_buf = io.BytesIO()
    seen: dict[str, int] = {}
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for raw_name, data in results:
            n = seen.get(raw_name, 0)
            seen[raw_name] = n + 1
            entry = raw_name if n == 0 else f"{raw_name.rsplit('.', 1)[0]}_{n}.jpg"
            zf.writestr(entry, data)
    zip_buf.seek(0)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="plates_{ts}.zip"'},
    )


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "7862")),
        log_level="info"
    )

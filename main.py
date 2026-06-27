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
import tempfile
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


_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
_VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _expand_one(name, raw):
    """Yield (source_name, kind, payload) for one file. kind: 'image' (ndarray)
    or 'video' (list of frames). Zips are extracted recursively; unknown
    extensions are sniffed by trying image- then video-decode."""
    ext = os.path.splitext(name)[1].lower()
    if ext == ".zip":
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
        except Exception:
            return
        for n in zf.namelist():
            if n.endswith("/"):
                continue
            yield from _expand_one(n, zf.read(n))
        return
    if ext in _IMG_EXT or ext not in _VID_EXT:          # image, or sniff unknown
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            yield (name, "image", arr)
            return
        if ext in _IMG_EXT:                              # named image but undecodable
            return
    if ext in _VID_EXT or ext not in _IMG_EXT:           # video, or sniff unknown
        frames = _read_video_frames(raw)
        if frames:
            yield (name, "video", frames)


def _expand_inputs(uploads):
    for up in uploads:
        yield from _expand_one(up.filename or "file", up.file.read())


def _vehicle_entry(track_id, vehicle):
    b = vehicle.bbox_xyxy
    entry = {
        "track_id": int(track_id),
        "vehicle_type": vehicle.vehicle_type or None,
        "license_plate": vehicle.plate_number or None,
        "confidence": float(vehicle.ocr_conf) if vehicle.ocr_conf > 0 else None,
        "bbox": {"x1": float(b[0]), "y1": float(b[1]),
                 "x2": float(b[2]), "y2": float(b[3])},
        "plate_bbox": None,
    }
    if vehicle.license_plate_bbox is not None:
        pb = vehicle.license_plate_bbox
        entry["plate_bbox"] = {"x1": float(pb[0]), "y1": float(pb[1]),
                               "x2": float(pb[2]), "y2": float(pb[3])}
    return entry


def _detections_image(img):
    traffic_service.reset()
    traffic_service.process_frame(img)
    return [_vehicle_entry(tid, v) for tid, v in traffic_service.vehicles.items()]


def _tracks_video(name, frames, stride):
    traffic_service.reset()
    tracks: dict[int, dict] = {}
    for idx in range(0, len(frames), stride):
        traffic_service.process_frame(frames[idx])
        for tid, v in traffic_service.vehicles.items():
            e = tracks.setdefault(int(tid), {"frames_seen": 0})
            e.update(_vehicle_entry(tid, v))
            e["frames_seen"] = e.get("frames_seen", 0) + 1
    return {"source": name, "kind": "video", "n_frames": len(frames),
            "stride": stride, "tracks": list(tracks.values())}


def _annotate_video(frames, stride):
    """Annotate (detect) every Nth frame, write an mp4, return its bytes."""
    h, w = frames[0].shape[:2]
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    try:
        for idx in range(0, len(frames), stride):
            vw.write(traffic_service.detect_vehicles_only(frames[idx]))
    finally:
        vw.release()
    with open(out_path, "rb") as fh:
        data = fh.read()
    os.unlink(out_path)
    return data


@app.post("/predict/batch")
def predict_batch(
    files: List[UploadFile] = File(...),
    format: str = "media",
    frame_stride: int = 1,
):
    """Detect vehicles across one or more files of mixed type → annotated media or JSON.

    Accepts **multiple files** of any mix of: images (jpg/png/bmp/webp/tif/gif/...),
    videos (mp4/avi/mov/mkv/webm/m4v), and **zips** (extracted recursively).
    Unknown extensions are sniffed (image-decode, then video-decode).

    `format=media` (default): annotated JPEG per image, annotated MP4 per video.
      one output → that file; multiple → a zip.
    `format=json`: `{results:[{source, kind, detections|tracks}]}` — no images.
    `frame_stride`: process every Nth video frame (speed vs coverage).
    """
    if format not in ("media", "json"):
        raise HTTPException(status_code=400, detail="format must be 'media' or 'json'")
    stride = max(1, frame_stride)

    inputs = list(_expand_inputs(files))
    if not inputs:
        raise HTTPException(status_code=400, detail="no decodable images/videos in upload")

    if format == "json":
        results = []
        for name, kind, payload in inputs:
            if kind == "image":
                results.append({"source": name, "kind": "image",
                                "detections": _detections_image(payload)})
            else:
                results.append(_tracks_video(name, payload, stride))
        return {"results": results}

    # media: annotate each input
    outputs: list[tuple[str, bytes, str]] = []
    for name, kind, payload in inputs:
        stem = pathlib.PurePosixPath(name.replace("\\", "/")).stem or "out"
        try:
            if kind == "image":
                ok, buf = cv2.imencode(".jpg", traffic_service.detect_vehicles_only(payload))
                if ok:
                    outputs.append((f"{stem}_pred.jpg", buf.tobytes(), "image/jpeg"))
            else:
                outputs.append((f"{stem}_pred.mp4", _annotate_video(payload, stride), "video/mp4"))
        except Exception as e:
            logger.warning("Skipping %s: %s", name, e)

    if not outputs:
        raise HTTPException(status_code=500, detail="annotation produced no output")

    if len(outputs) == 1:
        fn, data, mt = outputs[0]
        return StreamingResponse(io.BytesIO(data), media_type=mt,
                                 headers={"Content-Disposition": f'attachment; filename="{fn}"'})

    zip_buf = io.BytesIO()
    seen: dict[str, int] = {}
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn, data, _ in outputs:
            n = seen.get(fn, 0)
            seen[fn] = n + 1
            if n:
                base, ext = fn.rsplit(".", 1)
                fn = f"{base}_{n}.{ext}"
            zf.writestr(fn, data)
    zip_buf.seek(0)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="predictions_{ts}.zip"'},
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
    scale: int = 1,
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


@app.post("/fuse")
def fuse_plates(
    files: List[UploadFile] = File(...),
    engine: str = "mflpr2",
    scale: int = 1,
):
    """Multi-frame plate fusion (image-only, no OCR). Returns a restored plate PNG.

    Mirrors the former fusion sidecar's POST /fuse, now served in-process on the
    same port. N ordered same-size plate crops -> one restored plate (BGR PNG).
    """
    if engine not in ("mflpr2", "eott"):
        raise HTTPException(status_code=400, detail=f"unknown engine: {engine!r}")
    crops = _decode_uploads(files)
    if not crops:
        raise HTTPException(status_code=400, detail="no decodable frames")
    try:
        out = fusion_fuse(crops, engine=engine, scale=scale)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{engine} failed: {e}")
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(status_code=500, detail="png encode failed")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


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
    scale: int = 1,
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


@app.post("/predict/vehicles/video")
def predict_vehicles_video(
    file: UploadFile = File(...),
    frame_stride: int = 1,
):
    """Detect+track vehicles across a video; return per-track vehicle data.

    Mirrors /predict/plates/video but returns vehicle type + plate per track
    (not fusion OCR). Tracking is maintained across frames (one track_id per
    vehicle); the latest seen state for each track is returned. `frame_stride`
    processes every Nth frame (speed vs coverage).
    """
    data = file.file.read()
    frames = _read_video_frames(data)
    if not frames:
        raise HTTPException(status_code=400, detail="could not decode video")

    stride = max(1, frame_stride)
    traffic_service.reset()  # fresh track ids for this clip
    tracks: dict[int, dict] = {}
    for idx in range(0, len(frames), stride):
        _ = traffic_service.process_frame(frames[idx])
        for track_id, vehicle in traffic_service.vehicles.items():
            entry = tracks.setdefault(int(track_id), {
                "track_id": int(track_id), "frames_seen": 0,
                "vehicle_type": None, "license_plate": None,
                "confidence": None, "bbox": None, "plate_bbox": None,
            })
            entry["frames_seen"] += 1
            b = vehicle.bbox_xyxy
            entry["bbox"] = {"x1": float(b[0]), "y1": float(b[1]),
                             "x2": float(b[2]), "y2": float(b[3])}
            if vehicle.vehicle_type:
                entry["vehicle_type"] = vehicle.vehicle_type
            if vehicle.plate_number:
                entry["license_plate"] = vehicle.plate_number
                entry["confidence"] = float(vehicle.ocr_conf) if vehicle.ocr_conf > 0 else None
            if vehicle.license_plate_bbox is not None:
                pb = vehicle.license_plate_bbox
                entry["plate_bbox"] = {"x1": float(pb[0]), "y1": float(pb[1]),
                                       "x2": float(pb[2]), "y2": float(pb[3])}
    return {"n_frames": len(frames), "stride": stride, "tracks": list(tracks.values())}


@app.post("/predict/vehicles/image")
def predict_vehicles_image(file: UploadFile = File(...)):
    """Detect+track vehicles in one image; return per-track vehicle type + plate.

    Single-image analogue of /predict/vehicles/video (same per-track schema).
    Resets the tracker first (stateless single shot).
    """
    contents = file.file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    traffic_service.reset()
    _ = traffic_service.process_frame(img)

    tracks = []
    for track_id, vehicle in traffic_service.vehicles.items():
        b = vehicle.bbox_xyxy
        entry = {
            "track_id": int(track_id),
            "vehicle_type": vehicle.vehicle_type or None,
            "license_plate": vehicle.plate_number or None,
            "confidence": float(vehicle.ocr_conf) if vehicle.ocr_conf > 0 else None,
            "bbox": {"x1": float(b[0]), "y1": float(b[1]),
                     "x2": float(b[2]), "y2": float(b[3])},
            "plate_bbox": None,
        }
        if vehicle.license_plate_bbox is not None:
            pb = vehicle.license_plate_bbox
            entry["plate_bbox"] = {"x1": float(pb[0]), "y1": float(pb[1]),
                                   "x2": float(pb[2]), "y2": float(pb[3])}
        tracks.append(entry)
    return {"tracks": tracks}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "7862")),
        log_level="info"
    )

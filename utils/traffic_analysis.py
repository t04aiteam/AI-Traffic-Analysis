from typing import Dict, List, Optional
import re
import inspect
import threading

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tracking.deep_sort import DeepSort
from tracking.sort import Sort
from utils.utils import (
    BGR_COLORS,
    VEHICLES,   
    Vehicle,
    check_image_size,
    check_legit_plate,
    compute_color,
    crop_expanded_plate,
    draw_text,
    gettime,
    map_label,
)


class TrafficAnalysisService:
    """
    AI Traffic Analysis Service - Core processing pipeline.
    
    Analyzes traffic scenes with vehicle detection, license plate recognition,
    and multi-object tracking capabilities.

    - Detects vehicles and plates with YOLO
    - Tracks with SORT/DeepSORT
    - OCR with PaddleOCR
    - Exposes `process_frame` and `process_image`
    """

    def __init__(self, opts):
        self.opts = opts

        requested_device = getattr(self.opts, "device", "auto")
        self.opts.device = self._resolve_device(requested_device)
        self._is_cuda = str(self.opts.device).lower().startswith("cuda")

        # Detectors
        self.vehicle_detector = self._create_vehicle_detector(self.opts.vehicle_weight)
        self.plate_detector = YOLO(self.opts.plate_weight, task="detect")
        try:
            self.plate_detector.to(self.opts.device)
        except Exception:
            pass

        # OCR
        self.read_plate = bool(getattr(self.opts, "read_plate", True))
        self.ocr_engine_name = str(getattr(self.opts, "ocr_engine", "paddle")).strip().lower()
        self._init_ocr_engine()
        self.ocr_thres: float = float(getattr(self.opts, "ocr_thres", 0.9))

        # Super-resolution (optional, applied to plate crop before OCR)
        self.sr_engine_name = str(getattr(self.opts, "sr_engine", "none")).strip().lower()
        self.sr_scale = int(getattr(self.opts, "sr_scale", 2))
        self._init_sr_engine()

        # Dual OCR engines (lazy-init on first call to detect_plates_dual_ocr)
        self._dual_ocr_lock = threading.Lock()
        self._dual_fpo = None
        self._dual_ppocr = None
        self._dual_ocr_ready = False

        # Tracking
        self.deepsort: bool = bool(getattr(self.opts, "deepsort", False))
        self.dsort_weight: str = str(getattr(self.opts, "dsort_weight", "weights/tracking/deepsort/ckpt.t7"))
        self.vehicles: Dict[int, Vehicle] = {}
        self._init_tracker()

        # Misc
        self.color = BGR_COLORS
        self.lang = getattr(self.opts, "lang", "en")

    def _create_vehicle_detector(self, weight_path: str) -> YOLO:
        detector = YOLO(weight_path, task="detect")
        try:
            detector.to(self.opts.device)
        except Exception:
            pass
        self.opts.vehicle_weight = weight_path
        return detector

    def _resolve_device(self, requested: Optional[str]) -> str:
        if requested is None:
            requested = "auto"
        requested = str(requested).strip().lower()
        if requested in {"auto", ""}:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.isdigit():
            return f"cuda:{requested}" if torch.cuda.is_available() else "cpu"
        if requested == "cuda":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested

    def _init_tracker(self) -> None:
        if self.deepsort:
            self.tracker = DeepSort(
                self.dsort_weight,
                max_dist=0.2,
                min_confidence=0.3,
                nms_max_overlap=0.5,
                max_iou_distance=0.7,
                max_age=70,
                n_init=3,
                nn_budget=100,
                use_cuda=self._is_cuda,
            )
        else:
            self.tracker = Sort()
        self.vehicles = {}

    def reset(self) -> None:
        self._init_tracker()

    def set_vehicle_weight(self, weight_path: str) -> None:
        if not weight_path:
            raise ValueError("Vehicle weight path must be provided")
        self.vehicle_detector = self._create_vehicle_detector(weight_path)
        # reset tracker state so IDs restart for the new model
        self.reset()

    def _init_ocr_engine(self) -> None:
        engine = self.ocr_engine_name
        if engine == "paddle":
            from paddleocr import PaddleOCR
            ocr_kwargs = dict(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,  # oneDNN PIR path crashes on PP-OCRv6 (ConvertPirAttribute2RuntimeAttribute)
            )
            try:
                if "use_gpu" in inspect.signature(PaddleOCR.__init__).parameters:
                    ocr_kwargs["use_gpu"] = self._is_cuda
            except (ValueError, TypeError):
                pass
            self.ocr = PaddleOCR(**ocr_kwargs)
            self._ocr_predict = self._ocr_predict_paddle
        elif engine in ("fpo", "fast-plate-ocr", "fastplateocr"):
            from fast_plate_ocr import LicensePlateRecognizer
            model_name = str(getattr(self.opts, "fpo_model", "cct-s-v2-global-model"))
            device = "cuda" if self._is_cuda else "cpu"
            self.ocr = LicensePlateRecognizer(model_name, device=device)
            self._ocr_predict = self._ocr_predict_fpo
        elif engine in ("ppv6", "ppocr-v6", "ppocrv6", "ppocr_v6"):
            import os as _os
            _os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="PP-OCRv6_medium_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,  # oneDNN PIR path crashes on PP-OCRv6 (ConvertPirAttribute2RuntimeAttribute)
            )
            self._ocr_predict = self._ocr_predict_paddle
        elif engine == "none":
            self.ocr = None
            self._ocr_predict = lambda img: ("", 0.0)
        else:
            raise ValueError(f"Unknown OCR_ENGINE: {engine!r} (expected 'paddle', 'ppv6', 'fpo', or 'none')")

    def _ocr_predict_paddle(self, plate_image):
        results = self.ocr.predict(input=plate_image)
        if len(results) > 0:
            plate_info = " ".join(results[0].get("rec_texts", []))
            rec_scores = results[0].get("rec_scores", [])
            conf_val = sum(rec_scores) / len(rec_scores) if rec_scores else 0.0
            return plate_info, conf_val
        return "", 0.0

    def _ocr_predict_fpo(self, plate_image):
        if plate_image is None or plate_image.size == 0:
            return "", 0.0
        # fast-plate-ocr expects RGB ndarray; pipeline gives BGR
        if plate_image.ndim == 3 and plate_image.shape[2] == 3:
            img = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
        else:
            img = plate_image
        preds = self.ocr.run(img, return_confidence=True)
        if not preds:
            return "", 0.0
        first = preds[0]
        text = getattr(first, "plate", "") or ""
        probs = getattr(first, "char_probs", None)
        if probs is not None and len(probs) > 0 and len(text) > 0:
            conf = float(np.mean(probs[: len(text)]))
        else:
            conf = 0.0
        return text, conf

    def _init_sr_engine(self) -> None:
        from utils.sr import create_sr_engine
        self.sr_engine = create_sr_engine(
            self.sr_engine_name,
            device=str(self.opts.device),
            scale=self.sr_scale,
            realesrgan_weight=getattr(self.opts, "realesrgan_weight", None),
        )

    def _extract_plate_text(self, plate_image):
        plate_info, conf_val = self._ocr_predict(plate_image)
        if not plate_info:
            return "", 0.0
        plate_info = re.sub(r"[^A-Za-z0-9\-.]", "", plate_info)
        if plate_info and len(plate_info) > 2 and plate_info[0].isalpha() and plate_info[2] == 'C':
            plate_info = plate_info[:2] + '0' + plate_info[3:]
        return plate_info, conf_val

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        displayed_frame = frame.copy()
        t0 = gettime()

        # Vehicle detection
        detection = self.vehicle_detector(
            frame,
            verbose=False,
            imgsz=640,
            device=self.opts.device,
            conf=self.opts.vconf,
        )[0]
        boxes = detection.boxes

        det_xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.empty((0, 4))
        det_cls = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.empty((0,), dtype=int)

        label_lookup = VEHICLES.get(self.lang, VEHICLES.get("en", []))

        def resolve_label(cls_idx: int) -> str:
            try:
                return map_label(cls_idx, label_lookup)
            except Exception:
                return str(cls_idx)

        # Tracking
        try:
            if self.deepsort:
                outputs = self.tracker.update(boxes.cpu().xywh, boxes.cpu().conf, frame)
            else:
                outputs = self.tracker.update(boxes.cpu().xyxy).astype(int)
        except Exception:
            outputs = np.empty((0, 5), dtype=int)

        def _iou(a, b):
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if inter <= 0:
                return 0.0
            area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
            area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            union = area_a + area_b - inter
            return float(inter) / float(union + 1e-6)

        in_frame_ids: List[int] = []
        det_label_for_track: Dict[int, str] = {}

        for i in range(len(outputs)):
            tid = int(outputs[i, -1])
            in_frame_ids.append(tid)
            if tid not in self.vehicles:
                self.vehicles[tid] = Vehicle(track_id=tid)
            v = self.vehicles[tid]
            v.bbox_xyxy = outputs[i, :4]

            x1, y1, x2, y2 = v.bbox_xyxy
            cv2.rectangle(
                displayed_frame, (int(x1), int(y1)), (int(x2), int(y2)), compute_color(tid), 1
            )

            if det_xyxy.shape[0] > 0:
                tb = np.array([float(x1), float(y1), float(x2), float(y2)])
                best_iou, best_idx = 0.0, None
                for j in range(det_xyxy.shape[0]):
                    iou_val = _iou(tb, det_xyxy[j])
                    if iou_val > best_iou:
                        best_iou, best_idx = iou_val, j
                if best_idx is not None and best_iou > 0.1:
                    label_text = resolve_label(int(det_cls[best_idx]))
                    det_label_for_track[tid] = label_text
                    v.vehicle_type = label_text

        if det_xyxy.shape[0] > 0 and det_cls.size > 0:
            for idx in range(det_xyxy.shape[0]):
                box = det_xyxy[idx].astype(int)
                label_text = resolve_label(int(det_cls[idx]))
                try:
                    draw_text(
                        img=displayed_frame,
                        text=str(label_text),
                        pos=(int(box[0]), int(box[1])),
                        text_color=self.color["blue"],
                        text_color_bg=self.color["green"],
                    )
                except Exception:
                    continue

        for tid in in_frame_ids:
            if tid not in det_label_for_track:
                v = self.vehicles[tid]
                label_text = v.vehicle_type if v.vehicle_type else f"ID {tid}"
                x1, y1, x2, _ = v.bbox_xyxy.astype(int)
                try:
                    draw_text(
                        img=displayed_frame,
                        text=str(label_text),
                        pos=(int(x1), int(y1)),
                        text_color=self.color["blue"],
                        text_color_bg=self.color["green"],
                    )
                except Exception:
                    continue
        # Plate recognition
        if self.read_plate and in_frame_ids:
            active: List[Vehicle] = []
            crops: List[np.ndarray] = []
            for tid in in_frame_ids:
                v = self.vehicles[tid]
                box = v.bbox_xyxy.astype(int)
                success = (
                    (v.ocr_conf > self.ocr_thres)
                    and len(v.plate_number) > 5
                    and check_legit_plate(v.plate_number)
                )
                if success:
                    draw_text(
                        img=displayed_frame,
                        text=v.plate_number,
                        pos=(box[0], box[1] + 26),
                        text_color=self.color["blue"],
                        text_color_bg=self.color["green"],
                    )
                    continue

                crop = frame[box[1] : box[3], box[0] : box[2], :]
                v.vehicle_image = crop
                if not check_image_size(crop, 112, 112):
                    continue
                active.append(v)
                crops.append(crop)

            if crops:
                detections = self.plate_detector(
                    crops,
                    verbose=False,
                    imgsz=640,
                    device=self.opts.device,
                    conf=self.opts.pconf,
                )
                with_plate: List[Vehicle] = []
                for idx, det in enumerate(detections):
                    v = active[idx]
                    crop = crops[idx]
                    box = v.bbox_xyxy.astype(int)
                    plate_xyxy = det.boxes.xyxy
                    if len(plate_xyxy) < 1:
                        continue
                    pxyxy = plate_xyxy[0].cpu().numpy().astype(int)
                    src = (int(pxyxy[0] + box[0]), int(pxyxy[1] + box[1]))
                    dst = (int(pxyxy[2] + box[0]), int(pxyxy[3] + box[1]))
                    cv2.rectangle(displayed_frame, src, dst, self.color["green"], 2)

                    try:
                        cropped_plate = crop_expanded_plate(pxyxy, crop, 0.15)
                    except Exception:
                        cropped_plate = np.zeros((8, 8, 3), dtype=np.uint8)

                    v.plate_image = cropped_plate
                    v.license_plate_bbox = pxyxy + np.array([box[0], box[1], box[0], box[1]])
                    with_plate.append(v)

                for v in with_plate:
                    ocr_input = v.plate_image
                    if self.sr_engine is not None and v.ocr_conf < self.ocr_thres:
                        try:
                            ocr_input = self.sr_engine.enhance(v.plate_image)
                        except Exception:
                            ocr_input = v.plate_image
                    text, conf = self._extract_plate_text(ocr_input)
                    if conf > v.ocr_conf:
                        v.plate_number = text
                        v.ocr_conf = conf

        # FPS overlay
        dt = gettime() - t0
        fps = int(round(1.0 / dt, 0)) if dt > 0 else 0
        draw_text(
            img=displayed_frame,
            text=f"FPS: {fps}",
            font_scale=1,
            font_thickness=2,
            text_color=self.color["blue"],
            text_color_bg=self.color["white"],
        )

        return displayed_frame

    def process_image(self, img: np.ndarray) -> np.ndarray:
        self.reset()
        return self.process_frame(img)

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

    def _ensure_dual_ocr(self) -> None:
        if self._dual_ocr_ready:
            return
        with self._dual_ocr_lock:
            if self._dual_ocr_ready:
                return
            from fast_plate_ocr import LicensePlateRecognizer
            device = "cuda" if self._is_cuda else "cpu"
            model_name = str(getattr(self.opts, "fpo_model", "cct-s-v2-global-model"))
            self._dual_fpo = LicensePlateRecognizer(model_name, device=device)
            import os as _os
            _os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR
            self._dual_ppocr = PaddleOCR(
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="PP-OCRv6_medium_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,  # oneDNN PIR path crashes on PP-OCRv6 (ConvertPirAttribute2RuntimeAttribute)
            )
            self._dual_ocr_ready = True

    def _ocr_plates_fpo(self, crop: np.ndarray) -> tuple:
        if crop.ndim == 3 and crop.shape[2] == 3:
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            img = crop
        preds = self._dual_fpo.run(img, return_confidence=True)
        if not preds:
            return "", 0.0
        first = preds[0]
        text = getattr(first, "plate", "") or ""
        probs = getattr(first, "char_probs", None)
        if probs is not None and len(probs) > 0 and len(text) > 0:
            conf = float(np.mean(probs[: len(text)]))
        else:
            conf = 0.0
        return text, conf

    def _ocr_plates_ppocr(self, crop: np.ndarray) -> tuple:
        results = self._dual_ppocr.predict(input=crop)
        if results:
            texts = results[0].get("rec_texts", [])
            scores = results[0].get("rec_scores", [])
            text = " ".join(texts)
            conf = float(sum(scores) / len(scores)) if scores else 0.0
            return text, conf
        return "", 0.0

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
                if self.deepsort:
                    outputs = self.tracker.update(boxes.cpu().xywh, boxes.cpu().conf, frame)
                else:
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

    def detect_plates_dual_ocr(self, frame: np.ndarray) -> list:
        """Detect plates and run FPO + PPOCRv6-medium on each crop."""
        if frame is None or frame.size == 0:
            return []
        self._ensure_dual_ocr()
        detection = self.plate_detector(
            frame,
            verbose=False,
            imgsz=getattr(self.opts, "plate_imgsz", 1280),
            device=self.opts.device,
            conf=self.opts.pconf,
        )[0]
        results = []
        for box in detection.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            conf = float(box.conf[0])
            crop = frame[max(0, y1) : y2, max(0, x1) : x2]
            if crop.size == 0:
                continue
            fpo_text, fpo_conf = self._ocr_plates_fpo(crop)
            ppocr_text, ppocr_conf = self._ocr_plates_ppocr(crop)
            results.append({
                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "confidence": conf,
                "fpo": {"text": fpo_text, "confidence": fpo_conf},
                "ppocr": {"text": ppocr_text, "confidence": ppocr_conf},
            })
        return results

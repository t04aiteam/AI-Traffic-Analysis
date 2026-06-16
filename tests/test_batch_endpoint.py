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

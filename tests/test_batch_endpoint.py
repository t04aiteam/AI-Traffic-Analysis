import cv2
import io
import numpy as np
import pytest
import zipfile
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
        assert out is not frame

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


# ---------------------------------------------------------------------------
# Integration: POST /predict/batch
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    import sys
    from utils.traffic_analysis import TrafficAnalysisService

    # Force re-import of main under patches so module-level TrafficAnalysisService()
    # call doesn't try to load real weights from disk.
    sys.modules.pop("main", None)
    with (
        patch("utils.traffic_analysis.YOLO"),
        patch("utils.traffic_analysis.Sort"),
        patch("utils.traffic_analysis.DeepSort"),
        patch.object(TrafficAnalysisService, "_init_ocr_engine"),
        patch.object(TrafficAnalysisService, "_init_sr_engine"),
    ):
        import main
        main.traffic_service.detect_vehicles_only = MagicMock(side_effect=lambda f: f)
        from fastapi.testclient import TestClient
        yield TestClient(main.app)
    sys.modules.pop("main", None)


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
        assert resp.headers["content-type"] == "application/zip"
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

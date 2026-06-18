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

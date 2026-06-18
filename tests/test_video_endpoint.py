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

import sys
import types
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
        svc.opts = types.SimpleNamespace(device="cpu", vconf=0.6, pconf=0.25)
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

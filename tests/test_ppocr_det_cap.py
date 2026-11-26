"""PP-OCR text detection must run on a working copy capped at 960 px (longest side).

Uncapped, detection ran at full resolution: 16-240 s per photo on box 222 CPU.
Capped: about 8.6 s per image across 168 lipla-ocr images.
"""

import sys
import threading
import types
from unittest import mock

from utils.traffic_analysis import TrafficAnalysisService


def _fake_modules(captured: list) -> dict:
    """Build stand-in paddleocr / fast_plate_ocr modules that record PaddleOCR kwargs.

    Args:
        captured: list each PaddleOCR(**kwargs) call appends its kwargs to.

    Returns:
        Mapping of module name to fake module, for mock.patch.dict(sys.modules).
    """
    paddle = types.SimpleNamespace(
        PaddleOCR=lambda **kw: captured.append(kw) or object()
    )
    fpo = types.SimpleNamespace(LicensePlateRecognizer=lambda *_a, **_kw: object())
    return {"paddleocr": paddle, "fast_plate_ocr": fpo}


def _bare_service(**attrs: object) -> TrafficAnalysisService:
    """Return a TrafficAnalysisService without running its heavy __init__.

    Args:
        **attrs: attributes to set on the instance.

    Returns:
        The service instance.
    """
    svc = TrafficAnalysisService.__new__(TrafficAnalysisService)
    svc.opts = types.SimpleNamespace()
    svc._is_cuda = False
    for k, v in attrs.items():
        setattr(svc, k, v)
    return svc


def _assert_capped(kw: dict) -> None:
    assert kw["text_det_limit_type"] == "max"
    assert kw["text_det_limit_side_len"] == 960


def test_dual_ocr_ppocr_detection_is_capped():
    captured = []
    svc = _bare_service(_dual_ocr_ready=False, _dual_ocr_lock=threading.Lock())
    with mock.patch.dict(sys.modules, _fake_modules(captured)):
        svc._ensure_dual_ocr()
    assert len(captured) == 1
    _assert_capped(captured[0])


def test_single_engine_paddle_paths_are_capped():
    for engine in ("paddle", "ppv6"):
        captured = []
        svc = _bare_service(ocr_engine_name=engine)
        with mock.patch.dict(sys.modules, _fake_modules(captured)):
            svc._init_ocr_engine()
        assert len(captured) == 1, engine
        _assert_capped(captured[0])

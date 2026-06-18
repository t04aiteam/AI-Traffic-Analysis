import cv2
import numpy as np
import pytest
import httpx

from utils import fusion_client
from utils.fusion_client import fuse, FusionUnavailable


def _crop(h=32, w=96, val=120):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _png(arr):
    ok, buf = cv2.imencode(".png", arr)
    return buf.tobytes()


def test_fuse_posts_and_decodes(monkeypatch):
    restored = np.full((64, 192, 3), 77, dtype=np.uint8)

    class FakeResp:
        status_code = 200
        content = _png(restored)
        def raise_for_status(self): pass

    captured = {}

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, files=None, params=None):
            captured["url"] = url
            captured["params"] = params
            captured["nfiles"] = len(files)
            return FakeResp()

    monkeypatch.setattr(fusion_client.httpx, "Client", FakeClient)
    out = fuse([_crop(), _crop(), _crop()], engine="eott", scale=2,
               base_url="http://x:8100")
    assert out.shape == (64, 192, 3)
    assert captured["url"] == "http://x:8100/fuse"
    assert captured["params"] == {"engine": "eott", "scale": 2}
    assert captured["nfiles"] == 3


def test_fuse_unreachable_raises_fusion_unavailable(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **k):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(fusion_client.httpx, "Client", FakeClient)
    with pytest.raises(FusionUnavailable):
        fuse([_crop()], base_url="http://x:8100")


def test_fuse_empty_crops_raises_value_error():
    with pytest.raises(ValueError):
        fuse([], base_url="http://x:8100")

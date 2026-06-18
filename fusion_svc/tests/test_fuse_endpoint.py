import cv2
import numpy as np
from fastapi.testclient import TestClient
from fusion_svc.app import app

client = TestClient(app)


def _png_bytes(h=32, w=96, val=200):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[8:24, 10:86] = val
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _files(n=6):
    return [("files", (f"c{i}.png", _png_bytes(val=180 + i), "image/png")) for i in range(n)]


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_fuse_mflpr2_returns_png():
    r = client.post("/fuse?engine=mflpr2&scale=2", files=_files())
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    arr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None and arr.shape == (64, 192, 3)


def test_fuse_eott_returns_png():
    r = client.post("/fuse?engine=eott&scale=2", files=_files())
    assert r.status_code == 200
    arr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None and arr.shape == (64, 192, 3)


def test_fuse_unknown_engine_400():
    r = client.post("/fuse?engine=bogus", files=_files())
    assert r.status_code == 400


def test_fuse_no_files_400():
    r = client.post("/fuse?engine=mflpr2", files=[])
    assert r.status_code in (400, 422)

#!/usr/bin/env python3
"""Smoke-test every endpoint of the main Traffic AI API (7862) and the fusion
sidecar (8100). Synthetic inputs — checks status codes + response shape, not
OCR accuracy. Prints one PASS/FAIL line per check and a final tally.

Run:  uv run scripts/smoke_api_all.py
Env:  MAIN_URL (default http://127.0.0.1:7862), FUSION_URL (default :8100)
"""
import io
import os
import sys
import tempfile
import zipfile

import cv2
import httpx
import numpy as np

MAIN = os.environ.get("MAIN_URL", "http://127.0.0.1:7862").rstrip("/")
# Fusion is served in-process on the main port now; override only for a split deploy.
FUSION = os.environ.get("FUSION_URL", MAIN).rstrip("/")

_passed = 0
_failed = 0


def check(name, cond, extra=""):
    global _passed, _failed
    mark = "PASS" if cond else "FAIL"
    if cond:
        _passed += 1
    else:
        _failed += 1
    print(f"[{mark}] {name}{(' — ' + extra) if extra else ''}")


def _png(h=32, w=96, val=200):
    img = np.zeros((h, w, 3), np.uint8)
    img[8:24, 10:86] = val
    return cv2.imencode(".png", img)[1].tobytes()


def _scene(w=640, h=480):
    """Fake road scene: gray bg, a dark 'vehicle' box, a bright 'plate'."""
    img = np.full((h, w, 3), 120, np.uint8)
    cv2.rectangle(img, (220, 200), (420, 400), (60, 60, 60), -1)
    cv2.rectangle(img, (290, 360), (360, 390), (240, 240, 240), -1)
    cv2.putText(img, "51A-123", (292, 383), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _video(n=12, w=320, h=240):
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 120, np.uint8)
        x = 40 + i * 5
        cv2.rectangle(f, (x, 100), (x + 120, 200), (60, 60, 60), -1)
        cv2.rectangle(f, (x + 35, 170), (x + 85, 195), (240, 240, 240), -1)
        vw.write(f)
    vw.release()
    return path


def main():
    c = httpx.Client(timeout=180.0)
    img = _scene()
    print(f"== main API {MAIN} ==")

    try:
        r = c.get(f"{MAIN}/")
        check("GET /", r.status_code == 200 and "service" in r.json(), str(r.status_code))
    except Exception as e:
        check("GET /", False, repr(e))
        print("main API unreachable — start it: uv run main.py")
        c.close()
        sys.exit(1)

    r = c.get(f"{MAIN}/health")
    check("GET /health", r.status_code == 200 and r.json().get("status") == "healthy", str(r.status_code))

    r = c.get(f"{MAIN}/config")
    check("GET /config", r.status_code == 200 and "device" in r.json(), str(r.status_code))

    r = c.post(f"{MAIN}/reset")
    check("POST /reset", r.status_code == 200, str(r.status_code))

    r = c.post(f"{MAIN}/predict/image", files={"file": ("s.jpg", img, "image/jpeg")})
    check("POST /predict/image", r.status_code == 200 and "detections" in r.json(), str(r.status_code))

    r = c.post(f"{MAIN}/predict/frame?frame_number=1", files={"file": ("s.jpg", img, "image/jpeg")})
    check("POST /predict/frame", r.status_code == 200 and "detections" in r.json(), str(r.status_code))

    r = c.post(f"{MAIN}/predict/vehicles/image", files={"file": ("s.jpg", img, "image/jpeg")})
    check("POST /predict/vehicles/image", r.status_code == 200 and isinstance(r.json().get("tracks"), list), str(r.status_code))

    r = c.post(f"{MAIN}/predict/batch", files={"files": ("s.jpg", img, "image/jpeg")})
    check("POST /predict/batch (1 img -> jpeg)",
          r.status_code == 200 and r.headers["content-type"] == "image/jpeg", str(r.status_code))

    multi = [("files", (f"s{i}.jpg", _scene(), "image/jpeg")) for i in range(3)]
    r = c.post(f"{MAIN}/predict/batch", files=multi)
    is_zip = r.headers.get("content-type") == "application/zip"
    n = len(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) if is_zip else 0
    check("POST /predict/batch (3 imgs -> zip)", r.status_code == 200 and is_zip and n == 3,
          f"{r.status_code}, {n} entries")

    r = c.post(f"{MAIN}/predict/batch", files={"files": ("x.txt", b"notimage", "text/plain")})
    check("POST /predict/batch (garbage -> 400)", r.status_code == 400, str(r.status_code))

    # batch: JSON mode over 2 images
    r = c.post(f"{MAIN}/predict/batch?format=json",
               files=[("files", (f"s{i}.jpg", _scene(), "image/jpeg")) for i in range(2)])
    j = r.json() if r.status_code == 200 else {}
    ok = r.status_code == 200 and len(j.get("results", [])) == 2 and j["results"][0]["kind"] == "image"
    check("POST /predict/batch (format=json, 2 imgs)", ok, str(r.status_code))

    # batch: a video file -> annotated mp4
    bvid = _video()
    with open(bvid, "rb") as fh:
        r = c.post(f"{MAIN}/predict/batch?frame_stride=3", files={"files": ("clip.mp4", fh.read(), "video/mp4")})
    check("POST /predict/batch (video -> mp4)",
          r.status_code == 200 and r.headers.get("content-type") == "video/mp4", str(r.status_code))

    # batch: a zip of 2 images -> zip of annotated
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        for i in range(2):
            zf.writestr(f"z{i}.jpg", _scene())
    r = c.post(f"{MAIN}/predict/batch", files={"files": ("imgs.zip", zbuf.getvalue(), "application/zip")})
    nz = len(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) if r.headers.get("content-type") == "application/zip" else 0
    check("POST /predict/batch (zip -> zip of 2)", r.status_code == 200 and nz == 2, f"{r.status_code}, {nz}")
    os.unlink(bvid)

    r = c.post(f"{MAIN}/predict/plates/batch", files={"files": ("s.jpg", img, "image/jpeg")})
    check("POST /predict/plates/batch (1 img -> jpeg)",
          r.status_code == 200 and r.headers["content-type"] == "image/jpeg", str(r.status_code))

    crops = [("files", (f"c{i}.png", _png(val=180 + i), "image/png")) for i in range(4)]
    for eng in ("mflpr2", "eott"):
        try:
            r = c.post(f"{MAIN}/predict/plates/multiframe?engine={eng}&scale=2", files=crops)
            ok = r.status_code == 200 and r.json().get("engine") == eng and "ppocr" in r.json()
            check(f"POST /predict/plates/multiframe ({eng})", ok, str(r.status_code))
        except Exception as e:
            check(f"POST /predict/plates/multiframe ({eng})", False, repr(e))

    try:
        r = c.post(f"{MAIN}/predict/plates/multiframe?engine=bogus", files=crops)
        check("POST /predict/plates/multiframe (bad engine -> 400)", r.status_code == 400, str(r.status_code))
    except Exception as e:
        check("POST /predict/plates/multiframe (bad engine -> 400)", False, repr(e))

    vpath = _video()
    try:
        with open(vpath, "rb") as fh:
            r = c.post(f"{MAIN}/predict/plates/video?engine=mflpr2",
                       files={"file": ("v.mp4", fh.read(), "video/mp4")})
        check("POST /predict/plates/video", r.status_code == 200 and isinstance(r.json(), list), str(r.status_code))
    except Exception as e:
        check("POST /predict/plates/video", False, repr(e))

    try:
        with open(vpath, "rb") as fh:
            r = c.post(f"{MAIN}/predict/vehicles/video?frame_stride=2",
                       files={"file": ("v.mp4", fh.read(), "video/mp4")})
        ok = r.status_code == 200 and isinstance(r.json().get("tracks"), list)
        check("POST /predict/vehicles/video", ok, str(r.status_code))
    except Exception as e:
        check("POST /predict/vehicles/video", False, repr(e))
    os.unlink(vpath)

    label = "fusion (in-process)" if FUSION == MAIN else f"fusion sidecar {FUSION}"
    print(f"\n== {label} ==")
    try:
        for eng in ("mflpr2", "eott"):
            r = c.post(f"{FUSION}/fuse?engine={eng}",
                       files=[("files", (f"c{i}.png", _png(val=180 + i), "image/png")) for i in range(4)])
            check(f"POST /fuse ({eng})",
                  r.status_code == 200 and r.headers["content-type"] == "image/png", str(r.status_code))
        r = c.post(f"{FUSION}/fuse?engine=bogus", files=[("files", ("c.png", _png(), "image/png"))])
        check("POST /fuse (bad engine -> 400)", r.status_code == 400, str(r.status_code))
    except Exception as e:
        check("fusion sidecar reachable", False, repr(e))

    c.close()
    print(f"\n=== {_passed} passed, {_failed} failed ===")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

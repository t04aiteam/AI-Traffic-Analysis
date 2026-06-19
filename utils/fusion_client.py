"""HTTP client for the fusion-svc multi-frame restoration microservice."""
import os

import cv2
import httpx
import numpy as np

DEFAULT_BASE_URL = "http://127.0.0.1:8100"


class FusionUnavailable(Exception):
    """Raised when fusion-svc cannot be reached."""


def _png_bytes(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("failed to PNG-encode crop")
    return buf.tobytes()


def fuse(crops, engine="mflpr2", scale=1, base_url=None, timeout=60.0):
    """POST PNG-encoded BGR crops to fusion-svc; return decoded BGR restored plate."""
    if not crops:
        raise ValueError("fuse requires at least one crop")
    url_base = base_url or os.environ.get("FUSION_URL", DEFAULT_BASE_URL)
    files = [
        ("files", (f"c{i}.png", _png_bytes(c), "image/png"))
        for i, c in enumerate(crops)
    ]
    params = {"engine": engine, "scale": int(scale)}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{url_base}/fuse", files=files, params=params)
            resp.raise_for_status()
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise FusionUnavailable(f"fusion-svc unreachable at {url_base}: {e}") from e
    arr = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("fusion-svc returned undecodable image")
    return arr

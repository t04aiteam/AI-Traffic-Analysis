"""fusion-svc: multi-frame plate restoration microservice (no OCR).

POST /fuse?engine=mflpr2|eott&scale=N
  body: multipart `files` — N ordered same-size plate crops
  returns: image/png (restored plate, BGR-encoded)
"""
import io

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from fusion_svc.adapters.eott_adapter import fuse_eott
from fusion_svc.adapters.mflpr2_adapter import fuse_mflpr2

app = FastAPI(title="fusion-svc", version="0.1.0")

_ENGINES = {"mflpr2": fuse_mflpr2, "eott": fuse_eott}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/fuse")
def fuse(
    files: list[UploadFile] = File(...),
    engine: str = Query("mflpr2"),
    scale: int = Query(1, ge=1, le=8),
):
    if engine not in _ENGINES:
        raise HTTPException(status_code=400, detail=f"unknown engine: {engine!r}")
    frames: list[np.ndarray] = []
    for up in files:
        data = up.file.read()
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            frames.append(arr)
    if not frames:
        raise HTTPException(status_code=400, detail="no decodable frames")
    try:
        out = _ENGINES[engine](frames, scale=scale)
    except Exception as e:  # engine-internal failure
        raise HTTPException(status_code=500, detail=f"{engine} failed: {e}")
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(status_code=500, detail="png encode failed")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

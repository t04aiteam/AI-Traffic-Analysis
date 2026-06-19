"""Adapter: fuse N BGR plate crops into one restored BGR image via mf-lpr2.

Aligned to mf-lpr2's NATIVE pre-OCR pipeline: this is exactly what native
`MFLPR2().run()` computes before its recognizer — `restore().image`. Defaults
match native `MFLPR2()`: AffineFlowEstimator flow and `scale=1`. Only the OCR
stage differs (native feeds RGB to its own recognizer; here we hand the restored
plate as BGR to this repo's dual-OCR).

Restore-only path — never constructs a recognizer, so torch/transformers are
never imported.
"""
import cv2
import numpy as np
from mf_lpr2 import MFLPR2, AffineFlowEstimator


def fuse_mflpr2(frames: list[np.ndarray], scale: int = 1) -> np.ndarray:
    """Fuse same-size BGR uint8 crops into one BGR uint8 restored plate.

    Returns shape (H*scale, W*scale, 3), uint8 BGR. `scale=1` (native default)
    reproduces native `MFLPR2().run()`'s restoration exactly.
    """
    if not frames:
        raise ValueError("fuse_mflpr2 requires at least one frame")
    # mf-lpr2 works in RGB; convert in, convert out.
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    pipe = MFLPR2(flow_estimator=AffineFlowEstimator(), scale=int(scale))
    result = pipe.restore(rgb)            # center_idx defaults to len//2
    img = np.clip(result.image, 0.0, 1.0)  # float32 [0,1] RGB
    img_u8 = (img * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

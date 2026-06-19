"""Adapter: fuse N BGR plate crops into one restored BGR image via eott.

Aligned to eott's NATIVE pre-OCR pipeline (everything `eott.run()` does except
recognition): track -> register -> reconstruct -> postprocess. The returned
image is the binarized plate native eott feeds to its OCR, promoted to 3-channel
BGR so this repo's dual-OCR can consume it. Only the OCR stage differs from
native; all restoration/reconstruction steps match.

Each frame's initial ROI is the full-image rectangle (frames are pre-cropped
plates). Uses eott's default Config (gsr4, hr_size=(300,98), otsu, max_frames=5).
The `scale` arg is accepted for interface parity but ignored: native eott has no
scale knob — it reconstructs onto a fixed HR grid (cfg.hr_size).
"""
import cv2
import numpy as np
from eott import Config
from eott.geometry import order_corners
from eott.stages import tracking, registration, reconstruction, postprocessing


def _full_rect(h: int, w: int) -> np.ndarray:
    # TL, TR, BR, BL
    return order_corners(
        np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    )


def fuse_eott(frames: list[np.ndarray], scale: int = 1) -> np.ndarray:
    """Fuse same-size BGR uint8 crops into one BGR uint8 binarized plate.

    Mirrors `eott.run()` minus the recognize() call (native defaults via
    `Config()`). Returns the post-processed binary plate as BGR uint8 — the
    same image native eott hands to its OCR.
    """
    if not frames:
        raise ValueError("fuse_eott requires at least one frame")
    cfg = Config()  # native defaults: gsr4 / hr_size (300,98) / otsu / max_frames 5
    cfg.validate()

    frames = frames[:: max(1, cfg.frame_step)][: cfg.max_frames]
    h, w = frames[0].shape[:2]
    corners0 = _full_rect(h, w)

    corners = tracking.track(frames, corners0, cfg)
    corners = registration.register(frames, corners, cfg)
    sr = reconstruction.reconstruct(frames, corners, cfg)   # BGR uint8
    binr = postprocessing.postprocess(sr, cfg)              # 2D uint8 (0/255)

    if binr.ndim == 2:
        binr = cv2.cvtColor(binr, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(binr)

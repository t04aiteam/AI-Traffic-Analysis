"""Adapter: fuse N BGR plate crops into one restored BGR image via eott.

Calls the reconstruction stage directly (skips tracking/registration/OCR).
Each frame's ROI is the full-image rectangle (frames are pre-cropped plates).
"""
import numpy as np
from eott import Config
from eott.geometry import order_corners
from eott.stages.reconstruction import reconstruct


def _full_rect(h: int, w: int) -> np.ndarray:
    # TL, TR, BR, BL
    return order_corners(
        np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    )


def fuse_eott(frames: list[np.ndarray], scale: int = 2) -> np.ndarray:
    """Fuse same-size BGR uint8 crops into one BGR uint8 restored plate.

    Returns shape (H*scale, W*scale, 3), uint8 BGR.
    """
    if not frames:
        raise ValueError("fuse_eott requires at least one frame")
    h, w = frames[0].shape[:2]
    corners = [_full_rect(h, w) for _ in frames]
    cfg = Config(
        reconstruction="gsr4",
        hr_size=(w * int(scale), h * int(scale)),  # eott hr_size is (width, height)
        max_frames=len(frames),
    )
    out = reconstruct(frames, corners, cfg)  # BGR uint8, shape (H*scale, W*scale, 3)
    return np.ascontiguousarray(out)

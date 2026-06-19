import numpy as np
from eott import Config
from fusion_svc.adapters.eott_adapter import fuse_eott


def _synthetic_burst(n=8, h=32, w=96):
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[8:24, 10:86] = 200
    return [base.copy() for _ in range(n)]


def _expected_shape():
    """Native eott pre-OCR output: postprocess() of the fixed hr_size grid.

    reconstruct -> (H, W) = hr_size (height, width); postprocess trims
    crop_margin from each side (ocr_upscale=1.0, binarize keeps size).
    """
    cfg = Config()
    w, h = cfg.hr_size  # (width, height)
    mx, my = int(w * cfg.crop_margin), int(h * cfg.crop_margin)
    return (h - 2 * my, w - 2 * mx, 3)


def test_fuse_eott_returns_binarized_bgr():
    frames = _synthetic_burst(n=8, h=32, w=96)
    out = fuse_eott(frames)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    # output is the native binarized plate on a FIXED hr grid, not input-sized
    assert out.shape == _expected_shape()


def test_fuse_eott_scale_is_ignored():
    """Native eott has no scale knob (fixed hr_size); scale must not change output."""
    frames = _synthetic_burst(n=8, h=32, w=96)
    out1 = fuse_eott(frames, scale=1)
    out2 = fuse_eott(frames, scale=2)
    assert out1.shape == out2.shape == _expected_shape()


def test_fuse_eott_single_frame_ok():
    frames = _synthetic_burst(n=1, h=32, w=96)
    out = fuse_eott(frames)
    assert out.shape == _expected_shape()
    assert out.dtype == np.uint8

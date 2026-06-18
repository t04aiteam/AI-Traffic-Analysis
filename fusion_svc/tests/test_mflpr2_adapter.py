import numpy as np
from fusion_svc.adapters.mflpr2_adapter import fuse_mflpr2


def _synthetic_burst(n=8, h=32, w=96):
    """N near-identical BGR crops with tiny sub-pixel-ish shifts."""
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[8:24, 10:86] = 200  # a bright bar standing in for plate glyphs
    frames = []
    for i in range(n):
        shifted = np.roll(base, shift=i % 2, axis=1)
        frames.append(shifted.copy())
    return frames


def test_fuse_mflpr2_returns_bgr_uint8_scaled():
    frames = _synthetic_burst(n=8, h=32, w=96)
    out = fuse_mflpr2(frames, scale=2)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape[0] == 64 and out.shape[1] == 192  # (H*2, W*2)


def test_fuse_mflpr2_single_frame_ok():
    frames = _synthetic_burst(n=1, h=32, w=96)
    out = fuse_mflpr2(frames, scale=1)
    assert out.shape == (32, 96, 3)
    assert out.dtype == np.uint8

import numpy as np
from fusion_svc.adapters.eott_adapter import fuse_eott


def _synthetic_burst(n=8, h=32, w=96):
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[8:24, 10:86] = 200
    return [base.copy() for _ in range(n)]


def test_fuse_eott_returns_bgr_uint8_scaled():
    frames = _synthetic_burst(n=8, h=32, w=96)
    out = fuse_eott(frames, scale=2)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape[0] == 64 and out.shape[1] == 192  # (H*2, W*2)


def test_fuse_eott_single_frame_ok():
    frames = _synthetic_burst(n=1, h=32, w=96)
    out = fuse_eott(frames, scale=1)
    assert out.shape == (32, 96, 3)
    assert out.dtype == np.uint8

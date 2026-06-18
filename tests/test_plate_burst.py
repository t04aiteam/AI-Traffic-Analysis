import numpy as np
from utils.plate_burst import resize_burst_to_common, select_burst_window


def _crop(h, w, val=100):
    return np.full((h, w, 3), val, dtype=np.uint8)


def test_resize_to_common_median():
    crops = [_crop(30, 90), _crop(32, 96), _crop(34, 102)]
    out = resize_burst_to_common(crops)
    sizes = {c.shape[:2] for c in out}
    assert len(sizes) == 1                      # all identical now
    assert out[0].shape[:2] == (32, 96)         # per-axis median
    assert all(c.dtype == np.uint8 for c in out)


def test_resize_to_explicit_target():
    crops = [_crop(30, 90), _crop(50, 50)]
    out = resize_burst_to_common(crops, target=(40, 80))
    assert all(c.shape[:2] == (40, 80) for c in out)


def test_resize_empty_returns_empty():
    assert resize_burst_to_common([]) == []


def test_select_window_under_cap_returns_all():
    crops = [_crop(10, 10) for _ in range(5)]
    assert len(select_burst_window(crops, max_frames=32)) == 5


def test_select_window_over_cap_downsamples_inclusive():
    crops = [_crop(10, 10, val=i) for i in range(40)]
    out = select_burst_window(crops, max_frames=8)
    assert len(out) == 8
    assert out[0][0, 0, 0] == 0     # first kept
    assert out[-1][0, 0, 0] == 39   # last kept

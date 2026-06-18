"""Helpers for preparing a multi-frame plate burst before fusion."""
import cv2
import numpy as np


def resize_burst_to_common(crops, target=None):
    """Resize all BGR crops to a common (H, W).

    target: (H, W) to force; if None, uses the per-axis median of input sizes.
    Returns a new list; input list is not mutated. Empty in -> empty out.
    """
    if not crops:
        return []
    if target is None:
        heights = [c.shape[0] for c in crops]
        widths = [c.shape[1] for c in crops]
        th = int(np.median(heights))
        tw = int(np.median(widths))
    else:
        th, tw = int(target[0]), int(target[1])
    out = []
    for c in crops:
        if c.shape[:2] == (th, tw):
            out.append(c)
        else:
            out.append(cv2.resize(c, (tw, th), interpolation=cv2.INTER_CUBIC))
    return out


def select_burst_window(crops, max_frames):
    """Cap a burst to max_frames evenly-spaced crops (keeps first and last)."""
    n = len(crops)
    if n <= max_frames:
        return list(crops)
    idx = np.linspace(0, n - 1, max_frames).round().astype(int)
    return [crops[i] for i in idx]

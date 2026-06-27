"""In-process multi-frame plate restoration (fusion engines).

Formerly an HTTP client to the fusion-svc sidecar (port 8100). The vendored
mf-lpr2 / eott engines are now installed into this venv and called directly, so
fusion runs in the same process as the main API — single-port deployment, no
sidecar. The public signature (`fuse`, `FusionUnavailable`) is unchanged for
backwards compatibility.
"""
import numpy as np

from fusion_svc.adapters.eott_adapter import fuse_eott
from fusion_svc.adapters.mflpr2_adapter import fuse_mflpr2

_ENGINES = {"mflpr2": fuse_mflpr2, "eott": fuse_eott}


class FusionUnavailable(Exception):
    """Kept for API compatibility. No longer raised — engines run in-process."""


def fuse(crops, engine="mflpr2", scale=1, **_ignored):
    """Fuse N BGR crops of one plate into a single restored BGR image.

    `_ignored` absorbs legacy kwargs (base_url/timeout) from old call sites.
    """
    if not crops:
        raise ValueError("fuse requires at least one crop")
    if engine not in _ENGINES:
        raise ValueError(f"unknown engine: {engine!r}")
    out = _ENGINES[engine](crops, scale=int(scale))
    if not isinstance(out, np.ndarray):
        raise ValueError(f"{engine} returned a non-array result")
    return out

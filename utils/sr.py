"""
Super-resolution / image-restoration module for the ALPR pipeline.

Usage pattern (cascade):
    sr = create_sr_engine("bicubic", scale=2)
    if sr is not None:
        plate_crop = sr.enhance(plate_crop)   # upscale before OCR
    text, conf = ocr_engine.predict(plate_crop)
    # Optionally apply SR only when OCR confidence is low:
    #   if conf < threshold: plate_crop = sr.enhance(plate_crop); retry OCR

BGR convention:
    All engines accept and return BGR uint8 HxWx3 ndarrays (OpenCV native).
    Internal colour-space conversions are the engine's responsibility.

Zero-dep baseline:
    "bicubic" works with only cv2 + numpy — install nothing extra.
    "realesrgan" requires:  uv pip install realesrgan basicsr
    "lcofl" is a stub pending Vietnamese-plate fine-tuning data.
"""

from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SREngine:
    """Abstract base for all super-resolution engines."""

    def enhance(self, plate_bgr: np.ndarray) -> np.ndarray:
        """
        Upscale / restore a licence-plate crop.

        Parameters
        ----------
        plate_bgr : np.ndarray
            BGR uint8 array of shape HxWx3 (OpenCV convention).

        Returns
        -------
        np.ndarray
            BGR uint8 array, upscaled.  Input is NEVER mutated.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Engine implementations
# ---------------------------------------------------------------------------

class BicubicSR(SREngine):
    """
    Pure-OpenCV bicubic upscaler.  Zero extra dependencies.

    This is the C0 baseline / control condition — guaranteed to work
    immediately without any additional pip installs.
    """

    def __init__(self, scale: int = 2, **kwargs) -> None:
        self.scale = scale

    def enhance(self, plate_bgr: np.ndarray) -> np.ndarray:
        if plate_bgr is None or plate_bgr.size == 0:
            return plate_bgr
        h, w = plate_bgr.shape[:2]
        return cv2.resize(
            plate_bgr,
            (w * self.scale, h * self.scale),
            interpolation=cv2.INTER_CUBIC,
        )


class RealESRGANSR(SREngine):
    """
    Real-ESRGAN upscaler using RealESRGAN_x2plus or x4plus weights.

    Requires:  uv pip install realesrgan basicsr

    Parameters
    ----------
    scale : int
        Upscale factor — 2 selects x2plus weights, 4 selects x4plus weights.
    device : str
        PyTorch device string, e.g. "cpu", "cuda", "cuda:0".
    realesrgan_weight : str, optional
        Override path to a custom .pth weight file.
    """

    def __init__(
        self,
        scale: int = 2,
        device: str = "cpu",
        realesrgan_weight: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Compat shim: basicsr imports `torchvision.transforms.functional_tensor`,
        # removed in torchvision>=0.17. It only needs `rgb_to_grayscale`, which
        # still lives in `functional`. Alias the old path before basicsr loads.
        import sys as _sys
        if "torchvision.transforms.functional_tensor" not in _sys.modules:
            try:
                import types as _types
                from torchvision.transforms import functional as _tvF
                _shim = _types.ModuleType("torchvision.transforms.functional_tensor")
                _shim.rgb_to_grayscale = _tvF.rgb_to_grayscale
                _sys.modules["torchvision.transforms.functional_tensor"] = _shim
            except Exception:
                pass

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            raise ImportError(
                "RealESRGAN engine requires extra packages. "
                "Install them with:  uv pip install realesrgan basicsr"
            ) from exc

        self.scale = scale

        # Official Real-ESRGAN release weights (RealESRGANer downloads + caches
        # a URL model_path automatically; a bare filename would NOT download).
        _WEIGHT_URL = {
            2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        }

        # Select architecture and default weight by scale
        if scale == 4:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            netscale = 4
        else:
            # Default / fallback: x2plus
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            netscale = 2

        # Custom weight overrides the auto-downloaded release weight.
        model_path: str = realesrgan_weight or _WEIGHT_URL.get(netscale, _WEIGHT_URL[2])

        # Normalise device: RealESRGANer accepts integer GPU index
        gpu_id: Optional[int]
        if device.startswith("cuda"):
            parts = device.split(":")
            gpu_id = int(parts[1]) if len(parts) > 1 else 0
        else:
            gpu_id = None  # CPU

        self._upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            gpu_id=gpu_id,
        )

    def enhance(self, plate_bgr: np.ndarray) -> np.ndarray:
        if plate_bgr is None or plate_bgr.size == 0:
            return plate_bgr
        # RealESRGAN expects RGB uint8
        img_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
        try:
            output_rgb, _ = self._upsampler.enhance(img_rgb, outscale=self.scale)
        except RuntimeError:
            # Fallback: if the model fails (e.g. CUDA OOM), return original
            return plate_bgr
        return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)


class LCOFLSR(SREngine):
    """
    LCOFL / LCDNet layout-aware super-resolution — STUB.

    This engine implements LCOFL (Layout-Consistent OCR-Feedback Loss) /
    LCDNet, a GAN-based SR method that incorporates an OCR-loss term so the
    network learns to restore character-discriminative detail.  It represents
    the current academic SOTA for plate SR, but is restricted to
    non-commercial use due to its training data licence.

    Status:
        NOT IMPLEMENTED.  The model requires fine-tuning on Vietnamese
        licence-plate data before it can be used in this pipeline.
        See the install guide for details:
            aiml-research-lowres-vn-lpr/install-guides/repo-3-lpsr-lacd-LCOFL.md

    Once fine-tuning is complete, replace this stub with a real
    implementation that lazy-imports lcofl/lcdnet and calls its API.
    """

    def __init__(self, scale: int = 2, device: str = "cpu", **kwargs) -> None:
        raise NotImplementedError(
            "LCOFL/LCDNet SR requires retraining on Vietnamese plate data — "
            "see aiml-research-lowres-vn-lpr/install-guides/repo-3-lpsr-lacd-LCOFL.md"
        )

    def enhance(self, plate_bgr: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(
            "LCOFL/LCDNet SR requires retraining on Vietnamese plate data — "
            "see aiml-research-lowres-vn-lpr/install-guides/repo-3-lpsr-lacd-LCOFL.md"
        )


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------

_REGISTRY = {
    "bicubic": BicubicSR,
    "realesrgan": RealESRGANSR,
    "lcofl": LCOFLSR,
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_sr_engine(
    name,
    device: str = "cpu",
    scale: int = 2,
    **kwargs,
) -> Optional[SREngine]:
    """
    Factory for SR engines.

    Parameters
    ----------
    name : str | None
        Engine name (case-insensitive).  Passing None, "", or "none"
        disables SR and returns None — the caller should skip enhancement.
    device : str
        PyTorch device string, e.g. "cpu", "cuda", "cuda:0".
    scale : int
        Integer upscale factor (2 or 4 for most engines).
    **kwargs
        Forwarded verbatim to the engine constructor.
        Notable: ``realesrgan_weight`` (str) overrides the weight path for
        RealESRGANSR.

    Returns
    -------
    SREngine or None
        Engine instance, or None when name resolves to "none" / "".

    Raises
    ------
    ValueError
        When ``name`` is not a registered engine and not a no-op sentinel.
    """
    # Normalise: treat None, "", "none" identically
    if name is None:
        normalised = "none"
    else:
        normalised = str(name).strip().lower()

    if normalised in ("", "none"):
        return None

    if normalised not in _REGISTRY:
        raise ValueError(
            f"Unknown SR engine: {name!r}. "
            f"Expected one of: {sorted(_REGISTRY.keys())} or None/\"none\" to disable."
        )

    engine_cls = _REGISTRY[normalised]
    return engine_cls(scale=scale, device=device, **kwargs)

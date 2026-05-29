"""Smoke test for plate_ocr_api.

Layers:
  1. Import the API module -> loads YOLO12n plate detector + fast-plate-ocr
     CCT-S v2 (downloads the OCR model on first run). Prints wiring info.
  2. OCR-only check on real cropped plate images shipped with fast-plate-ocr
     (test/assets/test_plate_*.png) -- exercises the colour conversion + run().
  3. Full detect -> crop -> OCR on a user-supplied scene image, if a path is
     passed as an argument.

Usage:
    uv run python scripts/smoke_plate_ocr_api.py [SCENE_IMAGE_PATH ...]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# --- Layer 1: import + model load --------------------------------------------
print("=== Layer 1: import plate_ocr_api (loads detector + OCR) ===")
import plate_ocr_api as api  # noqa: E402

print(f"plate_weight    : {api.CFG.plate_weight}")
print(f"ocr_model       : {api.CFG.ocr_model}")
print(f"ocr_color_mode  : {api.OCR_COLOR_MODE}")
print(f"ocr providers   : {api.ocr.providers}")
print(f"ocr img (HxW)   : {api.ocr.config.img_height}x{api.ocr.config.img_width}")
print(f"yolo names      : {api.plate_detector.names}")
print("Layer 1 OK\n")

# --- Layer 2: OCR-only on real plate crops -----------------------------------
print("=== Layer 2: OCR on real cropped plates (fast-plate-ocr assets) ===")
assets = Path("/Users/logan/Developer/vibes/WORK/LIPLA/fast-plate-ocr/test/assets")
sample_paths = sorted(assets.glob("test_plate_*.png"))
if not sample_paths:
    print("WARN: no sample plate crops found, skipping Layer 2")
else:
    crops = []
    for p in sample_paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        crops.append(api._to_ocr_color(bgr))
    preds = api.ocr.run(crops, return_confidence=True)
    for p, pred in zip(sample_paths, preds):
        conf = float(np.mean(pred.char_probs)) if pred.char_probs is not None else None
        conf_s = f"{conf:.3f}" if conf is not None else "n/a"
        print(f"  {p.name}: '{pred.plate}' (conf={conf_s})")
    print("Layer 2 OK\n")

# --- Layer 3: full pipeline on user scene image(s) ---------------------------
scene_args = sys.argv[1:]
print("=== Layer 3: full detect -> OCR on scene image(s) ===")
if not scene_args:
    print("No scene image path passed -> skipping (pass a path to test detection).")
else:
    for arg in scene_args:
        path = Path(arg)
        if not path.exists():
            print(f"  SKIP {arg}: missing")
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  SKIP {arg}: failed to decode")
            continue
        plates = api._recognize(img)
        print(f"  {path.name}: {len(plates)} plate(s)")
        for pl in plates:
            oc = f"{pl.ocr_conf:.3f}" if pl.ocr_conf is not None else "n/a"
            print(f"    text='{pl.text}' det={pl.det_conf:.3f} ocr={oc} legit={pl.legit}")

print("\nSmoke test complete.")

"""Batch-run the plate_ocr_api pipeline over a directory of images.

Loads the YOLO12n detector + fast-plate-ocr CCT-S v2 once, runs detect ->
crop -> OCR on every image, writes a results JSON and annotated images, and
prints a summary (detection rate, confidence stats, per-prefix breakdown,
no-detection list).

Usage:
    uv run python scripts/batch_plate_ocr.py IMAGE_DIR OUTPUT_DIR
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import plate_ocr_api as api

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def annotate(img, plates):
    for p in plates:
        x1, y1, x2, y2 = int(p.bbox.x1), int(p.bbox.y1), int(p.bbox.x2), int(p.bbox.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{p.text or '?'} {p.ocr_conf:.2f}" if p.ocr_conf is not None else (p.text or "?")
        cv2.putText(img, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img


def main(argv):
    if len(argv) < 2:
        print("usage: batch_plate_ocr.py IMAGE_DIR OUTPUT_DIR")
        return 2
    img_dir, out_dir = Path(argv[0]), Path(argv[1])
    ann_dir = out_dir / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
    print(f"Found {len(files)} images in {img_dir}")
    print(f"OCR model={api.CFG.ocr_model} color={api.OCR_COLOR_MODE} "
          f"plate_weight={api.CFG.plate_weight} conf={api.CFG.pconf} imgsz={api.CFG.imgsz}\n")

    results = []
    per_prefix = defaultdict(lambda: {"n": 0, "detected": 0, "plates": 0})
    no_detect = []
    confs = []
    t0 = time.time()

    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        prefix = f.name.split("_")[0]
        per_prefix[prefix]["n"] += 1
        if img is None:
            results.append({"file": f.name, "error": "decode_failed"})
            no_detect.append(f.name)
            continue
        plates = api._recognize(img)
        if plates:
            per_prefix[prefix]["detected"] += 1
            per_prefix[prefix]["plates"] += len(plates)
            # best = highest ocr_conf
            best = max(plates, key=lambda p: (p.ocr_conf or 0.0))
            confs.append(best.ocr_conf or 0.0)
            annotate(img, plates)
            cv2.imwrite(str(ann_dir / f"{f.stem}.jpg"), img)
        else:
            no_detect.append(f.name)
        results.append({
            "file": f.name,
            "prefix": prefix,
            "count": len(plates),
            "plates": [
                {"text": p.text, "ocr_conf": p.ocr_conf, "det_conf": p.det_conf,
                 "region": p.region, "legit": p.legit,
                 "bbox": [p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2]}
                for p in plates
            ],
        })
        if i % 50 == 0:
            print(f"  ...{i}/{len(files)}  ({time.time()-t0:.0f}s)")

    dt = time.time() - t0
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    detected = sum(1 for r in results if r.get("count"))
    confs_np = np.array(confs) if confs else np.array([0.0])
    print("\n================ SUMMARY ================")
    print(f"images            : {len(files)}")
    print(f"with >=1 detection: {detected} ({100*detected/max(1,len(files)):.1f}%)")
    print(f"no detection      : {len(no_detect)}")
    print(f"total plate reads : {sum(r.get('count',0) for r in results)}")
    print(f"time              : {dt:.0f}s  ({dt/max(1,len(files))*1000:.0f} ms/img)")
    print(f"best-ocr-conf      mean={confs_np.mean():.3f} "
          f"min={confs_np.min():.3f} p10={np.percentile(confs_np,10):.3f} "
          f"median={np.median(confs_np):.3f}")
    lowconf = sum(1 for c in confs if c < 0.8)
    print(f"best-conf < 0.80  : {lowconf}")
    print("\nper-camera:")
    for pfx in sorted(per_prefix):
        d = per_prefix[pfx]
        print(f"  {pfx}: n={d['n']:>3} detected={d['detected']:>3} "
              f"({100*d['detected']/max(1,d['n']):.0f}%) plates={d['plates']}")

    # sample reads
    print("\nsample reads (first 20 detected):")
    shown = 0
    for r in results:
        if r.get("count"):
            p = max(r["plates"], key=lambda x: (x["ocr_conf"] or 0.0))
            oc = f"{p['ocr_conf']:.3f}" if p["ocr_conf"] is not None else "n/a"
            print(f"  {r['file']:<40} '{p['text']}' conf={oc} region={p['region']}")
            shown += 1
            if shown >= 20:
                break

    if no_detect:
        print(f"\nno-detection files ({len(no_detect)}):")
        for n in no_detect[:30]:
            print(f"  {n}")
        if len(no_detect) > 30:
            print(f"  ...(+{len(no_detect)-30} more)")

    print(f"\nresults.json + annotated/ written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

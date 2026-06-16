"""A/B eval: SR engine OFF vs ON over a folder of frames, same detectors + OCR.

Runs the real TrafficAnalysisService pipeline (vehicle-detect -> plate-detect ->
[SR] -> OCR) per frame, for each SR engine, and reports plate-read count + mean
OCR confidence. Same OCR engine across arms, so the delta isolates SR's effect on
the plate crop.

Usage:
    uv run --with-requirements requirements.txt --with fast-plate-ocr \
        scripts/eval_sr_ab.py <frames_dir> [sr_engine ...]
Default arms: none bicubic
"""
import os
import sys
import glob
from types import SimpleNamespace

import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils.traffic_analysis import TrafficAnalysisService  # noqa: E402


def build_service(sr_engine: str) -> TrafficAnalysisService:
    opts = SimpleNamespace(
        vehicle_weight=os.path.join(REPO, "weights/vehicle/vehicle_yolov9s_640_30oct2025.pt"),
        plate_weight=os.path.join(REPO, "weights/plate/plate_yolov9s_640_2025.pt"),
        dsort_weight=os.path.join(REPO, "weights/tracking/deepsort/ckpt.t7"),
        vconf=0.4,
        pconf=0.25,
        ocr_thres=0.9,
        ocr_engine="fpo",                      # ONNX, M1-friendly; paddle needs paddlepaddle
        fpo_model="cct-s-v2-global-model",
        device="cpu",
        deepsort=False,
        read_plate=True,
        lang="en",
        sr_engine=sr_engine,
        sr_scale=2,
    )
    return TrafficAnalysisService(opts)


def eval_arm(sr_engine: str, frames: list) -> dict:
    svc = build_service(sr_engine)
    rows = []
    for fp in frames:
        img = cv2.imread(fp)
        if img is None:
            continue
        svc.process_image(img)  # resets tracker, single-frame
        for tid, v in svc.vehicles.items():
            if v.ocr_conf > 0 and v.plate_number:
                rows.append((os.path.basename(fp), v.plate_number, round(float(v.ocr_conf), 3)))
    n = len(rows)
    mean_conf = round(sum(r[2] for r in rows) / n, 3) if n else 0.0
    return {"engine": sr_engine, "reads": n, "mean_conf": mean_conf, "rows": rows}


def main() -> None:
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "data/overhead-frames")
    arms = sys.argv[2:] if len(sys.argv) > 2 else ["none", "bicubic"]
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")) +
                    glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"frames: {len(frames)} from {frames_dir}")
    print(f"arms:   {arms}\n")

    results = []
    for arm in arms:
        print(f"--- arm: SR_ENGINE={arm} ---")
        res = eval_arm(arm, frames)
        for fn, plate, conf in res["rows"]:
            print(f"    {fn:28s} {plate:14s} conf={conf}")
        print(f"    => reads={res['reads']} mean_conf={res['mean_conf']}\n")
        results.append(res)

    print("=== SUMMARY ===")
    print(f"{'engine':12s} {'reads':>6s} {'mean_conf':>10s}")
    for r in results:
        print(f"{r['engine']:12s} {r['reads']:>6d} {r['mean_conf']:>10.3f}")


if __name__ == "__main__":
    main()

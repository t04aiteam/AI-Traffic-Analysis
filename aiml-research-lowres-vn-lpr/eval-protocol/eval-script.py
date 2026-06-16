"""Eval harness — low-res VN LPR. Chạy: uv run eval-script.py --gt gt.csv --pred preds/ --out results.csv
Tính full-plate exact-match + partial(>=6,>=5) + char-acc cho mỗi cấu hình.
KHÔNG bao gồm model inference — chỉ chấm điểm output đã có (mỗi cấu hình 1 file pred csv: image,pred_plate).
"""
import argparse, csv, glob, os
from collections import defaultdict


def norm(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch.isalnum())


def char_correct(pred: str, gt: str) -> int:
    # số ký tự đúng theo vị trí (so theo độ dài gt)
    return sum(1 for i, c in enumerate(gt) if i < len(pred) and pred[i] == c)


def score(gt_map: dict, pred_csv: str):
    full = ge6 = ge5 = 0
    char_ok = char_tot = 0
    n = 0
    with open(pred_csv, newline="") as f:
        for row in csv.DictReader(f):
            img, pred = row["image"], norm(row.get("pred_plate", ""))
            if img not in gt_map:
                continue
            gt = norm(gt_map[img])
            n += 1
            cc = char_correct(pred, gt)
            char_ok += cc
            char_tot += len(gt)
            if pred == gt:
                full += 1
            if cc >= 6:
                ge6 += 1
            if cc >= 5:
                ge5 += 1
    if n == 0:
        return None
    return {
        "n_samples": n,
        "full_plate_acc": round(100 * full / n, 2),
        "partial_ge6": round(100 * ge6 / n, 2),
        "partial_ge5": round(100 * ge5 / n, 2),
        "char_acc": round(100 * char_ok / max(char_tot, 1), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="CSV: image,plate (ground truth)")
    ap.add_argument("--pred", required=True, help="dir chứa <config_id>.csv (image,pred_plate)")
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    gt_map = {}
    with open(a.gt, newline="") as f:
        for row in csv.DictReader(f):
            gt_map[row["image"]] = row["plate"]

    rows = []
    for pred_csv in sorted(glob.glob(os.path.join(a.pred, "*.csv"))):
        cid = os.path.splitext(os.path.basename(pred_csv))[0]
        s = score(gt_map, pred_csv)
        if s:
            s["config_id"] = cid
            rows.append(s)
            print(f"{cid}: full={s['full_plate_acc']}% ge6={s['partial_ge6']}% "
                  f"ge5={s['partial_ge5']}% char={s['char_acc']}% n={s['n_samples']}")

    if rows:
        cols = ["config_id", "n_samples", "full_plate_acc", "partial_ge6", "partial_ge5", "char_acc"]
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {a.out}")
    else:
        print("No matching predictions scored.")


if __name__ == "__main__":
    main()

"""POST given image path(s) to running ALPR API. Usage: detect_image.py PATH [PATH...]"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:7862"


def post_image(image_path: Path) -> dict:
    boundary = "----TonAIBoundary"
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else "image/jpeg"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + image_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + "/predict/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_error_code": e.code, "_error_body": e.read().decode(errors="replace")}


def main(args: list[str]) -> int:
    if not args:
        print("usage: detect_image.py PATH [PATH...]")
        return 2
    for p in args:
        path = Path(p)
        if not path.exists():
            print(f"SKIP: {p} missing")
            continue
        urllib.request.urlopen(BASE + "/reset", data=b"")  # reset tracker per image
        print(f"=== {path} ===")
        res = post_image(path)
        if "_error_code" in res:
            print(json.dumps(res, indent=2))
            continue
        dets = res.get("detections", [])
        print(f"total={len(dets)}")
        plates = [d for d in dets if d.get("license_plate") and d["license_plate"] != "nan"]
        print(f"with_plate={len(plates)}")
        for d in plates:
            print(f"  id={d['track_id']:>3} type={d['vehicle_type']:<10} plate={d['license_plate']:<14} conf={d.get('confidence')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

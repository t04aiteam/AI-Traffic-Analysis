"""End-to-end HTTP test against a running plate_ocr_api server.

Posts real plate image(s) to /predict and /predict/annotated and prints the
result. Defaults to the fast-plate-ocr sample crops if no path is given.

Usage:
    uv run python scripts/e2e_plate_ocr_api.py [IMAGE ...]
Env:
    BASE (default http://127.0.0.1:7863)
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

BASE = os.environ.get("BASE", "http://127.0.0.1:7863")
DEFAULT_IMAGES = [
    "/Users/logan/Developer/vibes/WORK/LIPLA/fast-plate-ocr/test/assets/test_plate_1.png",
    "/Users/logan/Developer/vibes/WORK/LIPLA/fast-plate-ocr/test/assets/test_plate_2.png",
]


def _multipart(path: Path) -> tuple[bytes, str]:
    boundary = "----TonAIBoundary"
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def post(endpoint: str, path: Path) -> bytes:
    body, ctype = _multipart(path)
    req = urllib.request.Request(
        BASE + endpoint, data=body, headers={"Content-Type": ctype}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main(argv: list[str]) -> int:
    images = argv or DEFAULT_IMAGES

    print(f"=== GET {BASE}/health ===")
    with urllib.request.urlopen(BASE + "/health", timeout=30) as r:
        print(r.read().decode())
    print(f"\n=== GET {BASE}/config ===")
    with urllib.request.urlopen(BASE + "/config", timeout=30) as r:
        print(r.read().decode())

    for img in images:
        p = Path(img)
        if not p.exists():
            print(f"\nSKIP {img}: missing")
            continue
        print(f"\n=== POST /predict  ({p.name}) ===")
        print(json.dumps(json.loads(post("/predict", p)), indent=2))

        # annotated -> save bytes, confirm it's a JPEG
        ann = post("/predict/annotated", p)
        out = Path(os.environ.get("ANNOT_DIR", ".")) / f"annotated_{p.stem}.jpg"
        out.write_bytes(ann)
        is_jpeg = ann[:2] == b"\xff\xd8"
        print(f"  annotated -> {out} ({len(ann)} bytes, jpeg={is_jpeg})")

    print("\nE2E complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

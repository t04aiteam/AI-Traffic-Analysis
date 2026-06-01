"""Hit local API endpoints + run prediction on demo image."""
import json
import urllib.request
import sys
from pathlib import Path

BASE = "http://127.0.0.1:7862"


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())


def post_image(path: str, image_path: Path) -> dict:
    boundary = "----TonAIBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_error_code": e.code, "_error_body": e.read().decode(errors="replace")}


def main() -> int:
    print("HEALTH:", json.dumps(get("/health"), indent=2))
    print("CONFIG:", json.dumps(get("/config"), indent=2))
    demo = Path("data/assets/demo.jpg")
    if not demo.exists():
        print(f"FAIL: missing demo image {demo}")
        return 1
    print(f"POST /predict/image with {demo} ...")
    result = post_image("/predict/image", demo)
    print("PREDICT:", json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

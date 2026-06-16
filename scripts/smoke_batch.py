"""
Smoke test for POST /predict/batch using TestClient (in-process, no server needed).
Usage: uv run python /tmp/claude/smoke_batch.py /path/to/images/
"""
import io
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Add project root to path so utils/tracking packages resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

IMAGE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
images = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in EXTS)

if not images:
    print(f"No images found in {IMAGE_DIR}")
    sys.exit(1)

print(f"Found {len(images)} images: {[p.name for p in images]}\n")

# --- Boot app with mocked models ---
from utils.traffic_analysis import TrafficAnalysisService

sys.modules.pop("main", None)
with (
    patch("utils.traffic_analysis.YOLO"),
    patch("utils.traffic_analysis.Sort"),
    patch("utils.traffic_analysis.DeepSort"),
    patch.object(TrafficAnalysisService, "_init_ocr_engine"),
    patch.object(TrafficAnalysisService, "_init_sr_engine"),
):
    import main
    # Patch detect_vehicles_only to draw a red border so we can confirm it ran
    import cv2, numpy as np
    def fake_detect(frame):
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        return out
    main.traffic_service.detect_vehicles_only = fake_detect

    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    # --- Health check ---
    r = client.get("/health")
    print("GET /health:", r.status_code, r.json())

    # --- Batch predict ---
    files = [
        ("files", (img.name, img.read_bytes(), "image/jpeg"))
        for img in images
    ]
    r = client.post("/predict/batch", files=files)
    print(f"\nPOST /predict/batch ({len(images)} files): status={r.status_code}")
    ct = r.headers.get("content-type", "")
    print(f"Content-Type: {ct}")

    if r.status_code != 200:
        print("FAIL:", r.text)
        sys.exit(1)

    if "zip" in ct:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            names = zf.namelist()
            print(f"ZIP entries ({len(names)}): {names}")
            # Verify each entry is a decodable JPEG
            for name in names:
                data = zf.read(name)
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                ok = img is not None
                size = f"{img.shape[1]}x{img.shape[0]}" if ok else "DECODE FAIL"
                print(f"  {name}: {'OK' if ok else 'FAIL'} {size}")
        # Save zip for inspection
        out_path = Path("/tmp/claude/smoke_out.zip")
        out_path.write_bytes(r.content)
        print(f"\nSaved zip → {out_path}")
    else:
        # Single image
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ok = img is not None
        size = f"{img.shape[1]}x{img.shape[0]}" if ok else "DECODE FAIL"
        print(f"Single JPEG: {'OK' if ok else 'FAIL'} {size}")
        out_path = Path("/tmp/claude/smoke_out.jpg")
        out_path.write_bytes(r.content)
        print(f"Saved → {out_path}")

    print("\nSMOKE TEST PASSED")

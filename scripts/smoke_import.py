"""Smoke test: import main app and report status."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import app, traffic_service
    print("IMPORT_OK")
    print(f"device={traffic_service.opts.device}")
    print(f"vehicle_weight={traffic_service.opts.vehicle_weight}")
    print(f"plate_weight={traffic_service.opts.plate_weight}")
except Exception as e:
    print(f"IMPORT_FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

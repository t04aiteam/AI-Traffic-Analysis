#!/usr/bin/env bash
# Install the vendored fusion engines (mf-lpr2 + eott) into the MAIN venv so
# multi-frame plate fusion runs in-process on the single API port (7862) — no
# separate sidecar.
#
# --no-deps is deliberate: mf-lpr2 declares `opencv-contrib-python`, which would
# clash with this project's `opencv-python-headless` (both provide `cv2`). The
# engines only use core cv2 (eott's SURF path is hasattr-guarded), so the
# existing headless build is sufficient. scipy is tracked in pyproject.
#
# Run AFTER `uv sync` (which prunes packages not in pyproject, i.e. these three).
set -euo pipefail
cd "$(dirname "$0")/.."

git submodule update --init --recursive
uv pip install --no-deps -e fusion_svc/external/mf-lpr2 \
                         -e fusion_svc/external/eott \
                         -e fusion_svc

uv run python -c "from fusion_svc.adapters.eott_adapter import fuse_eott; \
from fusion_svc.adapters.mflpr2_adapter import fuse_mflpr2; \
print('fusion engines installed in-process: mflpr2, eott')"

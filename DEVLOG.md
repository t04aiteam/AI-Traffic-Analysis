# DEVLOG · AI-Traffic-Analysis

## Open ideas
- [ ] Cap PP-OCR text-detection input (`text_det_limit_type="max"`, `text_det_limit_side_len=960`) in `_ensure_dual_ocr`: measured 1.1TL.jpg 49.8s -> 5.8s, 14TLmscah.JPG 161s -> 10.7s on box 222 CPU, and reads more of the plate (added 2026-11-26)
- [ ] Plate endpoint for full photos: detect plate first, OCR the crop, return JSON text; fall back to whole-image OCR when no plate is found (pinned 2026-11-26, revisit later)

## Rejected
- 2026-11-26: Ask API users to shrink images before upload · users won't do it; the fix belongs server-side

## Log

### 2026-11-26 · Deploy Traffic AI on box 222 (CPU) + OCR speed investigation [in-progress]
- **Goal:** shareable API for still-plate OCR over the c09 VPN at `http://10.8.0.16:7862`
- **Steps:** container `traffic-ai-7862` (image `traffic-ai:cpu-01787c7`) published on `0.0.0.0:7862`; Postman collection + guides point at 10.8.0.16; memory limit 8 GiB -> 24 GiB
- **Result:** small plate crop 1.9s. On `lipla-ocr` photos (median 1.2 MP, max 17.7 MP) one request took 16s / 57s / 240s. 24 GiB only helped the largest (240s -> 172s). Cause: PP-OCRv6 detection runs at full resolution; fast-plate-ocr takes 0.2-0.3s at every size
- **Next:** apply the detection cap (see Open ideas)

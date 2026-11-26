# DEVLOG · AI-Traffic-Analysis

## Open ideas
- [ ] Clean PP-OCR text server-side (drop `NNNmm`, return plate-shaped string): 45/45 on front views (added 2026-11-26)
- [ ] Plate endpoint for full photos: detect plate first, OCR the crop, return JSON text; fall back to whole-image OCR when no plate is found (pinned 2026-11-26, revisit later)

## Rejected
- 2026-11-26: Pad close-ups 2-3x before plate detection · 32/45 vs 45/45 for whole-image OCR + cleaning, and adds stray digits
- 2026-11-26: Ask API users to shrink images before upload · users won't do it; the fix belongs server-side

## Log

### 2026-11-26 · Deploy Traffic AI on box 222 (CPU) + OCR speed investigation [done]
- **Goal:** shareable API for still-plate OCR over the c09 VPN at `http://10.8.0.16:7862`
- **Steps:** container `traffic-ai-7862` (image `traffic-ai:cpu-01787c7`) published on `0.0.0.0:7862`; Postman collection + guides point at 10.8.0.16; memory limit 8 GiB -> 24 GiB
- **Result:** small plate crop 1.9s. On `lipla-ocr` photos (median 1.2 MP, max 17.7 MP) one request took 16s / 57s / 240s. 24 GiB only helped the largest (240s -> 172s). Cause: PP-OCRv6 detection runs at full resolution; fast-plate-ocr takes 0.2-0.3s at every size
- **Capped run (960 px det, all 168 `lipla-ocr` images, offline script in the container, API untouched):** 24.0 min total, 8.6s/image avg, max 13.1s; >4MP images avg 7.9s (was 57-240s). Plate-like text read on 36/168; both engines agree on all 34 that fast-plate-ocr read. Misses are mostly not readable plates: `.2` = mirrored back side 1/38, `.3`/`.4` = emblem close-ups with rulers 0/76; `.1` front views 27/38
- **Shipped cap** (`f396168`, image `traffic-ai:cpu-f396168`): live API over VPN 1.1TL.jpg 57s -> 10.6s, 14TLmscah.JPG 240s -> 24.8s, 6.4.JPG 16s -> 15.5s
- **/predict/plates/batch on all 168 (one request):** 310s total (~1.8s/image incl. 140 MB upload). Plate detector misfires on these close-up plate shots: boxes single characters or half a plate (1.1TL: one box on the `0`, both OCR empty; 2.1: boxes `CD` and `.34`)
- **Experiment, 45 front-view `.1` images (solutions 1+2):** text cleaning (drop `NNNmm`, keep VN plate shape incl. `19-636-NN-01`) on whole-image OCR: 45/45 plate read by both engines, engines agree 43/45 (2 diffs: fast-plate-ocr drops a digit on NN-012/NN-345, PP-OCR correct). Plate detection: pad 1x 1/45, 2x 18/45, 3x 32/45 (adds a stray digit on 17.1, 18.1). Whole image + cleaning wins; padding not worth shipping. Earlier 27/38 front-view count was low because the regex missed NN plates
- **Next:** plate-JSON endpoint (Open ideas) needs a whole-image fallback when the plate fills the frame

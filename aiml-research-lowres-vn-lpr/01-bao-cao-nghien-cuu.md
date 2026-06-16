# Báo cáo Khảo sát AI/ML — Nhận dạng Biển số Xe Việt Nam Độ phân giải thấp (Low-Resolution VN LPR)

> **Trí Nam — AI Engineer R&D** · Ngày: 2026-06-08 · Người thực hiện: Senior AI Engineer (Claude)
> **NotebookLM notebook:** `AIML-lowres-VN-LPR-20260608` (id `310ccde0-1ca0-4717-98d9-b5577dbb4731`)
> Mọi metric có `verification_status`: **VERIFIED** (citation NBLM) / **NBLM_ONLY** (chưa verify) / LOẠI nếu HALLUCINATED.

---

## 1. Tóm tắt điều hành

Đọc biển số xe VN từ ảnh độ phân giải thấp (camera giám sát, phạt nguội, ảnh xa/mờ/nén) là **pipeline 3 mắt xích**: **Detection → Super-Resolution (SR) → OCR**. Với ảnh low-res, SR là mắt xích quyết định: OCR chạy thẳng trên ảnh LR cho recognition rate **chỉ 1.1%–9.9%**, sau SR tăng lên **39%–49.8%** full-plate exact-match [VERIFIED].

**Khuyến nghị cốt lõi:**
1. **Pipeline production nền:** `fast-alpr` + `fast-plate-ocr` (MIT, ONNX/CUDA, ~3000 plates/s @RTX3090) — POC chạy trong ngày.
2. **Module SR low-res:** **LCOFL/LCDNet** (layout-aware + GAN + OCR-loss) — SOTA recognition-after-SR, **phải retrain trên data VN** (dataset gốc non-commercial).
3. **Fine-tune bắt buộc:** Không model nào train sẵn biển VN ở mức production. Biển VN (1+2 line, font riêng) ≠ CCPD(TQ)/RodoSol(Brazil)/Saudi. **Domain shift = rủi ro #1.**
4. **License:** detector họ YOLO Ultralytics (v5/v8/v11) = **AGPL-3.0**; openalpr = AGPL. Cần commercial license hoặc thay detector.

**Feasibility: HIGH.** Blocker duy nhất = chưa có data VN → phải thu thập + label.

---

## 2. Problem Statement (BƯỚC 0)

| Field | Value |
|---|---|
| Task | Low-resolution Vietnamese License Plate Recognition |
| Pipeline | Detection → Super-Resolution → OCR |
| Metric | Full-plate recog acc (exact-match %); phụ: char-acc, ≥6/≥5 char; SR: PSNR/SSIM; detect: mAP@0.5 |
| Deploy | GPU NVIDIA server/cloud |
| Data | Chưa có → thu thập VN + public pretrain |
| Domain risk | CAO |
| Compliance | Biển số = PII, NĐ 13/2023 |
| Mục tiêu | Baseline 3-6 tháng + paper refs + full hand-off |

---

## 3. Tại sao low-res LPR khó (VERIFIED — paper LCOFL 2408.15103, RodoSol degraded)

| Test | Full-plate | ≥6 char | ≥5 char | Status |
|---|---|---|---|---|
| HR gốc | 98.5% | 99.9% | 99.9% | VERIFIED |
| **LR degraded** | **1.1%** | 5.3% | 14.3% | VERIFIED |
| LR+SR PLNET | 39.0% | 59.9% | 74.2% | VERIFIED |
| LR+SR SR3 (diffusion) | 43.1% | 67.5% | 82.2% | VERIFIED |
| **LR+SR LCOFL** | **49.8%** | 71.2% | 83.3% | VERIFIED |

Real-world (3,723 cặp LR-HR thật): LR=9.9% → SR3=31.7% → PLNET=36.3% → **LCOFL=39.5%** [VERIFIED].
UFPR-SR-Plates (2505.06393, 100k cặp thật): LR=1.7% → LCDNet=31.1% → **LCDNet+MVCP(vote 5 ảnh)=44.7%** [VERIFIED] → **đa khung + vote = đòn bẩy lớn cho video.**

**Lỗi đặc trưng low-res** [VERIFIED]: nhầm `B↔8`, `G↔6/0`, `T↔7`, `S↔5`, `2↔Z`, `R↔B` → lý do SR layout-aware thắng SR thường.

---

## 4. So sánh 3 paradigm SR (cross-verify)

| Paradigm | Đại diện | Best recog-after-SR | Ưu | Nhược | Status |
|---|---|---|---|---|---|
| **Layout/Character-driven** | LCOFL, LCDNet | **49.8%** RodoSol / 44.7% UFPR | Cao nhất; phạt nhầm ký tự+layout | Test ở Brazil/Mercosur, retrain layout VN | VERIFIED acc; tổng quát = single-source |
| **Diffusion** | SR3 | 43.1% RodoSol | Ảnh đẹp, 92% người thích | **33h/4k ảnh, 52× chậm** → loại realtime | VERIFIED |
| **Attention+sub-pixel** | PLNET (2305.17313) | 39.0% | Nhanh, nền LCOFL | Nhầm ký tự S→5,G→0 | VERIFIED |
| **Scene-text SR** | TATSR (2210.06924) | NOT_FOUND | Đa ngôn ngữ | Không tối ưu layout biển | NBLM_ONLY (chỉ abstract) |

**Kết luận SR (GPU server):** ưu tiên **LCOFL/LCDNet** (cao nhất + nhanh hơn diffusion ~52×). SR3 chỉ batch offline. Video → cộng **MVCP vote nhiều khung** (+13.6 điểm % UFPR).

---

## 5. TOP 5 REPO (BƯỚC 4)

| # | Repo | Vai trò | License | Commercial | Pretrained | Điểm |
|---|---|---|---|---|---|---|
| 1 | **ankandrew/fast-alpr** | Pipeline ANPR production | MIT | ✅ | ✅ global models | **84** |
| 2 | **ankandrew/fast-plate-ocr** | OCR engine fine-tune VN | MIT | ✅ | ✅ cct models | **82** |
| 3 | **valfride/lpsr-lacd** (LCOFL) | SR low-res SOTA | Code: chưa rõ; **Data: non-commercial** | ⚠️ retrain | academic-only | **74** |
| 4 | **mrzaizai2k/...YOLOv7-and-CNN** | Tham chiếu VN 1&2-line | MIT (repo) | ✅¹ | ✅ .pt+.h5 | **70** |
| 5 | **detectRecog/CCPD** (RPnet) | Dataset+baseline | MIT | ✅ | ✅ fh02.pth | **72** |

¹ Repo MIT nhưng dùng YOLOv7 (GPL-3.0 WongKinYiu) — kiểm khi nhúng thương mại.

### Rubric chi tiết (7 tiêu chí / 100đ)

| Tiêu chí (trọng số) | fast-alpr | fast-plate-ocr | lpsr-lacd | mrzaizai2k | CCPD |
|---|---|---|---|---|---|
| Accuracy vs SOTA (20) | 15 | 15 | 19 | 11 | 12 |
| Reproducibility (20) | 18 | 18 | 15 | 13 | 12 |
| Pretrained public (15) | 14 | 14 | 8 | 13 | 13 |
| Code & docs (15) | 14 | 13 | 12 | 10 | 9 |
| Cộng đồng (10) | 8 (552★,3mo) | 9 (578★,3mo) | 6 (SIBGRAPI'24) | 5 (39★,4y) | 9 (2.6k★) |
| License TM (10) | 10 | 10 | 4 | 9 | 10 |
| Phù hợp HW (10) | 5 | 3 | 10 | 9 | 7 |
| **Tổng** | **84** | **82** | **74** | **70** | **72** |

### Repo LOẠI sớm

| Repo/Model | Lý do |
|---|---|
| openalpr | **AGPL-3.0**; engine cũ 6+ năm, Tesseract, yếu low-res (có commercial license qua info@openalpr.com) |
| morsetechlab/yolov11 | **AGPL-3.0** + README tự thú **train/test contamination → metric thổi phồng** |
| trungdinh22 (VN YOLOv5) | **Không LICENSE** = all-rights-reserved; YOLOv5 4 năm |
| winter2897 (VN Jetson) | Không LICENSE; target Jetson edge ≠ GPU server; code 6 năm |
| keremberke/yolov5m | Không khai license; YOLOv5 Ultralytics = AGPL ngầm |
| DunnBC22/trocr | Cho biển in sạch (CER 0.0368) không low-res; base trocr ràng buộc |
| hezarai/crnn-fa | **Persian only** — sai domain |

---

## 6. Khuyến nghị triển khai (thứ tự thử)

```
GĐ1 (tuần 1) POC: fast-alpr trên ảnh test VN → đo recog gốc, xác định ngưỡng "đủ phân giải"
GĐ2 (tuần 2-4) Data+fine-tune: thu 5-10k ảnh VN (1&2-line), label; fine-tune fast-plate-ocr ký tự VN; tham chiếu mrzaizai2k cho 2-line
GĐ3 (tháng 2-3) SR: cài LCOFL/LCDNet, RETRAIN trên cặp LR-HR VN tự tạo; chèn SR giữa detect↔OCR; A/B vs bicubic; video→thêm MVCP
GĐ4 Benchmark: pretrain detector trên CCPD-Blur/DB trước fine-tune VN; đo theo eval-protocol/
```

### Baseline đơn giản (đối chứng)
`YOLO detect → bicubic ×2 → fast-plate-ocr/PaddleOCR`. Mọi SR phải vượt baseline. Tham chiếu bicubic≈1.1–9.9% [VERIFIED] → SR phải ≥35–50%.

---

## 7. Datasets công khai

| Dataset | Vùng | Đặc điểm | License | Dùng cho |
|---|---|---|---|---|
| **CCPD2019** (300k) | TQ | Subset **Blur/DB/Challenge** = low-quality thật | MIT | Pretrain detector+recog |
| **UFPR-SR-Plates** (100k cặp) | Brazil | Cặp LR-HR thật, 5 khung/xe | Academic (2505.06393) | Benchmark SR + học MVCP |
| **RodoSol-SR/ALPR** | Brazil/Mercosur | Cặp LR-HR cho LCOFL | **Non-commercial, .edu email** | Chỉ nghiên cứu |
| **(cần tạo) VN-LR-HR** | **VN** | Tự thu HR rồi degrade | Nội bộ | **Bắt buộc** production |

> Tạo cặp LR-HR VN [VERIFIED, pp Nascimento]: ảnh HR → lặp nhiễu Gaussian + bicubic tới SSIM<0.1 → pad gray giữ tỉ lệ → LR 16×48, HR 32×96 (×2).

---

## 8. Plan B (nếu 5 repo fail)

1. OCR: PaddleOCR PP-OCRv4 (Apache) fine-tune VN — hỗ trợ tiếng Việt, cộng đồng lớn.
2. Detector tránh AGPL: RT-DETR/DETR (Apache) hoặc mua Ultralytics Enterprise.
3. SR thay: Real-ESRGAN (BSD) + OCR-loss fine-tune; hoặc TATSR.
4. End-to-end không SR: recognizer robust train thẳng trên LR + heavy aug (CCPD-Blur style).
5. API thương mại (Plate Recognizer) cho POC nhanh khi build in-house.

---

## 9. Rủi ro & Compliance (chi tiết 03-tco-risk-register.md)

- ⚠️ **AGPL** (YOLO Ultralytics, openalpr) → SaaS phải mở source hoặc mua commercial. Dùng ONNX export riêng / RT-DETR.
- ⚠️ **Data non-commercial** (RodoSol-SR) → chỉ học method LCOFL, retrain data VN.
- ⚠️ **Domain shift VN** → không zero-shot; bắt buộc fine-tune.
- ⚠️ **NĐ 13/2023 (PII):** biển số = dữ liệu cá nhân → cơ sở pháp lý, mã hóa, hạn chế truy cập.
- ⚠️ **Train/test contamination** (morsetechlab) → tự giữ held-out test VN.

---

## 10. Nguồn & verification

Xem `02-bang-20-bai-viet.md`, `nblm-output.json` (metric+citation), `nblm-audit.log` (compliance). Số liệu chính VERIFIED qua NBLM citation; NBLM_ONLY/single-source đã đánh dấu. Không số nào từ trí nhớ.

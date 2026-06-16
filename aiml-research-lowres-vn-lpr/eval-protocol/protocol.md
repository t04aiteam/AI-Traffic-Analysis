# Eval Protocol — Low-res VN LPR (đối chứng 5 cấu phần)

## Mục tiêu
So sánh fair pipeline ± SR trên cùng dữ liệu/HW/metric. Trả lời: SR giúp bao nhiêu? LCOFL vs bicubic vs diffusion?

## 1. Eval dataset
- **Public benchmark:** UFPR-SR-Plates (cặp LR-HR thật) + CCPD-Blur (low-quality).
- **VN holdout:** 100-500 ảnh biển VN tự thu (1&2-line), label tay. KHÔNG để lọt vào train (tránh contamination như morsetechlab).
- Tạo cặp LR-HR VN: HR → Gaussian noise + bicubic downscale tới SSIM<0.1 → LR 16×48, HR 32×96 (×2).

## 2. Common metric (1 cho tất cả)
- **Chính:** full-plate recognition accuracy (exact-match, mọi ký tự đúng).
- **Phụ:** ≥6 char, ≥5 char (forensic); char-level accuracy; SR: PSNR/SSIM; detect: mAP@0.5; latency FPS.

## 3. Common HW
- 1×GPU cố định (vd T4/L4 hoặc RTX), batch=1, fp16. Ghi rõ trong results.csv.

## 4. Same preprocessing
- Cùng crop+rectify biển trước OCR. Cùng input size cho SR (LR 16×48). Tránh A resize khác B.

## 5. Cấu hình so sánh (A/B)
| ID | Pipeline | Ghi chú |
|---|---|---|
| C0 | detect → bicubic ×2 → OCR | **Baseline** |
| C1 | detect → OCR (no SR) | Đối chứng "không SR" |
| C2 | detect → LCOFL SR → OCR | SOTA đề xuất |
| C3 | detect → SR3 diffusion → OCR | Chất lượng cao, chậm |
| C4 | detect → LCOFL SR → OCR + MVCP vote (video, 5 khung) | Nếu là video |

## 6. Paired test
Cùng input → so output từng cấu hình. Lưu predictions/cấu hình.

## 7. Failure case
Lưu 20 ca sai/cấu hình để inspect (đặc biệt nhầm B↔8, G↔6, 2↔Z). Soi SR có sửa được lỗi ký tự không.

## 8. Output
`results.csv` thống nhất (schema ở results-template.csv). Chạy `eval-script.py`.

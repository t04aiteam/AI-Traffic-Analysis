# TCO & Risk Register — Low-res VN LPR

## A. TCO (ước tính, GPU NVIDIA server)

> Tham chiếu tốc độ từ NBLM [VERIFIED]: fast-plate-ocr cct-xs ~3094 plates/s @RTX3090; LCOFL SR ~1.75 FPS (38min/4k ảnh) @Quadro RTX8000; SR3 diffusion ~0.03 FPS (33h/4k) — loại realtime.

### A.1 Inference cost (production)

| Cấu phần | Throughput | HW | $/giờ cloud¹ | $/1k ảnh |
|---|---|---|---|---|
| Detect+OCR (fast-alpr, ONNX/CUDA) | ~3000 plate/s | 1×T4/L4 | ~$0.35–0.6 | ~$0.00003 |
| + SR LCOFL (chèn trước OCR) | ~1.75 FPS² | 1×A10/RTX | ~$0.75–1.2 | ~$0.16–0.19 |
| SR3 diffusion (KHÔNG khuyến nghị realtime) | ~0.03 FPS | 1×A100 | ~$1.8–3 | ~$16–28 |

¹ Giá tham chiếu cloud GPU 2026 (T4/L4 rẻ, A100 đắt); on-prem rẻ hơn dài hạn.
² SR là nút cổ chai. Tối ưu: batch>1, TensorRT/fp16, hoặc chỉ SR ảnh OCR-fail (cascade).

### A.2 Training/Fine-tune cost (one-time)

| Task | Data | GPU-hour ước tính | Ghi chú |
|---|---|---|---|
| Fine-tune OCR ký tự VN (fast-plate-ocr) | 5-10k ảnh label | 5-20 GPU-h | Rẻ; cct nhỏ |
| Retrain LCOFL SR trên cặp LR-HR VN | 5-10k cặp tự tạo | 30-80 GPU-h | GAN + OCR-loss, nhiều epoch |
| Pretrain detector CCPD-Blur→fine-tune VN | CCPD 300k + VN | 20-60 GPU-h | YOLO/RT-DETR |

### A.3 On-prem vs Cloud
- **On-prem** (đã có GPU server): chỉ tốn điện+vận hành; phù hợp tải ổn định, dữ liệu PII nhạy cảm (NĐ13/2023 ưu tiên on-prem).
- **Cloud**: linh hoạt POC, scale spike phạt nguội; lưu ý compliance khi đẩy biển số (PII) lên cloud nước ngoài.

---

## B. Risk Register

| # | Rủi ro | Mức | Tác động | Giảm thiểu |
|---|---|---|---|---|
| R1 | **License AGPL** (YOLO Ultralytics v5/v8/v11, openalpr) | CAO | Buộc mở source SaaS hoặc vi phạm | Dùng detector ONNX export độc lập / RT-DETR (Apache) / mua Ultralytics Enterprise |
| R2 | **Data non-commercial** (RodoSol-SR academic-only) | CAO | Không được dùng weight/data thương mại | Chỉ học method LCOFL; **retrain trên data VN tự thu** |
| R3 | **Domain shift VN** (biển VN ≠ mọi benchmark) | CAO | Zero-shot fail; acc thấp | Thu thập 5-10k ảnh VN 1&2-line + fine-tune bắt buộc |
| R4 | **PII / NĐ 13/2023** (biển số = dữ liệu cá nhân) | CAO | Pháp lý | Cơ sở pháp lý xử lý, mã hóa lưu trữ, hạn chế truy cập, ưu tiên on-prem |
| R5 | **Train/test contamination** (metric repo thổi phồng — morsetechlab tự thú) | TRUNG | Đánh giá sai năng lực | Luôn tự giữ held-out test set VN; không tin metric repo |
| R6 | **SR là nút cổ chai latency** | TRUNG | Không đạt realtime | TensorRT/fp16, batch, cascade chỉ-SR-khi-OCR-fail; tránh diffusion |
| R7 | **2-line VN plate** (xe máy) | TRUNG | Recog sai layout | Tham chiếu mrzaizai2k; train riêng head 2-line; tránh model chỉ 1-line |
| R8 | **Code lpsr-lacd license chưa rõ (chỉ data non-commercial)** | TRUNG | Rủi ro dùng code | Liên hệ tác giả xác nhận code license; hoặc tự implement LCOFL từ paper |
| R9 | **Bitrot repo cũ** (trungdinh22/winter2897 4-6 năm) | THẤP | Dependency hell | Đã loại; dùng fast-alpr (commit 3 tháng) |
| R10 | **Drift** (camera/điều kiện thay đổi) | THẤP-TRUNG | Acc giảm theo thời gian | Monitor recog-rate, retrain định kỳ, uncertainty (paper 2302.01427) để flag low-confidence |

---

## C. License Matrix (VERIFIED qua NBLM + web_fetch LICENSE)

| Repo/Model | License | Commercial | Nguồn verify |
|---|---|---|---|
| fast-alpr | MIT | ✅ | NBLM (LICENSE=MIT) |
| fast-plate-ocr | MIT | ✅ | NBLM (MIT) |
| mrzaizai2k YOLOv7+CNN | MIT (repo); YOLOv7=GPL-3.0 | ⚠️ kiểm phần YOLOv7 | NBLM |
| CCPD (detectRecog) | MIT | ✅ | NBLM |
| valfride/lpsr-lacd code | Chưa khai SPDX | ⚠️ | web_fetch (không thấy LICENSE file) |
| RodoSol-SR dataset | **Non-commercial academic** (.edu email) | ❌ | web_fetch (license-agreement.pdf) |
| openalpr | AGPL-3.0 (có commercial qua liên hệ) | ⚠️ | NBLM |
| morsetechlab yolov11 | AGPL-3.0 | ⚠️ | NBLM |
| keremberke yolov5m | Không khai (YOLOv5=AGPL) | ⚠️ | NBLM NOT_FOUND |
| DunnBC22 trocr | MIT (+ base trocr license) | ⚠️ | NBLM |
| hezarai crnn-fa | Không khai (Persian) | n/a | NBLM NOT_FOUND |

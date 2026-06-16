# Install Guide — repo #1: ankandrew/fast-alpr (MIT) ⭐ TOP

**Vai trò:** Pipeline ANPR production (detect + OCR, ONNX Runtime). Điểm khởi đầu POC.
**URL:** https://github.com/ankandrew/fast-alpr · **License:** MIT (commercial ✅) · 552★, commit 3 tháng.

## Cài (GPU NVIDIA — uv, theo convention Trí Nam)
```bash
uv init lpr-poc && cd lpr-poc
uv add "fast-alpr[onnx-gpu]"        # NVIDIA CUDA (Linux/Windows)
# uv add "fast-alpr[onnx]"          # CPU cross-platform
# uv add "fast-alpr[onnx-openvino]" # Intel
```
> Upstream README dùng `pip` + install extra; ở đây map sang `uv add`. Backend ONNX bắt buộc 1 extra, không có sẵn.

## Demo inference (verbatim API NBLM)
```python
from fast_alpr import ALPR

alpr = ALPR(
    detector_model="yolo-v8-tiny-960-none-gray",
    ocr_model="ch_PP-OCRv3_det_rec-640-v3-none-gray",
)
prediction = alpr.predict("path/to/your/image.jpg")
print(prediction)
```
Chạy: `uv run demo.py`

## Lưu ý cho VN low-res
- Detector mặc định họ YOLOv8 (Ultralytics) = **AGPL ngầm** → production thương mại: export ONNX riêng hoặc đổi detector (fast-alpr cho swap model).
- OCR mặc định chưa biết ký tự VN → **fine-tune fast-plate-ocr** (repo #2).
- Low-res: chèn module SR (repo #3) **trước** OCR.

## Troubleshoot
- Thiếu ONNX backend → lỗi "no runtime": phải có 1 extra `[onnx-gpu]`/`[onnx]`.
- CUDA mismatch → khớp onnxruntime-gpu với CUDA driver.

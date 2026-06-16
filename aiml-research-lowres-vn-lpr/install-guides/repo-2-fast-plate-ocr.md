# Install Guide — repo #2: ankandrew/fast-plate-ocr (MIT) ⭐ OCR engine

**Vai trò:** OCR biển nhẹ/nhanh (chạy SAU detector, nhận ảnh crop biển). Fine-tune ký tự VN.
**URL:** https://github.com/ankandrew/fast-plate-ocr · **License:** MIT ✅ · 578★. cct-xs ~3094 plates/s @RTX3090 [VERIFIED].

## Cài (uv)
```bash
uv add "fast-plate-ocr[onnx-gpu]"   # inference GPU
# uv add "fast-plate-ocr[onnx]"     # inference CPU
# uv add "fast-plate-ocr[train]"    # train/fine-tune
```

## Inference (verbatim NBLM)
```bash
fast-plate-ocr predict --image-path /path/to/plate.png --model-config /path/to/model_config.yaml
```
```python
from fast_plate_ocr import ONNXPlateRecognizer
recognizer = ONNXPlateRecognizer("model_config.yaml", "plate_config.yaml", "model.onnx")
predictions = recognizer.predict(numpy_images)  # list ảnh crop biển (BGR)
```

## Fine-tune ký tự VN
- Tutorial: `examples/fine_tune_workflow.ipynb` (data → train → export ONNX/TFLite/CoreML).
- Backend Keras 3 (TF/JAX/PyTorch). Region recognition optional.
- Chuẩn bị tập ký tự VN đầy đủ (chữ + số + ký tự tỉnh). Biển 2-line: cân nhắc tách dòng trước OCR.

## Troubleshoot
- OCR kỳ vọng ảnh **đã crop biển** → luôn chạy detector trước.

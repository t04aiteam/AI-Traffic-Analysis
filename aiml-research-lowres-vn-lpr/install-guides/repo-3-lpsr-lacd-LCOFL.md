# Install Guide — repo #3: valfride/lpsr-lacd (LCOFL/LCDNet) ⚠️ SR SOTA, license cẩn trọng

**Vai trò:** Super-Resolution low-res biển (layout-aware + GAN + OCR-loss). Recog-after-SR SOTA: 49.8% RodoSol / 44.7% UFPR [VERIFIED].
**URL:** https://github.com/valfride/lpsr-lacd (SIBGRAPI 2024).
**License:** ⚠️ Code chưa khai SPDX; **dataset RodoSol-SR NON-COMMERCIAL academic-only (.edu email + agreement)**.
→ **Trí Nam: học method, RETRAIN trên cặp LR-HR VN tự tạo. Liên hệ tác giả xác nhận code license trước khi nhúng thương mại.**

## Cài
```bash
git clone https://github.com/valfride/lpsr-lacd && cd lpsr-lacd
uv venv && source .venv/bin/activate
uv pip sync requirements.txt        # hoặc: uv add $(cat requirements.txt) — PyTorch + deps
```

## Train from scratch (verbatim NBLM, map sang uv run)
Trong config đặt:
```yaml
LOAD_PRE_TRAINED_OCR: null
resume: null
```
```bash
uv run ParallelNetTrain.py --config ./config/Sibgrapi_ablation/cgnetV2_deformable_test.yaml --save True
```

## Train/fine-tune trên data VN (verbatim format split)
```
path/to/HR1.jpg;path/to/LR1.jpg;training
path/to/HR2.jpg;path/to/LR2.jpg;validation
path/to/HR3.jpg;path/to/LR3.jpg;testing
```
Config wrapper: `imgW: 48, imgH: 16, image_aspect_ratio: 3, background: (127,127,127)` (LR 16×48, HR 32×96, ×2).

## Tạo cặp LR-HR VN (pp Nascimento [VERIFIED])
HR biển VN → lặp Gaussian noise + bicubic downscale tới SSIM<0.1 → pad gray giữ tỉ lệ → LR 16×48 / HR 32×96.

## Lưu ý
- SR ~1.75 FPS @RTX8000 → nút cổ chai. Tối ưu batch/fp16/TensorRT, hoặc cascade (chỉ SR khi OCR-fail).
- Chèn SR **giữa** detect và OCR.

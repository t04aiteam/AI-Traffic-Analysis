# Install Guide — repo #5: detectRecog/CCPD (RPnet) (MIT) — dataset + baseline

**Vai trò:** Dataset 300k biển TQ (ECCV2018) + baseline end-to-end RPnet. Có **subset low-quality: Blur, DB, Challenge** → pretrain cho low-res.
**URL:** https://github.com/detectRecog/CCPD · **License:** MIT ✅ · 2.6k★.
**Số liệu [VERIFIED]:** SSD512+HC full-plate CCPD-Base 43.42% / CCPD-Blur 25.83%; YOLOv3-320 detect 52 FPS.

## Lấy dataset
```bash
git clone https://github.com/detectRecog/CCPD && cd CCPD
# Tải CCPD2019.tar.xz từ Google Drive / BaiduYun (code: hm0u) — link trong README
tar xf CCPD2019.tar.xz
# split file ở thư mục split/
```
Annotation nhúng trong tên file (area, tilt, bbox, 4 vertices, LP number index).

## Baseline RPnet (verbatim NBLM, map sang uv run)
```bash
# Môi trường gốc rất cũ: pytorch(0.3.1), numpy(1.14.3), cv2(2.4.9.1), Cuda 9.1
# 1) Train localization wR2 (hoặc tải pretrained wR2 từ drive) theo wR2.py
# 2) Train RPnet (rpnet.py) — set wR2Path = path wR2 đã train
# 3) Demo (cần fh02.pth):
uv run demo.py
```

## Giá trị cho dự án VN
- **Pretrain detector + recognizer trên CCPD-Blur/DB** (low-quality thật) → rồi fine-tune data VN. Warm-start tốt cho low-res.
- ⚠️ Biển TQ (1 ký tự tỉnh Hán + 6) khác VN → CHỈ pretrain, không deploy thẳng.
- ⚠️ Code PyTorch 0.3.1 rất cũ → cân nhắc chỉ lấy dataset, dùng detector hiện đại (fast-alpr/RT-DETR).

# Install Guide — repo #4: mrzaizai2k/License-Plate-Recognition-YOLOv7-and-CNN (MIT) — tham chiếu VN

**Vai trò:** Pipeline VN tham chiếu — YOLOv7 detect + Hough align + CNN char-recog. Xử lý **biển VN 1 & 2 line** (dataset 1000 ảnh VN).
**URL:** https://github.com/mrzaizai2k/License-Plate-Recognition-YOLOv7-and-CNN · **License:** MIT (repo) ✅; ⚠️ YOLOv7=GPL-3.0.

## Cài
```bash
git clone https://github.com/mrzaizai2k/License-Plate-Recognition-YOLOv7-and-CNN
cd License-Plate-Recognition-YOLOv7-and-CNN
uv venv && source .venv/bin/activate
uv pip sync requirements.txt
# Tải weights từ Git RELEASES: LP_detect_yolov7_500img.pt + weight.h5 → đặt đúng path trong code
```

## Demo (verbatim NBLM, map sang uv run)
```bash
uv run main_image.py    # sửa path ảnh trong script
uv run main_video.py    # video 1920x1080, 24fps khuyến nghị
```

## Giá trị cho dự án
- **Tham chiếu xử lý 2-line VN** (xe máy) + alignment Hough.
- Dataset VN 1000 ảnh + data.yaml để train YOLOv7 (trên Kaggle tác giả).
- ⚠️ Repo 4 năm, 39★ → tham chiếu logic, KHÔNG làm nền production (nền = fast-alpr).
- ⚠️ Thương mại: YOLOv7 GPL-3.0 — thay detector Apache nếu cần.

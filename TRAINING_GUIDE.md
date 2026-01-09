# Detector Training Quick Reference

Quick reference for training object detection models in the license-plate-recognition project.

## Available Detectors

| Detector | Location | Best For | Speed | Accuracy |
|----------|----------|----------|-------|----------|
| **YOLO** | `detectors/yolo/` | General detection, real-time | ⚡⚡⚡ Fast | ⭐⭐⭐ High |
| **RF-DETR** | `detectors/rf-detr/` | Transformer-based detection | ⚡⚡ Moderate | ⭐⭐⭐ High |

## Quick Start

### YOLO Training

```bash
# Simple training
./scripts/train_yolo.sh \
  --data data/license-plates/data.yaml \
  --model yolov8n.yaml

# Advanced training
./scripts/train_yolo.sh \
  --data data/license-plates/data.yaml \
  --model yolov9m.yaml \
  --epochs 200 \
  --batch 32 \
  --imgsz 640 \
  --device 0 \
  --name plate-detector
```

### RF-DETR Training

```bash
# Simple training
python detectors/rf-detr/train.py \
  --data data/license-plates \
  --epochs 50

# Advanced training
python detectors/rf-detr/train.py \
  --data data/license-plates \
  --model rfdetr-base \
  --epochs 100 \
  --batch 32 \
  --lr 1e-4 \
  --output runs/rf-detr/plates
```

## YOLO Models Available

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolov8n.yaml | Nano | Fastest | Good | Edge devices, RPi |
| yolov8s.yaml | Small | Very Fast | Better | Mobile, embedded |
| yolov8m.yaml | Medium | Fast | High | General use |
| yolov8l.yaml | Large | Moderate | Very High | Server deployment |
| yolov9c.yaml | Compact | Fast | High | Balanced |
| yolov9m.yaml | Medium | Moderate | Very High | High accuracy |
| yolo11n.yaml | Nano | Fastest | Good+ | Latest nano |
| yolo11m.yaml | Medium | Fast | Very High | Latest balanced |

## Common Parameters

### Training Parameters
```bash
--data         # Path to data.yaml file (required)
--model        # Model architecture (e.g., yolov8n.yaml)
--epochs       # Number of epochs (default: 200 for YOLO, 50 for RF-DETR)
--batch        # Batch size (default: 16)
--imgsz        # Image size (default: 640)
--device       # GPU device: '0', '0,1', or 'cpu' (default: '0')
--workers      # Dataloader workers (default: 8)
--name         # Experiment name
--output       # Output directory (RF-DETR only)
--resume       # Resume from checkpoint
```

### Inference Parameters (RF-DETR)
```bash
--source       # Image/video/directory path
--model        # Model name or weights path
--conf         # Confidence threshold (default: 0.5)
--save-img     # Save annotated images
--save-txt     # Save detection results as text
--output       # Output directory
```

## Dataset Format

Both detectors expect YOLO format:

```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── ...
    └── val/
        └── ...
```

**data.yaml:**
```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
nc: 2
names: ['car', 'plate']
```

**Label format (normalized):**
```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.2
```

## Inference

### YOLO Inference
```python
from ultralytics import YOLO
model = YOLO('runs/train/weights/best.pt')
results = model('image.jpg')
```

### RF-DETR Inference
```bash
python detectors/rf-detr/infer.py \
  --source test_images/ \
  --weights runs/rf-detr/best.pth \
  --conf 0.5 \
  --save-img
```

## Export Models

### YOLO to ONNX
```bash
python detectors/yolo/exporter.py \
  --weights runs/train/weights/best.pt \
  --dynamic
```

### Use Exported Model
```python
import onnxruntime
session = onnxruntime.InferenceSession("model.onnx")
outputs = session.run(None, {'images': input_data})
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch 8

# Or use gradient accumulation (RF-DETR)
--batch 8 --grad-accum-steps 4  # Effective batch: 32
```

### Slow Training
```bash
# Reduce workers if CPU bottleneck
--workers 4

# Use smaller image size
--imgsz 416
```

### Poor Results
```bash
# Increase epochs
--epochs 300

# Try different model size
--model yolov8m.yaml  # or yolov9m.yaml

# Lower confidence threshold for inference
--conf 0.3
```

## Performance Tips

1. **Batch Size**: Use largest that fits in GPU memory
2. **Image Size**: 640 is standard, 1280 for small objects
3. **Epochs**: 100-200 usually sufficient for small datasets
4. **Learning Rate**: Default works well, adjust if loss is unstable
5. **Data Augmentation**: Enabled by default, adjust in data.yaml

## File Locations

```
detectors/
├── README.md              # Complete documentation
├── yolo/                  # YOLO implementation
│   ├── train_ultralytics.py
│   ├── exporter.py
│   └── YOLOv8.py
├── rf-detr/              # RF-DETR implementation
│   ├── README.md         # RF-DETR docs
│   ├── train.py
│   └── infer.py
├── train_yolo.sh         # YOLO training wrapper
└── train_yolo.example.sh # Example config

scripts/
├── train_yolo.sh         # Alternative YOLO wrapper
└── train_yolo.example.sh # Example config
```

## Documentation

- **Detectors Overview**: [detectors/README.md](detectors/README.md)
- **RF-DETR Guide**: [detectors/rf-detr/README.md](detectors/rf-detr/README.md)
- **Refactoring Notes**: [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
- **Main README**: [README.md](README.md)

## Examples

### Train License Plate Detector
```bash
./scripts/train_yolo.sh \
  --data data/license-plates/data.yaml \
  --model yolov8n.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 320 \
  --name plate-detector
```

### Train Vehicle Detector
```bash
./scripts/train_yolo.sh \
  --data data/vehicles/data.yaml \
  --model yolo11m.yaml \
  --epochs 100 \
  --batch 32 \
  --imgsz 640 \
  --name vehicle-detector
```

### Train with RF-DETR
```bash
python detectors/rf-detr/train.py \
  --data data/vehicles \
  --model rfdetr-base \
  --epochs 100 \
  --batch 16 \
  --output runs/rf-detr/vehicles
```

### Run Batch Inference
```bash
python detectors/rf-detr/infer.py \
  --source test_images/ \
  --weights runs/rf-detr/vehicles/best.pth \
  --conf 0.5 \
  --save-img \
  --save-txt
```

---

**For detailed information, see**: [detectors/README.md](detectors/README.md)

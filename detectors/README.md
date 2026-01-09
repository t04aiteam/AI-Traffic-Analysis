# Detectors

This directory contains object detection models and training scripts for the license plate recognition system.

## Available Detectors

### 1. YOLO (Ultralytics)
**Location**: [`detectors/yolo/`](yolo/)

Unified YOLO implementation supporting YOLOv8, YOLOv9, YOLOv10, and YOLO11 models through Ultralytics framework.

**Features**:
- State-of-the-art accuracy
- Fast inference speed
- Easy to train and deploy
- Supports all YOLO variants (v8, v9, v10, v11)

**Quick Start**:
```bash
# Train a YOLOv8 model
python detectors/yolo/train_ultralytics.py \
    --data data/vehicle-detection/data.yaml \
    --model yolov8m.yaml \
    --epochs 100 \
    --batch 16
```

See [YOLO Training Documentation](yolo/train_ultralytics.py) for more details.

### 2. RF-DETR (Realtime Detection Transformer)
**Location**: [`detectors/rf-detr/`](rf-detr/)

Transformer-based object detector with real-time inference capabilities.

**Features**:
- End-to-end trainable
- No anchor boxes required
- Competitive accuracy with transformers
- Real-time inference

**Quick Start**:
```bash
# Train RF-DETR model
python detectors/rf-detr/train.py \
    --data data/vehicle-detection \
    --epochs 50 \
    --batch 16
```

See [RF-DETR README](rf-detr/README.md) for complete documentation.

## Training Scripts

### YOLO Training

Use the wrapper scripts in `scripts/` directory:

```bash
# Basic training
./scripts/train_yolo.sh \
    --data data/license-plates/data.yaml \
    --model yolov8n.yaml

# Advanced training with custom parameters
./scripts/train_yolo.sh \
    --data data/license-plates/data.yaml \
    --model yolov9m.yaml \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --device 0 \
    --workers 8 \
    --name license-plate-detector
```

Or use the example scripts in `detectors/`:
```bash
cd detectors
./train_yolo.sh  # Edit this file to customize for your dataset
```

### RF-DETR Training

```bash
python detectors/rf-detr/train.py \
    --data data/vehicle-detection-3 \
    --model rfdetr-base \
    --epochs 100 \
    --batch 32 \
    --output runs/rf-detr/vehicle
```

## Model Comparison

| Model | Speed (FPS) | Accuracy | Size | Use Case |
|-------|-------------|----------|------|----------|
| YOLOv8n | ~250 | Good | Small | Edge devices, real-time |
| YOLOv8m | ~150 | Better | Medium | Balanced speed/accuracy |
| YOLOv9m | ~120 | Better | Medium | High accuracy needed |
| YOLOv10m | ~140 | Better | Medium | Latest improvements |
| YOLO11m | ~160 | Best | Medium | Best overall |
| RF-DETR Base | ~45 | Better | Medium | Transformer architecture |

*Benchmarks on RTX 3090 with 640x640 input*

## Dataset Format

Both YOLO and RF-DETR expect datasets in YOLO format:

```
dataset_name/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── ...
    └── val/
        └── ...
```

**data.yaml** format:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 2  # number of classes
names: ['car', 'plate']  # class names
```

**Label format** (normalized YOLO):
```
class_id x_center y_center width height
```

## Directory Structure

```
detectors/
├── README.md              # This file
├── yolo/                  # Ultralytics YOLO implementation
│   ├── train_ultralytics.py
│   ├── exporter.py
│   └── YOLOv8.py
├── rf-detr/              # RF-DETR implementation
│   ├── README.md
│   ├── train.py
│   └── infer.py
├── train_yolo.sh         # YOLO training wrapper script
└── train_yolo.example.sh # Example training script
```

## Choosing a Detector

### Use YOLO when:
- You need real-time performance (>100 FPS)
- You want mature, well-documented models
- You need to deploy on edge devices
- You want the latest SOTA models (YOLO11)

### Use RF-DETR when:
- You want to experiment with transformers
- You have sufficient compute resources
- You want end-to-end trainable architecture
- You need competitive accuracy with moderate speed

## Common Training Parameters

Both detectors support similar parameters:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--data` | Path to data.yaml | Required |
| `--epochs` | Training epochs | 100-300 |
| `--batch` | Batch size | 8-32 |
| `--imgsz` | Image size | 640, 1280 |
| `--device` | GPU device | '0', '0,1', 'cpu' |
| `--workers` | Dataloader workers | 4-8 |

## Tips for Best Results

1. **Dataset Quality**:
   - Use high-quality, diverse images
   - Ensure accurate labels
   - Balance class distribution

2. **Hyperparameters**:
   - Start with default parameters
   - Increase batch size if GPU allows
   - Use learning rate warmup

3. **Training Duration**:
   - Small datasets: 50-100 epochs
   - Large datasets: 100-300 epochs
   - Monitor validation metrics

4. **Data Augmentation**:
   - Both frameworks use default augmentation
   - Adjust for specific use cases

## GPU Memory Requirements

Typical memory usage with batch size 16:

| Model | Memory (GB) | Recommended GPU |
|-------|-------------|-----------------|
| YOLOv8n | 2-3 | GTX 1660 Ti+ |
| YOLOv8m | 4-6 | RTX 3060+ |
| YOLOv9m | 6-8 | RTX 3070+ |
| YOLO11m | 5-7 | RTX 3060 Ti+ |
| RF-DETR Base | 8-10 | RTX 3080+ |

Reduce batch size or use gradient accumulation if OOM errors occur.

## Exporting Models

### YOLO Export
```bash
python detectors/yolo/exporter.py \
    --weights runs/train/best.pt \
    --format onnx  # or torchscript, tflite, etc.
```

### RF-DETR Export
RF-DETR models can be exported using standard PyTorch export:
```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

## Support

For issues or questions:
1. Check the README in each detector directory
2. Review training logs for errors
3. Ensure dataset format is correct
4. Verify GPU memory availability

## References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YOLOv9 Paper](https://arxiv.org/abs/2402.13616)
- [RF-DETR](https://github.com/roboflow/RF-DETR)
- [DETR Paper](https://arxiv.org/abs/2005.12872)

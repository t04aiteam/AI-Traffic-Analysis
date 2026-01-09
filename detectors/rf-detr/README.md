# RF-DETR Detector

RF-DETR (Realtime-DETR) is a real-time detection transformer that provides fast inference with competitive accuracy. This directory contains training and inference scripts for RF-DETR models.

## Overview

RF-DETR is a transformer-based object detector that offers:
- Real-time inference speed
- High accuracy on detection tasks
- End-to-end trainable architecture
- No need for hand-crafted anchors or NMS post-processing

## Installation

Install the required dependencies:

```bash
pip install rfdetr inference supervision pillow
```

## Dataset Format

RF-DETR expects datasets in the following structure:

```
dataset_name/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        └── ...
```

Label format (YOLO format):
```
class_id x_center y_center width height
```

Where coordinates are normalized (0-1).

## Training

### Basic Training

Train with default parameters:

```bash
python detectors/rf-detr/train.py \
    --data data/vehicle-detection-3 \
    --epochs 50
```

### Advanced Training

Train with custom parameters:

```bash
python detectors/rf-detr/train.py \
    --data data/vehicle-detection-3 \
    --model rfdetr-base \
    --epochs 100 \
    --batch 32 \
    --lr 1e-4 \
    --grad-accum-steps 4 \
    --output runs/rf-detr/vehicle \
    --device 0 \
    --workers 8
```

### Resume Training

Resume from a checkpoint:

```bash
python detectors/rf-detr/train.py \
    --data data/vehicle-detection-3 \
    --resume runs/rf-detr/vehicle/checkpoint.pth
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Path to dataset directory |
| `--model` | `rfdetr-base` | Model variant (rfdetr-base, rfdetr-small, rfdetr-large) |
| `--epochs` | 50 | Number of training epochs |
| `--batch` | 16 | Batch size |
| `--grad-accum-steps` | 4 | Gradient accumulation steps |
| `--lr` | 1e-4 | Learning rate |
| `--imgsz` | 640 | Input image size |
| `--output` | `./runs/rf-detr` | Output directory |
| `--name` | None | Experiment name |
| `--device` | `0` | CUDA device or 'cpu' |
| `--workers` | 8 | Number of dataloader workers |
| `--resume` | None | Checkpoint path to resume from |
| `--verbose` | False | Enable verbose logging |

## Inference

### Single Image Inference

Run inference on a single image:

```bash
python detectors/rf-detr/infer.py \
    --source path/to/image.jpg \
    --model rfdetr-base \
    --conf 0.5 \
    --save-img
```

### Video Inference

Run inference on a video:

```bash
python detectors/rf-detr/infer.py \
    --source path/to/video.mp4 \
    --model rfdetr-base \
    --save-video
```

### Batch Inference

Run inference on a directory of images:

```bash
python detectors/rf-detr/infer.py \
    --source path/to/images/ \
    --conf 0.5 \
    --output results/rf-detr \
    --save-img \
    --save-txt
```

### Custom Model Inference

Use a custom trained model:

```bash
python detectors/rf-detr/infer.py \
    --source test_images/ \
    --weights runs/rf-detr/vehicle/best.pth \
    --conf 0.6
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source` | Required | Image file, video file, directory, or URL |
| `--model` | `rfdetr-base` | Model name |
| `--weights` | None | Path to custom model weights |
| `--conf` | 0.5 | Confidence threshold |
| `--iou` | 0.45 | IOU threshold for NMS |
| `--imgsz` | 640 | Input image size |
| `--output` | `./runs/rf-detr/inference` | Output directory |
| `--save-img` | False | Save annotated images |
| `--save-video` | False | Save annotated video |
| `--save-txt` | False | Save detection results as text |
| `--no-show` | False | Don't display results |
| `--device` | `0` | CUDA device or 'cpu' |
| `--verbose` | False | Enable verbose logging |

## Model Variants

RF-DETR offers three model variants:

| Model | Parameters | Speed | Accuracy |
|-------|-----------|-------|----------|
| rfdetr-small | Lowest | Fastest | Good |
| rfdetr-base | Medium | Fast | Better |
| rfdetr-large | Highest | Slower | Best |

Choose based on your speed/accuracy requirements.

## Tips for Best Results

### Training Tips

1. **Batch Size**: Use the largest batch size that fits in GPU memory. Use gradient accumulation if needed:
   ```bash
   --batch 8 --grad-accum-steps 4  # Effective batch size: 32
   ```

2. **Learning Rate**: Start with 1e-4 and adjust based on loss curves:
   - Loss decreasing too slowly → increase LR
   - Loss unstable/NaN → decrease LR

3. **Image Size**: Larger images improve small object detection but are slower:
   ```bash
   --imgsz 640   # Standard
   --imgsz 1280  # Better for small objects
   ```

4. **Early Stopping**: Monitor validation metrics and stop if no improvement

### Inference Tips

1. **Confidence Threshold**: Adjust based on your use case:
   - Higher (0.7+): Fewer false positives, may miss objects
   - Lower (0.3-0.5): More detections, more false positives

2. **Speed Optimization**: 
   - Use smaller model variants for real-time applications
   - Reduce image size with `--imgsz`

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training
```bash
# Solution: Reduce batch size or use gradient accumulation
--batch 8 --grad-accum-steps 4
```

**Issue**: Model not detecting objects
```bash
# Solution: Lower confidence threshold
--conf 0.3
```

**Issue**: Training loss is NaN
```bash
# Solution: Reduce learning rate
--lr 5e-5
```

## Examples

### Complete Training Pipeline

```bash
# 1. Train model
python detectors/rf-detr/train.py \
    --data data/license-plates \
    --epochs 100 \
    --batch 16 \
    --output runs/rf-detr/plates

# 2. Run inference on test images
python detectors/rf-detr/infer.py \
    --source data/license-plates/images/test \
    --weights runs/rf-detr/plates/best.pth \
    --conf 0.5 \
    --output results/plates \
    --save-img --save-txt
```

### Quick Test

```bash
# Test on a single image with pre-trained model
python detectors/rf-detr/infer.py \
    --source test.jpg \
    --model rfdetr-base \
    --conf 0.5
```

## Performance Benchmarks

Typical training times (on single GPU):
- Small dataset (1k images): 1-2 hours
- Medium dataset (10k images): 8-12 hours
- Large dataset (100k images): 2-3 days

Inference speed (on RTX 3090):
- rfdetr-small: ~60 FPS (640x640)
- rfdetr-base: ~45 FPS (640x640)
- rfdetr-large: ~30 FPS (640x640)

## References

- [RF-DETR Paper](https://arxiv.org/abs/2304.08069)
- [Roboflow RF-DETR](https://github.com/roboflow/RF-DETR)
- [DETR: End-to-End Object Detection](https://arxiv.org/abs/2005.12872)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training/inference logs with `--verbose`
3. Ensure dataset format is correct
4. Verify GPU memory availability

## License

This implementation uses the RF-DETR model from Roboflow. Please refer to their license terms for usage restrictions.

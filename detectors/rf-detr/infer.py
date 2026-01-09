"""
RF-DETR Inference Script

Run inference with RF-DETR models on images, videos, or directories.

Examples:
    Inference on single image:
        python detectors/rf-detr/infer.py --source image.jpg --model rfdetr-base
    
    Inference on video:
        python detectors/rf-detr/infer.py --source video.mp4 --model rfdetr-base --save-video
    
    Inference on directory:
        python detectors/rf-detr/infer.py --source images/ --conf 0.5 --output results/
    
    Use custom trained model:
        python detectors/rf-detr/infer.py --source test/ --weights runs/rf-detr/best.pth
"""

import argparse
from pathlib import Path
import sys
from typing import Union

import supervision as sv
from PIL import Image
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RF-DETR inference on images or videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input parameters
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source: image file, video file, directory, or URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rfdetr-base",
        help="Model name or path to custom weights"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to custom model weights (overrides --model)"
    )
    
    # Inference parameters
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    
    # Output parameters
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/rf-detr/inference",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-img",
        action="store_true",
        help="Save annotated images"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated video"
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results as text files"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display results"
    )
    
    # Other parameters
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (cuda device id or 'cpu')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_model(args):
    """Load RF-DETR model."""
    try:
        from inference import get_model
    except ImportError:
        print("Error: inference package not found. Please install it:")
        print("  pip install inference")
        sys.exit(1)
    
    if args.weights:
        model = get_model(args.weights)
        if args.verbose:
            print(f"Loaded custom weights from: {args.weights}")
    else:
        model = get_model(args.model)
        if args.verbose:
            print(f"Loaded model: {args.model}")
    
    return model


def process_image(image_path: Union[str, Path], model, args):
    """Process a single image."""
    # Load image
    if isinstance(image_path, str) and image_path.startswith("http"):
        import requests
        from io import BytesIO
        image = Image.open(BytesIO(requests.get(image_path).content))
    else:
        image = Image.open(image_path)
    
    # Run inference
    predictions = model.infer(image, confidence=args.conf)[0]
    detections = sv.Detections.from_inference(predictions)
    
    # Extract labels
    labels = [
        f"{pred.class_name} {pred.confidence:.2f}"
        for pred in predictions.predictions
    ]
    
    # Annotate image
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator(
        color=sv.ColorPalette.ROBOFLOW
    ).annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator(
        color=sv.ColorPalette.ROBOFLOW
    ).annotate(annotated_image, detections, labels)
    
    return annotated_image, detections, labels


def save_results(image, detections, labels, output_path: Path, save_img, save_txt):
    """Save inference results."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save annotated image
    if save_img:
        img_path = output_path / "annotated.jpg"
        image.save(img_path)
        print(f"Saved annotated image to: {img_path}")
    
    # Save detection results as text
    if save_txt:
        txt_path = output_path / "detections.txt"
        with open(txt_path, "w") as f:
            for i, (det, label) in enumerate(zip(detections.xyxy, labels)):
                x1, y1, x2, y2 = det
                f.write(f"{i}: {label} [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n")
        print(f"Saved detections to: {txt_path}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if args.verbose:
        print("Loading model...")
    model = load_model(args)
    
    # Process source
    source_path = Path(args.source)
    
    if source_path.is_file():
        # Single image or video
        if args.verbose:
            print(f"Processing: {source_path}")
        
        annotated_image, detections, labels = process_image(source_path, model, args)
        
        # Print results
        print(f"\nDetected {len(labels)} objects:")
        for label in labels:
            print(f"  - {label}")
        
        # Save results
        save_results(
            annotated_image,
            detections,
            labels,
            output_path,
            args.save_img or not args.no_show,
            args.save_txt
        )
        
        # Display results
        if not args.no_show:
            try:
                annotated_image.show()
            except Exception as e:
                print(f"Could not display image: {e}")
    
    elif source_path.is_dir():
        # Directory of images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            f for f in source_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No image files found in {source_path}")
            return
        
        print(f"Found {len(image_files)} images")
        
        for i, img_path in enumerate(image_files, 1):
            if args.verbose:
                print(f"Processing {i}/{len(image_files)}: {img_path.name}")
            
            annotated_image, detections, labels = process_image(img_path, model, args)
            
            # Save results for each image
            img_output = output_path / img_path.stem
            save_results(
                annotated_image,
                detections,
                labels,
                img_output,
                args.save_img,
                args.save_txt
            )
        
        print(f"\nProcessing complete! Results saved to: {output_path}")
    
    else:
        print(f"Error: Source not found: {args.source}")
        sys.exit(1)


if __name__ == "__main__":
    main()
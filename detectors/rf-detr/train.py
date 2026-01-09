"""
RF-DETR Training Script

Train RF-DETR (Realtime-DETR) models for object detection.
RF-DETR is a real-time detection transformer that provides fast inference
with competitive accuracy.

Examples:
    Train with default parameters:
        python detectors/rf-detr/train.py --data data/vehicle-detection-3 --epochs 10
    
    Train with custom batch size and learning rate:
        python detectors/rf-detr/train.py --data data/vehicle-detection-3 \
            --epochs 50 --batch 32 --lr 1e-4 --output runs/rf-detr/vehicle
    
    Resume training from checkpoint:
        python detectors/rf-detr/train.py --data data/vehicle-detection-3 \
            --resume runs/rf-detr/vehicle/checkpoint.pth
"""

import argparse
import os
from pathlib import Path
import sys


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RF-DETR model for object detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Dataset parameters
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset directory (contains images/ and labels/)"
    )
    
    # Training parameters
    parser.add_argument(
        "--model",
        type=str,
        default="rfdetr-base",
        choices=["rfdetr-base", "rfdetr-small", "rfdetr-large"],
        help="Model variant to train"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
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
        default="./runs/rf-detr",
        help="Output directory for training results"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: model name + timestamp)"
    )
    
    # Other parameters
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (cuda device id or 'cpu')"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def validate_paths(args):
    """Validate input paths and create output directory."""
    # Validate dataset path
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data}")
    
    # Check for expected dataset structure
    expected_subdirs = ["images", "labels"]
    missing = [d for d in expected_subdirs if not (data_path / d).exists()]
    if missing:
        print(f"Warning: Missing expected subdirectories: {missing}")
        print(f"Expected structure: {args.data}/{{images,labels}}/{{train,val}}/")
    
    # Create output directory
    output_path = Path(args.output)
    if args.name:
        output_path = output_path / args.name
    output_path.mkdir(parents=True, exist_ok=True)
    
    return str(data_path.resolve()), str(output_path.resolve())


def main():
    """Main training function."""
    args = parse_args()
    
    # Validate paths
    try:
        dataset_dir, output_dir = validate_paths(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Import RF-DETR (lazy import after arg parsing for --help speed)
    try:
        from rfdetr import RFDETRBase
    except ImportError:
        print("Error: rfdetr package not found. Please install it:")
        print("  pip install rfdetr")
        sys.exit(1)
    
    # Configure device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # Initialize model
    if args.verbose:
        print(f"Initializing {args.model} model...")
    
    model = RFDETRBase()
    
    # Training configuration
    train_config = {
        "dataset_dir": dataset_dir,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "output_dir": output_dir,
    }
    
    if args.resume:
        train_config["resume"] = args.resume
    
    # Print configuration
    print("\n" + "="*50)
    print("RF-DETR Training Configuration")
    print("="*50)
    for key, value in train_config.items():
        print(f"  {key:20s}: {value}")
    print("="*50 + "\n")
    
    # Train model
    try:
        model.train(**train_config)
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {output_dir}")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Example client for Traffic AI Service
Demonstrates how to use the API for image and video processing
"""
import requests
import cv2
import sys
import argparse
from pathlib import Path


class TrafficAIClient:
    """Client for interacting with Traffic AI Service"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self):
        """Check if service is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def process_image(self, image_path):
        """Process a single image and return detections"""
        try:
            with open(image_path, "rb") as f:
                files = {"file": (Path(image_path).name, f, "image/jpeg")}
                response = self.session.post(
                    f"{self.base_url}/predict/image",
                    files=files
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error processing image: {e}")
            return None
    
    def process_frame(self, frame_bytes, frame_number=None):
        """Process a video frame and return detections"""
        try:
            files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
            params = {"frame_number": frame_number} if frame_number is not None else {}
            response = self.session.post(
                f"{self.base_url}/predict/frame",
                files=files,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error processing frame: {e}")
            return None
    
    def reset_tracker(self):
        """Reset the tracking state"""
        try:
            response = self.session.post(f"{self.base_url}/reset")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error resetting tracker: {e}")
            return None
    
    def get_config(self):
        """Get current service configuration"""
        try:
            response = self.session.get(f"{self.base_url}/config")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting config: {e}")
            return None


def process_image_example(client, image_path):
    """Example: Process a single image"""
    print(f"\n=== Processing Image: {image_path} ===")
    
    result = client.process_image(image_path)
    if result:
        print(f"Found {len(result['detections'])} vehicles")
        for det in result['detections']:
            print(f"  Vehicle ID: {det['track_id']}")
            print(f"    Type: {det.get('vehicle_type', 'Unknown')}")
            print(f"    BBox: ({det['bbox']['x1']:.1f}, {det['bbox']['y1']:.1f}) - "
                  f"({det['bbox']['x2']:.1f}, {det['bbox']['y2']:.1f})")
            if det.get('license_plate'):
                print(f"    License Plate: {det['license_plate']} "
                      f"(confidence: {det.get('confidence', 0):.2f})")
            print()


def process_video_example(client, video_path, max_frames=None):
    """Example: Process a video file"""
    print(f"\n=== Processing Video: {video_path} ===")
    
    # Reset tracker before starting new video
    client.reset_tracker()
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info: {total_frames} frames @ {fps:.2f} FPS")
    
    frame_num = 0
    detected_plates = set()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and frame_num >= max_frames:
            break
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Send to API
        result = client.process_frame(buffer.tobytes(), frame_num)
        
        if result:
            detections = result['detections']
            if detections:
                print(f"Frame {frame_num}: {len(detections)} vehicles")
                for det in detections:
                    plate = det.get('license_plate')
                    if plate and plate not in detected_plates:
                        detected_plates.add(plate)
                        print(f"  NEW PLATE: {plate} "
                              f"(Vehicle {det['track_id']}, {det.get('vehicle_type', 'Unknown')})")
        
        frame_num += 1
        
        # Print progress
        if frame_num % 30 == 0:
            print(f"Processed {frame_num}/{total_frames if max_frames is None else max_frames} frames...")
    
    cap.release()
    
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_num}")
    print(f"Unique license plates detected: {len(detected_plates)}")
    if detected_plates:
        print("Detected plates:", sorted(detected_plates))


def main():
    parser = argparse.ArgumentParser(
        description="Traffic AI Service Client - Example Usage"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the Traffic AI service"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file to process"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file to process"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to process from video"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check service health"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show service configuration"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = TrafficAIClient(args.url)
    
    # Health check
    if args.health or (not args.image and not args.video and not args.config):
        print("=== Health Check ===")
        health = client.health_check()
        if health:
            print(f"Status: {health['status']}")
            print(f"Device: {health['device']}")
            print(f"Models Loaded: {health['models_loaded']}")
        else:
            print("Service is not available!")
            sys.exit(1)
    
    # Show config
    if args.config:
        print("\n=== Service Configuration ===")
        config = client.get_config()
        if config:
            for key, value in config.items():
                print(f"{key}: {value}")
    
    # Process image
    if args.image:
        if not Path(args.image).exists():
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        process_image_example(client, args.image)
    
    # Process video
    if args.video:
        if not Path(args.video).exists():
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        process_video_example(client, args.video, args.max_frames)


if __name__ == "__main__":
    main()

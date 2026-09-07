import sys
import json
import argparse
import cv2
import numpy as np

from src.pipeline import YOLOPipeline
from src.schemas import PreprocessConfig


def create_sample_synthetic_image():
    """Generates a test image with shapes and text for instant demo testing."""
    img = np.zeros((400, 600, 3), dtype=np.uint8) + 240
    # Draw geometric shapes simulating objects
    cv2.circle(img, (200, 200), 80, (0, 0, 255), -1)  # Red circle
    cv2.rectangle(img, (350, 100), (520, 300), (255, 0, 0), -1)  # Blue rectangle
    cv2.putText(img, "SpectraDef YOLO Pipeline", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description="SpectraDef - YOLO Image Input & Detection CLI Test Tool")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--blur", type=int, default=0, help="Gaussian blur kernel size (e.g. 3, 5)")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE contrast enhancement")
    parser.add_argument("--sharpen", action="store_true", help="Enable image sharpening")
    parser.add_argument("--save-output", type=str, default="output_detected.jpg", help="Filename to save annotated image")
    
    args = parser.parse_args()

    print("==================================================")
    print("🚀 SpectraDef: YOLO Image Input + Detection Pipeline")
    print("==================================================")

    # Initialize Pipeline
    pipeline = YOLOPipeline(model_name="yolov8n.pt", conf_threshold=args.conf)

    # Preprocessing Config
    config = PreprocessConfig(
        gaussian_blur=args.blur,
        clahe_contrast=args.clahe,
        sharpen=args.sharpen
    )

    # Load input image or sample
    if args.image:
        print(f"📷 Loading image from: {args.image}")
        input_data = args.image
    else:
        print("💡 No input image specified. Generating synthetic test image...")
        input_data = create_sample_synthetic_image()

    # Process through pipeline
    print("⚡ Running preprocessor + YOLOv8 inference...")
    result = pipeline.process(
        image_input=input_data,
        config=config,
        conf_threshold=args.conf,
        return_annotated_base64=False
    )

    print("\n--------------------------------------------------")
    print("📊 STANDARDIZED SYSTEM OUTPUT SUMMARY")
    print("--------------------------------------------------")
    summary = result.to_summary()
    print(json.dumps(summary, indent=2))

    print("\n⏱️ PERFORMANCE METRICS:")
    print(f" - Preprocessing Latency: {result.preprocess_time_ms} ms")
    print(f" - YOLO Inference Latency: {result.inference_time_ms} ms")
    print(f" - Total Pipeline Latency: {result.total_time_ms} ms")
    print(f" - Total Detections Found: {result.total_detections}")

    print("\n✅ Execution completed successfully!")


if __name__ == "__main__":
    main()

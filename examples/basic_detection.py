#!/usr/bin/env python3
"""
Basic Road Objects Detection Example

This example shows the simplest way to detect road objects in an image.
Make sure you have a valid MareArts license before running this code.

Get your license at: https://www.marearts.com/products/anpr
"""

import cv2
import os
from marearts_road_objects import ma_road_object_detector

def basic_detection_example():
    """Basic example of road object detection"""

    # License credentials - Option 1: Hardcoded (for testing)
    # Uncomment and replace with your actual credentials
    username = "your-email@domain.com"
    serial_key = "your-serial-key"
    signature = "your-signature"

    # License credentials - Option 2: From environment variables
    # ma-robj config
    # source ~/.marearts/.marearts_env
    # username = os.getenv("MAREARTS_ANPR_USERNAME")
    # serial_key = os.getenv("MAREARTS_ANPR_SERIAL_KEY")
    # signature = os.getenv("MAREARTS_ANPR_SIGNATURE")

    print("🚗 MareArts Road Objects Detection - Basic Example")
    print("=" * 50)

    try:
        # Initialize detector (model downloads automatically on first use)
        print("🔧 Initializing detector...")
        print("📦 Model will download automatically on first use (~107MB)")
        detector = ma_road_object_detector(
            model_size="small_fp32",
            user_name=username,
            serial_key=serial_key,
            signature=signature,
            conf_thres=0.5,  # Confidence threshold
            iou_thres=0.5    # NMS IoU threshold
        )
        print("✅ Detector initialized successfully!")

        # Load test image (replace with your image path)
        image_path = "test_traffic.jpg"

        # Create a test image if it doesn't exist
        if not os.path.exists(image_path):
            print(f"\n⚠️  Test image '{image_path}' not found.")
            print("   Please place a traffic scene image in the current directory")
            print("   or modify the 'image_path' variable to point to your image.")
            return

        print(f"\n📸 Loading image: {image_path}")
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Error: Could not load image '{image_path}'")
            print("   Please check the file path and format (JPG, PNG, etc.)")
            return

        # Detect objects
        print("🔍 Detecting road objects...")
        result = detector.detector(image)

        # Extract results from ANPR-style format
        detections = result['results']
        proc_time = result['ltrb_proc_sec']

        # Display results
        print(f"\n🎯 Detection Results:")
        print(f"   Processing time: {proc_time:.3f}s")
        print(f"   Found {len(detections)} objects")
        print("-" * 40)

        for i, detection in enumerate(detections, 1):
            class_name = detection['class']
            confidence = detection['ltrb_conf']  # 0-100 integer
            ltrb = detection['ltrb']  # [left, top, right, bottom]

            print(f"{i}. {class_name}")
            print(f"   Confidence: {confidence}%")
            print(f"   Bounding Box: {ltrb}")

        # Draw detections on image and save
        output_image = draw_detections(image.copy(), detections)
        output_path = "detected_objects.jpg"
        cv2.imwrite(output_path, output_image)

        print(f"\n💾 Results saved to: {output_path}")
        print("✅ Detection complete!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check your license credentials")
        print("   2. Ensure you have internet connectivity for first-time model download")
        print("   3. Verify the image path is correct")
        print("   4. Run 'ma-robj validate' to test your license")

def draw_detections(image, detections):
    """Draw bounding boxes and labels on the image"""

    # Color mapping for different object types (8 classes)
    colors = {
        "person": (0, 255, 0),          # Green
        "bicycle": (255, 255, 0),       # Cyan
        "motorcycle": (255, 0, 255),    # Magenta
        "car": (255, 0, 0),             # Blue
        "bus": (0, 165, 255),           # Orange
        "truck": (0, 100, 255),         # Dark orange
        "traffic_light": (255, 255, 255), # White
        "stop_sign": (0, 0, 255)        # Red
    }

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['ltrb'])
        confidence = detection['ltrb_conf']  # 0-100
        class_name = detection['class']

        # Get color for this class
        color = colors.get(class_name, (255, 255, 255))  # White as default

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Create label with class name and confidence
        label = f"{class_name}: {confidence}%"

        # Calculate label size for background rectangle
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            2
        )

    return image

if __name__ == "__main__":
    basic_detection_example()
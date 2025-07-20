#!/usr/bin/env python3
"""
Basic Road Objects Detection Example

This example shows the simplest way to detect road objects in an image.
Make sure you have a valid MareArts license before running this code.

Get your license at: https://study.marearts.com/p/anpr-lpr-solution.html
"""

import cv2
import os
from marearts_road_objects import create_detector, download_model

def basic_detection_example():
    """Basic example of road object detection"""
    
    # License credentials (replace with your actual credentials)
    username = "your-email@domain.com"
    serial_key = "your-serial-key"
    
    print("🚗 MareArts Road Objects Detection - Basic Example")
    print("=" * 50)
    
    try:
        # Step 1: Download model (only needed once, then cached locally)
        print("📦 Downloading model (this may take a few minutes on first run)...")
        model_path = download_model("small", username, serial_key)
        print(f"✅ Model downloaded to: {model_path}")
        
        # Step 2: Create detector
        print("🔧 Initializing detector...")
        detector = create_detector(model_path, username, serial_key, model_size="small")
        print("✅ Detector initialized successfully!")
        
        # Step 3: Load test image (replace with your image path)
        image_path = "test_traffic.jpg"
        
        # Create a test image if it doesn't exist
        if not os.path.exists(image_path):
            print(f"⚠️  Test image '{image_path}' not found.")
            print("   Please place a traffic scene image in the current directory")
            print("   or modify the 'image_path' variable to point to your image.")
            return
        
        print(f"📸 Loading image: {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"❌ Error: Could not load image '{image_path}'")
            print("   Please check the file path and format (JPG, PNG, etc.)")
            return
        
        # Step 4: Detect objects
        print("🔍 Detecting road objects...")
        result = detector.detect(image, confidence_threshold=0.5)
        
        # Step 5: Display results
        print(f"\n🎯 Detection Results:")
        print(f"   Processing time: {result['processing_time_ms']}ms")
        print(f"   Found {result['total_objects']} objects")
        print("-" * 40)
        
        for detection in result['detections']:
            class_name = detection['class']
            subclass = detection['subclass']
            confidence = detection['confidence']
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            
            print(f"{detection['id']}. {class_name} ({subclass})")
            print(f"   Confidence: {confidence}")
            print(f"   Bounding Box: {bbox}")
        
        # Step 6: Draw detections on image and save
        output_image = draw_detections(image.copy(), result['detections'])
        output_path = "detected_objects.jpg"
        cv2.imwrite(output_path, output_image)
        
        print(f"\n💾 Results saved to: {output_path}")
        print("✅ Detection complete!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check your license credentials")
        print("   2. Ensure you have internet connectivity")
        print("   3. Verify the image path is correct")
        print("   4. Run 'marearts-robj validate' to test your license")

def draw_detections(image, detections):
    """Draw bounding boxes and labels on the image"""
    
    # Color mapping for different object types
    colors = {
        "person": (0, 255, 0),      # Green for persons
        "4-wheels": (255, 0, 0),    # Blue for cars/trucks
        "2-wheels": (0, 0, 255)     # Red for bikes/motorcycles
    }
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class']
        
        # Get color for this class
        color = colors.get(class_name, (255, 255, 255))  # White as default
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Create label with class name and confidence
        label = f"{class_name}: {confidence:.2f}"
        
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
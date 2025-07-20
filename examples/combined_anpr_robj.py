#!/usr/bin/env python3
"""
Combined ANPR + Road Objects Detection Example

This example demonstrates using both marearts-anpr and marearts-road-objects
packages together with the same license. This provides comprehensive traffic
scene analysis including vehicle detection and license plate recognition.

Requirements:
- marearts-road-objects (this package)
- marearts-anpr (optional, install separately)
- Same license works for both packages!

Get your license at: https://study.marearts.com/p/anpr-lpr-solution.html
"""

import cv2
import json
import os
from datetime import datetime
from marearts_road_objects import create_detector, download_model

# Try to import ANPR - gracefully handle if not installed
try:
    from marearts_anpr import create_anpr_detector
    ANPR_AVAILABLE = True
    print("✅ ANPR package available - full functionality enabled")
except ImportError:
    ANPR_AVAILABLE = False
    print("⚠️  ANPR package not installed - road objects only")
    print("   Install with: pip install marearts-anpr")

class ComprehensiveTrafficAnalyzer:
    """
    Comprehensive traffic analyzer using both road objects and ANPR detection
    """
    
    def __init__(self, username, serial_key, model_size="medium"):
        """
        Initialize both detectors with the same license
        
        Args:
            username: Your MareArts license username
            serial_key: Your MareArts license serial key
            model_size: Road objects model size ('small', 'medium', 'large')
        """
        self.username = username
        self.serial_key = serial_key
        
        print("🔧 Initializing traffic analyzer...")
        
        # Initialize Road Objects detector
        print("📦 Setting up road objects detection...")
        model_path = download_model(model_size, username, serial_key)
        self.road_detector = create_detector(model_path, username, serial_key, model_size)
        print("✅ Road objects detector ready")
        
        # Initialize ANPR detector if available
        if ANPR_AVAILABLE:
            print("📦 Setting up ANPR detection...")
            self.anpr_detector = create_anpr_detector(username, serial_key)
            print("✅ ANPR detector ready")
        else:
            self.anpr_detector = None
            print("⚠️  ANPR detector not available")
        
        # Statistics tracking
        self.stats = {
            "total_analyzed": 0,
            "vehicles_detected": 0,
            "persons_detected": 0,
            "plates_detected": 0
        }
        
        print("🚀 Traffic analyzer initialized successfully!")
    
    def analyze_image(self, image, confidence_threshold=0.6):
        """
        Comprehensive analysis of a traffic image
        
        Args:
            image: OpenCV image (BGR format)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            dict: Complete analysis results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "road_objects": [],
            "license_plates": [],
            "summary": {},
            "processing_info": {
                "anpr_enabled": ANPR_AVAILABLE,
                "confidence_threshold": confidence_threshold
            }
        }
        
        # Detect road objects (vehicles, persons, bikes)
        print("🔍 Detecting road objects...")
        road_objects = self.road_detector.detect(image, confidence_threshold=confidence_threshold)
        results["road_objects"] = road_objects
        
        # Detect license plates if ANPR is available
        if self.anpr_detector:
            print("🔍 Detecting license plates...")
            license_plates = self.anpr_detector.detect(image)
            results["license_plates"] = license_plates
        else:
            print("⚠️  Skipping license plate detection (ANPR not available)")
            results["license_plates"] = []
        
        # Generate summary statistics
        results["summary"] = self._generate_summary(road_objects, results["license_plates"])
        
        # Update global statistics
        self._update_stats(results["summary"])
        
        return results
    
    def _generate_summary(self, road_objects, license_plates):
        """Generate summary statistics from detections"""
        
        # Count objects by type
        object_counts = {"person": 0, "4-wheels": 0, "2-wheels": 0}
        for obj in road_objects:
            class_name = obj['class']
            if class_name in object_counts:
                object_counts[class_name] += 1
        
        summary = {
            "total_road_objects": len(road_objects),
            "vehicles": object_counts["4-wheels"] + object_counts["2-wheels"],
            "cars": object_counts["4-wheels"],
            "bikes": object_counts["2-wheels"],
            "persons": object_counts["person"],
            "license_plates": len(license_plates),
            "objects_by_type": object_counts
        }
        
        return summary
    
    def _update_stats(self, summary):
        """Update global statistics"""
        self.stats["total_analyzed"] += 1
        self.stats["vehicles_detected"] += summary["vehicles"]
        self.stats["persons_detected"] += summary["persons"] 
        self.stats["plates_detected"] += summary["license_plates"]
    
    def draw_detections(self, image, results):
        """
        Draw all detections on the image
        
        Args:
            image: OpenCV image to draw on
            results: Analysis results from analyze_image()
            
        Returns:
            OpenCV image with drawn detections
        """
        output_image = image.copy()
        
        # Draw road objects
        for obj in results["road_objects"]:
            x1, y1, x2, y2 = obj['bbox']
            confidence = obj['confidence']
            class_name = obj['class']
            
            # Color coding for different object types
            colors = {
                "person": (0, 255, 0),      # Green
                "4-wheels": (255, 0, 0),    # Blue  
                "2-wheels": (0, 0, 255)     # Red
            }
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(output_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw license plates (if ANPR is available)
        for plate in results["license_plates"]:
            x1, y1, x2, y2 = plate['bbox']
            plate_text = plate.get('text', 'Unknown')
            confidence = plate['confidence']
            
            # Yellow for license plates
            color = (0, 255, 255)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw plate text
            label = f"LP: {plate_text} ({confidence:.2f})"
            cv2.putText(output_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add summary information
        self._draw_summary(output_image, results["summary"])
        
        return output_image
    
    def _draw_summary(self, image, summary):
        """Draw summary statistics on the image"""
        
        # Background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Summary text
        y_offset = 35
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(image, "Traffic Analysis Summary:", (20, y_offset), 
                   font, 0.6, text_color, 2)
        
        y_offset += 25
        cv2.putText(image, f"Vehicles: {summary['vehicles']}", (20, y_offset), 
                   font, 0.5, text_color, 1)
        
        y_offset += 20
        cv2.putText(image, f"Cars: {summary['cars']}, Bikes: {summary['bikes']}", 
                   (20, y_offset), font, 0.5, text_color, 1)
        
        y_offset += 20
        cv2.putText(image, f"Persons: {summary['persons']}", (20, y_offset), 
                   font, 0.5, text_color, 1)
        
        if ANPR_AVAILABLE:
            y_offset += 20
            cv2.putText(image, f"License Plates: {summary['license_plates']}", 
                       (20, y_offset), font, 0.5, text_color, 1)
    
    def get_statistics(self):
        """Get global statistics"""
        return self.stats.copy()

def main():
    """Main example function"""
    
    print("🚗 MareArts Combined Traffic Analysis Example")
    print("=" * 50)
    
    # License credentials (replace with your actual credentials)
    username = "your-email@domain.com"
    serial_key = "your-serial-key"
    
    try:
        # Initialize analyzer
        analyzer = ComprehensiveTrafficAnalyzer(username, serial_key, "medium")
        
        # Test image path
        image_path = "traffic_scene.jpg"
        
        # Check if test image exists
        if not os.path.exists(image_path):
            print(f"\n⚠️  Test image '{image_path}' not found.")
            print("   Please place a traffic scene image in the current directory")
            print("   or modify the 'image_path' variable.")
            return
        
        # Load and analyze image
        print(f"\n📸 Loading image: {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"❌ Error: Could not load image '{image_path}'")
            return
        
        print("🔍 Analyzing traffic scene...")
        results = analyzer.analyze_image(image, confidence_threshold=0.6)
        
        # Display results
        print("\n🎯 Analysis Results:")
        print("-" * 30)
        summary = results["summary"]
        print(f"Total Objects: {summary['total_road_objects']}")
        print(f"Vehicles: {summary['vehicles']} (Cars: {summary['cars']}, Bikes: {summary['bikes']})")
        print(f"Persons: {summary['persons']}")
        
        if ANPR_AVAILABLE:
            print(f"License Plates: {summary['license_plates']}")
            
            # Show detected license plate texts
            if results["license_plates"]:
                print("\nDetected License Plates:")
                for i, plate in enumerate(results["license_plates"], 1):
                    text = plate.get('text', 'Unknown')
                    conf = plate['confidence']
                    print(f"  {i}. {text} (confidence: {conf:.2f})")
        
        # Draw detections and save result
        print("\n💾 Saving annotated image...")
        annotated_image = analyzer.draw_detections(image, results)
        output_path = "comprehensive_analysis.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Annotated image saved: {output_path}")
        
        # Save detailed results as JSON
        json_path = "analysis_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Detailed results saved: {json_path}")
        
        # Show statistics
        stats = analyzer.get_statistics()
        print(f"\n📊 Session Statistics:")
        print(f"   Images analyzed: {stats['total_analyzed']}")
        print(f"   Total vehicles detected: {stats['vehicles_detected']}")
        print(f"   Total persons detected: {stats['persons_detected']}")
        if ANPR_AVAILABLE:
            print(f"   Total license plates detected: {stats['plates_detected']}")
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Verify your license credentials")
        print("   2. Check internet connectivity for model download")
        print("   3. Ensure the image file exists and is readable")
        print("   4. Run 'marearts-robj validate' to test your license")
        if not ANPR_AVAILABLE:
            print("   5. Install ANPR package: pip install marearts-anpr")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Batch Processing Example

This example demonstrates how to process multiple images in a directory,
generate detailed reports, and save annotated results. Perfect for analyzing
large datasets or processing surveillance footage frames.

Features:
- Process entire directories of images
- Generate detailed JSON reports
- Save annotated images
- Progress tracking
- Performance statistics
- Error handling for corrupted images

Get your license at: https://study.marearts.com/p/anpr-lpr-solution.html
"""

import cv2
import json
import os
import time
from pathlib import Path
from datetime import datetime
from marearts_road_objects import create_detector, download_model

class BatchProcessor:
    """Batch image processing for road object detection"""
    
    def __init__(self, username, serial_key, model_size="medium"):
        """
        Initialize batch processor
        
        Args:
            username: MareArts license username
            serial_key: MareArts license serial key
            model_size: Model size ('small', 'medium', 'large')
        """
        self.username = username
        self.serial_key = serial_key
        
        print("🔧 Initializing batch processor...")
        
        # Initialize detector
        print(f"📦 Loading {model_size} model...")
        model_path = download_model(model_size, username, serial_key)
        self.detector = create_detector(model_path, username, serial_key, model_size)
        print("✅ Detector ready!")
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Processing statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "total_detections": 0,
            "processing_time": 0,
            "detection_counts": {"person": 0, "4-wheels": 0, "2-wheels": 0}
        }
    
    def process_directory(self, input_dir, output_dir, confidence_threshold=0.6, 
                         save_annotations=True, save_reports=True):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            confidence_threshold: Minimum confidence for detections
            save_annotations: Whether to save annotated images
            save_reports: Whether to save JSON reports
            
        Returns:
            dict: Processing results and statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Validate input directory
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = self._find_image_files(input_path)
        
        if not image_files:
            print(f"⚠️  No supported image files found in {input_dir}")
            print(f"   Supported formats: {', '.join(self.supported_formats)}")
            return None
        
        print(f"📁 Found {len(image_files)} images to process")
        print(f"📂 Output directory: {output_dir}")
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        
        # Initialize processing
        self.stats["start_time"] = datetime.now()
        self.stats["total_images"] = len(image_files)
        
        results = {
            "processing_info": {
                "input_directory": str(input_path),
                "output_directory": str(output_path),
                "confidence_threshold": confidence_threshold,
                "model_size": self.detector.model_size,
                "timestamp": self.stats["start_time"].isoformat()
            },
            "images": {},
            "summary": {}
        }
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            print(f"🔍 Processing [{i}/{len(image_files)}]: {image_file.name}")
            
            try:
                # Process single image
                image_result = self._process_single_image(
                    image_file, output_path, confidence_threshold, 
                    save_annotations
                )
                
                results["images"][image_file.name] = image_result
                self.stats["processed_images"] += 1
                
                # Update detection counts
                for detection in image_result["detections"]:
                    class_name = detection["class"]
                    if class_name in self.stats["detection_counts"]:
                        self.stats["detection_counts"][class_name] += 1
                        self.stats["total_detections"] += 1
                
                # Progress indicator
                if i % 10 == 0 or i == len(image_files):
                    progress = (i / len(image_files)) * 100
                    print(f"   📊 Progress: {progress:.1f}% ({i}/{len(image_files)})")
            
            except Exception as e:
                print(f"   ❌ Error processing {image_file.name}: {str(e)}")
                results["images"][image_file.name] = {
                    "status": "failed",
                    "error": str(e)
                }
                self.stats["failed_images"] += 1
        
        # Finalize processing
        self.stats["end_time"] = datetime.now()
        self.stats["processing_time"] = (
            self.stats["end_time"] - self.stats["start_time"]
        ).total_seconds()
        
        # Generate summary
        results["summary"] = self._generate_summary()
        
        # Save reports if requested
        if save_reports:
            self._save_reports(output_path, results)
        
        # Print final statistics
        self._print_final_stats()
        
        return results
    
    def _find_image_files(self, directory):
        """Find all supported image files in directory"""
        image_files = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        
        return sorted(image_files)
    
    def _process_single_image(self, image_file, output_dir, confidence_threshold, 
                             save_annotation):
        """Process a single image"""
        start_time = time.time()
        
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            raise ValueError("Could not load image")
        
        # Detect objects
        detections = self.detector.detect(image, confidence_threshold=confidence_threshold)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare result
        result = {
            "status": "success",
            "detections": detections,
            "detection_count": len(detections),
            "processing_time_seconds": processing_time,
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "objects_by_type": self._count_objects_by_type(detections)
        }
        
        # Save annotated image if requested
        if save_annotation and detections:
            annotated_image = self._draw_detections(image.copy(), detections)
            annotation_path = output_dir / f"annotated_{image_file.name}"
            cv2.imwrite(str(annotation_path), annotated_image)
            result["annotation_file"] = str(annotation_path)
        
        return result
    
    def _count_objects_by_type(self, detections):
        """Count detections by object type"""
        counts = {"person": 0, "4-wheels": 0, "2-wheels": 0}
        
        for detection in detections:
            class_name = detection["class"]
            if class_name in counts:
                counts[class_name] += 1
        
        return counts
    
    def _draw_detections(self, image, detections):
        """Draw detection bounding boxes and labels"""
        
        # Color mapping
        colors = {
            "person": (0, 255, 0),      # Green
            "4-wheels": (255, 0, 0),    # Blue
            "2-wheels": (0, 0, 255)     # Red
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Get color
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Label background
            cv2.rectangle(image, (x1, y1-label_height-10), 
                         (x1+label_width, y1), color, -1)
            
            # Label text
            cv2.putText(image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def _generate_summary(self):
        """Generate processing summary"""
        success_rate = (self.stats["processed_images"] / self.stats["total_images"]) * 100
        avg_processing_time = self.stats["processing_time"] / max(1, self.stats["processed_images"])
        
        summary = {
            "total_images": self.stats["total_images"],
            "processed_successfully": self.stats["processed_images"],
            "failed_images": self.stats["failed_images"],
            "success_rate_percent": round(success_rate, 2),
            "total_processing_time_seconds": round(self.stats["processing_time"], 2),
            "average_time_per_image_seconds": round(avg_processing_time, 3),
            "total_detections": self.stats["total_detections"],
            "detections_by_type": self.stats["detection_counts"].copy()
        }
        
        if self.stats["processed_images"] > 0:
            summary["average_detections_per_image"] = round(
                self.stats["total_detections"] / self.stats["processed_images"], 2
            )
        
        return summary
    
    def _save_reports(self, output_dir, results):
        """Save detailed reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        full_report_path = output_dir / f"batch_results_{timestamp}.json"
        with open(full_report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Full report saved: {full_report_path}")
        
        # Save summary only
        summary_path = output_dir / f"batch_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(results["summary"], f, indent=2, default=str)
        print(f"📄 Summary report saved: {summary_path}")
        
        # Save CSV summary for easy analysis
        csv_path = output_dir / f"batch_summary_{timestamp}.csv"
        self._save_csv_summary(csv_path, results)
        print(f"📄 CSV summary saved: {csv_path}")
    
    def _save_csv_summary(self, csv_path, results):
        """Save summary as CSV"""
        import csv
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Headers
            writer.writerow([
                "Image", "Status", "Detections", "Persons", "Cars", "Bikes", 
                "Processing_Time_Seconds"
            ])
            
            # Data rows
            for image_name, image_data in results["images"].items():
                if image_data.get("status") == "success":
                    objects = image_data.get("objects_by_type", {})
                    writer.writerow([
                        image_name,
                        image_data["status"],
                        image_data["detection_count"],
                        objects.get("person", 0),
                        objects.get("4-wheels", 0),
                        objects.get("2-wheels", 0),
                        round(image_data["processing_time_seconds"], 3)
                    ])
                else:
                    writer.writerow([
                        image_name,
                        "failed",
                        0, 0, 0, 0, 0
                    ])
    
    def _print_final_stats(self):
        """Print final processing statistics"""
        print("\n" + "="*50)
        print("📊 BATCH PROCESSING COMPLETE")
        print("="*50)
        
        print(f"📁 Total images: {self.stats['total_images']}")
        print(f"✅ Successfully processed: {self.stats['processed_images']}")
        print(f"❌ Failed: {self.stats['failed_images']}")
        
        if self.stats['total_images'] > 0:
            success_rate = (self.stats['processed_images'] / self.stats['total_images']) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
        
        print(f"⏱️  Total processing time: {self.stats['processing_time']:.2f} seconds")
        
        if self.stats['processed_images'] > 0:
            avg_time = self.stats['processing_time'] / self.stats['processed_images']
            print(f"⚡ Average time per image: {avg_time:.3f} seconds")
        
        print(f"🎯 Total detections: {self.stats['total_detections']}")
        print(f"👤 Persons: {self.stats['detection_counts']['person']}")
        print(f"🚗 Vehicles: {self.stats['detection_counts']['4-wheels']}")
        print(f"🏍️  Bikes: {self.stats['detection_counts']['2-wheels']}")

def main():
    """Main function for batch processing example"""
    
    print("📁 MareArts Road Objects - Batch Processing")
    print("=" * 50)
    
    # License credentials (replace with your actual credentials)
    username = "your-email@domain.com"
    serial_key = "your-serial-key"
    
    # Directory paths (modify these for your use case)
    input_directory = "input_images"
    output_directory = "batch_results"
    
    try:
        # Check input directory
        if not os.path.exists(input_directory):
            print(f"⚠️  Input directory '{input_directory}' not found.")
            print("Creating example directory structure...")
            os.makedirs(input_directory, exist_ok=True)
            print(f"📁 Created: {input_directory}")
            print("   Please add your images to this directory and run again.")
            return
        
        # Initialize processor
        processor = BatchProcessor(username, serial_key, model_size="medium")
        
        # Process directory
        results = processor.process_directory(
            input_dir=input_directory,
            output_dir=output_directory,
            confidence_threshold=0.6,
            save_annotations=True,
            save_reports=True
        )
        
        if results:
            print(f"\n✅ Batch processing completed successfully!")
            print(f"📂 Results saved to: {output_directory}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Verify your license credentials")
        print("   2. Ensure input directory contains supported image files")
        print("   3. Check that you have write permissions for output directory")
        print("   4. Run 'marearts-robj validate' to test your license")

if __name__ == "__main__":
    main()
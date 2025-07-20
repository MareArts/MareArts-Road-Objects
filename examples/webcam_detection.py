#!/usr/bin/env python3
"""
Real-time Webcam Road Objects Detection

This example demonstrates real-time road object detection using your webcam.
Great for testing the system and seeing detection performance in real-time.

Requirements:
- Webcam or external camera
- Valid MareArts license

Get your license at: https://study.marearts.com/p/anpr-lpr-solution.html

Controls:
- Press 'q' to quit
- Press 's' to save current frame
- Press 'c' to toggle confidence display
- Press SPACE to pause/resume
"""

import cv2
import time
import os
from datetime import datetime
from marearts_road_objects import create_detector, download_model

class WebcamDetector:
    """Real-time webcam object detection"""
    
    def __init__(self, username, serial_key, model_size="small"):
        """
        Initialize webcam detector
        
        Args:
            username: MareArts license username
            serial_key: MareArts license serial key  
            model_size: Model size ('small' recommended for real-time)
        """
        self.username = username
        self.serial_key = serial_key
        
        print("🔧 Initializing webcam detector...")
        
        # Use small model for best real-time performance
        print(f"📦 Loading {model_size} model for real-time detection...")
        model_path = download_model(model_size, username, serial_key)
        self.detector = create_detector(model_path, username, serial_key, model_size)
        print("✅ Detector ready!")
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.show_confidence = True
        self.paused = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "total_detections": 0,
            "vehicles": 0,
            "persons": 0,
            "bikes": 0
        }
    
    def run(self, camera_id=0, width=640, height=480):
        """
        Start real-time detection
        
        Args:
            camera_id: Camera ID (0 for default webcam)
            width: Frame width
            height: Frame height
        """
        print(f"📹 Starting webcam detection (Camera {camera_id})")
        print("Controls: 'q'=quit, 's'=save, 'c'=toggle confidence, SPACE=pause")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            print("💡 Try different camera_id (0, 1, 2, etc.)")
            return
        
        print("✅ Camera initialized")
        print("🚀 Starting real-time detection...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Process frame if not paused
                if not self.paused:
                    processed_frame = self._process_frame(frame)
                else:
                    processed_frame = self._draw_paused_overlay(frame)
                
                # Add UI overlay
                processed_frame = self._draw_ui_overlay(processed_frame)
                
                # Display frame
                cv2.imshow("MareArts Road Objects - Webcam Detection", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key_input(key, processed_frame):
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  Detection interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self._print_session_stats()
    
    def _process_frame(self, frame):
        """Process a single frame"""
        start_time = time.time()
        
        # Detect objects
        detections = self.detector.detect(frame, confidence_threshold=self.confidence_threshold)
        
        # Update statistics
        self.stats["frames_processed"] += 1
        self.stats["total_detections"] += len(detections)
        
        # Count objects by type
        for detection in detections:
            class_name = detection['class']
            if class_name == "person":
                self.stats["persons"] += 1
            elif class_name == "4-wheels":
                self.stats["vehicles"] += 1
            elif class_name == "2-wheels":
                self.stats["bikes"] += 1
        
        # Draw detections
        processed_frame = self._draw_detections(frame, detections)
        
        # Update FPS
        self._update_fps()
        
        return processed_frame
    
    def _draw_detections(self, frame, detections):
        """Draw detection results on frame"""
        
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
            
            # Get color for this class
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label (with or without confidence)
            if self.show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # Label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(frame, (x1, y1-label_height-5), 
                         (x1+label_width, y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def _draw_ui_overlay(self, frame):
        """Draw UI information overlay"""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # FPS and performance info
        y = 30
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y += 25
        cv2.putText(frame, f"Confidence: {self.confidence_threshold:.2f}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        cv2.putText(frame, f"Detections: {self.stats['total_detections']}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        status = "PAUSED" if self.paused else "RUNNING"
        color = (0, 255, 255) if self.paused else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _draw_paused_overlay(self, frame):
        """Draw pause overlay"""
        height, width = frame.shape[:2]
        
        # Dim the frame
        frame = cv2.addWeighted(frame, 0.5, frame, 0, 0)
        
        # Add pause text
        text = "PAUSED - Press SPACE to resume"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return frame
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        
        if self.fps_counter >= 30:  # Update every 30 frames
            elapsed = time.time() - self.fps_start_time
            self.current_fps = 30 / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def _handle_key_input(self, key, frame):
        """
        Handle keyboard input
        
        Returns:
            bool: True to continue, False to quit
        """
        if key == ord('q'):
            return False
        
        elif key == ord('s'):
            # Save current frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"webcam_detection_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Frame saved: {filename}")
        
        elif key == ord('c'):
            # Toggle confidence display
            self.show_confidence = not self.show_confidence
            status = "ON" if self.show_confidence else "OFF"
            print(f"🔧 Confidence display: {status}")
        
        elif key == ord(' '):
            # Toggle pause
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RESUMED"
            print(f"⏯️  Detection {status}")
        
        elif key == ord('=') or key == ord('+'):
            # Increase confidence threshold
            self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
            print(f"🔧 Confidence threshold: {self.confidence_threshold:.2f}")
        
        elif key == ord('-'):
            # Decrease confidence threshold
            self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
            print(f"🔧 Confidence threshold: {self.confidence_threshold:.2f}")
        
        return True
    
    def _print_session_stats(self):
        """Print session statistics"""
        print("\n📊 Session Statistics:")
        print("-" * 30)
        print(f"Frames processed: {self.stats['frames_processed']}")
        print(f"Total detections: {self.stats['total_detections']}")
        print(f"Vehicles detected: {self.stats['vehicles']}")
        print(f"Persons detected: {self.stats['persons']}")
        print(f"Bikes detected: {self.stats['bikes']}")
        
        if self.stats['frames_processed'] > 0:
            avg_detections = self.stats['total_detections'] / self.stats['frames_processed']
            print(f"Average detections per frame: {avg_detections:.2f}")

def main():
    """Main function"""
    
    print("📹 MareArts Road Objects - Webcam Detection")
    print("=" * 50)
    
    # License credentials (replace with your actual credentials)
    username = "your-email@domain.com"
    serial_key = "your-serial-key"
    
    try:
        # Initialize detector
        detector = WebcamDetector(username, serial_key, model_size="small")
        
        # Start detection
        detector.run(camera_id=0, width=640, height=480)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check your license credentials")
        print("   2. Verify webcam is connected and working")
        print("   3. Try different camera_id values (0, 1, 2, etc.)")
        print("   4. Ensure sufficient lighting for detection")
        print("   5. Run 'marearts-robj validate' to test your license")

if __name__ == "__main__":
    main()
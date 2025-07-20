#!/bin/bash
# CLI Examples for MareArts Road Objects Detection
#
# This script demonstrates various ways to use the marearts-robj CLI tool.
# Make sure you have a valid license before running these commands.
#
# Get your license at: https://study.marearts.com/p/anpr-lpr-solution.html

echo "🚗 MareArts Road Objects - CLI Examples"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "\n${BLUE}$1${NC}"
}

print_command() {
    echo -e "${GREEN}$ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}💡 $1${NC}"
}

# Step 1: Initial Setup
print_step "1. 🔧 Initial Setup"
print_info "First, configure your license credentials"
print_command "marearts-robj config"
echo "This will securely prompt for your username and serial key"

print_info "Or set environment variables (recommended for automation):"
print_command "export MAREARTS_ROBJ_USERNAME=\"your-email@domain.com\""
print_command "export MAREARTS_ROBJ_SERIAL_KEY=\"your-serial-key\""

# Step 2: License Validation
print_step "2. ✅ License Validation"
print_info "Verify your license is working"
print_command "marearts-robj validate"

print_info "Check current configuration"
print_command "marearts-robj config --show"

# Step 3: GPU Support
print_step "3. 🚀 GPU Support Check"
print_info "Check if GPU acceleration is available"
print_command "marearts-robj gpu-info"

print_info "Expected output with GPU:"
echo "🚀 CUDAExecutionProvider (GPU)"
echo "⚡ CPUExecutionProvider"
echo "GPU Acceleration: ENABLED"

# Step 4: Model Management
print_step "4. 📦 Model Management"
print_info "Download specific models"
print_command "marearts-robj download --model small"
print_command "marearts-robj download --model medium"
print_command "marearts-robj download --model large"

print_info "Force redownload of a model"
print_command "marearts-robj download --model medium --force"

# Step 5: Basic Detection
print_step "5. 🔍 Basic Detection"
print_info "Detect objects in a single image"
print_command "marearts-robj detect traffic_scene.jpg"

print_info "Use different model sizes"
print_command "marearts-robj detect image.jpg --model small"   # Fastest
print_command "marearts-robj detect image.jpg --model medium"  # Balanced
print_command "marearts-robj detect image.jpg --model large"   # Most accurate

# Step 6: Advanced Detection Options
print_step "6. ⚙️ Advanced Detection Options"
print_info "Adjust confidence threshold"
print_command "marearts-robj detect image.jpg --confidence 0.8"  # More strict
print_command "marearts-robj detect image.jpg --confidence 0.3"  # More detections

print_info "Save annotated output image"
print_command "marearts-robj detect input.jpg --output detected_output.jpg"

print_info "Combine multiple options"
print_command "marearts-robj detect highway.jpg --model large --confidence 0.7 --output highway_detected.jpg"

# Step 7: Batch Processing
print_step "7. 📁 Batch Processing"
print_info "Process multiple images using shell scripting"

print_command "# Process all JPG files in current directory"
cat << 'EOF'
for image in *.jpg; do
    if [ -f "$image" ]; then
        echo "Processing: $image"
        marearts-robj detect "$image" --output "detected_$image"
    fi
done
EOF

print_command "# Process images in a specific directory"
cat << 'EOF'
input_dir="input_images"
output_dir="output_results"
mkdir -p "$output_dir"

for image in "$input_dir"/*.{jpg,png,jpeg}; do
    if [ -f "$image" ]; then
        filename=$(basename "$image")
        echo "Processing: $filename"
        marearts-robj detect "$image" --output "$output_dir/detected_$filename"
    fi
done
EOF

print_command "# Process with different models based on image size"
cat << 'EOF'
for image in *.jpg; do
    if [ -f "$image" ]; then
        # Get image dimensions (requires imagemagick: identify command)
        if command -v identify >/dev/null; then
            width=$(identify -ping -format "%w" "$image")
            if [ "$width" -gt 1920 ]; then
                model="large"
            elif [ "$width" -gt 1280 ]; then
                model="medium"
            else
                model="small"
            fi
            echo "Processing $image with $model model (width: ${width}px)"
            marearts-robj detect "$image" --model "$model" --output "detected_$image"
        else
            marearts-robj detect "$image" --output "detected_$image"
        fi
    fi
done
EOF

# Step 8: Automation Examples
print_step "8. 🤖 Automation Examples"
print_info "Watch directory for new images (requires inotify-tools)"
print_command "# Install: sudo apt-get install inotify-tools"
cat << 'EOF'
#!/bin/bash
watch_dir="watch_folder"
output_dir="auto_results"
mkdir -p "$watch_dir" "$output_dir"

inotifywait -m -e create -e moved_to "$watch_dir" --format '%f' |
while read filename; do
    if [[ "$filename" =~ \.(jpg|jpeg|png)$ ]]; then
        echo "New image detected: $filename"
        sleep 1  # Wait for file to be fully written
        marearts-robj detect "$watch_dir/$filename" --output "$output_dir/detected_$filename"
        echo "Processing complete: $filename"
    fi
done
EOF

print_info "Scheduled processing with cron"
print_command "# Add to crontab (crontab -e):"
cat << 'EOF'
# Process images every hour
0 * * * * /path/to/process_images.sh

# Process images every day at 2 AM
0 2 * * * cd /path/to/images && for img in *.jpg; do marearts-robj detect "$img" --output "daily_$(date +%Y%m%d)_$img"; done
EOF

# Step 9: Performance Optimization
print_step "9. ⚡ Performance Optimization"
print_info "For real-time applications, use small model"
print_command "marearts-robj detect webcam_frame.jpg --model small --confidence 0.6"

print_info "For batch processing, use medium/large models"
print_command "marearts-robj detect high_res_image.jpg --model large --confidence 0.8"

print_info "Monitor GPU usage during processing"
print_command "# In another terminal:"
print_command "watch -n 1 nvidia-smi"

# Step 10: Troubleshooting
print_step "10. 🛠️ Troubleshooting"
print_info "Common troubleshooting commands"

print_command "# Test license"
print_command "marearts-robj validate"

print_command "# Check GPU support"
print_command "marearts-robj gpu-info"

print_command "# Reconfigure license"
print_command "marearts-robj config"

print_command "# Check if image file is valid"
print_command "file your_image.jpg"

print_command "# Check Python installation"
print_command "python -c \"import marearts_road_objects; print('✅ Package installed')\""

# Step 11: Integration Examples
print_step "11. 🔗 Integration Examples"
print_info "Integrate with other tools"

print_command "# Combine with FFmpeg for video processing"
cat << 'EOF'
# Extract frames from video
ffmpeg -i input_video.mp4 -vf fps=1 frame_%04d.jpg

# Process frames
for frame in frame_*.jpg; do
    marearts-robj detect "$frame" --output "detected_$frame"
done

# Create video from processed frames
ffmpeg -framerate 1 -pattern_type glob -i 'detected_frame_*.jpg' -c:v libx264 output_video.mp4
EOF

print_command "# Combine with ImageMagick for preprocessing"
cat << 'EOF'
# Resize large images before processing
for image in large_*.jpg; do
    # Resize to max 1280px width
    convert "$image" -resize '1280x>' "resized_$image"
    # Process resized image
    marearts-robj detect "resized_$image" --output "detected_$image"
done
EOF

print_command "# Integration with database logging"
cat << 'EOF'
#!/bin/bash
# Process image and log results to database
image="$1"
result_file="temp_result.json"

# Process image (assuming JSON output capability)
if marearts-robj detect "$image" --output-json "$result_file"; then
    # Log to database (example with sqlite3)
    detections=$(jq '.detections | length' "$result_file")
    sqlite3 detection_log.db "INSERT INTO detections (image, timestamp, count) VALUES ('$image', datetime('now'), $detections);"
    echo "Logged $detections detections for $image"
fi
EOF

# Final tips
print_step "💡 Final Tips"
echo "• Use environment variables for automation scripts"
echo "• Start with small model for testing, upgrade to larger for production"
echo "• Monitor GPU memory usage for optimal batch sizes"
echo "• Keep your license credentials secure"
echo "• Use appropriate confidence thresholds for your use case"

print_info "For more examples, check the Python scripts in the examples/ directory"
print_info "Get your license at: https://study.marearts.com/p/anpr-lpr-solution.html"

echo -e "\n✅ CLI examples complete!"
echo "Run any of these commands to get started with road object detection."
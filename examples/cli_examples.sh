#!/bin/bash
# MareArts Road Objects - CLI Examples

# Basic Commands
ma-robj version         # Show package version
ma-robj validate        # Validate license
ma-robj gpu-info        # Check GPU support

# Detection
ma-robj detect test_image.jpg                 # Detect objects in image
ma-robj detect test_image.jpg -o output_test_image.jpg   # Save annotated output

# Model Selection
ma-robj detect test_image.jpg --model small_fp32    # Fast (recommended)
ma-robj detect test_image.jpg --model medium_fp32   # Balanced
ma-robj detect test_image.jpg --model large_fp32    # Most accurate

# Configuration
ma-robj config                      # Configure license credentials
source ~/.marearts/.marearts_env    # Load environment variables
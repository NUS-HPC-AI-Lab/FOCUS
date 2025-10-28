#!/bin/bash

# FOCUS Keyframe Extraction Example Script
# This script demonstrates how to use FOCUS for keyframe extraction

echo "FOCUS: Frame-Optimistic Confidence Upper-bound Selection"
echo "========================================================"

# Set default parameters
DATASET_NAME="longvideobench"
DATASET_PATH="./datasets/longvideobench"
OUTPUT_DIR="focus_example"
NUM_KEYFRAMES=64
BATCH_SIZE=32
BLIP_MODEL="large"

# FOCUS-specific parameters
COARSE_EVERY_SEC=16.0
FINE_EVERY_SEC=1.0
ZOOM_RATIO=0.25
INTERPOLATION_METHOD="nearest"
FINAL_MIN_ARMS=4
FINAL_MAX_ARMS=32
TOP_RATIO=0.2
TEMPERATURE=0.06

echo "Configuration:"
echo "  Dataset: $DATASET_NAME"
echo "  Dataset Path: $DATASET_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Number of Keyframes: $NUM_KEYFRAMES"
echo "  Batch Size: $BATCH_SIZE"
echo "  BLIP Model: $BLIP_MODEL"
echo ""
echo "FOCUS Parameters:"
echo "  Coarse Sampling Interval: ${COARSE_EVERY_SEC}s"
echo "  Fine Sampling Interval: ${FINE_EVERY_SEC}s"
echo "  Zoom Ratio: $ZOOM_RATIO"
echo "  Interpolation Method: $INTERPOLATION_METHOD"
echo "  Final Arm Bounds: [$FINAL_MIN_ARMS, $FINAL_MAX_ARMS]"
echo "  Top Ratio: $TOP_RATIO"
echo "  Temperature: $TEMPERATURE"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path '$DATASET_PATH' does not exist!"
    echo "Please download the dataset first following the AKS instructions."
    exit 1
fi

# Check if select_keyframe.py exists
if [ ! -f "select_keyframe.py" ]; then
    echo "Error: select_keyframe.py not found in current directory!"
    echo "Please make sure you're running this script from the opensource directory."
    exit 1
fi

echo "Starting FOCUS keyframe extraction..."
echo ""

# Run FOCUS
python select_keyframe.py \
    --dataset_name "$DATASET_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_keyframes "$NUM_KEYFRAMES" \
    --batch_size "$BATCH_SIZE" \
    --blip_model "$BLIP_MODEL" \
    --coarse_every_sec "$COARSE_EVERY_SEC" \
    --fine_every_sec "$FINE_EVERY_SEC" \
    --zoom_ratio "$ZOOM_RATIO" \
    --interpolation_method "$INTERPOLATION_METHOD" \
    --final_min_arms "$FINAL_MIN_ARMS" \
    --final_max_arms "$FINAL_MAX_ARMS" \
    --top_ratio "$TOP_RATIO" \
    --temperature "$TEMPERATURE" \
    --seed 42

# Check if extraction was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "FOCUS keyframe extraction completed successfully!"
    echo ""
    echo "Output files:"
    echo "  - ./selected_frames/$DATASET_NAME/$OUTPUT_DIR/selected_frames.json"
    echo "  - ./selected_frames/$DATASET_NAME/$OUTPUT_DIR/sampling_details.json"
    echo "  - ./selected_frames/$DATASET_NAME/$OUTPUT_DIR/extraction_stats.json"
    echo ""
    echo "You can now use these results for evaluation with lmms-eval."
else
    echo ""
    echo "FOCUS keyframe extraction failed!"
    echo "Please check the error messages above and ensure all dependencies are installed."
fi

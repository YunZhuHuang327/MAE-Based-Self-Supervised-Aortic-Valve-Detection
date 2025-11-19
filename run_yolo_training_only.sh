#!/bin/bash

# Quick script to train YOLO for valve detection
# Assumes data is already prepared

echo "=========================================="
echo "YOLO Training for Valve Detection"
echo "=========================================="

# Configuration
ARGS_YAML="/DATA1/yunzhu/SSL/args.yaml"
MAE_CHECKPOINT="./mae_valve_output/best_model.pth"

# Check if dataset exists
DATASET_YAML="/DATA1/yunzhu/SSL/yolo_dataset/valve_detection.yaml"

if [ ! -f "$DATASET_YAML" ]; then
    echo "❌ Error: Dataset not found: $DATASET_YAML"
    echo ""
    echo "Please prepare dataset first:"
    echo "  python prepare_yolo_data.py"
    echo ""
    exit 1
fi

echo "Configuration:"
echo "  Args: $ARGS_YAML"
echo "  Dataset: $DATASET_YAML"
echo "  MAE checkpoint: ${MAE_CHECKPOINT:-None (standard YOLO)}"
echo "=========================================="
echo ""

# Check for MAE checkpoint
MAE_ARG=""
if [ -f "$MAE_CHECKPOINT" ]; then
    echo "✅ Using MAE pre-trained features"
    MAE_ARG="--mae_checkpoint $MAE_CHECKPOINT"
else
    echo "📌 Using standard YOLO backbone"
fi

# Start training
python train_valve_yolo.py \
    --args_yaml "$ARGS_YAML" \
    $MAE_ARG

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="

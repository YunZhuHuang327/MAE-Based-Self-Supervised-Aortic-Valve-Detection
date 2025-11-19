#!/bin/bash

# Complete Pipeline: MAE Pre-training + YOLO Valve Detection
# Runs the entire workflow from data preparation to training

echo "=========================================="
echo "Complete Pipeline for Valve Detection"
echo "MAE Pre-training + YOLO Training"
echo "=========================================="

# Configuration
SOURCE_DIR="/DATA1/yunzhu/SSL/training_image/training_image"
YOLO_DATASET="/DATA1/yunzhu/SSL/yolo_dataset"
MAE_CHECKPOINT="./mae_valve_output/best_model.pth"
ARGS_YAML="/DATA1/yunzhu/SSL/args.yaml"

# Step 1: Prepare YOLO Dataset
echo ""
echo "=========================================="
echo "Step 1: Preparing YOLO Dataset"
echo "=========================================="
echo "  Source: $SOURCE_DIR"
echo "  Output: $YOLO_DATASET"
echo "  Train: patient 1-30"
echo "  Val: patient 31-40"
echo ""

python prepare_yolo_data.py \
    --source_dir "$SOURCE_DIR" \
    --output_dir "$YOLO_DATASET" \
    --validate

if [ $? -ne 0 ]; then
    echo "❌ Error: Data preparation failed"
    exit 1
fi

echo "✅ Dataset preparation complete"

# Step 2: Check if MAE model exists
echo ""
echo "=========================================="
echo "Step 2: Checking MAE Model"
echo "=========================================="

if [ -f "$MAE_CHECKPOINT" ]; then
    echo "✅ Found MAE checkpoint: $MAE_CHECKPOINT"

    # Load checkpoint info
    python -c "
import torch
ckpt = torch.load('$MAE_CHECKPOINT', map_location='cpu')
print(f'  Epoch: {ckpt[\"epoch\"]}')
print(f'  Loss: {ckpt[\"loss\"]:.4f}')
"
else
    echo "⚠️  MAE checkpoint not found: $MAE_CHECKPOINT"
    echo ""
    echo "Would you like to train MAE first? (recommended)"
    echo "  1) Yes - train MAE now (3-5 hours)"
    echo "  2) No - skip MAE and use standard YOLO backbone"
    echo ""
    read -p "Choice [1/2]: " choice

    if [ "$choice" = "1" ]; then
        echo ""
        echo "=========================================="
        echo "Training MAE Model..."
        echo "=========================================="

        ./run_training_valve_optimized.sh

        if [ $? -ne 0 ]; then
            echo "❌ Error: MAE training failed"
            exit 1
        fi

        echo "✅ MAE training complete"
    else
        echo "Proceeding without MAE pre-training"
        MAE_CHECKPOINT=""
    fi
fi

# Step 3: Train YOLO
echo ""
echo "=========================================="
echo "Step 3: Training YOLO for Valve Detection"
echo "=========================================="
echo "  Config: $ARGS_YAML"
echo "  MAE checkpoint: ${MAE_CHECKPOINT:-None}"
echo ""

python train_valve_yolo.py \
    --args_yaml "$ARGS_YAML" \
    --mae_checkpoint "$MAE_CHECKPOINT"

if [ $? -ne 0 ]; then
    echo "❌ Error: YOLO training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Complete Pipeline Finished!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  YOLO dataset: $YOLO_DATASET"
echo "  Training results: /DATA1/yunzhu/SSL/valve_training_results"
echo ""
echo "To evaluate the model:"
echo "  python evaluate_valve_yolo.py --weights ./valve_training_results/train/weights/best.pt"
echo ""

#!/bin/bash

# YOLO Training with MAE pre-trained features
# Training: patient 1-30, 41-50 (labeled only)
# Validation: patient 31-40 (ALL images)

echo "=========================================="
echo "YOLO Training for Valve Detection"
echo "With MAE Pre-trained Features"
echo "=========================================="

# Configuration
ARGS_YAML="/DATA1/yunzhu/SSL/args.yaml"
MAE_CHECKPOINT="/DATA1/yunzhu/SSL/mae_valve_output/best_model.pth"
DATASET_YAML="/DATA1/yunzhu/SSL/yolo_dataset_all_val/valve_detection.yaml"

# Check if dataset exists
if [ ! -f "$DATASET_YAML" ]; then
    echo "❌ Error: Dataset not found: $DATASET_YAML"
    echo ""
    echo "Please prepare dataset first:"
    echo "  python prepare_yolo_data_all_val.py --validate"
    echo ""
    exit 1
fi

# Check if MAE checkpoint exists
if [ ! -f "$MAE_CHECKPOINT" ]; then
    echo "❌ Error: MAE checkpoint not found: $MAE_CHECKPOINT"
    echo ""
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Args: $ARGS_YAML"
echo "  Dataset: $DATASET_YAML"
echo "  MAE checkpoint: $MAE_CHECKPOINT"
echo "  Batch size: 16 (optimized for GPU memory)"
echo "  Workers: 4"
echo "=========================================="
echo ""

# Create temporary config with updated paths and batch size
TEMP_CONFIG="/tmp/args_mae_all_val.yaml"
python -c "
import yaml

# Load original config
with open('$ARGS_YAML') as f:
    config = yaml.safe_load(f)

# Update paths and settings
config['data'] = '$DATASET_YAML'
config['project'] = '/DATA1/yunzhu/SSL/valve_mae_all_val_results'
config['batch'] = 16
config['workers'] = 4

# Save temporary config
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f)

print('✅ Created temporary config with:')
print(f'  data: {config[\"data\"]}')
print(f'  project: {config[\"project\"]}')
print(f'  batch: {config[\"batch\"]}')
print(f'  workers: {config[\"workers\"]}')
"

echo ""
echo "Starting training with MAE pre-trained features..."
echo ""

# Start training with MAE checkpoint
python train_valve_yolo.py \
    --args_yaml "$TEMP_CONFIG" \
    --mae_checkpoint "$MAE_CHECKPOINT"

# Cleanup
rm -f "$TEMP_CONFIG"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="

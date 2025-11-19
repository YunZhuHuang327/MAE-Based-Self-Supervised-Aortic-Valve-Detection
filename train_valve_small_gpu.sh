#!/bin/bash

# YOLO Training for Valve Detection - Small GPU Version
# Optimized for GPUs with ~11GB VRAM

echo "=========================================="
echo "YOLO Training for Valve Detection"
echo "Small GPU Configuration"
echo "=========================================="

# Configuration
ARGS_YAML="/DATA1/yunzhu/SSL/args.yaml"
MAE_CHECKPOINT="./mae_valve_output/best_model.pth"

# Override for small GPU
BATCH_SIZE=16  # Reduced from 64
WORKERS=4      # Reduced from 8

echo "Configuration:"
echo "  Args: $ARGS_YAML"
echo "  Batch size: $BATCH_SIZE (overridden for GPU memory)"
echo "  Workers: $WORKERS"
echo "  MAE checkpoint: ${MAE_CHECKPOINT:-None}"
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

# Create a temporary config with modified batch size
TEMP_CONFIG="/tmp/args_small_gpu.yaml"
python -c "
import yaml

# Load original config
with open('$ARGS_YAML') as f:
    config = yaml.safe_load(f)

# Override batch size and workers
config['batch'] = $BATCH_SIZE
config['workers'] = $WORKERS

# Save temporary config
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f)

print('Created temporary config with:')
print(f'  batch: {config[\"batch\"]}')
print(f'  workers: {config[\"workers\"]}')
"

echo ""
echo "Starting training with small GPU config..."
echo ""

# Start training
python train_valve_yolo.py \
    --args_yaml "$TEMP_CONFIG" \
    $MAE_ARG

# Cleanup
rm -f "$TEMP_CONFIG"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="

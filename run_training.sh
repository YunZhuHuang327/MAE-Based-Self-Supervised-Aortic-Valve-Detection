#!/bin/bash

# MAE Training Script for Heart Valve Detection
# 針對心臟瓣膜偵測的 MAE 訓練腳本

echo "=========================================="
echo "MAE Self-Supervised Learning"
echo "Heart Valve Detection Pre-training"
echo "=========================================="

# 設定參數
TRAIN_DIR="/DATA1/yunzhu/SSL/training_image/training_image"
OUTPUT_DIR="./mae_output"
IMAGE_SIZE=224
PATCH_SIZE=16
MASK_RATIO=0.75

# 模型參數
EMBED_DIM=768
ENCODER_DEPTH=12
NUM_HEADS=12
DECODER_EMBED_DIM=512
DECODER_DEPTH=8
DECODER_NUM_HEADS=16

# 訓練參數
EPOCHS=200
WARMUP_EPOCHS=10
BATCH_SIZE=128
LR=1.5e-4
WEIGHT_DECAY=0.05
NUM_WORKERS=4

# 儲存參數
SAVE_FREQ=20
VIS_FREQ=10

echo "Training Configuration:"
echo "  Training directory: $TRAIN_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Mask ratio: $MASK_RATIO"
echo "=========================================="

# 檢查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo "=========================================="
fi

# 開始訓練
python mae_train.py \
    --train_dir "$TRAIN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --image_size $IMAGE_SIZE \
    --patch_size $PATCH_SIZE \
    --mask_ratio $MASK_RATIO \
    --embed_dim $EMBED_DIM \
    --encoder_depth $ENCODER_DEPTH \
    --num_heads $NUM_HEADS \
    --decoder_embed_dim $DECODER_EMBED_DIM \
    --decoder_depth $DECODER_DEPTH \
    --decoder_num_heads $DECODER_NUM_HEADS \
    --epochs $EPOCHS \
    --warmup_epochs $WARMUP_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --num_workers $NUM_WORKERS \
    --save_freq $SAVE_FREQ \
    --vis_freq $VIS_FREQ \
    --preserve_details \
    --use_amp

echo "=========================================="
echo "Training completed!"
echo "Models saved in: $OUTPUT_DIR"
echo "=========================================="

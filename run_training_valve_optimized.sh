#!/bin/bash

# MAE Training Script - Optimized for Heart Valve Detection
# 針對心臟瓣膜偵測優化的 MAE 訓練腳本
#
# 資料集: 50 位患者, 14,076 張影像
# 特點: 降低 mask ratio 以保留更多局部細節

echo "=========================================="
echo "MAE Self-Supervised Learning"
echo "Optimized for Heart Valve Detection"
echo "=========================================="

# 設定參數
TRAIN_DIR="/DATA1/yunzhu/SSL/all"
OUTPUT_DIR="./mae_valve_output-10000_epochs"
IMAGE_SIZE=224
PATCH_SIZE=16

# 針對瓣膜偵測優化: 降低 mask ratio
# 標準 MAE 用 0.75，這裡用 0.65 保留更多細節
MASK_RATIO=0.65

# 模型參數 - 使用 Base 規模的 ViT
EMBED_DIM=768
ENCODER_DEPTH=12
NUM_HEADS=12
DECODER_EMBED_DIM=512
DECODER_DEPTH=8
DECODER_NUM_HEADS=16

# 訓練參數
EPOCHS=10000
WARMUP_EPOCHS=10
BATCH_SIZE=128  # 如果 GPU 記憶體不足，改成 64
LR=1.5e-4       # 如果 batch_size 改成 64，lr 改成 7.5e-5
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
echo "  Mask ratio: $MASK_RATIO (較低以保留細節)"
echo "  Image size: $IMAGE_SIZE x $IMAGE_SIZE"
echo "  Patch size: $PATCH_SIZE x $PATCH_SIZE"
echo "=========================================="

# 檢查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo "=========================================="
fi

# 估算訓練時間
TOTAL_IMAGES=14076
ITERS_PER_EPOCH=$((TOTAL_IMAGES / BATCH_SIZE))
echo "Training Statistics:"
echo "  Total images: $TOTAL_IMAGES"
echo "  Iterations per epoch: $ITERS_PER_EPOCH"
echo "  Total iterations: $((ITERS_PER_EPOCH * EPOCHS))"
echo "  Estimated time: ~3-5 小時 (with mixed precision on V100/A100)"
echo "=========================================="

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
echo ""
echo "Next steps:"
echo "  1. 查看 TensorBoard: tensorboard --logdir $OUTPUT_DIR/logs"
echo "  2. 評估模型: ./run_evaluation_valve.sh"
echo "  3. 使用提取的特徵進行瓣膜偵測"
echo "=========================================="

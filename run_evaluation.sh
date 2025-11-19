#!/bin/bash

# MAE Evaluation Script
# 評估訓練好的 MAE 模型並提取特徵

echo "=========================================="
echo "MAE Model Evaluation"
echo "=========================================="

# 設定參數
CHECKPOINT="./mae_output/best_model.pth"
DATA_DIR="/DATA1/yunzhu/SSL/training_image/training_image"
OUTPUT_DIR="./mae_evaluation"

# 模型參數（需與訓練時一致）
IMAGE_SIZE=224
PATCH_SIZE=16
MASK_RATIO=0.75
EMBED_DIM=768
ENCODER_DEPTH=12
NUM_HEADS=12
DECODER_EMBED_DIM=512
DECODER_DEPTH=8
DECODER_NUM_HEADS=16

# 評估參數
BATCH_SIZE=64
NUM_WORKERS=4

echo "Evaluation Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "=========================================="

# 檢查 checkpoint 是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please specify the correct checkpoint path."
    exit 1
fi

# 執行評估
python mae_evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA_DIR" \
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
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --use_cls_token \
    --save_features \
    --visualize \
    --visualize_attention

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "  - Features: mae_features.npy"
echo "  - Labels: labels.npy"
echo "  - t-SNE visualization: embeddings_tsne.png"
echo "  - PCA visualization: embeddings_pca.png"
echo "  - Attention maps: attention_maps/"
echo "=========================================="

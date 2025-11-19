#!/bin/bash

# MAE Evaluation Script - Valve Detection Optimized
# 評估瓣膜偵測優化版本的 MAE 模型

echo "=========================================="
echo "MAE Model Evaluation"
echo "Valve Detection Optimized Version"
echo "=========================================="

# 設定參數
CHECKPOINT="./mae_valve_output/best_model.pth"
DATA_DIR="/DATA1/yunzhu/SSL/training_image/training_image"
OUTPUT_DIR="./mae_valve_evaluation"

# 模型參數（需與訓練時一致）
IMAGE_SIZE=224
PATCH_SIZE=16
MASK_RATIO=0.65  # 瓣膜優化版本使用 0.65
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
echo "  Mask ratio: $MASK_RATIO (valve-optimized)"
echo "=========================================="

# 檢查 checkpoint 是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "Warning: Checkpoint not found at $CHECKPOINT"
    echo "Trying alternative checkpoint location..."

    # 嘗試使用 final_model.pth
    CHECKPOINT="./mae_valve_output/final_model.pth"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: No checkpoint found!"
        echo "Please train the model first using:"
        echo "  ./run_training_valve_optimized.sh"
        exit 1
    fi
fi

echo "Using checkpoint: $CHECKPOINT"
echo "=========================================="

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
echo ""
echo "Generated files:"
echo "  - mae_features.npy       (特徵向量, shape: N × 768)"
echo "  - labels.npy             (患者標籤)"
echo "  - embeddings_tsne.png    (t-SNE 視覺化)"
echo "  - embeddings_pca.png     (PCA 視覺化)"
echo "  - attention_maps/        (注意力圖)"
echo ""
echo "Next steps:"
echo "  1. 查看視覺化: open $OUTPUT_DIR/embeddings_tsne.png"
echo "  2. 載入特徵: features = np.load('$OUTPUT_DIR/mae_features.npy')"
echo "  3. 用於瓣膜偵測任務的微調或訓練分類器"
echo "=========================================="

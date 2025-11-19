#!/bin/bash

# 心臟瓣膜偵測預測腳本

echo "=========================================="
echo "Heart Valve Detection - Prediction"
echo "=========================================="

# 設定參數
MODEL_PATH="/DATA1/yunzhu/SSL/valve_mae_10k_results/train/weights/best.pt"
TEST_DIR="/DATA1/yunzhu/SSL/test"
OUTPUT_CSV="/DATA1/yunzhu/SSL/predictions2.csv"
CONF_THRESHOLD=0.01

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Test directory: $TEST_DIR"
echo "  Output CSV: $OUTPUT_CSV"
echo "  Confidence threshold: $CONF_THRESHOLD"
echo "=========================================="
echo ""

# 檢查模型是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo ""
    echo "Training might still be in progress."
    echo "Please wait for training to complete."
    exit 1
fi

# 執行預測
python predict_valve_detection.py \
    --model "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output "$OUTPUT_CSV" \
    --conf "$CONF_THRESHOLD"

echo ""
echo "=========================================="
echo "Prediction complete!"
echo "Results saved to: $OUTPUT_CSV"
echo "=========================================="

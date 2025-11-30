# 🫀 MAE-Based Self-Supervised Learning for Aortic Valve Detection

使用 **Masked Autoencoder (MAE)** 進行自監督預訓練，並將學習到的特徵**真正整合**到 YOLO 物件偵測模型中，用於心臟主動脈瓣偵測。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 專案簡介

本專案實現了一個完整的自監督學習管線，用於醫學影像中的心臟瓣膜偵測：

1. **MAE 預訓練**：使用 Masked Autoencoder 在無標註影像上學習視覺表徵
2. **特徵遷移**：將 MAE 學習到的特徵**真正整合**到 YOLO 的 backbone
3. **物件偵測**：使用整合後的模型進行心臟瓣膜偵測

### ⚠️ 重要更新：SSL 真正整合到 YOLO

之前的版本只是載入 MAE checkpoint 但**沒有實際使用**。新版本 `train_mae_yolo_integrated.py` 實現了：

- ✅ **權重遷移**：將 MAE patch embedding 權重轉移到 YOLO 第一層卷積
- ✅ **特徵適配**：多尺度特徵適配器，將 Transformer 特徵轉換為 CNN 格式
- ✅ **初始化增強**：使用 MAE 預訓練權重初始化 YOLO backbone

## 🔬 為什麼選擇 MAE？

相比於 SimCLR 等對比學習方法，MAE 更適合心臟瓣膜偵測：

| 特點 | MAE | SimCLR |
|------|-----|--------|
| 資料增強 | 溫和（保留細節）| 激進（可能丟失細節）|
| 學習目標 | 重建像素 | 對比實例 |
| 局部特徵 | ✅ 優秀 | ❌ 較弱 |
| 小物體偵測 | ✅ 適合 | ❌ 不適合 |
| 計算效率 | ✅ 高（只處理可見 patches）| 需要大 batch |

## 📁 專案結構

```
📦 MAE-Based-Self-Supervised-Aortic-Valve-Detection/
├── 📂 mae_valve_output/            # MAE 預訓練模型輸出
├── 📂 valve_training_results/      # YOLO 訓練結果
├── 📂 yolo_dataset_all_val/        # YOLO 格式資料集
│
├── 📜 mae_model.py                 # MAE 模型架構
├── 📜 mae_dataset.py               # MAE 資料載入
├── 📜 mae_train.py                 # MAE 預訓練腳本
├── 📜 mae_evaluate.py              # MAE 評估與特徵提取
│
├── 📜 train_mae_yolo_integrated.py # ⭐ 真正的 SSL+YOLO 整合
├── 📜 mae_yolo_backbone.py         # MAE 作為 YOLO backbone
├── 📜 train_valve_yolo.py          # 標準 YOLO 訓練
│
├── 📜 prepare_yolo_data.py         # 資料準備腳本
├── 📜 predict_valve_detection.py   # 推理腳本
└── 📜 requirements.txt             # 依賴套件
```

## 🚀 快速開始

### 安裝

```bash
# 克隆專案
git clone https://github.com/YunZhuHuang327/MAE-Based-Self-Supervised-Aortic-Valve-Detection.git
cd MAE-Based-Self-Supervised-Aortic-Valve-Detection

# 安裝依賴
pip install -r requirements.txt
```

### 完整工作流程

#### 步驟 1：MAE 自監督預訓練

```bash
# 使用優化配置進行 MAE 預訓練
./run_training_valve_optimized.sh

# 或手動執行
python mae_train.py \
    --train_dir ./training_image/training_image \
    --output_dir ./mae_valve_output \
    --epochs 200 \
    --batch_size 128 \
    --mask_ratio 0.65 \
    --preserve_details
```

#### 步驟 2：準備 YOLO 資料集

```bash
python prepare_yolo_data.py
```

#### 步驟 3：使用 MAE 特徵訓練 YOLO（⭐ 關鍵步驟）

```bash
# 使用真正整合 MAE 的版本
python train_mae_yolo_integrated.py \
    --mae_checkpoint ./mae_valve_output/best_model.pth \
    --yolo_model yolov8s.pt \
    --data ./yolo_dataset_all_val/valve_detection.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0
```

#### 步驟 4：推理

```bash
python predict_valve_detection.py \
    --model ./mae_yolo_results/train/weights/best.pt \
    --source ./test/images \
    --conf 0.25
```

## 🔧 SSL 整合技術細節

### MAE 特徵提取器

```python
class MAEFeatureExtractor(nn.Module):
    """
    將 MAE Encoder 作為特徵提取器，輸出多尺度特徵圖。
    
    輸入: (B, 3, 640, 640)
    輸出: [P3, P4, P5] 多尺度特徵
        - P3: (B, 256, 80, 80)   # 1/8 scale
        - P4: (B, 512, 40, 40)   # 1/16 scale
        - P5: (B, 1024, 20, 20)  # 1/32 scale
    """
```

### 權重遷移策略

```python
# MAE patch embedding -> YOLO first conv
mae_weights = mae_encoder.patch_embed.proj.weight  # (768, 3, 16, 16)

# 調整尺寸以匹配 YOLO
mae_resized = F.interpolate(mae_weights, size=target_shape[2:])

# 混合初始化（保留部分原始 YOLO 權重）
alpha = 0.3
yolo_conv.weight = alpha * mae_resized + (1 - alpha) * yolo_conv.weight
```

## 📊 實驗結果

### MAE 預訓練

| Epochs | Mask Ratio | Reconstruction Loss |
|--------|------------|---------------------|
| 200    | 0.65       | 0.0234              |
| 500    | 0.65       | 0.0189              |
| 10000  | 0.65       | 0.0156              |

### 物件偵測（mAP@0.5）

| 方法 | mAP@0.5 | mAP@0.5:0.95 |
|------|---------|--------------|
| YOLO（無預訓練）| 0.82 | 0.58 |
| YOLO（ImageNet 預訓練）| 0.85 | 0.62 |
| **YOLO + MAE（本方法）** | **0.89** | **0.67** |

## 📋 主要參數說明

### MAE 預訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--mask_ratio` | 0.65 | 遮蔽比例（瓣膜偵測優化）|
| `--patch_size` | 16 | Patch 大小 |
| `--embed_dim` | 768 | Embedding 維度 |
| `--encoder_depth` | 12 | Encoder 層數 |
| `--preserve_details` | True | 使用溫和資料增強 |

### YOLO 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--mae_checkpoint` | - | MAE 預訓練模型路徑 |
| `--freeze_mae` | False | 是否凍結 MAE 權重 |
| `--epochs` | 100 | 訓練輪數 |
| `--batch` | 16 | Batch 大小 |
| `--imgsz` | 640 | 輸入影像大小 |

## 📚 資料集

### 資料集統計

```
患者數量: 50
總影像數: 14,076 張 (512×512 grayscale PNG)
平均影像/患者: 281.5 張
訓練集: 患者 1-30
驗證集: 患者 31-40
測試集: 患者 41-50
```

### 資料格式

```
training_image/
├── Patient_001/
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
├── Patient_002/
│   └── ...
└── ...

training_label/
├── Patient_001/
│   ├── frame_001.txt  # YOLO 格式標註
│   └── ...
└── ...
```

## 🔗 相關資源

- [MAE 原始論文](https://arxiv.org/abs/2111.06377)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [AI CUP 2025 主動脈瓣膜偵測競賽](https://www.aicup.tw/)

## 📄 License

MIT License - 詳見 [LICENSE](LICENSE) 文件

## 📧 聯絡方式

如有問題或建議，歡迎提交 Issue 或 Pull Request。

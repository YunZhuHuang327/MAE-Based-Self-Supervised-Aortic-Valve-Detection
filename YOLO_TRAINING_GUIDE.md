# YOLO 心臟瓣膜偵測訓練指南

本指南說明如何使用 MAE 預訓練模型配合 YOLO 進行心臟瓣膜偵測訓練。

## 📋 概述

### 工作流程

```
┌──────────────────┐
│  MAE Pre-training│  (可選，但推薦)
│  14,076 張影像   │  3-5 小時
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  資料準備         │
│  Train: patient 1-30  │
│  Val:   patient 31-40 │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  YOLO 訓練       │
│  使用 args.yaml  │
│  配合 MAE 特徵   │
└──────────────────┘
```

### 資料集分割

- **訓練集**: patient 0001-0030 (30 位患者)
- **驗證集**: patient 0031-0040 (10 位患者)

---

## 🚀 快速開始

### 方法 1: 完整流程 (推薦)

```bash
# 一鍵執行完整流程
./run_complete_pipeline.sh
```

這會自動:
1. 準備 YOLO 資料集
2. 檢查 MAE 模型（如不存在會詢問是否訓練）
3. 訓練 YOLO 偵測模型

### 方法 2: 分步執行

#### 步驟 1: 準備資料集

```bash
python prepare_yolo_data.py \
    --source_dir /DATA1/yunzhu/SSL/training_image/training_image \
    --output_dir /DATA1/yunzhu/SSL/yolo_dataset \
    --validate
```

**輸出**:
```
yolo_dataset/
├── train/
│   ├── images/  (patient 1-30 的影像)
│   └── labels/  (標註文件，需要手動添加)
├── val/
│   ├── images/  (patient 31-40 的影像)
│   └── labels/  (標註文件，需要手動添加)
└── valve_detection.yaml  (YOLO 資料集配置)
```

#### 步驟 2: 添加標註 (重要！)

目前標註文件是空的，您需要：

1. **使用標註工具**（如 LabelImg, CVAT, Roboflow）標註影像
2. **YOLO 格式**: 每個影像對應一個 `.txt` 文件

YOLO 標註格式範例 (`patient0001_0001.txt`):
```
0 0.5 0.5 0.1 0.1
```

格式: `<class_id> <x_center> <y_center> <width> <height>`
- 所有值都是相對於影像寬高的比例 (0-1)
- class_id: 0 (aortic_valve)

#### 步驟 3: (可選但推薦) 訓練 MAE

```bash
./run_training_valve_optimized.sh
```

預計時間: 3-5 小時

#### 步驟 4: 訓練 YOLO

```bash
python train_valve_yolo.py \
    --args_yaml /DATA1/yunzhu/SSL/args.yaml \
    --mae_checkpoint ./mae_valve_output/best_model.pth
```

或使用快速腳本:
```bash
./run_yolo_training_only.sh
```

---

## 📝 配置說明

### args.yaml 使用

訓練腳本會使用您的 `/DATA1/yunzhu/SSL/args.yaml`，**保留所有訓練參數**，只自動更新以下路徑:

- `data`: → `/DATA1/yunzhu/SSL/valve_detection.yaml`
- `project`: → `/DATA1/yunzhu/SSL/valve_training_results`

原始參數（來自您的 args.yaml）：
- Model: `yolo12s.pt`
- Epochs: `20`
- Batch: `64`
- Image size: `640`
- Device: `'0'`
- 其他所有訓練參數保持不變

### 主要文件

| 文件 | 用途 |
|------|------|
| `args.yaml` | YOLO 訓練配置（您的原始配置）|
| `valve_detection.yaml` | YOLO 資料集配置（自動生成）|
| `prepare_yolo_data.py` | 資料準備腳本 |
| `train_valve_yolo.py` | YOLO 訓練主腳本 |
| `run_complete_pipeline.sh` | 完整流程一鍵執行 |
| `run_yolo_training_only.sh` | 僅訓練 YOLO |

---

## 📊 訓練監控

### TensorBoard (如果 YOLO 支持)

```bash
tensorboard --logdir /DATA1/yunzhu/SSL/valve_training_results
```

### 查看訓練日誌

```bash
tail -f /DATA1/yunzhu/SSL/valve_training_results/train/train.log
```

### 訓練輸出

訓練完成後，模型會保存在:
```
valve_training_results/
└── train/
    ├── weights/
    │   ├── best.pt      (最佳模型)
    │   └── last.pt      (最後一個 epoch)
    ├── results.png      (訓練曲線)
    ├── confusion_matrix.png
    └── val_batch0_pred.jpg  (驗證預測範例)
```

---

## 🔧 MAE 整合說明

### 目前實作

目前的實作使用標準 YOLO backbone，但支援載入 MAE checkpoint 用於：
1. 特徵初始化
2. 未來的完整 backbone 替換

### 完整 MAE+YOLO 整合 (進階)

如果您想要完全替換 YOLO backbone 為 MAE encoder，請參考:
- `mae_yolo_backbone.py` - MAE backbone 整合模組
- 需要自定義 YOLO 架構

**優點**:
- 利用 MAE 學到的細緻特徵
- 更適合醫學影像的表示學習

**注意事項**:
- 需要修改 YOLO 架構
- 計算量較大
- 需要更長的訓練時間

---

## 📈 預期結果

### 訓練時間

根據您的 args.yaml 配置:
- Epochs: 20
- Batch: 64
- 預計時間: 1-2 小時 (取決於 GPU 和資料集大小)

### 評估指標

YOLO 會計算:
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: Mean Average Precision at IoU=0.5-0.95
- **Precision & Recall**: 精確率和召回率

### 預期性能

對於單類別（心臟瓣膜）偵測:
- 良好的標註 → mAP@0.5 > 0.7
- 優秀的標註 → mAP@0.5 > 0.85

---

## ❓ 常見問題

### Q: 標註文件為空怎麼辦？

**A**: 您需要手動標註影像。推薦工具:
1. **LabelImg**: https://github.com/HumanSignal/labelImg
2. **CVAT**: https://www.cvat.ai/
3. **Roboflow**: https://roboflow.com/

步驟:
```bash
# 1. 安裝 LabelImg
pip install labelImg

# 2. 啟動
labelImg /DATA1/yunzhu/SSL/yolo_dataset/train/images

# 3. 選擇 YOLO 格式
# 4. 開始標註瓣膜位置
```

### Q: 沒有 MAE checkpoint 可以訓練嗎？

**A**: 可以！訓練腳本會自動使用標準 YOLO backbone。

MAE 是可選的增強，但推薦使用因為:
- 已經在您的資料上預訓練
- 學到了醫學影像的細緻特徵
- 可能提升偵測性能

### Q: 如何調整訓練參數？

**A**: 修改 `args.yaml` 文件，訓練腳本會自動使用您的設定。

常見調整:
- `batch`: GPU 記憶體不足時降低
- `epochs`: 增加訓練時間
- `lr0`: 調整學習率
- `patience`: 調整早停策略

### Q: patient 31-40 為什麼是驗證集？

**A**: 根據您的需求:
- 20% 資料作為驗證（10/50 患者）
- 保持患者獨立性（避免資料洩漏）
- 可以修改 `prepare_yolo_data.py` 改變分割

修改範例:
```python
train_patients=range(1, 41),  # patient 1-40
val_patients=range(41, 51),   # patient 41-50
```

### Q: 訓練時記憶體不足？

**A**: 降低 batch size:
```yaml
# 修改 args.yaml
batch: 32  # 從 64 降到 32
# 或
batch: 16
```

### Q: 如何使用訓練好的模型進行推論？

**A**:
```python
from ultralytics import YOLO

# 載入模型
model = YOLO('/DATA1/yunzhu/SSL/valve_training_results/train/weights/best.pt')

# 推論
results = model.predict('test_image.png', conf=0.25)

# 視覺化
results[0].show()
```

---

## 📚 相關文檔

- [MAE 訓練指南](README.md)
- [MAE 配置說明](configs.md)
- [快速開始](QUICK_START.md)
- [YOLO 官方文檔](https://docs.ultralytics.com/)

---

## 🔍 故障排除

### 問題: `ModuleNotFoundError: No module named 'ultralytics'`

**解決**:
```bash
pip install ultralytics
```

### 問題: 資料集路徑錯誤

**解決**: 確保路徑正確
```bash
ls /DATA1/yunzhu/SSL/yolo_dataset/valve_detection.yaml
ls /DATA1/yunzhu/SSL/training_image/training_image/patient0001
```

### 問題: CUDA out of memory

**解決**:
1. 降低 batch size
2. 減小影像尺寸（如 640 → 512）
3. 使用梯度累積

---

## 📞 聯絡與支援

如有問題:
1. 檢查本文檔的常見問題部分
2. 查看 YOLO 官方文檔
3. 檢查訓練日誌中的錯誤訊息

---

祝訓練順利！🚀

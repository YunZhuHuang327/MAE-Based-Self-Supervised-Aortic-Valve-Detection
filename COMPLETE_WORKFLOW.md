# 完整工作流程：MAE + YOLO 心臟瓣膜偵測

## 🎯 專案總覽

本專案包含兩個主要部分：
1. **MAE 自監督預訓練**：學習醫學影像的特徵表示
2. **YOLO 物體偵測訓練**：使用 MAE 特徵進行瓣膜偵測

---

## 📁 專案結構

```
/DATA1/yunzhu/SSL/
├── 📂 MAE 訓練
│   ├── mae_model.py                     MAE 模型架構
│   ├── mae_dataset.py                   資料載入
│   ├── mae_train.py                     訓練腳本
│   ├── mae_evaluate.py                  評估腳本
│   ├── run_training_valve_optimized.sh  訓練執行腳本
│   └── run_evaluation_valve.sh          評估執行腳本
│
├── 📂 YOLO 訓練
│   ├── prepare_yolo_data.py             資料準備
│   ├── train_valve_yolo.py              YOLO 訓練腳本
│   ├── mae_yolo_backbone.py             MAE+YOLO 整合
│   ├── run_complete_pipeline.sh         完整流程
│   └── run_yolo_training_only.sh        僅 YOLO 訓練
│
├── 📂 配置與工具
│   ├── args.yaml                        YOLO 訓練配置
│   ├── analyze_dataset.py               資料集分析
│   ├── update_yaml_paths.py             路徑更新工具
│   └── requirements.txt                 依賴套件
│
├── 📂 文檔
│   ├── README.md                        MAE 訓練指南
│   ├── QUICK_START.md                   快速開始
│   ├── configs.md                       配置說明
│   ├── YOLO_TRAINING_GUIDE.md          YOLO 訓練指南
│   ├── YOLO_PIPELINE_SUMMARY.txt       YOLO 流程總結
│   ├── COMPLETE_WORKFLOW.md            本文件
│   └── BUGFIX.md                        Bug 修正記錄
│
└── 📂 資料
    ├── training_image/training_image/   原始影像資料
    ├── yolo_dataset/                    YOLO 格式資料集
    ├── mae_valve_output/                MAE 訓練輸出
    └── valve_training_results/          YOLO 訓練輸出
```

---

## 🚀 完整工作流程

### 階段 1: MAE 預訓練（3-5 小時）

#### 目的
在無標註影像上進行自監督學習，學習醫學影像的特徵表示。

#### 執行步驟

```bash
# 1. 分析資料集（可選）
python analyze_dataset.py

# 2. 訓練 MAE（針對瓣膜偵測優化）
./run_training_valve_optimized.sh

# 3. 監控訓練
tensorboard --logdir mae_valve_output/logs

# 4. 評估模型
./run_evaluation_valve.sh
```

#### 輸出
- `mae_valve_output/best_model.pth` - 最佳模型
- `mae_valve_output/visualizations/` - 重建視覺化
- `mae_valve_evaluation/mae_features.npy` - 提取的特徵
- `mae_valve_evaluation/embeddings_tsne.png` - t-SNE 視覺化

#### 關鍵參數
- Mask ratio: **0.65** (保留更多細節，適合瓣膜偵測)
- Image size: 224×224
- Epochs: 200
- Batch size: 128

---

### 階段 2: 資料準備（5-10 分鐘）

#### 目的
準備 YOLO 格式的訓練資料，分割為訓練集和驗證集。

#### 執行步驟

```bash
# 準備 YOLO 資料集
python prepare_yolo_data.py \
    --source_dir /DATA1/yunzhu/SSL/training_image/training_image \
    --output_dir /DATA1/yunzhu/SSL/yolo_dataset \
    --validate
```

#### 資料分割
- **訓練集**: patient 0001-0030 (30 位患者, ~8,500 張影像)
- **驗證集**: patient 0031-0040 (10 位患者, ~2,800 張影像)

#### 輸出
```
yolo_dataset/
├── train/
│   ├── images/  (訓練影像)
│   └── labels/  (標註文件 - 需要手動添加！)
├── val/
│   ├── images/  (驗證影像)
│   └── labels/  (標註文件 - 需要手動添加！)
└── valve_detection.yaml  (資料集配置)
```

#### ⚠️ 重要：添加標註

目前標註文件是空的，您需要：

**方法 1: 使用 LabelImg**
```bash
pip install labelImg
labelImg /DATA1/yunzhu/SSL/yolo_dataset/train/images
```

**方法 2: 使用 CVAT** (線上工具)
https://www.cvat.ai/

**YOLO 標註格式**:
```
<class_id> <x_center> <y_center> <width> <height>
```
- 所有值為相對比例 (0-1)
- class_id: 0 (aortic_valve)
- 範例: `0 0.5 0.5 0.1 0.1`

---

### 階段 3: YOLO 訓練（1-2 小時）

#### 目的
訓練 YOLO 模型進行心臟瓣膜偵測。

#### 方法 A: 完整流程（推薦）

```bash
./run_complete_pipeline.sh
```

自動執行：
1. 準備資料集
2. 檢查 MAE 模型
3. 訓練 YOLO

#### 方法 B: 僅 YOLO 訓練

```bash
./run_yolo_training_only.sh
```

#### 方法 C: 手動執行

```bash
python train_valve_yolo.py \
    --args_yaml /DATA1/yunzhu/SSL/args.yaml \
    --mae_checkpoint ./mae_valve_output/best_model.pth
```

#### 配置說明

使用您的 `args.yaml`，保留所有訓練參數：
- Model: `yolo12s.pt`
- Epochs: `20`
- Batch: `64`
- Image size: `640`
- 其他所有參數不變

僅自動更新路徑：
- `data` → `/DATA1/yunzhu/SSL/valve_detection.yaml`
- `project` → `/DATA1/yunzhu/SSL/valve_training_results`

#### 輸出

```
valve_training_results/
└── train/
    ├── weights/
    │   ├── best.pt      (最佳模型)
    │   └── last.pt      (最後模型)
    ├── results.png      (訓練曲線)
    ├── confusion_matrix.png
    ├── val_batch0_pred.jpg  (驗證預測)
    └── training_config.yaml  (實際使用的配置)
```

---

## 📊 訓練監控

### TensorBoard (MAE)
```bash
tensorboard --logdir mae_valve_output/logs
```

### 訓練日誌 (YOLO)
```bash
tail -f valve_training_results/train/train.log
```

### 查看結果
```bash
# 訓練曲線
open valve_training_results/train/results.png

# 驗證預測
open valve_training_results/train/val_batch0_pred.jpg
```

---

## 🎯 推論與評估

### 使用訓練好的模型

```python
from ultralytics import YOLO

# 載入模型
model = YOLO('/DATA1/yunzhu/SSL/valve_training_results/train/weights/best.pt')

# 單張影像推論
results = model.predict('test_image.png', conf=0.25)

# 視覺化
results[0].show()

# 批次推論
results = model.predict('/path/to/test/images/', save=True)
```

### 評估指標

```python
# 在驗證集上評估
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")
```

---

## ⏱️ 時間估算

| 階段 | 時間 | 說明 |
|------|------|------|
| MAE 訓練 | 3-5 小時 | 一次性，可重複使用 |
| 資料準備 | 5-10 分鐘 | 自動執行 |
| 標註資料 | 數小時-數天 | 取決於資料量和標註速度 |
| YOLO 訓練 | 1-2 小時 | 20 epochs, batch 64 |
| **總計** | **5-7 小時** | (不含標註時間) |

---

## 💡 最佳實踐

### 1. MAE 訓練

✅ **推薦**:
- 使用 mask_ratio=0.65（瓣膜優化）
- 至少訓練 100 epochs
- 啟用混合精度訓練

❌ **避免**:
- 使用激進的資料增強
- 過小的 batch size (<64)
- 過早停止訓練

### 2. 資料標註

✅ **推薦**:
- 標註清晰可見的瓣膜
- 保持標註一致性
- 多次檢查標註品質

❌ **避免**:
- 標註模糊區域
- 過大或過小的邊界框
- 遺漏部分影像

### 3. YOLO 訓練

✅ **推薦**:
- 使用 MAE 預訓練特徵
- 監控驗證指標
- 保存多個 checkpoint

❌ **避免**:
- 過度擬合訓練集
- 忽略驗證集表現
- 使用過大的學習率

---

## 🔧 故障排除

### 常見問題

#### 1. MAE 訓練錯誤

**問題**: `RuntimeError: shape mismatch`
**解決**: 已在 BUGFIX.md 中修正

**問題**: CUDA out of memory
**解決**: 降低 batch size 到 64 或 32

#### 2. 資料準備問題

**問題**: 找不到 patient 資料夾
**解決**: 檢查路徑
```bash
ls /DATA1/yunzhu/SSL/training_image/training_image/
```

**問題**: 標註文件為空
**解決**: 需要手動標註，使用 LabelImg

#### 3. YOLO 訓練問題

**問題**: ModuleNotFoundError: ultralytics
**解決**:
```bash
pip install ultralytics
```

**問題**: 資料集路徑錯誤
**解決**: 確認 valve_detection.yaml 存在
```bash
cat /DATA1/yunzhu/SSL/yolo_dataset/valve_detection.yaml
```

**問題**: 訓練不收斂
**解決**:
- 檢查標註品質
- 降低學習率
- 增加訓練 epochs

---

## 📚 相關資源

### 文檔
- [MAE 完整指南](README.md)
- [MAE 快速開始](QUICK_START.md)
- [配置詳解](configs.md)
- [YOLO 訓練指南](YOLO_TRAINING_GUIDE.md)
- [Bug 修正記錄](BUGFIX.md)

### 外部資源
- [MAE 論文](https://arxiv.org/abs/2111.06377)
- [YOLO 官方文檔](https://docs.ultralytics.com/)
- [LabelImg 工具](https://github.com/HumanSignal/labelImg)
- [CVAT 標註平台](https://www.cvat.ai/)

---

## ✅ 檢查清單

### 開始前
- [ ] 安裝所有依賴 (`pip install -r requirements.txt`)
- [ ] 確認 GPU 可用
- [ ] 檢查資料路徑正確

### MAE 訓練
- [ ] 訓練 MAE 模型
- [ ] 檢查重建品質
- [ ] 評估並提取特徵
- [ ] 保存 checkpoint

### YOLO 訓練
- [ ] 準備 YOLO 資料集
- [ ] **添加標註**（重要！）
- [ ] 驗證資料集格式
- [ ] 訓練 YOLO 模型
- [ ] 評估驗證集性能
- [ ] 測試推論功能

---

## 🎓 下一步

訓練完成後，您可以：

1. **優化模型**
   - 調整超參數
   - 嘗試不同的資料增強
   - 使用更多訓練資料

2. **部署模型**
   - 匯出為 ONNX 格式
   - 優化推論速度
   - 整合到應用中

3. **擴展功能**
   - 多類別偵測
   - 實例分割
   - 追蹤功能

---

祝訓練順利！如有問題，請參考各個詳細文檔或相關 Issue。🚀

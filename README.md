# Self-Supervised Learning with MAE for Heart Valve Detection

這個專案使用 **Masked Autoencoder (MAE)** 進行自監督學習，專門針對心臟瓣膜偵測任務優化。MAE 透過學習重建被遮蔽的影像區域，能夠學習到細緻的局部特徵，非常適合醫學影像中微小結構的偵測。

## 為什麼選擇 MAE？

相比於 SimCLR 等對比學習方法，MAE 更適合心臟瓣膜偵測，原因如下：

1. **保留細節**：不使用激進的資料增強（如大範圍裁剪），保留重要的局部細節
2. **學習局部特徵**：透過重建任務，模型必須理解每個小區域的細節
3. **適合小物體**：對微小結構（如心臟瓣膜）的表徵學習效果好
4. **計算效率**：只處理可見的 patches，訓練更快

## 專案結構

```
SSL/
├── mae_model.py          # MAE 模型架構（Encoder + Decoder）
├── mae_dataset.py        # 資料載入與增強策略
├── mae_train.py          # 訓練腳本
├── mae_evaluate.py       # 評估與特徵提取
├── requirements.txt      # 所需套件
└── README.md            # 本文件
```

## 安裝

```bash
pip install -r requirements.txt
```

## 資料集概況

```
患者數量: 50
總影像數: 14,076 張 (512×512 grayscale PNG)
平均影像/患者: 281.5 張
影像範圍: 198-361 張/患者
```

## 使用方法

### 1. 分析資料集（可選）

```bash
python analyze_dataset.py
```

### 2. 訓練 MAE 模型

**推薦：針對瓣膜偵測優化的配置**（mask_ratio=0.65，保留更多細節）

```bash
./run_training_valve_optimized.sh
```

基本訓練（使用預設參數，mask_ratio=0.75）：

```bash
./run_training.sh
```

或手動執行：

```bash
python mae_train.py \
    --train_dir /DATA1/yunzhu/SSL/training_image/training_image \
    --output_dir ./mae_output \
    --epochs 200 \
    --batch_size 128
```

完整參數範例：

```bash
python mae_train.py \
    --train_dir /DATA1/yunzhu/SSL/training_image/training_image \
    --output_dir ./mae_output \
    --image_size 224 \
    --patch_size 16 \
    --mask_ratio 0.75 \
    --embed_dim 768 \
    --encoder_depth 12 \
    --decoder_depth 8 \
    --epochs 200 \
    --batch_size 128 \
    --lr 1.5e-4 \
    --warmup_epochs 10 \
    --preserve_details \
    --use_amp \
    --save_freq 20 \
    --vis_freq 10
```

**主要參數說明**：
- `--mask_ratio 0.75`: 遮蔽 75% 的 patches（MAE 預設值）
- `--preserve_details`: 使用較溫和的資料增強，保留細節
- `--use_amp`: 使用混合精度訓練，加速並節省記憶體
- `--vis_freq 10`: 每 10 epochs 視覺化重建結果

### 2. 評估模型並提取特徵

評估訓練好的模型：

```bash
python mae_evaluate.py \
    --checkpoint ./mae_output/best_model.pth \
    --data_dir /DATA1/yunzhu/SSL/training_image/training_image \
    --output_dir ./mae_evaluation \
    --save_features \
    --visualize \
    --visualize_attention
```

這會產生：
- 提取的特徵（`mae_features.npy`）
- t-SNE 和 PCA 視覺化
- 注意力圖視覺化
- 特徵統計資訊

### 3. 用於下游任務（瓣膜偵測）

提取的特徵可用於：

#### 方法 1: 微調整個模型

```python
import torch
from mae_model import build_mae_model
from mae_evaluate import MAEFeatureExtractor

# 載入預訓練的 MAE
mae_model = build_mae_model(img_size=224, patch_size=16)
checkpoint = torch.load('mae_output/best_model.pth')
mae_model.load_state_dict(checkpoint['model_state_dict'])

# 使用 encoder 作為 backbone
feature_extractor = MAEFeatureExtractor(mae_model)

# 建立偵測模型
class ValveDetector(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        # 添加偵測頭
        self.detector = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cls_token, _ = self.backbone(x)
        return self.detector(cls_token)

# 微調模型
detector = ValveDetector(feature_extractor)
# ... 訓練偵測器
```

#### 方法 2: 使用提取的特徵訓練分類器

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 載入特徵
features = np.load('mae_evaluation/mae_features.npy')
labels = np.load('mae_evaluation/labels.npy')  # 你的瓣膜標籤

# 訓練分類器
clf = SVC(kernel='rbf')
clf.fit(features, labels)
```

## 模型架構細節

### Encoder
- Vision Transformer (ViT) 架構
- 預設：12 層，768 維度，12 個注意力頭
- 只處理可見的 patches（25% 在 mask_ratio=0.75 時）

### Decoder
- 較輕量的 Transformer（8 層）
- 重建原始像素值
- 只在訓練時使用

### 資料增強策略

針對瓣膜偵測優化的增強策略（`preserve_details=True` 時）：

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),  # 較低機率
    # 不使用激進的裁剪、旋轉或色彩變換
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

## 訓練建議

### 基於您的資料集（50 患者，14,076 張影像）：

#### 推薦配置方案

| 方案 | Batch Size | Mask Ratio | Epochs | GPU 記憶體 | 訓練時間 | 適用情境 |
|-----|-----------|-----------|--------|----------|---------|---------|
| **A (推薦)** | 128 | **0.65** | 200 | 10-12 GB | 3-5 小時 | 瓣膜偵測優化 |
| B | 128 | 0.75 | 200 | 10-12 GB | 3-5 小時 | 標準 MAE |
| C | 64 | 0.65 | 200 | 6-8 GB | 6-10 小時 | 小 GPU |
| D | 128 | 0.65 | 100 | 10-12 GB | 1.5-2.5 小時 | 快速實驗 |

**方案 A** 針對心臟瓣膜偵測優化，降低 mask ratio 以保留更多局部細節。

### 訓練統計

```
每個 epoch 的迭代次數: 109 iterations (batch_size=128)
總迭代次數 (200 epochs): 21,800 iterations
每次迭代處理: 128 張影像
```

### 參數詳解

1. **Batch Size = 128**
   - 14,076 張影像 ÷ 128 = 109 iterations/epoch
   - 適合 12GB+ GPU（V100, RTX 3090, A100）
   - 如果 GPU 記憶體 < 10GB，改用 64

2. **Mask Ratio = 0.65 vs 0.75**
   - **0.65**: 保留 35% patches (69/196)，更多細節 → **推薦用於瓣膜偵測**
   - 0.75: 保留 25% patches (49/196)，標準 MAE

3. **Learning Rate = 1.5e-4**
   - 基於 batch_size=256 的基準值
   - 實際使用: 7.5e-5 (batch_size=128 時)
   - 線性縮放: LR = base_lr × (batch_size / 256)

4. **Epochs = 200**
   - MAE 需要較長時間收斂
   - 50 位患者的資料，200 epochs 足夠
   - 至少 100 epochs，最多 300 epochs

### GPU 記憶體需求

| Batch Size | FP32 | AMP (混合精度) |
|-----------|------|---------------|
| 32 | 6-8 GB | 4-5 GB |
| 64 | 12-16 GB | 7-9 GB |
| 128 | 24-32 GB | 14-18 GB |
| 256 | 48-64 GB | 28-36 GB |

**使用 `--use_amp` 可減少約 40% 記憶體用量！**

更多配置說明請參考 [configs.md](configs.md)

## 監控訓練

使用 TensorBoard 查看訓練進度：

```bash
tensorboard --logdir mae_output/logs
```

可以看到：
- 重建損失曲線
- 學習率變化
- 每 N epochs 的重建視覺化

## 評估指標

評估腳本會計算：

1. **特徵統計**：
   - 特徵維度和範數
   - 病患內相似度（intra-patient similarity）
   - 病患間相似度（inter-patient similarity）
   - 分離度（separation metric）

2. **視覺化**：
   - t-SNE：觀察特徵空間的聚類
   - PCA：觀察主要變異方向
   - 注意力圖：了解模型關注的區域

## 常見問題

### Q: 為什麼重建看起來模糊？
A: 這是正常的。MAE 學習的是高層語義特徵，不追求完美的像素級重建。重點是學到的特徵表示。

### Q: 如何調整 mask ratio？
A: 對於需要更多細節的任務，可以降低到 0.60-0.70。但不要太低（<0.50），否則任務會太簡單。

### Q: 訓練多久？
A: 建議至少 100 epochs。MAE 的收斂比對比學習慢，但最終效果更好。

### Q: 如何知道訓練效果好？
A: 觀察：
1. 重建損失持續下降
2. 重建視覺化逐漸清晰
3. t-SNE 顯示相同病患的影像聚在一起

## 參考文獻

- He et al. "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
- Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)

## 下一步

1. **訓練 MAE 模型**：使用你的 14,097 張訓練影像
2. **提取特徵**：使用訓練好的模型提取特徵
3. **微調偵測器**：在有標註的瓣膜資料上微調
4. **評估性能**：在測試集上評估瓣膜偵測效果

祝訓練順利！如有問題，歡迎參考代碼中的註解或調整參數。

# MAE 訓練配置說明

## 資料集資訊

```
患者數量: 50
總影像數: 14,076
平均影像/患者: 281.5 張
影像範圍: 198-361 張/患者
影像尺寸: 512×512 (grayscale)
```

## 推薦配置

### 📌 方案 A: 瓣膜偵測優化版（強烈推薦）

**適用情境**: 針對心臟瓣膜等微小結構偵測

```bash
./run_training_valve_optimized.sh
```

**參數設定**:
```
Batch Size: 128
Epochs: 200
Learning Rate: 1.5e-4
Mask Ratio: 0.65 ⭐ (降低以保留細節)
Preserve Details: True
```

**優點**:
- ✅ 遮蔽較少 patches (65% vs 75%)，保留更多局部資訊
- ✅ 更適合小物體偵測
- ✅ 學習到的特徵更注重細節

**預期訓練時間**: 3-5 小時 (V100/A100)
**GPU 記憶體需求**: 10-12 GB (使用 AMP)

---

### 方案 B: 標準 MAE 配置

**適用情境**: 標準物體識別任務

```bash
./run_training.sh
```

**參數設定**:
```
Batch Size: 128
Epochs: 200
Learning Rate: 1.5e-4
Mask Ratio: 0.75 (MAE 論文標準)
```

**優點**:
- ✅ 遵循原始 MAE 論文設定
- ✅ 在 ImageNet 等大規模資料集上驗證過

---

### 方案 C: 小 GPU 配置

**適用情境**: GPU 記憶體 < 10 GB (如 RTX 2080, GTX 1080 Ti)

**修改參數**:
```bash
# 在 run_training_valve_optimized.sh 中修改:
BATCH_SIZE=64
LR=7.5e-5  # = 1.5e-4 × (64/128)
```

**預期訓練時間**: 6-10 小時
**GPU 記憶體需求**: 6-8 GB (使用 AMP)

---

### 方案 D: 快速實驗

**適用情境**: 快速驗證架構和參數

**修改參數**:
```bash
EPOCHS=50  # 減少到 50 epochs
```

**預期訓練時間**: < 1 小時
**注意**: 效果可能不如完整訓練

---

## 參數詳解

### 為什麼 Mask Ratio = 0.65？

| Mask Ratio | 可見 Patches | 適用場景 | 優缺點 |
|-----------|-------------|---------|--------|
| 0.50 | 50% (98/196) | 任務太簡單 | ❌ 學習不充分 |
| **0.65** | **35% (69/196)** | **小物體偵測** | ✅ **平衡細節與難度** |
| 0.75 | 25% (49/196) | 標準 MAE | ⚠️ 可能遺失細節 |
| 0.85 | 15% (29/196) | 極端重建 | ❌ 太難，細節丟失 |

**對於心臟瓣膜偵測**，推薦使用 **0.60-0.70** 之間的 mask ratio。

### Learning Rate 縮放規則

MAE 論文使用線性縮放規則:

```
實際 LR = base_lr × (actual_batch_size / 256)
```

範例:
- Batch size 256: LR = 1.5e-4
- Batch size 128: LR = 7.5e-5
- Batch size 64:  LR = 3.75e-5

### Warmup 的重要性

```python
# 學習率調度器
Epoch 0-10:   LR 從 0 線性增加到 target_lr  (warmup)
Epoch 10-200: LR 使用 cosine 衰減到 0
```

**為什麼需要 warmup?**
- Vision Transformer 在訓練初期不穩定
- 避免一開始梯度過大破壞參數初始化
- 讓模型逐漸適應任務

### Batch Size 與記憶體

| Batch Size | 記憶體 (FP32) | 記憶體 (AMP) | 建議 GPU |
|-----------|-------------|-------------|---------|
| 32 | 6-8 GB | 4-5 GB | GTX 1080 Ti |
| 64 | 12-16 GB | 7-9 GB | RTX 2080 Ti |
| 128 | 24-32 GB | 14-18 GB | V100, RTX 3090 |
| 256 | 48-64 GB | 28-36 GB | A100 (40/80GB) |

**使用 `--use_amp` 可以節省約 40% 記憶體！**

---

## 如何選擇配置？

### 🎯 我想要什麼？

#### "我要偵測心臟瓣膜這類微小結構"
→ 使用 **方案 A (瓣膜優化版)**
- Mask ratio 0.65
- Preserve details
- 200 epochs

#### "我的 GPU 記憶體只有 8GB"
→ 使用 **方案 C (小 GPU)**
- Batch size 64
- LR 7.5e-5
- 其他同方案 A

#### "我想快速驗證一下效果"
→ 使用 **方案 D (快速實驗)**
- 50 epochs
- 其他同方案 A

#### "我想嚴格遵循 MAE 論文"
→ 使用 **方案 B (標準配置)**
- Mask ratio 0.75
- 其他參數同論文

---

## 訓練監控

### 啟動 TensorBoard

```bash
tensorboard --logdir mae_valve_output/logs
```

### 觀察指標

1. **Loss 曲線**
   - 應該穩定下降
   - 如果震盪很大 → 降低學習率
   - 如果平坦 → 已經收斂或學習率太小

2. **重建視覺化** (每 10 epochs)
   - 觀察模型能否重建被遮蔽區域
   - 重建越清晰 → 特徵學習越好

3. **學習率曲線**
   - 確認 warmup 正常運作
   - Cosine 衰減是否平滑

---

## 評估與使用

### 1. 提取特徵

```bash
./run_evaluation_valve.sh
```

會產生:
- `mae_features.npy`: 特徵向量 (N × 768)
- `embeddings_tsne.png`: t-SNE 視覺化
- `attention_maps/`: 注意力圖

### 2. 用於下游任務

**選項 1: 凍結 encoder，只訓練分類頭**
```python
feature_extractor = MAEFeatureExtractor(mae_model)
for param in feature_extractor.parameters():
    param.requires_grad = False  # 凍結

# 只訓練新的分類頭
classifier = nn.Linear(768, num_classes)
```

**選項 2: 微調整個模型**
```python
# 全部參數都可訓練
feature_extractor = MAEFeatureExtractor(mae_model)
classifier = nn.Linear(768, num_classes)

# 通常對 encoder 使用較小的學習率
optimizer = torch.optim.Adam([
    {'params': feature_extractor.parameters(), 'lr': 1e-5},
    {'params': classifier.parameters(), 'lr': 1e-3}
])
```

---

## 常見問題

### Q: 訓練多久可以看到效果？

A:
- **50 epochs**: 可以看到初步的重建能力
- **100 epochs**: 特徵質量已經不錯
- **200 epochs**: 接近最佳效果
- **300+ epochs**: 通常不會再有顯著提升

### Q: 如何判斷是否過擬合？

A: MAE 是自監督學習，不容易過擬合。如果擔心:
1. 保留一部分影像作為驗證集
2. 觀察驗證集的重建 loss
3. 查看 t-SNE，看是否學到有意義的聚類

### Q: Mask ratio 應該選多少？

A:
- **0.50**: 太簡單，學習不充分
- **0.60-0.70**: 適合小物體偵測 ⭐
- **0.75**: 標準 MAE，適合一般物體
- **0.80+**: 太難，可能損失細節

對於心臟瓣膜，建議 **0.65**。

### Q: 能不能用更大的 image size？

A: 可以，但要注意:
- Image size 256: 記憶體增加約 33%
- Image size 384: 記憶體增加約 2x
- Patch size 保持 16，會產生更多 patches

如果原始影像 > 512，可以考慮 image_size=384。

---

## 硬體建議

### 最低配置
- GPU: 8GB VRAM (GTX 1080 Ti, RTX 2070)
- Batch size: 32-64
- 訓練時間: 10-15 小時

### 推薦配置
- GPU: 12GB+ VRAM (RTX 3090, V100)
- Batch size: 128
- 訓練時間: 3-5 小時

### 理想配置
- GPU: A100 40GB/80GB
- Batch size: 256
- 訓練時間: 1-2 小時

---

## 下一步

1. ✅ 選擇適合的配置方案
2. ✅ 執行訓練腳本
3. ⏳ 監控訓練過程 (TensorBoard)
4. ⏳ 評估模型並提取特徵
5. ⏳ 用於瓣膜偵測任務

祝訓練順利！🚀

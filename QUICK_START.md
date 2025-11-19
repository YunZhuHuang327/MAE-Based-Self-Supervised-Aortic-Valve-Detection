# 快速開始指南

## 📊 您的資料集

- **50 位患者**，共 **14,076 張** 512×512 灰階醫學影像
- 平均每位患者 281 張影像
- 適合使用 MAE 進行自監督學習預訓練

---

## 🎯 為什麼選擇 MAE？

對於**心臟瓣膜偵測**這類需要識別微小結構的任務：

| 方法 | 優點 | 缺點 | 適合瓣膜偵測？ |
|-----|------|------|--------------|
| **MAE** | ✅ 保留細節<br>✅ 學習局部特徵<br>✅ 適合小物體 | 需要較長訓練時間 | ⭐ **強烈推薦** |
| SimCLR | 訓練快 | ❌ 激進增強<br>❌ 可能破壞細節 | ⚠️ 不推薦 |

---

## ⚡ 3 步驟開始訓練

### 步驟 1: 安裝依賴

```bash
cd /DATA1/yunzhu/SSL
pip install -r requirements.txt
```

### 步驟 2: 開始訓練（推薦配置）

```bash
./run_training_valve_optimized.sh
```

**這個配置**:
- ✅ 針對瓣膜偵測優化（mask_ratio=0.65）
- ✅ 保留更多局部細節
- ✅ 使用混合精度訓練（節省記憶體）
- ⏱️ 預計訓練時間: 3-5 小時 (12GB+ GPU)

### 步驟 3: 評估並提取特徵

```bash
./run_evaluation_valve.sh
```

**會產生**:
- `mae_features.npy` - 特徵向量 (14076 × 768)
- `embeddings_tsne.png` - 視覺化聚類效果
- `attention_maps/` - 模型關注的區域

---

## 💻 根據 GPU 選擇配置

### 我有 12GB+ GPU (RTX 3090, V100, A100)
```bash
# 使用推薦配置（已設定好）
./run_training_valve_optimized.sh
```

### 我只有 8GB GPU (RTX 2080, GTX 1080 Ti)
修改 `run_training_valve_optimized.sh`:
```bash
BATCH_SIZE=64    # 改成 64
LR=7.5e-5        # 對應調整學習率
```

### 我想快速實驗（1-2 小時）
修改 `run_training_valve_optimized.sh`:
```bash
EPOCHS=50        # 減少到 50 epochs
```

---

## 📈 訓練監控

### 啟動 TensorBoard

```bash
tensorboard --logdir mae_valve_output/logs
```

在瀏覽器打開 `http://localhost:6006`

### 觀察重點

1. **Loss 曲線** - 應該穩定下降
2. **重建視覺化** - 每 10 epochs 自動保存在 `mae_valve_output/visualizations/`
3. **學習率** - 確認 warmup 和 cosine 衰減正常

---

## 🔍 訓練完成後

### 1. 查看學到的特徵

```bash
python -c "
import numpy as np
features = np.load('mae_valve_evaluation/mae_features.npy')
print(f'特徵形狀: {features.shape}')
print(f'特徵範圍: [{features.min():.2f}, {features.max():.2f}]')
"
```

### 2. 視覺化聚類效果

```bash
# 查看 t-SNE 圖，觀察同一患者的影像是否聚在一起
open mae_valve_evaluation/embeddings_tsne.png
```

### 3. 用於瓣膜偵測

參考 [README.md](README.md) 的「用於下游任務」章節

---

## 🛠️ 常見問題

### Q: 訓練時出現 CUDA out of memory

**A**: 降低 batch size
```bash
# 在腳本中修改
BATCH_SIZE=64  # 或 32
LR=7.5e-5      # 記得同步調整 LR
```

### Q: 訓練很慢，每個 iteration 要 5 秒以上

**A**: 檢查:
1. 是否啟用了 `--use_amp`（混合精度）
2. `--num_workers` 是否設為 4-8
3. 資料是否在 SSD 上（HDD 會慢很多）

### Q: Loss 不下降或震盪很大

**A**:
1. 降低學習率（例如改成 `LR=3.75e-5`）
2. 確認 warmup 正常運作
3. 檢查資料載入是否正確

### Q: 我該用多少個 epochs？

**A**:
- **最少**: 100 epochs（看到初步效果）
- **推薦**: 200 epochs（接近最佳）
- **最多**: 300 epochs（通常不會再提升）

### Q: Mask ratio 該用多少？

**A**: 針對心臟瓣膜偵測：
- **0.65** ⭐ 推薦（保留更多細節）
- 0.70 可嘗試
- 0.75 標準 MAE（可能損失細節）
- 0.60 以下太簡單

---

## 📚 更多資訊

- **詳細文檔**: [README.md](README.md)
- **配置說明**: [configs.md](configs.md)
- **資料集分析**: 執行 `python analyze_dataset.py`

---

## 🚀 完整流程

```bash
# 1. 安裝
pip install -r requirements.txt

# 2. 訓練（3-5 小時）
./run_training_valve_optimized.sh

# 3. 監控（另一個終端）
tensorboard --logdir mae_valve_output/logs

# 4. 評估
./run_evaluation_valve.sh

# 5. 使用特徵進行瓣膜偵測
# （根據你的下游任務實作）
```

---

## 🎓 重要概念

### MAE 是如何運作的？

1. **遮蔽** 75% 的影像區域（使用 0.65 可保留更多細節）
2. **編碼** 可見的 25% 區域
3. **解碼** 並重建被遮蔽的區域
4. **學習** 影像的局部和全局表示

### 為什麼 mask_ratio=0.65 更適合瓣膜偵測？

```
Mask Ratio 0.75: ████████████████████████████░░░░░░░  (保留 25%)
Mask Ratio 0.65: ████████████████████████████░░░░░░░░░░ (保留 35%)
                                            ↑
                                    多保留 10% 的細節
```

對於微小的心臟瓣膜結構，這 10% 的差異可能很關鍵！

---

祝訓練順利！💪

如有問題，請檢查:
1. [README.md](README.md) - 完整文檔
2. [configs.md](configs.md) - 配置細節
3. Issue tracker（如果設有）

# GPU 記憶體不足問題解決方案

## 問題
```
torch.OutOfMemoryError: CUDA out of memory
GPU 0 has a total capacity of 10.57 GiB
```

## 原因
您的 `args.yaml` 設定 `batch_size=64`，對於 ~11GB GPU 太大了。

## 解決方案

### 方案 1: 使用小 GPU 配置腳本（推薦）

```bash
./train_valve_small_gpu.sh
```

這會自動：
- 降低 batch size 到 16
- 降低 workers 到 4
- 保留所有其他參數

### 方案 2: 手動設定環境變數

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./run_yolo_training_only.sh
```

### 方案 3: 永久修改 args.yaml

如果您想永久改變設定：

```yaml
# 編輯 /DATA1/yunzhu/SSL/args.yaml
batch: 16  # 從 64 改成 16
workers: 4  # 從 8 改成 4
```

## Batch Size 建議

根據 GPU 記憶體大小：

| GPU VRAM | 推薦 Batch Size | 預期速度 |
|----------|----------------|---------|
| 6-8 GB   | 8-12          | 較慢     |
| 10-12 GB | 16-24         | 中等     |
| 16-24 GB | 32-48         | 較快     |
| 24+ GB   | 64+           | 最快     |

## 訓練時間影響

降低 batch size 會：
- ✅ 減少記憶體使用
- ⚠️ 增加訓練時間（約 3-4 倍）
- ⚠️ 可能略微影響收斂

原始設定（batch=64）：1-2 小時
小 GPU 設定（batch=16）：3-6 小時

## 監控 GPU 使用

訓練時監控 GPU：
```bash
watch -n 1 nvidia-smi
```

合理的使用率：
- GPU 使用率：80-95%
- 記憶體使用：8-10 GB（留 1-2 GB buffer）


import os
import numpy as np
from pathlib import Path

# 分析資料集
root_dir = Path('/DATA1/yunzhu/SSL/training_image/training_image')

patient_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith('patient')])

print("=" * 80)
print("資料集分析報告")
print("=" * 80)

counts = []
patient_info = []

for patient_dir in patient_dirs:
    imgs = list(patient_dir.glob('*.png'))
    counts.append(len(imgs))
    patient_info.append((patient_dir.name, len(imgs)))

counts = np.array(counts)

print(f"\n【基本資訊】")
print(f"  患者數量: {len(patient_dirs)}")
print(f"  總影像數: {counts.sum()}")
print(f"  每位患者平均影像數: {counts.mean():.1f}")
print(f"  每位患者中位數影像數: {np.median(counts):.0f}")
print(f"  最少影像數: {counts.min()}")
print(f"  最多影像數: {counts.max()}")
print(f"  標準差: {counts.std():.1f}")

print(f"\n【影像數量分佈】")
unique, counts_per = np.unique(counts, return_counts=True)
for num_imgs, num_patients in zip(unique, counts_per):
    print(f"  {num_imgs} 張影像: {num_patients} 位患者")

print(f"\n【前 10 位患者】")
for name, count in patient_info[:10]:
    print(f"  {name}: {count} 張")

print(f"\n【後 10 位患者】")
for name, count in patient_info[-10:]:
    print(f"  {name}: {count} 張")

# 計算訓練參數建議
total_images = counts.sum()
num_patients = len(patient_dirs)

print("\n" + "=" * 80)
print("訓練參數建議分析")
print("=" * 80)

# 計算不同 batch size 下的 iterations per epoch
print(f"\n【每個 epoch 的迭代次數】")
for bs in [32, 64, 128, 256]:
    iters = total_images // bs
    print(f"  Batch size {bs:3d}: {iters:4d} iterations/epoch ({iters * bs} 張影像)")

# 記憶體需求估算
print(f"\n【GPU 記憶體需求估算 (image_size=224)】")
print(f"  Batch size  32: ~3-4 GB")
print(f"  Batch size  64: ~6-8 GB")
print(f"  Batch size 128: ~10-12 GB")
print(f"  Batch size 256: ~20-24 GB")
print(f"  註: 使用 --use_amp 可減少約 40% 記憶體")

# 學習率建議
print(f"\n【學習率建議 (基於 batch size)】")
base_lr = 1.5e-4
base_batch = 256
for bs in [32, 64, 128, 256]:
    effective_lr = base_lr * (bs / base_batch)
    print(f"  Batch size {bs:3d}: {effective_lr:.2e} (base_lr × {bs}/{base_batch})")

# 訓練時間估算
print(f"\n【訓練時間估算】")
print(f"  假設每個 iteration 0.5 秒 (混合精度, V100/A100):")
for bs in [64, 128, 256]:
    iters = total_images // bs
    time_per_epoch = iters * 0.5 / 60  # 分鐘
    time_200_epochs = time_per_epoch * 200 / 60  # 小時
    print(f"    Batch size {bs:3d}: ~{time_per_epoch:.1f} 分鐘/epoch, 200 epochs = ~{time_200_epochs:.1f} 小時")

print("\n" + "=" * 80)
print("參數選擇理由")
print("=" * 80)

print(f"""
【為什麼選擇這些參數？】

1. **Batch Size = 128**
   - 資料集有 {total_images} 張影像，每個 epoch 有 {total_images // 128} iterations
   - 不會太大（避免記憶體不足）也不會太小（保證梯度穩定）
   - 如果你的 GPU 記憶體 < 10GB，建議改成 64

2. **Epochs = 200**
   - MAE 需要較長時間收斂（比對比學習慢）
   - 對於 {num_patients} 位患者的資料，200 epochs 足以學到良好特徵
   - 建議至少跑 100 epochs，最多 300 epochs

3. **Learning Rate = 1.5e-4**
   - 這是 MAE 論文的基準學習率（針對 batch_size=256）
   - 實際 LR = base_lr × (batch_size / 256)
   - Batch size 128 時，實際 LR ≈ 7.5e-5

4. **Warmup = 10 epochs**
   - 前 10 個 epochs 線性增加學習率
   - 避免一開始學習率太高導致訓練不穩定
   - 對於 Transformer 架構很重要

5. **Mask Ratio = 0.75**
   - MAE 的標準設定，遮蔽 75% 的 patches
   - 對於心臟瓣膜這種需要細節的任務，可以考慮降到 0.60-0.70
   - 但不要低於 0.50（任務會太簡單）

6. **Patch Size = 16**
   - 對於 224×224 影像，產生 14×14 = 196 個 patches
   - Patch 太小（8）會增加計算量
   - Patch 太大（32）會損失細節

7. **Image Size = 224**
   - Vision Transformer 的標準輸入大小
   - 可以使用 ImageNet 預訓練權重初始化
   - 如果原始影像很大（>512），可以考慮 256 或 384

8. **Preserve Details = True**
   - 使用較溫和的資料增強
   - 不使用激進的裁剪、旋轉、色彩變化
   - **對心臟瓣膜偵測非常重要！**

【建議的配置方案】

方案 A - 標準配置（推薦）:
  - Batch size: 128, Epochs: 200, LR: 1.5e-4
  - 適合: 有 12GB+ GPU (如 V100, RTX 3090, A100)
  - 訓練時間: 約 15-20 小時

方案 B - 小 GPU 配置:
  - Batch size: 64, Epochs: 200, LR: 7.5e-5
  - 適合: 8GB GPU (如 RTX 2080, GTX 1080 Ti)
  - 訓練時間: 約 20-30 小時

方案 C - 快速實驗:
  - Batch size: 128, Epochs: 100, LR: 1.5e-4
  - 適合: 快速驗證效果
  - 訓練時間: 約 7-10 小時

方案 D - 保留更多細節（針對瓣膜偵測）:
  - Batch size: 128, Epochs: 200, LR: 1.5e-4, **Mask ratio: 0.65**
  - 遮蔽較少的 patches，保留更多局部資訊
  - 可能對小物體偵測更好
""")

print("=" * 80)

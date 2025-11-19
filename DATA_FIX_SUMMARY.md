# 資料修復總結

## 問題描述

原始的資料集存在嚴重的標籤-圖片不匹配問題：
- **舊資料**：`/DATA1/yunzhu/SSL/training_image/training_image/` 和 `/DATA1/yunzhu/SSL/training_label/training_label/`
- **匹配率**：0% - 所有標籤檔案對應的圖片都不存在

### 具體問題（以 patient0031 為例）

**舊資料問題：**
- 圖片編號範圍：1-121, 230-249（缺少 122-172）
- 標籤編號範圍：122-172（共 51 個）
- **結果：0/51 匹配（0%）**

原因：
1. 圖片檔案中缺少編號 122-172 的圖片
2. 但標籤檔案恰好是針對這些缺失圖片的
3. 導致訓練時 YOLO 識別所有圖片為 "backgrounds"（無標註）

## 解決方案

上傳並使用完整的資料集：
- **新圖片**：`42_training_image (2).zip` → `/DATA1/yunzhu/SSL/training_image_new/training_image/`
- **新標籤**：`42_training_label (2).zip` → `/DATA1/yunzhu/SSL/training_label_new/training_label/`

### 新資料驗證結果

**Patient 0031（範例）：**
- 圖片總數：完整資料集
- 標籤總數：51 個
- **匹配：51/51 (100%)**

**所有患者統計：**
- 總標籤數：2,787 個
- 匹配數：2,787 個
- **匹配率：100%**

## YOLO 資料集配置

使用 `prepare_yolo_data_filtered.py` 創建僅包含有標註圖片的資料集：

### 資料分割
- **訓練集**：Patient 1-30 + 41-50 的所有有標註圖片
  - 樣本數：2,250
  - 所有樣本都有心臟瓣膜標註

- **驗證集**：Patient 31-40 的所有有標註圖片
  - 樣本數：537
  - 所有樣本都有心臟瓣膜標註

### 最終資料集統計

```
Train: 2250 images, 0 backgrounds ✅
Val: 537 images, 0 backgrounds ✅
```

**100% 的圖片都有有效標註！**

## 訓練配置

### 使用的腳本
- **準備資料**：`prepare_yolo_data_filtered.py`
- **訓練**：`train_valve_small_gpu.sh`
  - Batch size: 16（小 GPU 優化）
  - Workers: 4
  - Model: YOLOv12s
  - Epochs: 20

### 資料路徑
```yaml
Source images: /DATA1/yunzhu/SSL/training_image_new/training_image
Source labels: /DATA1/yunzhu/SSL/training_label_new/training_label
Output: /DATA1/yunzhu/SSL/yolo_dataset
Dataset YAML: /DATA1/yunzhu/SSL/yolo_dataset/valve_detection.yaml
```

### 訓練輸出
```
Results: /DATA1/yunzhu/SSL/valve_training_results/train/
Models: /DATA1/yunzhu/SSL/valve_training_results/train/weights/
  - best.pt
  - last.pt
```

## 驗證步驟

執行以下命令驗證新資料集：

```bash
# 檢查匹配率
python -c "
from pathlib import Path

img_base = Path('/DATA1/yunzhu/SSL/training_image_new/training_image')
lbl_base = Path('/DATA1/yunzhu/SSL/training_label_new/training_label')

for patient_id in [1, 2, 31, 35, 40]:
    img_dir = img_base / f'patient{patient_id:04d}'
    lbl_dir = lbl_base / f'patient{patient_id:04d}'

    if img_dir.exists() and lbl_dir.exists():
        img_nums = set([int(f.stem.split('_')[-1]) for f in img_dir.glob('*.png')])
        lbl_nums = set([int(f.stem.split('_')[-1]) for f in lbl_dir.glob('*.txt')])

        matching = len(img_nums & lbl_nums)
        total_lbls = len(lbl_nums)

        print(f'Patient {patient_id:04d}: {matching}/{total_lbls} match ({matching/total_lbls*100:.1f}%)')
"

# 驗證 YOLO 資料集
python prepare_yolo_data_filtered.py \
    --source_images /DATA1/yunzhu/SSL/training_image_new/training_image \
    --source_labels /DATA1/yunzhu/SSL/training_label_new/training_label \
    --output_dir /DATA1/yunzhu/SSL/yolo_dataset \
    --validate
```

## 重要提醒

### 使用正確的資料來源
❌ **不要使用**：
- `/DATA1/yunzhu/SSL/training_image/training_image/`
- `/DATA1/yunzhu/SSL/training_label/training_label/`

✅ **請使用**：
- `/DATA1/yunzhu/SSL/training_image_new/training_image/`
- `/DATA1/yunzhu/SSL/training_label_new/training_label/`

### MAE 訓練
如果需要重新訓練 MAE，請使用新的完整圖片集：
```bash
# 更新 MAE 訓練腳本中的路徑
python mae_train.py \
    --image_dir /DATA1/yunzhu/SSL/training_image_new/training_image \
    --output_dir ./mae_valve_output_new
```

## 問題排查

如果訓練時看到：
```
train: Scanning... 0 images, 11365 backgrounds
```

說明圖片和標籤不匹配，請：
1. 確認使用的是 `training_image_new` 和 `training_label_new`
2. 重新運行 `prepare_yolo_data_filtered.py`
3. 清除緩存：`rm -f /DATA1/yunzhu/SSL/yolo_dataset/*/labels.cache`

## 成功指標

正確的訓練啟動應該顯示：
```
train: Scanning... 2250 images, 0 backgrounds ✅
val: Scanning... 537 images, 0 backgrounds ✅
```

---

**修復完成日期：2024-11-04**
**資料來源：42_training_image (2).zip, 42_training_label (2).zip**

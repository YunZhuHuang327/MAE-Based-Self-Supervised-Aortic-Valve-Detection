# 心臟瓣膜偵測 - 預測指南

## 📋 預測結果格式

每一行為一個預測框，格式如下：

```
圖片名稱(不包含.png) 類別 信心分數 左上x 左上y 右下x 右下y
```

### 範例：
```
patient0051_9998 0 0.7777 100 100 200 200
patient0051_9999 0 0.8888 100 100 200 200
patient0075_9998 0 0.9999 100 100 200 200
```

### 欄位說明：
- **圖片名稱**：不含副檔名的圖片名稱
- **類別**：0 = aortic_valve (主動脈瓣膜)
- **信心分數**：0-1 之間的浮點數，表示模型的信心度
- **左上 x, y**：預測框左上角的 x, y 座標（整數，原始圖片尺寸）
- **右下 x, y**：預測框右下角的 x, y 座標（整數，原始圖片尺寸）

## 🚀 執行預測

### 方法 1：使用腳本（推薦）

```bash
./run_prediction.sh
```

### 方法 2：使用 Python 直接執行

```bash
python predict_valve_detection.py \
    --model /DATA1/yunzhu/SSL/valve_training_results/train/weights/best.pt \
    --test_dir /DATA1/yunzhu/SSL/test \
    --output /DATA1/yunzhu/SSL/predictions.csv \
    --conf 0.25
```

### 參數說明：

- `--model`: 訓練好的模型路徑（best.pt）
- `--test_dir`: 測試圖片目錄
- `--output`: 輸出 CSV 檔案路徑
- `--conf`: 信心度閾值（預設 0.25，範圍 0.0-1.0）

### 調整信心度閾值

如果想要更嚴格的預測（減少誤報）：
```bash
python predict_valve_detection.py --conf 0.5  # 只輸出信心度 > 0.5 的預測
```

如果想要更寬鬆的預測（增加召回率）：
```bash
python predict_valve_detection.py --conf 0.1  # 輸出信心度 > 0.1 的預測
```

## 📊 輸出檔案

### 預測結果 CSV
- **位置**：`/DATA1/yunzhu/SSL/predictions.csv`
- **格式**：空格分隔（參考上方格式說明）
- **內容**：所有信心度超過閾值的預測框

### 統計資訊

執行完成後會顯示：
- 處理的圖片總數
- 有偵測結果的圖片數量
- 總偵測框數量
- 平均信心分數

範例輸出：
```
📊 Statistics:
   Images with detections: 3250/16620
   Total detections: 4123
   Average confidence: 0.8542
```

## 🔍 查看結果

### 查看前 10 筆預測

```bash
head -10 /DATA1/yunzhu/SSL/predictions.csv
```

### 統計預測數量

```bash
# 總預測框數量
wc -l /DATA1/yunzhu/SSL/predictions.csv

# 各信心度區間的分布
awk '{print $3}' /DATA1/yunzhu/SSL/predictions.csv | sort -n | uniq -c
```

### 找出高信心度的預測

```bash
# 信心度 > 0.9 的預測
awk '$3 > 0.9' /DATA1/yunzhu/SSL/predictions.csv | head -20
```

## 📁 測試資料結構

程式會自動掃描測試目錄下的所有圖片（包含子目錄）：

```
/DATA1/yunzhu/SSL/test/
├── images1/
│   ├── patient_xxx.png
│   └── ...
├── images2/
│   ├── patient_yyy.png
│   └── ...
└── tmp/
    └── testing_image/
        └── ...
```

支援的圖片格式：`.png`, `.jpg`, `.jpeg`

## 🎯 模型資訊

### 訓練結果
- **mAP50**: 96.87%
- **mAP50-95**: 68.63%
- **Precision**: 92.15%
- **Recall**: 93.86%
- **訓練集**: 2,250 張有標註圖片
- **驗證集**: 537 張有標註圖片（patient 31-40）

### 模型路徑
```
/DATA1/yunzhu/SSL/valve_training_results/train/weights/
├── best.pt    # 最佳模型（依據 mAP50）
└── last.pt    # 最後一個 epoch 的模型
```

## 💡 使用建議

### 1. 選擇合適的信心度閾值

- **0.25**（預設）：平衡準確率和召回率
- **0.5-0.7**：提高精確度，減少誤報
- **0.1-0.2**：提高召回率，找出更多可能的目標

### 2. 批次處理大量圖片

如果圖片數量很大，預測可能需要較長時間：
- 16,620 張圖片約需 5-10 分鐘（取決於 GPU）
- 可以在背景執行：`nohup ./run_prediction.sh > prediction.log 2>&1 &`

### 3. 結果後處理

根據需求過濾結果：
```bash
# 只保留高信心度預測（> 0.8）
awk '$3 > 0.8' predictions.csv > predictions_high_conf.csv

# 按信心度排序
sort -k3 -rn predictions.csv > predictions_sorted.csv

# 統計每張圖片的偵測框數量
awk '{print $1}' predictions.csv | sort | uniq -c
```

## ❓ 常見問題

### Q: 為什麼有些圖片沒有預測結果？

A: 可能的原因：
1. 圖片中確實沒有瓣膜
2. 信心度低於閾值
3. 瓣膜太小或模糊

解決方法：降低 `--conf` 參數

### Q: 如何視覺化預測結果？

A: 可以使用以下程式碼：
```python
from PIL import Image, ImageDraw

img = Image.open('test_image.png')
draw = ImageDraw.Draw(img)

# 繪製預測框（x1, y1, x2, y2）
draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

img.save('result.png')
```

### Q: CSV 檔案太大怎麼辦？

A: 可以：
1. 提高信心度閾值減少預測數量
2. 只保留每張圖片信心度最高的預測
3. 壓縮 CSV：`gzip predictions.csv`

## 📝 相關檔案

- **預測腳本**：`predict_valve_detection.py`
- **執行腳本**：`run_prediction.sh`
- **訓練模型**：`valve_training_results/train/weights/best.pt`
- **資料修復說明**：`DATA_FIX_SUMMARY.md`

---

**更新日期**：2024-11-04

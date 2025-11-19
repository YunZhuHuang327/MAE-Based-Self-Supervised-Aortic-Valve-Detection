"""
使用訓練好的 YOLO 模型進行心臟瓣膜偵測並生成預測結果 CSV

輸出格式：
圖片名稱(不包含.png) -空格- 類別 -空格- 信心分數(0~1之間的小數) -空格-
預測框左上角原始 x 座標(整數) -空格- 預測框左上角原始 y 座標(整數) -空格-
預測框右下角原始 x 座標(整數) -空格- 預測框右下角原始 y 座標(整數)
"""

import os
import sys
from pathlib import Path
import csv
from ultralytics import YOLO
from tqdm import tqdm


def predict_and_save_csv(
    model_path,
    test_dir,
    output_csv,
    conf_threshold=0.25,
    iou_threshold=0.45
):
    """
    使用 YOLO 模型進行預測並保存為 CSV 格式

    Args:
        model_path: 訓練好的模型路徑 (best.pt)
        test_dir: 測試圖片目錄
        output_csv: 輸出 CSV 檔案路徑
        conf_threshold: 信心度閾值
        iou_threshold: IOU 閾值
    """
    print("=" * 80)
    print("Heart Valve Detection - Inference")
    print("=" * 80)

    # 載入模型
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)

    # 找到所有測試圖片
    test_path = Path(test_dir)
    print(f"\n🔍 Searching for images in: {test_path}")

    # 支援多種圖片格式
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(list(test_path.rglob(ext)))

    image_files = sorted(image_files)
    print(f"   Found {len(image_files)} images")

    if len(image_files) == 0:
        print("❌ No images found!")
        return

    # 進行預測
    print(f"\n🚀 Running inference...")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   IOU threshold: {iou_threshold}")

    # 準備輸出結果
    results_list = []

    # 對每張圖片進行預測
    for img_path in tqdm(image_files, desc="Predicting"):
        # 執行預測
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # 取得圖片名稱（不含副檔名）
        img_name = img_path.stem

        # 處理預測結果
        if len(results) > 0:
            result = results[0]

            # 檢查是否有檢測到物體
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                # 遍歷所有檢測框
                for i in range(len(boxes)):
                    # 取得邊界框座標 (xyxy format: x1, y1, x2, y2)
                    box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]

                    # 取得信心分數
                    conf = float(boxes.conf[i].cpu().numpy())

                    # 取得類別 (0 = aortic_valve)
                    cls = int(boxes.cls[i].cpu().numpy())

                    # 轉換座標為整數
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                    # 添加到結果列表
                    # 格式: 圖片名稱 類別 信心分數 左上x 左上y 右下x 右下y
                    results_list.append([
                        img_name,
                        cls,
                        conf,
                        x1,
                        y1,
                        x2,
                        y2
                    ])
            else:
                # 如果沒有檢測到任何物體，也可以選擇記錄
                # 這裡我們只記錄有檢測結果的
                pass

    # 保存為 CSV
    print(f"\n💾 Saving results to: {output_csv}")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')

        # 寫入所有預測結果
        for row in results_list:
            writer.writerow(row)

    print(f"✅ Saved {len(results_list)} predictions")

    # 統計資訊
    if results_list:
        # 按圖片分組統計
        images_with_detections = len(set([r[0] for r in results_list]))
        total_detections = len(results_list)
        avg_conf = sum([r[2] for r in results_list]) / len(results_list)

        print(f"\n📊 Statistics:")
        print(f"   Images with detections: {images_with_detections}/{len(image_files)}")
        print(f"   Total detections: {total_detections}")
        print(f"   Average confidence: {avg_conf:.4f}")
    else:
        print(f"\n⚠️  No detections found (confidence threshold might be too high)")

    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)

    return results_list


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Heart valve detection inference with YOLO'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='/DATA1/yunzhu/SSL/valve_training_results/train/weights/best.pt',
        help='Path to trained YOLO model'
    )

    parser.add_argument(
        '--test_dir',
        type=str,
        default='/DATA1/yunzhu/SSL/test',
        help='Directory containing test images'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='/DATA1/yunzhu/SSL/predictions.csv',
        help='Output CSV file path'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IOU threshold for NMS (default: 0.45)'
    )

    args = parser.parse_args()

    # 檢查模型是否存在
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print(f"   Please check the model path")
        sys.exit(1)

    # 檢查測試目錄是否存在
    if not os.path.exists(args.test_dir):
        print(f"❌ Test directory not found: {args.test_dir}")
        sys.exit(1)

    # 執行預測
    predict_and_save_csv(
        model_path=args.model,
        test_dir=args.test_dir,
        output_csv=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )


if __name__ == '__main__':
    main()

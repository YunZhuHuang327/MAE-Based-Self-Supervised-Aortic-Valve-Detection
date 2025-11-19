from postprocess.false_positive import summarize_contiguous_slices, detect_outlier_groups_from_summary, get_confidences_for_outliers, remove_outlier_entries, copy_file_safe

merged_file = "/DATA1/meowbase/aicup-aortic-valve-object-detection/submission/download/640_batch64_epoch200_20251018_1940_merged.txt"

copy_path = copy_file_safe(merged_file)

for round_idx in range(5):  # 最多跑5輪避免無窮迴圈
    print(f"\n=== 第 {round_idx+1} 輪離群偵測 ===")
    summaries = summarize_contiguous_slices(copy_path)
    outlier_summary = detect_outlier_groups_from_summary(summaries, gap_threshold=15, size_threshold=3)

    if not outlier_summary:
        print("✅ 沒有新的離群群組，結束清理。")
        break

    get_confidences_for_outliers(copy_path, outlier_summary)
    remove_outlier_entries(copy_path, outlier_summary)
else:
    print("⚠️ 達到最大迭代次數，可能仍有未清理完的離群值。")

print(f"清理完成，新檔案：{copy_path}")

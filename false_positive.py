from collections import defaultdict
import shutil
import os

def summarize_contiguous_slices(file_path):
    """
    讀取 merged.txt，回傳每個病人的連續區段。
    格式：
      { "0051": "50-80, 90-120, 156", "0052": "142-143, 149-202", ... }
    並同時印出摘要。
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    per_patient = defaultdict(list)
    for line in lines:
        parts = line.split()
        pid, sid = parts[0].split('_')
        sid = int(sid)
        per_patient[pid].append(sid)

    summaries = {}
    for pid, slices in sorted(per_patient.items()):
        slices = sorted(set(slices))
        if not slices:
            continue

        ranges = []
        start = slices[0]
        prev = slices[0]

        for s in slices[1:]:
            # if s == prev + 1:
            if s - prev <= 2:
                prev = s
            else:
                if start == prev:
                    ranges.append(f"{start}")
                else:
                    ranges.append(f"{start}-{prev}")
                start = s
                prev = s
        if start == prev:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{prev}")

        joined = ", ".join(ranges)
        summaries[pid] = joined
        print(f"Patient_{pid}: {joined}")

    return summaries


def detect_outlier_groups_from_summary(summaries, gap_threshold=30, size_threshold=3):
    """
    偵測病人內部的離群群組：
    若群與前一群距離 > gap_threshold，且前或後群長度 ≤ size_threshold，
    則視為離群群組。
    會回傳 dict，例如：
        {
          "0096": ["223"],
          "0054": ["1-2", "94"]
        }
    """
    outlier_summary = {}

    print("\n=== 離群群組偵測結果 ===")
    for pid, rest in sorted(summaries.items()):
        parts = [p.strip() for p in rest.split(',') if p.strip()]

        groups = []
        for p in parts:
            if '-' in p:
                s, e = map(int, p.split('-'))
            else:
                s = e = int(p)
            groups.append((s, e, e - s + 1))

        outliers = set()
        for i in range(1, len(groups)):
            prev_start, prev_end, prev_len = groups[i - 1]
            cur_start, cur_end, cur_len = groups[i]
            gap = cur_start - prev_end

            if gap > gap_threshold:
                if prev_len <= size_threshold:
                    label = f"{prev_start}" if prev_start == prev_end else f"{prev_start}-{prev_end}"
                    outliers.add(label)
                if cur_len <= size_threshold:
                    label = f"{cur_start}" if cur_start == cur_end else f"{cur_start}-{cur_end}"
                    outliers.add(label)

        if outliers:
            outlier_summary[pid] = sorted(outliers)
            print(f"Patient_{pid} 離群群組: {', '.join(sorted(outliers))}")

    print("\n✅ 離群檢測完成。")
    return outlier_summary


def get_confidences_for_outliers(merged_file, outlier_summary):
    """
    merged_file: 原始 YOLO 結果檔（merged.txt）
    outlier_summary: 來自 detect_outlier_groups_from_summary() 的字典或 list
                     格式如 {"0053": ["235"], "0054": ["1-2", "94"], ...}
    會印出離群群組中每張切片的信心分數。
    """
    # 讀原始結果
    with open(merged_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 建索引 { patient_id : { slice : conf } }
    from collections import defaultdict
    conf_map = defaultdict(dict)
    for line in lines:
        parts = line.split()
        pid, sid = parts[0].split('_')
        sid = int(sid)
        conf = float(parts[2])
        conf_map[pid][sid] = conf

    # 顯示離群群組的分數
    print("\n=== 離群群組信心分數 ===")
    for pid, group_list in outlier_summary.items():
        if pid not in conf_map:
            continue
        print(f"\nPatient_{pid}:")
        for group in group_list:
            if '-' in group:
                s, e = map(int, group.split('-'))
                slices = range(s, e + 1)
            else:
                slices = [int(group)]
            confs = [conf_map[pid].get(s, None) for s in slices if s in conf_map[pid]]
            confs_str = ", ".join(f"{s}:{c:.4f}" for s, c in zip(slices, confs) if c is not None)
            print(f"  {group} → {confs_str}")


def copy_file_safe(src_path, dst_path=None):
    """
    複製檔案以保護原始資料。
    若未指定 dst_path，會自動加上 '_copy' 後綴。
    例如 merged.txt -> merged_copy.txt
    """
    if dst_path is None:
        root, ext = os.path.splitext(src_path)
        dst_path = f"{root}_copy{ext}"
    shutil.copy(src_path, dst_path)
    print(f"✅ 已建立安全副本: {dst_path}")
    return dst_path

def remove_outlier_entries(clean_file, outlier_summary):
    """
    clean_file: 要修改的副本檔案路徑
    outlier_summary: 離群群組的 dict，如 {"0096": ["223"], "0054": ["1-2", "94"], ...}
    會直接刪除這些切片行。
    """
    # 讀取所有行
    with open(clean_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 決定要刪除的 (pid, sid)
    to_delete = set()
    for pid, group_list in outlier_summary.items():
        for group in group_list:
            if '-' in group:
                s, e = map(int, group.split('-'))
                for i in range(s, e + 1):
                    to_delete.add((pid, i))
            else:
                to_delete.add((pid, int(group)))

    # 過濾
    new_lines = []
    for line in lines:
        parts = line.split()
        pid, sid = parts[0].split('_')
        sid = int(sid)
        if (pid, sid) not in to_delete:
            new_lines.append(line)

    # 寫回
    with open(clean_file, 'w') as f:
        for l in new_lines:
            f.write(l + "\n")

    print(f"🧹 已從 {clean_file} 移除 {len(lines) - len(new_lines)} 行離群切片。")

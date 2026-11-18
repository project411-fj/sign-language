#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_video_fps.py
讀取 videos/keypoint_videos/<part>/<子資料夾>/*_upper_body.mp4 的 fps，寫成 .csv (filename,fps)。

用法
----
1. 修改下面「路徑設定」區。
2. python get_video_fps.py

輸出範例 (filename 是相對於該 part 的路徑)
--------
filename,fps
1/000600_001507_upper_body.mp4,30.0
1/004045_004735_upper_body.mp4,30.0
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import cv2

# ================================================================ 路徑設定
VIDEO_DIR = r"Data Preparation\videos\keypoint_videos"                     # 放裁切後影片的目錄
FPS_CSV = r"Data Preparation\Phase5-Pause\output\01fps"             # 要寫入的 .csv (跟 find_sign_pauses.py 的 FPS_CSV 一樣)


def get_fps(path: Path) -> float | None:
    """回傳影片的 fps；打不開或讀不到就回傳 None。"""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not fps or not math.isfinite(fps) or fps <= 0:
        return None
    return round(float(fps), 3)


def video_sort_key(path: Path):
    # <子資料夾>/<start>_<end>_upper_body.mp4
    parts = path.stem.split("_")
    return int(path.parent.name), int(parts[0]), int(parts[1])


def main() -> int:
    video_dir, csv_path = Path(VIDEO_DIR), Path(FPS_CSV)
    if not video_dir.is_dir():
        print(f"[錯誤] VIDEO_DIR 不存在：{video_dir}")
        return 1

    videos = sorted(video_dir.glob("*/*_upper_body.mp4"), key=video_sort_key)
    if not videos:
        print(f"{video_dir} 裡沒有 *_upper_body.mp4")
        return 1

    rows, failed = [], []
    for v in videos:
        name = v.relative_to(video_dir).as_posix()   # 例如 1/000600_001507_upper_body.mp4
        fps = get_fps(v)
        if fps is None:
            failed.append(name)
            print(f"[無法讀取] {name}")
            continue
        rows.append((name, fps))
        print(f"{name}: {fps:g}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig：用 Excel 開啟中文檔名才不會變亂碼
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(("filename", "fps"))
        w.writerows(rows)

    print(f"\n寫入 {len(rows)} 筆 -> {csv_path}" + (f"，{len(failed)} 個影片無法讀取" if failed else ""))
    return 0


if __name__ == "__main__":
    num = input("part: ")
    VIDEO_DIR = fr"{VIDEO_DIR}\{num}"
    FPS_CSV = fr"{FPS_CSV}\fps_{num}.csv"
    sys.exit(main())
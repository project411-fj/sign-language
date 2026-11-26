"""STT 版第 1 步：讀每個片段的 fps (和 OCR/01get_fps.py 的輸出一樣)。

輸入：videos/keypoint_videos/<part>/<n>/<start>_<end>_upper_body.mp4   (Phase4 01cropV2.py 的輸出)
輸出：output/01fps/fps_<part>.csv (欄位 filename, fps)；02find_sign_pauses.py 用它把語音的秒數換成格數。

OCR 和 STT 的結果都放在 output/ (一個 part 只會走其中一條)。這支只處理 STT 路徑的 part
(part_videos/<part>/<n>/ 底下有範圍資料夾的 part)，輸入 all 也不會動到 OCR 的 part。
執行後輸入 part：「13」、「13,14」或「all」。
"""
import csv
import math
import re
from pathlib import Path

import cv2
from tqdm import tqdm

# ================================================================ 路徑設定 (後面會再接 /<part>)
VIDEO_DIR = Path("Data Preparation/videos/keypoint_videos")
FPS_DIR = Path("Data Preparation/Phase5-Pause/output/01fps")          # 裡面是 fps_<part>.csv
PART_VIDEOS_DIR = Path("Data Preparation/videos/part_videos")     # 用來判斷是不是 STT 路徑的 part
BOX_RE = re.compile(r"\d+(\.\d+)?(_\d+(\.\d+)?){3}")   # STT 路徑的範圍資料夾，例如 0.20_0.73_0.10_0.33


def get_fps(path):
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


def video_sort_key(path):
    # <子資料夾>/<start>_<end>_upper_body.mp4
    parts = path.stem.split("_")
    return int(path.parent.name), int(parts[0]), int(parts[1])


def is_stt_part(part):
    """STT 路徑的 part：part_videos/<part>/<n>/ 底下有 <TOP>_<BOTTOM>_<LEFT>_<RIGHT>/ 範圍資料夾。
    OCR 和 STT 的結果都放在 output/，所以 OCR 的 part 不能跑 STT 的程式，否則會用語音的時間蓋掉 OCR 的切點。"""
    return any(d.is_dir() and BOX_RE.fullmatch(d.name) for d in (PART_VIDEOS_DIR / part).glob("*/*"))


def run_part(part):
    if not is_stt_part(part):
        print(f"part {part}：part_videos 裡沒有範圍資料夾，不是 STT 路徑的 part (OCR 的 part 請用 OCR/ 的程式)，跳過")
        return
    video_dir = VIDEO_DIR / part
    if not video_dir.is_dir():
        print(f"part {part}：找不到 {video_dir}，請先跑 Phase4 的 STT/01cropV2.py，跳過")
        return

    videos = sorted(video_dir.glob("*/*_upper_body.mp4"), key=video_sort_key)
    rows, no_fps = [], []
    for video in tqdm(videos, desc=f"part {part}", unit="video"):
        name = video.relative_to(video_dir).as_posix()            # 例如 1/000600_001507_upper_body.mp4
        fps = get_fps(video)
        if fps is None:
            no_fps.append(name)
            continue
        rows.append((name, fps))

    fps_csv = FPS_DIR / f"fps_{part}.csv"
    fps_csv.parent.mkdir(parents=True, exist_ok=True)
    with fps_csv.open("w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig：Excel 開啟中文才不會亂碼
        writer = csv.writer(f)
        writer.writerow(("filename", "fps"))
        writer.writerows(rows)

    print(f"part {part}：寫入 {len(rows)} 筆 → {fps_csv}" + (f"，{len(no_fps)} 支影片讀不到 fps：" if no_fps else ""))
    for name in no_fps:
        print(f"    {name}")


def parse_parts(answer, root):
    """「3」→ ["3"]；「1,2,5」→ ["1", "2", "5"]；「all」→ root 底下所有 part (依數字排序)。"""
    answer = answer.strip()
    if answer.lower() == "all":
        return sorted((p.name for p in root.iterdir() if p.is_dir()), key=lambda x: (len(x), x))
    return [p.strip() for p in answer.split(",") if p.strip()]


def main():
    answer = input("part (例如 13、13,14 或 all): ")
    parts = parse_parts(answer, VIDEO_DIR)
    if answer.strip().lower() == "all":
        parts = [part for part in parts if is_stt_part(part)]   # all 只取 STT 路徑的 part
    if not parts:
        print("沒有輸入 part" if answer.strip().lower() != "all" else "沒有 STT 路徑的 part")
        return
    print(f"要處理的 part：{'、'.join(parts)}")
    for part in parts:
        run_part(part)


if __name__ == "__main__":
    main()

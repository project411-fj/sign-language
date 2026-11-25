"""STT 版第 1 步：讀每個片段的 fps，把 Phase2 語音轉文字的秒數換成格數。

輸入：
    videos/part_videos/<part>/<n>/<start>_<end>.mp4                   (Phase1 的片段，用來讀 fps；STT 路徑的影片本身就只有手語員，不用裁切)
    Phase2-STT/context_breeze/<part>/<n>/<start>_<end>.csv            (Phase2 01voice.py 的輸出：start, end, text，單位是秒)
輸出 (output_STT/)：
    01fps/fps_<part>.csv                                          (欄位 filename, fps；格式和 OCR 版的 01get_fps.py 一樣)
    01frames/<part>/<n>/<start>_<end>_frames.txt                  (每行「起始格-結束格: 文字」，格式和 OCR 的字幕一樣)

Phase4 STT 版直接從 part_videos 取關鍵點，所以秒數 × fps 就是 .npy 的列索引。
執行後輸入 part：「3」、「1,2,5」或「all」(part_videos 底下所有 part)。
"""
import csv
import math
import re
from pathlib import Path

import cv2
from tqdm import tqdm

# ================================================================ 路徑設定 (後面會再接 /<part>)
VIDEO_DIR = Path("Data Preparation/videos/part_videos")
STT_DIR = Path("Data Preparation/Phase2-STT/context_breeze")          # 改成 context 就是讀 voice.py (Whisper large-v3) 的輸出
FPS_DIR = Path("Data Preparation/Phase5-Pause/output_STT/01fps")      # 裡面是 fps_<part>.csv
FRAMES_DIR = Path("Data Preparation/Phase5-Pause/output_STT/01frames")
TXT_SUFFIX = "_frames"     # 000600_001507.csv -> 000600_001507_frames.txt


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
    # <子資料夾>/<start>_<end>.mp4
    parts = path.stem.split("_")
    return int(path.parent.name), int(parts[0]), int(parts[1])


def to_frames(start, end, fps):
    """秒 → 格 (兩端都含)：第 i 格涵蓋 [i/fps, (i+1)/fps)。"""
    first = math.floor(start * fps + 1e-6)
    last = math.ceil(end * fps - 1e-6) - 1
    return first, max(first, last)


def convert(csv_path, txt_path, fps):
    """把一支片段的 STT .csv 轉成「起始格-結束格: 文字」的 .txt，回傳句數。"""
    lines = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            first, last = to_frames(float(row["start"]), float(row["end"]), fps)
            lines.append((first, last, text))
    lines.sort()
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("".join(f"{s}-{e}: {t}\n" for s, e, t in lines), encoding="utf-8")
    return len(lines)


def run_part(part):
    video_dir, stt_dir, frames_dir = VIDEO_DIR / part, STT_DIR / part, FRAMES_DIR / part
    if not video_dir.is_dir():
        print(f"part {part}：找不到 {video_dir}，跳過")
        return
    if not stt_dir.is_dir():
        print(f"part {part}：找不到 {stt_dir}，請先跑 Phase2 的 01voice.py，跳過")
        return

    videos = sorted((p for p in video_dir.glob("*/*.mp4") if re.fullmatch(r"\d+_\d+", p.stem)), key=video_sort_key)
    rows, no_fps, no_stt, empty = [], [], [], []
    for video in tqdm(videos, desc=f"part {part}", unit="video"):
        name = video.relative_to(video_dir).as_posix()            # 例如 1/000600_001507.mp4
        fps = get_fps(video)
        if fps is None:
            no_fps.append(name)
            continue
        rows.append((name, fps))

        clip = video.stem                                         # 000600_001507
        csv_path = stt_dir / video.parent.name / f"{clip}.csv"
        if not csv_path.exists():
            no_stt.append(name)
            continue
        if convert(csv_path, frames_dir / video.parent.name / f"{clip}{TXT_SUFFIX}.txt", fps) == 0:
            empty.append(name)

    fps_csv = FPS_DIR / f"fps_{part}.csv"
    fps_csv.parent.mkdir(parents=True, exist_ok=True)
    with fps_csv.open("w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig：Excel 開啟中文才不會亂碼
        writer = csv.writer(f)
        writer.writerow(("filename", "fps"))
        writer.writerows(rows)

    print(f"part {part}：{len(videos)} 支影片，fps → {fps_csv}，"
          f"轉出 {len(rows) - len(no_stt)} 個 .txt → {frames_dir}")
    for label, names in (("讀不到 fps", no_fps), ("沒有 STT 結果 (請先跑 01voice.py)", no_stt),
                         ("STT 沒有任何句子 (02 會跳過)", empty)):
        if names:
            print(f"  {label}：{len(names)} 支")
            for name in names:
                print(f"    {name}")


def parse_parts(answer, root):
    """「3」→ ["3"]；「1,2,5」→ ["1", "2", "5"]；「all」→ root 底下所有 part (依數字排序)。"""
    answer = answer.strip()
    if answer.lower() == "all":
        return sorted((p.name for p in root.iterdir() if p.is_dir()), key=lambda x: (len(x), x))
    return [p.strip() for p in answer.split(",") if p.strip()]


def main():
    parts = parse_parts(input("part (例如 3、1,2,5 或 all): "), VIDEO_DIR)
    if not parts:
        print("沒有輸入 part")
        return
    print(f"要處理的 part：{'、'.join(parts)}")
    for part in parts:
        run_part(part)


if __name__ == "__main__":
    main()

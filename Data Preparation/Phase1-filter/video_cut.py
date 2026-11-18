import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from moviepy import VideoFileClip

# ========== 設定 ==========
PART = "1"
CSV_PATH = Path("Data Preparation/Phase1-filter/intervals") / f"style_interval_{PART}.csv"
SOURCE_DIR = Path("Data Preparation/videos/source_videos") / PART    # 原始影片
OUTPUT_DIR = Path("Data Preparation/videos/part_videos") / PART      # 剪好的影片
WORKERS = 2                                          # 同時剪幾支

def to_sec(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def cut(name, row):
    with VideoFileClip(str(SOURCE_DIR / row["video"])) as source:
        clip = source.subclipped(to_sec(row["start"]), to_sec(row["end"]))
        clip.write_videofile(str(OUTPUT_DIR / name), logger=None)

with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

names = []
video_no, clip_no = {}, {}
for row in rows:
    v = video_no.setdefault(row["video"], len(video_no) + 1)
    clip_no[v] = clip_no.get(v, 0) + 1
    names.append(f"part_{v}_{clip_no[v]}.mp4")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with ThreadPoolExecutor(WORKERS) as pool:
    futures = [pool.submit(cut, name, row) for name, row in zip(names, rows)]
    for done, future in enumerate(as_completed(futures), start=1):
        future.result()   # 執行緒裡發生的錯誤會在這裡顯示出來
        print(f"\r[{'█' * (30 * done // len(rows)):<30}] {done}/{len(rows)} ({100 * done // len(rows)}%)", end="", flush=True)

print(f"\n完成，共 {len(rows)} 支影片，在 {OUTPUT_DIR}")
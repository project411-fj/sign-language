"""02style.py 的指定影片版：只處理 VIDEOS 列出的 .mp4，不用輸入 part。

對每一支 source_videos/<part>/<n>.mp4：
    讀 01frames_crop/<part>/<n>_frames/ 的字幕條畫面 (01crop_text_13~.py 的輸出)
    → 判斷每一格是不是目標節目的樣式 → 把連續的目標畫面合併成區間
    → 寫到 02intervals/<part>/<n>_frames.txt (每行「起始格 結束格」，03clipV2.py 會讀這個)
判斷方式 (粗掃 + 細掃) 跟 02style.py 完全相同。
每支影片用自己的模型：source_videos/<part>/<n>.mp4 → Phase1-filter/best<part>-<n>.pt (例如 13/3.mp4 → best13-3.pt)。
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

# ========== 設定 ==========
# 要處理的影片：直接列出 .mp4 的路徑 (通常跟 01crop_text_13~.py 的 VIDEOS 相同)
#   相對路徑從專案根目錄 (sign-language/) 算起，也可以寫完整路徑
VIDEOS = [
    r"Data Preparation\videos\source_videos\13\1.mp4",
    r"Data Preparation\videos\source_videos\13\3.mp4"
    ]

BASE = Path(__file__).resolve().parent          # Data Preparation/Phase1-filter/
PROJECT = BASE.parents[1]                       # sign-language/
FRAMES_DIR = BASE / "01frames_crop"             # 01crop_text_13~.py 的輸出
OUTPUT = BASE / "02intervals"

CROP_SIZE = 320
TARGET_THRESHOLD = 0.9
BATCH = 64
SAMPLE_SEC = 0.5  # 粗掃取樣間隔（秒）
DEVICE = 0        # 固定用第一張 GPU
WORKERS = 8       # 讀圖執行緒數


def crop_subtitle(img):
    """只取字幕條左邊 3 倍高度的寬度，再縮成正方形（必須和 02style.py、02style_train/train_style.py 一致）。"""
    img = img.convert("RGB")
    w, h = img.size
    return img.crop((0, 0, min(w, 3 * h), h)).resize((CROP_SIZE, CROP_SIZE))


def bar(done, total, width=20):
    return f"[{'█' * (width * done // total):<{width}}] {100 * done // total:3d}%"


def show(sub):
    """畫兩行：主進度條（影片）+ 副進度條（目前影片內），畫完游標回到第一行開頭。"""
    main = f"總進度 {bar(n - 1, len(jobs))} ({n}/{len(jobs)}) {video.name}"
    print(f"\033[2K{main}\n\033[2K  {sub}", end="\r\033[1A", flush=True)


def classify(paths, phase):
    """逐批分類，回傳每張是否為 target，並更新副進度條。"""
    out = []
    for b in range(0, len(paths), BATCH):
        batch = paths[b:b + BATCH]
        imgs = list(pool.map(lambda p: crop_subtitle(Image.open(p)), batch))  # 多執行緒讀圖 + 裁切
        results = model.predict(imgs, imgsz=CROP_SIZE, device=DEVICE, verbose=False)
        out += [float(r.probs.data[target_idx]) >= TARGET_THRESHOLD for r in results]
        show(f"{phase} {bar(len(out), len(paths))} {len(out)}/{len(paths)} 幀")
    return out


def weights_of(video):
    """source_videos/13/3.mp4 → BASE / "best13-3.pt" (每支影片各自的字幕條樣式模型)。"""
    return BASE / f"best{video.parent.name}-{video.stem}.pt"


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT / p


videos = [resolve(p) for p in VIDEOS]
not_found = [v for v in videos if not v.is_file()]
if not_found:
    raise SystemExit("找不到這些影片：\n" + "\n".join(f"  {v}" for v in not_found))

# 每支影片對應的畫面資料夾：source_videos/13/3.mp4 → 01frames_crop/13/3_frames
jobs, missing, no_weights = [], [], []
for v in videos:
    frames = FRAMES_DIR / v.parent.name / f"{v.stem}_frames"
    if not (frames.is_dir() and any(frames.glob("*.jpg"))):
        missing.append((v, frames))
    elif not weights_of(v).is_file():
        no_weights.append(v)
    else:
        jobs.append((v, frames))

print(f"共 {len(videos)} 支影片，其中 {len(jobs)} 支可以處理")
for v, frames in missing:
    print(f"  [略過] {v.parent.name}/{v.name}：找不到 {frames} (請先用 01crop_text_13~.py 裁切)")
for v in no_weights:
    print(f"  [略過] {v.parent.name}/{v.name}：找不到模型 {weights_of(v)}")
for v, _ in jobs:
    print(f"  {v.parent.name}/{v.name} → 模型 {weights_of(v).name}")
if not jobs:
    raise SystemExit("沒有可以處理的影片")

os.system("")  # 讓 Windows 終端機吃 ANSI 游標控制碼

results = []
pool = ThreadPoolExecutor(WORKERS)
models = {}   # 同一個模型檔只載入一次

for n, (video, d) in enumerate(jobs, 1):
    weights = weights_of(video)
    if weights not in models:
        loaded = YOLO(str(weights))
        models[weights] = (loaded, [k for k, v in loaded.names.items() if v == "target"][0])
    model, target_idx = models[weights]   # classify() 用的是這兩個全域變數
    out_dir = OUTPUT / video.parent.name              # source_videos/13/3.mp4 → 02intervals/13/
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(d.glob("*.jpg"))
    labels = [None] * len(paths)

    # 從影片本身讀 FPS
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    step = max(1, round(fps * SAMPLE_SEC))

    # 1. 粗掃：每 step 幀取樣一次（最後一幀一定取）
    samples = list(range(0, len(paths), step))
    if samples[-1] != len(paths) - 1:
        samples.append(len(paths) - 1)
    for i, v in zip(samples, classify([paths[i] for i in samples], f"粗掃 (fps={fps:.2f}, 每 {step} 幀)")):
        labels[i] = v

    # 2. 細掃：前後兩個取樣點結果不同 → 中間逐幀補測，找出真實邊界
    gaps = [i for a, b in zip(samples, samples[1:]) if labels[a] != labels[b] for i in range(a + 1, b)]
    for i, v in zip(gaps, classify([paths[i] for i in gaps], "細掃")):
        labels[i] = v

    # 3. 其餘沒測的幀：前後取樣點結果相同，直接沿用
    for i in range(len(paths)):
        if labels[i] is None:
            labels[i] = labels[i - 1]

    # 幀編號取自檔名 frame_000123.jpg → 123
    hits = [(int(p.stem.split("_")[-1]), v) for p, v in zip(paths, labels)]

    # 把連續的 target 幀合併成區間
    intervals = []
    start = prev = None
    for idx, is_target in hits:
        if is_target and start is None:
            start = idx
        elif not is_target and start is not None:
            intervals.append((start, prev))
            start = None
        prev = idx
    if start is not None:
        intervals.append((start, prev))

    with open(out_dir / f"{d.name}.txt", "w", encoding="utf-8") as f:
        for s, e in intervals:
            f.write(f"{s} {e}\n")

    show(f"完成 測了 {len(samples) + len(gaps)}/{len(paths)} 幀，{len(intervals)} 個區間")
    results.append(out_dir / f"{d.name}.txt")

print(f"\033[2K總進度 {bar(len(jobs), len(jobs))} ({len(jobs)}/{len(jobs)})\n\033[2K  全部完成，結果：")
for r in results:
    print(f"  {r}")

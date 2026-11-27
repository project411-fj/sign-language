import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

# ========== 設定 ==========
BASE = Path(__file__).resolve().parent
FRAMES_DIR = BASE / "01frames_crop"
VIDEO_DIR = BASE.parent / "videos" / "source_videos"  # 底下是 <part>/1.mp4、2.mp4 ...
WEIGHTS = BASE / "style_model" / "best~12.pt"
OUTPUT = BASE / "02intervals"

CROP_SIZE = 320
TARGET_THRESHOLD = 0.9
BATCH = 64
SAMPLE_SEC = 0.5  # 粗掃取樣間隔（秒）
DEVICE = 0        # 固定用第一張 GPU
WORKERS = 8       # 讀圖執行緒數


def crop_subtitle(img):
    """只取字幕條左邊 3 倍高度的寬度，再縮成正方形（必須和 style_sort.py 一致）。"""
    img = img.convert("RGB")
    w, h = img.size
    return img.crop((0, 0, min(w, 3 * h), h)).resize((CROP_SIZE, CROP_SIZE))


def bar(done, total, width=20):
    return f"[{'█' * (width * done // total):<{width}}] {100 * done // total:3d}%"


def show(sub):
    """畫兩行：主進度條（資料夾）+ 副進度條（目前資料夾內），畫完游標回到第一行開頭。"""
    main = f"總進度 {bar(n - 1, len(dirs))} ({n}/{len(dirs)}) {d.name}"
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


os.system("")  # 讓 Windows 終端機吃 ANSI 游標控制碼

pool = ThreadPoolExecutor(WORKERS)
model = YOLO(str(WEIGHTS))
target_idx = [k for k, v in model.names.items() if v == "target"][0]

num = input("part: ").strip()
dirs = sorted(d for d in (FRAMES_DIR / num).iterdir() if d.is_dir())
out_dir = OUTPUT / num
out_dir.mkdir(parents=True, exist_ok=True)

for n, d in enumerate(dirs, 1):
    paths = sorted(d.glob("*.jpg"))
    labels = [None] * len(paths)

    # 從對應影片讀 FPS：01frames_crop/<part>/3_frames → source_videos/<part>/3.mp4
    video = VIDEO_DIR / num / f"{d.name.removesuffix('_frames')}.mp4"
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

print(f"\033[2K總進度 {bar(len(dirs), len(dirs))} ({len(dirs)}/{len(dirs)})\n\033[2K  全部完成")
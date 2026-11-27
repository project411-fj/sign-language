"""找出影片裡「有手語員」的時段，寫成 02style.py 的區間格式，存到 02intervals/。

對 VIDEOS 列出的每一支 source_videos/<part>/<n>.mp4：
    逐格讀影片，只看手語員範圍 (TOP, BOTTOM, LEFT, RIGHT) 裡的畫面
    → 用 YOLO Pose 判斷每一格範圍裡有沒有手語員 (夠大的人 + 看得到手臂)
    → 把連續有手語員的格合併成區間，補掉短暫偵測不到的空隙、去掉太短的區間
    → 寫到 02intervals/<part>/<n>_frames.txt (每行「起始格 結束格」，跟 02style.py 一樣，03clipV2_13~.py 會讀這個)
另外也會建好 part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/ (跟 01crop_text_13~.py 相同)，
所以 STT 路徑的影片可以直接：02signer_track.py → 03clipV2_13~.py，不用先裁字幕條畫面、也不用訓練樣式模型。

判斷方式跟 02style.py 一樣是「粗掃 + 細掃」：每 SAMPLE_SEC 秒判斷一格，
前後兩次結果不同時，中間的每一格都再判斷一次，找出準確的邊界。影片只會讀一次。
"""
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ========== 設定 ==========
# 要處理的影片 → 手語員所在的範圍 (TOP, BOTTOM, LEFT, RIGHT)，都是佔畫面高 / 寬的比例；寫 None 就看整個畫面
#   通常跟 01crop_text_13~.py 的 VIDEOS 相同。相對路徑從專案根目錄 (sign-language/) 算起，也可以寫完整路徑
VIDEOS = {
    r"Data Preparation\videos\source_videos\13\1.mp4": (0.20, 0.73, 0.10, 0.33),
    r"Data Preparation\videos\source_videos\13\3.mp4": (0.20, 0.75, 0.63, 1.00),
}

BASE = Path(__file__).resolve().parent          # Data Preparation/Phase1-filter/
PROJECT = BASE.parents[1]                       # sign-language/
OUTPUT = BASE / "02intervals"                   # 02intervals/<part>/<n>_frames.txt
PART_VIDEOS_ROOT = BASE.parent / "videos" / "part_videos"
OVERWRITE = False      # 02intervals 裡已經有這支影片的結果 (例如 02style 的) 時：False = 跳過，True = 覆蓋

WEIGHTS = "yolo26n-pose.pt"   # 跟 Phase4 02keypoint_pose.py 同一個模型，第一次執行會自動下載
DEVICE = 0                    # 第一張 GPU；沒有 GPU 就改成 "cpu"
IMGSZ = 320                   # 範圍裁出來後縮到這個大小再判斷 (手語員佔滿範圍，320 就夠)
BATCH = 32                    # 一次判斷幾格

PERSON_CONF = 0.5      # 人的信心值至少多少
MIN_HEIGHT = 0.4       # 人的框至少要有範圍高度的多少 (排除背景裡的小人、觀眾)
REQUIRE_ARMS = True    # True：還要看得到至少一隻手肘或手腕 (排除只露出頭的人)
KPT_CONF = 0.5         # 手肘 / 手腕關鍵點的信心值至少多少

SAMPLE_SEC = 0.5       # 粗掃間隔 (秒)
MIN_GAP_SEC = 1.0      # 區間之間的空隙短於這個秒數就接起來 (偵測偶爾漏掉幾格)
MIN_LEN_SEC = 2.0      # 短於這個秒數的區間丟掉
FRAME_START = 1        # 寫出的格號 = 影片第幾格 + FRAME_START；跟 01frames_crop 的 frame_000001.jpg = 第 0 格一致

ARM_POINTS = [7, 8, 9, 10]   # COCO 關鍵點：左手肘、右手肘、左手腕、右手腕


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT / p


def check_box(path, box):
    top, bottom, left, right = box if box is not None else (0.0, 1.0, 0.0, 1.0)
    if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
        raise SystemExit(f"{path} 的範圍不合理：上 {top}、下 {bottom}、左 {left}、右 {right} "
                         "(需要 0 <= 上 < 下 <= 1、0 <= 左 < 右 <= 1，順序是 TOP, BOTTOM, LEFT, RIGHT)")
    return top, bottom, left, right


def box_name(box):
    """(0.2, 0.73, 0.1, 0.33) → "0.20_0.73_0.10_0.33" (跟 01crop_text_13~.py 相同)。"""
    def fmt(v):
        s = f"{v:.4f}".rstrip("0")
        return s if len(s.split(".")[1]) >= 2 else f"{v:.2f}"
    return "_".join(fmt(v) for v in box)


def make_clip_dir(video, box):
    """建立 part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/；舊範圍的空資料夾刪掉，有檔案的提醒。"""
    parent = PART_VIDEOS_ROOT / video.parent.name / video.stem
    clip_dir = parent / box_name(box)
    if parent.is_dir():
        for old in parent.iterdir():
            if old.is_dir() and old != clip_dir and re.fullmatch(r"\d+(\.\d+)?(_\d+(\.\d+)?){3}", old.name):
                if any(old.iterdir()):
                    print(f"  注意：{old} 是舊的範圍而且裡面有檔案，請確認後手動刪除 "
                          "(同一支影片有兩個範圍資料夾時，03clipV2_13~.py 不知道要用哪一個)")
                else:
                    old.rmdir()
    clip_dir.mkdir(parents=True, exist_ok=True)
    return clip_dir


def has_signer(model, crops):
    """每一張裁好的畫面裡有沒有手語員 → [True/False, ...]。"""
    out = []
    for b in range(0, len(crops), BATCH):
        results = model.predict(crops[b:b + BATCH], imgsz=IMGSZ, conf=PERSON_CONF, device=DEVICE, verbose=False)
        for r in results:
            found = False
            if r.boxes is not None and len(r.boxes):
                heights = (r.boxes.xyxyn[:, 3] - r.boxes.xyxyn[:, 1]).cpu().numpy()
                big = heights >= MIN_HEIGHT
                if REQUIRE_ARMS and r.keypoints is not None and r.keypoints.conf is not None:
                    arms = (r.keypoints.conf[:, ARM_POINTS].cpu().numpy() >= KPT_CONF).any(1)
                    big &= arms
                elif REQUIRE_ARMS:
                    big[:] = False
                found = bool(big.any())
            out.append(found)
    return out


def scan(model, video, box, bar):
    """逐格讀影片，回傳每一格有沒有手語員 (np.bool_ 陣列) 和 fps。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不開 {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * SAMPLE_SEC))
    top, bottom, left, right = box

    labels = []          # 已經確定的格
    buf, buf_label = [], None   # buf[0] 是上一個取樣點 (結果 = buf_label)，後面是還沒判斷的格
    tested = 0

    def crop(frame):
        h, w = frame.shape[:2]
        c = frame[round(top * h):round(bottom * h), round(left * w):round(right * w)]
        scale = IMGSZ / max(c.shape[:2])
        return cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else c

    def flush(final):
        """判斷 buf 裡的取樣點，結果不同的兩個取樣點之間逐格補判斷；最後一個取樣點留在 buf 給下一批。"""
        nonlocal buf, buf_label, tested
        pos = list(range(0, len(buf), step))
        if final and pos[-1] != len(buf) - 1:
            pos.append(len(buf) - 1)
        lab = dict(zip(pos[1:] if buf_label is not None else pos,
                       has_signer(model, [buf[i] for i in (pos[1:] if buf_label is not None else pos)])))
        if buf_label is not None:
            lab[0] = buf_label
        tested += len(pos) - (buf_label is not None)
        res = [None] * len(buf)
        for i in pos:
            res[i] = lab[i]
        gaps = [i for a, b in zip(pos, pos[1:]) if lab[a] != lab[b] for i in range(a + 1, b)]
        for i, v in zip(gaps, has_signer(model, [buf[i] for i in gaps])):
            res[i] = v
        tested += len(gaps)
        for i in range(len(res)):
            if res[i] is None:
                res[i] = res[i - 1]
        keep = len(buf) if final else pos[-1]
        labels.extend(res[:keep])
        buf, buf_label = buf[keep:], (None if final else res[keep])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(crop(frame))
        bar.update()
        if len(buf) > BATCH * step:
            flush(final=False)
    cap.release()
    if buf:
        flush(final=True)
    return np.array(labels, dtype=bool), fps, tested


def to_intervals(labels, fps):
    """連續 True 的格 → [(起, 迄), ...]，補短空隙、去短區間。"""
    edges = np.diff(np.r_[0, labels.astype(np.int8), 0])
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 < MIN_GAP_SEC * fps:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(int(s), int(e)) for s, e in merged if e - s + 1 >= MIN_LEN_SEC * fps]


def main():
    videos = {resolve(p): check_box(p, box) for p, box in VIDEOS.items()}
    not_found = [v for v in videos if not v.is_file()]
    if not_found:
        raise SystemExit("找不到這些影片：\n" + "\n".join(f"  {v}" for v in not_found))

    jobs = []
    for v, box in videos.items():
        out = OUTPUT / v.parent.name / f"{v.stem}_frames.txt"
        if out.exists() and not OVERWRITE:
            print(f"  [略過] {v.parent.name}/{v.name}：{out} 已經存在 (要重做就把 OVERWRITE 改成 True)")
            continue
        print(f"  {v.parent.name}/{v.name}：範圍 {box_name(box)}，片段資料夾 {make_clip_dir(v, box)}")
        jobs.append((v, box, out))
    if not jobs:
        raise SystemExit("沒有要處理的影片")

    model = YOLO(WEIGHTS)
    results = []
    for n, (video, box, out) in enumerate(jobs, 1):
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        with tqdm(total=total, desc=f"({n}/{len(jobs)}) {video.parent.name}/{video.name}", unit="格") as bar:
            labels, fps, tested = scan(model, video, box, bar)
        intervals = to_intervals(labels, fps)

        # 格數要跟 01frames_crop (ffmpeg 一格一張) 一致，不然格號會跟 03clipV2 換算的時間對不上
        frames_dir = BASE / "01frames_crop" / video.parent.name / f"{video.stem}_frames"
        if frames_dir.is_dir():
            jpgs = sum(1 for _ in frames_dir.glob("frame_*.jpg"))
            if jpgs and jpgs != len(labels):
                print(f"  注意：讀到 {len(labels)} 格，但 {frames_dir} 有 {jpgs} 張，格號可能有偏移")

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f"{s + FRAME_START} {e + FRAME_START}\n" for s, e in intervals), encoding="utf-8")
        covered = sum(e - s + 1 for s, e in intervals)
        print(f"  完成：{len(labels)} 格 ({len(labels) / fps / 60:.1f} 分鐘)，判斷了 {tested} 格，"
              f"{len(intervals)} 個區間，有手語員 {covered / max(len(labels), 1):.0%} ({covered / fps / 60:.1f} 分鐘) → {out}")
        results.append(out)

    print("\n全部完成，下一步：03clipV2_13~.py (輸入 part)")
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    main()

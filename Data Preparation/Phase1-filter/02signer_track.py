"""找出影片裡「有手語員」的時段，寫成 02style.py 的區間格式，存到 02intervals/。

對 VIDEOS 列出的每一支 source_videos/<part>/<n>.mp4：
    先自動找出手語員的範圍 (TOP, BOTTOM, LEFT, RIGHT)：在整支影片平均取 DETECT_SAMPLES 格，
    找出「一直待在同一個位置、看得到手臂的人」(鏡頭切換時，畫面上的其他人位置會變，手語員的子畫面不會)，
    再把範圍畫在畫面上存成 02intervals/<part>/<n>_box.jpg 讓你確認
    → 逐格讀影片，只看手語員範圍裡的畫面
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
# 要處理的影片 → 手語員所在的範圍
#   None：程式自己找 (建議)；自動找錯時 (看 <n>_box.jpg)，可以改成手動指定 (TOP, BOTTOM, LEFT, RIGHT)，
#   都是佔畫面高 / 寬的比例，例如 (0.20, 0.73, 0.10, 0.33)。相對路徑從專案根目錄 (sign-language/) 算起，也可以寫完整路徑
VIDEOS = {
    r"Data Preparation\videos\source_videos\13\1.mp4": None,
    r"Data Preparation\videos\source_videos\13\3.mp4": None,
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

# 自動找範圍
DETECT_SAMPLES = 300   # 在影片裡平均取幾格來找手語員
DETECT_IMGSZ = 960     # 找範圍時整個畫面縮到這個大小判斷 (子畫面裡的手語員比較小，用大一點)
SAME_PLACE_IOU = 0.5   # 兩個框重疊 (IoU) 超過這個就算同一個位置
MIN_PRESENCE = 0.2     # 手語員至少要出現在多少比例的取樣格裡，不然就當作找不到
MARGIN = 0.15          # 範圍往四周多留人的框寬 / 高的多少 (打手語時手會伸出身體的框)

ARM_POINTS = [7, 8, 9, 10]   # COCO 關鍵點：左手肘、右手肘、左手腕、右手腕


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT / p


def check_box(path, box):
    if box is None:
        return None          # 之後自動找
    top, bottom, left, right = box
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


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def detect_box(model, video, preview):
    """自動找手語員的範圍 → (TOP, BOTTOM, LEFT, RIGHT)，並把結果畫在畫面上存成 preview。"""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, people = [], []                 # people[i] = 第 i 個取樣格裡的人 [(x1, y1, x2, y2), ...] (比例)
    for t in tqdm(np.linspace(0, max(total - 1, 0), DETECT_SAMPLES).astype(int), desc="  找手語員範圍", unit="格", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame = cap.read()
        if not ok:
            continue
        r = model.predict(frame, imgsz=DETECT_IMGSZ, conf=PERSON_CONF, device=DEVICE, verbose=False)[0]
        boxes = []
        if r.boxes is not None and len(r.boxes) and r.keypoints is not None and r.keypoints.conf is not None:
            arms = (r.keypoints.conf[:, ARM_POINTS].cpu().numpy() >= KPT_CONF).any(1)
            boxes = [tuple(b) for b, a in zip(r.boxes.xyxyn.cpu().numpy().tolist(), arms) if a or not REQUIRE_ARMS]
        people.append(boxes)
        if len(people) % max(1, DETECT_SAMPLES // 6) == 1:   # 確認圖用：平均分散在整支影片的 6 格
            frames.append(frame)
    cap.release()
    if not people:
        raise RuntimeError(f"{video} 讀不到畫面")

    # 依位置分群：每一群 = 一個固定的位置，記下出現在哪些取樣格
    groups = []   # [{"boxes": [...], "seen": set(取樣格)}, ...]
    for i, boxes in enumerate(people):
        for b in boxes:
            best = max(groups, key=lambda g: iou(b, np.median(g["boxes"], 0)), default=None)
            if best is not None and iou(b, np.median(best["boxes"], 0)) >= SAME_PLACE_IOU:
                best["boxes"].append(b)
                best["seen"].add(i)
            else:
                groups.append({"boxes": [b], "seen": {i}})
    groups.sort(key=lambda g: -len(g["seen"]))
    for g in groups[:3]:
        m = np.median(g["boxes"], 0)
        print(f"    候選位置 上 {m[1]:.2f} 下 {m[3]:.2f} 左 {m[0]:.2f} 右 {m[2]:.2f}：出現在 {len(g['seen']) / len(people):.0%} 的取樣格")
    if not groups or len(groups[0]["seen"]) < MIN_PRESENCE * len(people):
        raise RuntimeError(f"{video}：找不到固定位置的手語員，請在 VIDEOS 手動指定範圍 (TOP, BOTTOM, LEFT, RIGHT)")

    # 範圍 = 這個位置大部分時候的框 (去掉最極端的 5%) 再往外留 MARGIN，取到 0.01
    b = np.array(groups[0]["boxes"])
    x1, y1 = np.percentile(b[:, 0], 5), np.percentile(b[:, 1], 5)
    x2, y2 = np.percentile(b[:, 2], 95), np.percentile(b[:, 3], 95)
    mw, mh = MARGIN * (x2 - x1), MARGIN * (y2 - y1)
    box = (np.floor(max(0.0, y1 - mh) * 100) / 100, np.ceil(min(1.0, y2 + mh) * 100) / 100,
           np.floor(max(0.0, x1 - mw) * 100) / 100, np.ceil(min(1.0, x2 + mw) * 100) / 100)
    box = tuple(float(v) for v in box)

    # 確認用的圖：幾張取樣畫面並排，紅框 = 手語員範圍
    tiles = []
    for f in frames[:6]:
        h, w = f.shape[:2]
        f = f.copy()
        cv2.rectangle(f, (round(box[2] * w), round(box[0] * h)), (round(box[3] * w), round(box[1] * h)), (0, 0, 255), max(2, w // 300))
        tiles.append(cv2.resize(f, (640, round(640 * h / w))))
    preview.parent.mkdir(parents=True, exist_ok=True)
    rows = [np.hstack(tiles[i:i + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[i:i + 3]))) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(preview), np.vstack(rows))
    return box


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
        jobs.append((v, box, out))
    if not jobs:
        raise SystemExit("沒有要處理的影片")

    model = YOLO(WEIGHTS)
    results, found = [], []
    for n, (video, box, out) in enumerate(jobs, 1):
        print(f"\n({n}/{len(jobs)}) {video.parent.name}/{video.name}")
        if box is None:
            preview = out.with_name(f"{video.stem}_box.jpg")
            try:
                box = detect_box(model, video, preview)
            except RuntimeError as exc:
                print(f"  [略過] {exc}")
                continue
            print(f"  自動找到範圍 {box} (TOP, BOTTOM, LEFT, RIGHT)，請打開 {preview} 確認紅框框住手語員")
            found.append((video, box))
        print(f"  範圍 {box_name(box)}，片段資料夾 {make_clip_dir(video, box)}")
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        with tqdm(total=total, desc="  逐格判斷", unit="格") as bar:
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
    if found:
        print("\n自動找到的範圍 (可以記到 note.md；找錯的話把它改好貼回 VIDEOS 再跑一次，記得 OVERWRITE = True)：")
        for v, box in found:
            print(f'    r"{v.relative_to(PROJECT) if v.is_relative_to(PROJECT) else v}": {box},')


if __name__ == "__main__":
    main()

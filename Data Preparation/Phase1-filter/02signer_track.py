"""找出影片裡「有手語員」的時段和手語員的位置，寫成 02style.py 的區間格式，存到 02intervals/。

執行後輸入 part：「13」或「13,14」→ 處理 source_videos/<part>/ 底下所有 .mp4；直接按 Enter → 只處理 VIDEOS 列出的影片。

手語員的位置可以在影片中途改變 (例如前段在右邊、後段在左邊的藍框裡)。對每一支 source_videos/<part>/<n>.mp4：
  1. 找位置：每 DETECT_SEC 秒取一格整個畫面，用 YOLO Pose 找看得到手臂的人，依位置分群。
     手語員會在同一個位置連續待很久 (MIN_RUN_SEC 以上)，新聞畫面裡的人鏡頭一換位置就變，
     所以「累積出現夠久、而且曾經連續出現夠久」的位置才算手語員的位置 (最多 MAX_POSITIONS 個)。
     每個位置的範圍畫在 02intervals/<part>/<n>_box.jpg 讓你確認。
  2. 逐格判斷：每 SAMPLE_SEC 秒判斷一格「每個位置有沒有人」，前後兩次結果不同時，
     中間的每一格都再判斷一次，邊界準確到格 (跟 02style.py 一樣的粗掃 + 細掃)。
     最後整支影片一起決定每一格手語員在哪個位置：換位置要付代價，所以新聞畫面裡剛好有人
     站在別的位置幾秒鐘，不會被當成手語員換位置 (見 decide())。
  3. 連續在同一個位置的格合併成區間，補掉短暫偵測不到的空隙、去掉太短的區間，
     寫到 02intervals/<part>/<n>_frames.txt，每行「起始格 結束格 範圍」，例如 1201 5400 0.12_0.95_0.62_0.93
     (範圍 = TOP_BOTTOM_LEFT_RIGHT，03clipV2_13~.py 會把這個區間剪到 part_videos/<part>/<n>/<範圍>/)。
另外會建好每個位置的 part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/ 資料夾。
STT 路徑的影片可以直接：02signer_track.py → 03clipV2_13~.py，不用先裁字幕條畫面、也不用訓練樣式模型。

GPU = True 時：影片用 ffmpeg 在顯卡上解碼 (NVDEC) 並縮小，只有縮小後的畫面搬回 CPU；YOLO 判斷也在顯卡上。
"""
import re
import subprocess
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ========== 設定 ==========
# 執行時直接按 Enter 才會用這裡：要處理的影片 → 手語員的位置
#   (輸入 part 時處理整個 part；這裡有列到的影片會用這裡的設定，沒列到的自動找)
#   None：程式自己找 (建議，可以找出多個位置)
#   自動找錯時 (看 <n>_box.jpg)，可以手動指定範圍 (TOP, BOTTOM, LEFT, RIGHT)，都是佔畫面高 / 寬的比例；
#   有多個位置就寫成 list：[(0.12, 0.95, 0.62, 0.93), (0.10, 0.80, 0.02, 0.30)]
#   相對路徑從專案根目錄 (sign-language/) 算起，也可以寫完整路徑
VIDEOS = {
    r"Data Preparation\videos\source_videos\13\1.mp4": None,
    r"Data Preparation\videos\source_videos\13\3.mp4": None,
}

BASE = Path(__file__).resolve().parent          # Data Preparation/Phase1-filter/
PROJECT = BASE.parents[1]                       # sign-language/
SOURCE_ROOT = BASE.parent / "videos" / "source_videos"   # 輸入 part 時找 source_videos/<part>/*.mp4
OUTPUT = BASE / "02intervals"                   # 02intervals/<part>/<n>_frames.txt
PART_VIDEOS_ROOT = BASE.parent / "videos" / "part_videos"
OVERWRITE = False      # 02intervals 裡已經有這支影片的結果 (例如 02style 的) 時：False = 跳過，True = 覆蓋

GPU = True             # True：影片在 NVIDIA 顯卡上解碼、縮小 (跟 01crop_text_13~.py 一樣)；False：用 CPU
FFMPEG = "ffmpeg"      # 不在 PATH 就改成 ffmpeg.exe 完整路徑
FFPROBE = "ffprobe"

WEIGHTS = "yolo26n-pose.pt"   # 跟 Phase4 02keypoint_pose.py 同一個模型，第一次執行會自動下載
DEVICE = 0                    # 第一張 GPU；沒有 GPU 就改成 "cpu"
SCAN_SIZE = 480               # 整個畫面縮到最長邊這麼大再判斷 (手語員通常佔畫面高度一半以上，480 就夠)
BATCH = 32                    # 一次判斷幾格

PERSON_CONF = 0.5      # 人的信心值至少多少
REQUIRE_ARMS = True    # True：要看得到至少一隻手肘或手腕才算 (排除只露出頭的人)
KPT_CONF = 0.5         # 手肘 / 手腕關鍵點的信心值至少多少
MIN_PERSON = 0.3       # 找位置時，人的框至少要有畫面高度的多少 (排除新聞畫面裡的小人)
MATCH_IOU = 0.4        # 逐格判斷時，人的框跟某個位置「平常手語員的框」重疊 (IoU) 至少這麼多，才算這個位置有人
SWITCH_SEC = 2.0       # 手語員「換位置」的代價 (秒)：越大越不容易換 (新聞畫面裡的人站在別的位置幾秒，不會被當成換位置)
APPEAR_SEC = 1.0       # 手語員「出現 / 消失」的代價 (秒)：「消失再出現」要付兩次，所以短於約 1 秒的漏偵測會被補起來
NONE_COST = 0.7        # 有人符合某個位置、卻判斷成「沒有手語員」的代價 (每格)

# 找位置
DETECT_SEC = 1.0          # 每幾秒取一格來找位置
SAME_PLACE_IOU = 0.5      # 兩個框重疊 (IoU) 超過這個就算同一個位置
MIN_PRESENCE_SEC = 30     # 一個位置累積至少出現這麼多秒才算手語員的位置
MIN_RUN_SEC = 20          # 而且至少有一次連續出現這麼多秒 (新聞畫面裡的人通常幾秒鏡頭就換了)
MAX_POSITIONS = 5         # 最多幾個位置
MERGE_IOU = 0.6           # 兩個位置的範圍重疊超過這個就合併成一個
MARGIN = 0.15             # 範圍往四周多留人的框寬 / 高的多少 (打手語時手會伸出身體的框)

# 逐格判斷
SAMPLE_SEC = 0.5       # 粗掃間隔 (秒)
MIN_GAP_SEC = 1.0      # 同一個位置的區間之間，空隙短於這個秒數就接起來 (偵測偶爾漏掉幾格)
MIN_LEN_SEC = 2.0      # 短於這個秒數的區間丟掉
FRAME_START = 1        # 寫出的格號 = 影片第幾格 + FRAME_START；跟 01frames_crop 的 frame_000001.jpg = 第 0 格一致

ARM_POINTS = [7, 8, 9, 10]   # COCO 關鍵點：左手肘、右手肘、左手腕、右手腕
COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255)]   # 確認圖裡各位置的顏色 (BGR)
BOX_RE = re.compile(r"\d+(\.\d+)?(_\d+(\.\d+)?){3}")


# ---------------------------------------------------------------- 小工具
def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT / p


def check_boxes(path, setting):
    """VIDEOS 的設定 → None (自動找) 或 [(TOP, BOTTOM, LEFT, RIGHT), ...]。"""
    if setting is None:
        return None
    boxes = [setting] if isinstance(setting[0], (int, float)) else list(setting)
    for box in boxes:
        top, bottom, left, right = box
        if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
            raise SystemExit(f"{path} 的範圍不合理：上 {top}、下 {bottom}、左 {left}、右 {right} "
                             "(需要 0 <= 上 < 下 <= 1、0 <= 左 < 右 <= 1，順序是 TOP, BOTTOM, LEFT, RIGHT)")
    return [tuple(float(v) for v in box) for box in boxes]


def box_name(box):
    """(0.2, 0.73, 0.1, 0.33) → "0.20_0.73_0.10_0.33" (跟 01crop_text_13~.py 相同)。"""
    def fmt(v):
        s = f"{v:.4f}".rstrip("0")
        return s if len(s.split(".")[1]) >= 2 else f"{v:.2f}"
    return "_".join(fmt(v) for v in box)


def make_clip_dirs(video, boxes):
    """建立每個位置的 part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/；不再使用的空資料夾刪掉，有檔案的提醒。"""
    parent = PART_VIDEOS_ROOT / video.parent.name / video.stem
    keep = {parent / box_name(b) for b in boxes}
    if parent.is_dir():
        for old in parent.iterdir():
            if old.is_dir() and old not in keep and BOX_RE.fullmatch(old.name):
                if any(old.iterdir()):
                    print(f"  注意：{old} 是舊的範圍而且裡面有檔案，請確認後手動刪除 "
                          "(舊片段會跟這次剪的片段重複)")
                else:
                    old.rmdir()
    for d in keep:
        d.mkdir(parents=True, exist_ok=True)


def iou(a, b):
    """兩個框 (x1, y1, x2, y2) 的 IoU。"""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def as_xyxy(box):
    top, bottom, left, right = box
    return left, top, right, bottom


# ---------------------------------------------------------------- 讀影片 (ffmpeg，GPU 解碼)
def video_info(path):
    """(編碼格式, 寬, 高, fps, 總格數)。"""
    codec, w, h = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")[:3]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return codec, int(w), int(h), fps, total


def read_frames(video, every=1):
    """用 ffmpeg 依序讀出整個畫面 (縮到最長邊 SCAN_SIZE，BGR)；every > 1 時只讀第 0、every、2×every… 格。
    GPU = True：解碼、縮小都在顯卡上，只有縮小後的畫面搬回 CPU。每個解碼出來的格都會算到，不補格也不丟格。"""
    codec, w, h, _, _ = video_info(video)
    scale = min(1.0, SCAN_SIZE / max(w, h))
    ow, oh = max(2, round(w * scale / 2) * 2), max(2, round(h * scale / 2) * 2)
    select = f"select=not(mod(n\\,{every}))," if every > 1 else ""
    if GPU:
        decode = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", f"{codec}_cuvid"]   # h264_cuvid…
        # 色彩標記改成一般的 BT.709 (有些 VP9 影片標錯會讓轉換失敗，跟 01crop_text_13~.py 相同的處理)
        vf = (f"{select}scale_cuda=w={ow}:h={oh}:interp_algo=bilinear,hwdownload,format=nv12,"
              "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709")
    else:
        decode = []
        vf = f"{select}scale={ow}:{oh}:flags=area"
    cmd = [FFMPEG, "-nostdin", "-loglevel", "error", *decode, "-i", str(video), "-an",
           # setpts=N 依畫面順序重新編時間戳記，並沿用原影片的時間單位 (-enc_time_base demux)；
           # 少了後者，29.97 fps 這類影片每一格都會印「non monotonically increasing dts」錯誤 (跟 01crop_text_13~.py 相同的處理)
           "-vf", vf + ",setpts=N", "-fps_mode", "passthrough", "-enc_time_base", "demux",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    size = ow * oh * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=size * 8)

    # ffmpeg 的錯誤訊息要邊跑邊讀掉：ffmpeg 一直印錯誤訊息時 (例如影片裡有壞掉的格)，
    # 沒人讀的話管道塞滿，ffmpeg 就卡住不再輸出畫面 (進度條停在某個 % 不動)
    errors, count = deque(maxlen=20), [0]

    def drain():
        for line in proc.stderr:
            errors.append(line.decode("utf-8", "replace").rstrip())
            count[0] += 1
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()

    finished = False
    try:
        while True:
            data = proc.stdout.read(size)
            if len(data) < size:
                break
            yield np.frombuffer(data, np.uint8).reshape(oh, ow, 3)
        finished = True
    finally:
        if not finished:                 # 呼叫端提早結束 (例如出錯)：把 ffmpeg 關掉
            proc.kill()
        proc.stdout.close()
        code = proc.wait()
        reader.join(timeout=5)
        if finished and code != 0:
            raise RuntimeError(f"ffmpeg 讀取 {video} 失敗" + (" (顯卡不支援這個影片格式的話，把 GPU 改成 False)" if GPU else "")
                               + "：\n" + "\n".join(errors))
        if finished and count[0]:
            tqdm.write(f"  注意：ffmpeg 讀 {video.name} 時有 {count[0]} 行錯誤訊息 (可能是影片裡有壞掉的格，"
                       f"格數可能跟影片時間對不上)，最後幾行：\n    " + "\n    ".join(list(errors)[-3:]))


# ---------------------------------------------------------------- YOLO
def people_in(model, frames):
    """每一格裡看得到手臂的人 → [[(x1, y1, x2, y2), ...], ...] (佔畫面的比例)。"""
    out = []
    for b in range(0, len(frames), BATCH):
        for r in model.predict(frames[b:b + BATCH], imgsz=SCAN_SIZE, conf=PERSON_CONF, device=DEVICE, verbose=False):
            boxes = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxyn.cpu().numpy()
                if REQUIRE_ARMS:
                    if r.keypoints is None or r.keypoints.conf is None:
                        xyxy = xyxy[:0]
                    else:
                        xyxy = xyxy[(r.keypoints.conf[:, ARM_POINTS].cpu().numpy() >= KPT_CONF).any(1)]
                boxes = [tuple(map(float, b)) for b in xyxy]
            out.append(boxes)
    return out


def person_box(position):
    """範圍 (TOP, BOTTOM, LEFT, RIGHT) 去掉往外留的 MARGIN → 平常手語員的框 (x1, y1, x2, y2)。"""
    top, bottom, left, right = position
    mw, mh = (right - left) * MARGIN / (1 + 2 * MARGIN), (bottom - top) * MARGIN / (1 + 2 * MARGIN)
    return left + mw, top + mh, right - mw, bottom - mh


def match_scores(people, positions):
    """這一格每個位置的符合程度 → np.array(位置數)：有人的框跟該位置「平常手語員的框」重疊 (IoU) 至少 MATCH_IOU 時
    是那個 IoU (有好幾個人取最大)，否則 0。"""
    out = np.zeros(len(positions), np.float32)
    for p in people:
        for k, pos in enumerate(positions):
            v = iou(p, person_box(pos))
            if v >= MATCH_IOU:
                out[k] = max(out[k], v)
    return out


def classify(model, frames, positions):
    """每一格每個位置的符合程度 → np.array(格數, 位置數)。"""
    return np.array([match_scores(p, positions) for p in people_in(model, frames)], np.float32).reshape(-1, len(positions))


# ---------------------------------------------------------------- 1. 找位置
def find_positions(model, video, preview):
    """自動找手語員的位置 → [(TOP, BOTTOM, LEFT, RIGHT), ...]，並畫確認圖。"""
    _, _, _, fps, total = video_info(video)
    every = max(1, round(fps * DETECT_SEC))
    n_samples = total // every + 1
    keep_every = max(1, n_samples // 200)            # 確認圖用：最多留 200 張縮圖
    people, thumbs = [], {}
    frames = []
    with tqdm(total=n_samples, desc="  找手語員位置", unit="格", leave=False) as bar:
        for i, frame in enumerate(read_frames(video, every)):
            frames.append(frame)
            if i % keep_every == 0:
                thumbs[i] = frame
            if len(frames) == BATCH:
                people += people_in(model, frames)
                bar.update(len(frames))
                frames = []
        if frames:
            people += people_in(model, frames)
            bar.update(len(frames))
    if not people:
        raise RuntimeError(f"{video} 讀不到畫面")

    # 依位置分群：每一群 = 一個位置，記下出現在哪些取樣格
    groups = []   # [{"boxes": [...], "seen": [取樣格, ...]}, ...]
    for i, boxes in enumerate(people):
        for b in boxes:
            if b[3] - b[1] < MIN_PERSON:
                continue
            scores = [iou(b, g["center"]) for g in groups]
            k = int(np.argmax(scores)) if scores else -1
            if k >= 0 and scores[k] >= SAME_PLACE_IOU:
                g = groups[k]
                g["boxes"].append(b)
                if g["seen"][-1] != i:
                    g["seen"].append(i)
                if len(g["boxes"]) % 20 == 0:      # 偶爾更新這群的中心位置
                    g["center"] = tuple(np.median(g["boxes"], 0))
            else:
                groups.append({"boxes": [b], "seen": [i], "center": b})

    def longest_run(seen):
        """最長連續出現幾個取樣格 (中間漏掉 1~2 格也算連續)。"""
        best = run = 1
        for a, b in zip(seen, seen[1:]):
            run = run + (b - a) if b - a <= 3 else 1
            best = max(best, run)
        return best

    def region(boxes):
        """這群大部分時候的框 (去掉最極端的 5%) 再往外留 MARGIN → (x1, y1, x2, y2)。"""
        b = np.array(boxes)
        x1, y1 = np.percentile(b[:, 0], 5), np.percentile(b[:, 1], 5)
        x2, y2 = np.percentile(b[:, 2], 95), np.percentile(b[:, 3], 95)
        mw, mh = MARGIN * (x2 - x1), MARGIN * (y2 - y1)
        return max(0.0, x1 - mw), max(0.0, y1 - mh), min(1.0, x2 + mw), min(1.0, y2 + mh)

    found = []
    for g in groups:
        presence, run = len(g["seen"]) * DETECT_SEC, longest_run(g["seen"]) * DETECT_SEC
        if presence >= MIN_PRESENCE_SEC and run >= MIN_RUN_SEC:
            found.append({"boxes": g["boxes"], "seen": set(g["seen"]), "region": region(g["boxes"])})
    # 範圍幾乎一樣的位置合併 (例如手語員只是稍微往旁邊站)
    found.sort(key=lambda f: -len(f["seen"]))
    merged = []
    for f in found:
        k = next((k for k, m in enumerate(merged) if iou(f["region"], m["region"]) >= MERGE_IOU), -1)
        if k >= 0:
            merged[k]["boxes"] += f["boxes"]
            merged[k]["seen"] |= f["seen"]
            merged[k]["region"] = region(merged[k]["boxes"])
        else:
            merged.append(f)
    merged = sorted(merged, key=lambda f: -len(f["seen"]))[:MAX_POSITIONS]
    if not merged:
        top = sorted(groups, key=lambda g: -len(g["seen"]))[:3]
        for g in top:
            print(f"    最常出現的位置：出現 {len(g['seen']) * DETECT_SEC:.0f} 秒，最長連續 {longest_run(g['seen']) * DETECT_SEC:.0f} 秒")
        raise RuntimeError(f"{video}：找不到固定位置的手語員 (需要累積 {MIN_PRESENCE_SEC} 秒、連續 {MIN_RUN_SEC} 秒)，"
                           "請在 VIDEOS 手動指定範圍")

    positions = []
    for f in merged:
        x1, y1, x2, y2 = f["region"]
        positions.append((float(np.floor(y1 * 100) / 100), float(np.ceil(y2 * 100) / 100),
                          float(np.floor(x1 * 100) / 100), float(np.ceil(x2 * 100) / 100)))
    for k, (pos, f) in enumerate(zip(positions, merged)):
        print(f"    位置 {k + 1}：{box_name(pos)}，出現在 {len(f['seen']) / len(people):.0%} 的取樣格")

    # 確認圖：每個位置一列 (該位置的取樣畫面，畫上它的框)，最後一列是「沒有手語員」的取樣畫面
    # 確認圖只挑「明顯是某個位置」的畫面：最符合的位置比第二名高 0.2 以上
    # (新聞畫面裡剛好有人站在別的位置、分不清楚的畫面不挑，逐格判斷時會參考前後的格來決定)
    labels = {}
    for i in thumbs:
        sc = np.sort(match_scores(people[i], positions))[::-1]
        k = int(match_scores(people[i], positions).argmax())
        labels[i] = -1 if sc[0] == 0 else (k if len(sc) == 1 or sc[0] - sc[1] >= 0.2 else None)
    rows = []
    for k in list(range(len(positions))) + [-1]:
        idx = [i for i, lab in labels.items() if lab == k]
        idx = [idx[j] for j in np.linspace(0, len(idx) - 1, min(4, len(idx))).astype(int)] if idx else []
        tiles = []
        for i in idx:
            t = thumbs[i].copy()
            h, w = t.shape[:2]
            for kk, pos in enumerate(positions):
                top, bottom, left, right = pos
                cv2.rectangle(t, (round(left * w), round(top * h)), (round(right * w), round(bottom * h)),
                              COLORS[kk % len(COLORS)], 3 if kk == k else 1)
            text = f"position {k + 1}" if k >= 0 else "no signer"
            cv2.putText(t, f"{text}  {i * DETECT_SEC / 60:.1f} min", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(t)
        if tiles:
            blank = np.zeros_like(tiles[0])
            rows.append(np.hstack(tiles + [blank] * (4 - len(tiles))))
    preview.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview), np.vstack(rows))
    return positions


# ---------------------------------------------------------------- 2. 逐格判斷
def scan(model, video, positions, bar):
    """逐格算每個位置的符合程度 → (np.array(格數, 位置數)，fps，實際判斷了幾格)。
    粗掃：每 SAMPLE_SEC 秒判斷一格；前後兩個取樣點「有人符合的位置」不同時，中間逐格判斷；其餘的格沿用前一個取樣點。"""
    _, _, _, fps, _ = video_info(video)
    step = max(1, round(fps * SAMPLE_SEC))
    K = len(positions)
    scores = []                 # 已經確定的格
    buf, buf_score = [], None   # buf[0] 是上一個取樣點 (結果 = buf_score)，後面是還沒判斷的格
    tested = 0

    def flush(final):
        nonlocal buf, buf_score, tested
        pos = list(range(0, len(buf), step))
        if final and pos[-1] != len(buf) - 1:
            pos.append(len(buf) - 1)
        todo = pos[1:] if buf_score is not None else pos
        res = np.zeros((len(buf), K), np.float32)
        known = np.zeros(len(buf), bool)
        res[todo] = classify(model, [buf[i] for i in todo], positions)
        if buf_score is not None:
            res[0] = buf_score
        known[pos] = True
        tested += len(todo)
        for a, b in zip(pos, pos[1:]):
            if b - a > 1 and ((res[a] > 0) != (res[b] > 0)).any():   # 有人符合的位置不同：中間逐格判斷
                res[a + 1:b] = classify(model, buf[a + 1:b], positions)
                known[a + 1:b] = True
                tested += b - a - 1
        for i in range(1, len(buf)):
            if not known[i]:
                res[i] = res[i - 1]
        keep = len(buf) if final else pos[-1]
        scores.append(res[:keep])
        buf, buf_score = buf[keep:], (None if final else res[keep])

    for frame in read_frames(video):
        buf.append(frame)
        bar.update()
        if len(buf) > BATCH * step:
            flush(final=False)
    if buf:
        flush(final=True)
    return (np.concatenate(scores) if scores else np.zeros((0, K), np.float32)), fps, tested


def decide(scores, fps):
    """依每一格的符合程度，整支影片一起決定每一格手語員在哪個位置 (-1 = 不在)。
    用動態規劃找總代價最低的結果：
      在位置 k 的代價 = 1 - 符合程度 (那個位置沒人符合 = 2)；不在任何位置的代價 = 0 (沒人符合) 或 NONE_COST (有人符合)；
      換位置要付 SWITCH_SEC 秒份的代價，出現 / 消失要付 APPEAR_SEC 秒份的代價。
    這樣新聞畫面裡剛好有人站在別的位置幾秒鐘，不會被當成手語員換位置；真的換版面 (持續很久) 才會換。"""
    T, K = scores.shape
    if T == 0:
        return np.zeros(0, np.int8)
    cost = np.empty((T, K + 1), np.float32)
    cost[:, :K] = np.where(scores > 0, 1 - scores, 2.0)
    cost[:, K] = np.where(scores.max(1) > 0, NONE_COST, 0.0)
    trans = np.full((K + 1, K + 1), SWITCH_SEC * fps, np.float32)      # trans[前一格, 這一格]
    trans[K, :] = trans[:, K] = APPEAR_SEC * fps
    np.fill_diagonal(trans, 0)
    back = np.empty((T, K + 1), np.int8)
    total = cost[0].copy()
    for t in range(1, T):
        m = total[:, None] + trans
        back[t] = m.argmin(0)
        total = m.min(0) + cost[t]
    labels = np.empty(T, np.int8)
    labels[-1] = int(total.argmin())
    for t in range(T - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    labels[labels == K] = -1
    return labels


def to_intervals(labels, fps):
    """連續在同一個位置的格 → [(起, 迄, 位置編號), ...]；同一個位置中間的短空隙接起來，太短的區間丟掉。"""
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] >= 0:
                runs.append([start, i - 1, int(labels[start])])
            start = i
    merged = []
    for s, e, k in runs:
        # 只有「空隙裡沒有其他位置」時才接起來 (中間換到別的位置就是不同的區間)
        if merged and merged[-1][2] == k and s - merged[-1][1] - 1 < MIN_GAP_SEC * fps \
                and (labels[merged[-1][1] + 1:s] < 0).all():
            merged[-1][1] = e
        else:
            merged.append([s, e, k])
    return [(s, e, k) for s, e, k in merged if e - s + 1 >= MIN_LEN_SEC * fps]


# ---------------------------------------------------------------- 主程式
def videos_of_parts(answer):
    """「13」、「13,14」→ {source_videos/<part>/<n>.mp4: 設定}；VIDEOS 有列到的用它的設定，其他自動找 (None)。"""
    listed = {resolve(p).resolve(): s for p, s in VIDEOS.items()}
    videos = {}
    for part in (x.strip() for x in answer.split(",") if x.strip()):
        folder = SOURCE_ROOT / part
        mp4s = sorted(folder.glob("*.mp4"), key=lambda p: (len(p.stem), p.stem)) if folder.is_dir() else []
        if not mp4s:
            print(f"  part {part}：{folder} 裡沒有 .mp4，跳過")
        for v in mp4s:
            videos[v] = listed.get(v.resolve())
    return videos


def main():
    answer = input("part (例如 13 或 13,14；直接按 Enter = 只處理 VIDEOS 列出的影片): ").strip()
    if answer.lower() == "all":
        raise SystemExit("不支援 all：part 1~12 是 OCR 路徑，請輸入 STT 路徑的 part，例如 13 或 13,14")
    if answer:
        videos = {v: check_boxes(v, s) for v, s in videos_of_parts(answer).items()}
    else:
        videos = {resolve(p): check_boxes(p, s) for p, s in VIDEOS.items()}
    if not videos:
        raise SystemExit("沒有要處理的影片")
    print(f"要處理 {len(videos)} 支影片：" + "、".join(f"{v.parent.name}/{v.name}" for v in videos))
    not_found = [v for v in videos if not v.is_file()]
    if not_found:
        raise SystemExit("找不到這些影片：\n" + "\n".join(f"  {v}" for v in not_found))

    jobs = []
    for v, positions in videos.items():
        out = OUTPUT / v.parent.name / f"{v.stem}_frames.txt"
        if out.exists() and not OVERWRITE:
            print(f"  [略過] {v.parent.name}/{v.name}：{out} 已經存在 (要重做就把 OVERWRITE 改成 True)")
            continue
        jobs.append((v, positions, out))
    if not jobs:
        raise SystemExit("沒有要處理的影片")

    model = YOLO(WEIGHTS)
    results, found = [], []
    for n, (video, positions, out) in enumerate(jobs, 1):
        print(f"\n({n}/{len(jobs)}) {video.parent.name}/{video.name}")
        if positions is None:
            preview = out.with_name(f"{video.stem}_box.jpg")
            try:
                positions = find_positions(model, video, preview)
            except RuntimeError as exc:
                print(f"  [略過] {exc}")
                continue
            print(f"  找到 {len(positions)} 個位置，請打開 {preview} 確認每個框都框住手語員")
            found.append((video, positions))
        else:
            print("  手動指定的位置：" + "、".join(box_name(p) for p in positions))

        total = video_info(video)[4]
        with tqdm(total=total, desc="  逐格判斷", unit="格") as bar:
            scores, fps, tested = scan(model, video, positions, bar)
        labels = decide(scores, fps)
        intervals = to_intervals(labels, fps)
        used = sorted({k for _, _, k in intervals})
        make_clip_dirs(video, [positions[k] for k in used])

        # 格數要跟 01frames_crop (ffmpeg 一格一張) 一致，不然格號會跟 03clipV2 換算的時間對不上
        frames_dir = BASE / "01frames_crop" / video.parent.name / f"{video.stem}_frames"
        if frames_dir.is_dir():
            jpgs = sum(1 for _ in frames_dir.glob("frame_*.jpg"))
            if jpgs and jpgs != len(labels):
                print(f"  注意：讀到 {len(labels)} 格，但 {frames_dir} 有 {jpgs} 張，格號可能有偏移")

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f"{s + FRAME_START} {e + FRAME_START} {box_name(positions[k])}\n"
                               for s, e, k in intervals), encoding="utf-8")
        covered = sum(e - s + 1 for s, e, _ in intervals)
        print(f"  完成：{len(labels)} 格 ({len(labels) / fps / 60:.1f} 分鐘)，判斷了 {tested} 格，"
              f"{len(intervals)} 個區間，有手語員 {covered / max(len(labels), 1):.0%} ({covered / fps / 60:.1f} 分鐘) → {out}")
        for k in used:
            sec = sum(e - s + 1 for s, e, kk in intervals if kk == k) / fps
            print(f"    位置 {k + 1} {box_name(positions[k])}：{sec / 60:.1f} 分鐘")
        results.append(out)

    print("\n全部完成，下一步：03clipV2_13~.py (輸入 part)")
    for r in results:
        print(f"  {r}")
    if found:
        print("\n自動找到的位置 (可以記到 note.md；找錯的話把它改好貼回 VIDEOS 再跑一次，記得 OVERWRITE = True)：")
        for v, positions in found:
            value = positions[0] if len(positions) == 1 else positions
            print(f'    r"{v.relative_to(PROJECT) if v.is_relative_to(PROJECT) else v}": {value},')


if __name__ == "__main__":
    main()

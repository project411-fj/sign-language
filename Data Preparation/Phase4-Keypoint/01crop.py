import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Lock, Thread

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


PART = "3"  # 要處理的 part：videos/part_videos/<PART>/

INPUT_DIR = Path("Data Preparation/videos/part_videos") / PART
OUTPUT_DIR = Path("Data Preparation/videos/keypoint_videos") / PART
FAILED_LOG = Path(__file__).parent / f"failed_part_{PART}.txt"

PERSON_MODEL = r"Data Preparation\models\Yolo_models\person.pt"
POSE_MODEL = r"Data Preparation\models\Yolo_models\yolo26n-pose.pt"

PERSON_CONF = 0.5
POSE_CONF = 0.5
KEYPOINT_CONF = 0.5

BATCH_SIZE = 32
FRAME_STRIDE = 3     # 分析時每 N 幀取 1 幀（ROI 是統計值，不需要逐幀）
QUANTIZE = 16        # GPU 上用 FP16 推論（32 = FP32）
PREFETCH = 2         # 預讀幾批幀，讓解碼和 GPU 推論同時進行
POSE_MAX_SIDE = 640  # 暫存給 pose 的裁切圖最長邊（pose 輸入本來就是 640，縮小可省記憶體）

# 裁切輸出改用 ffmpeg（需在 PATH 中）
MAX_WORKERS = 2      # ffmpeg 本身就是多執行緒，同時跑 2 支通常足夠
FFMPEG = "ffmpeg"
# GPU：解碼、編碼用 NVIDIA 顯卡（裁切、縮放在 CPU）
VIDEO_CODEC = "h264_nvenc"
VIDEO_CODEC_ARGS = ["-preset", "p4", "-cq", "19"]
HWACCEL = ["-hwaccel", "cuda"]
# 改回 CPU：
# VIDEO_CODEC = "libx264"
# VIDEO_CODEC_ARGS = ["-preset", "veryfast", "-crf", "18"]
# HWACCEL = []

OUTPUT_SIZE = 640
SIDE_PADDING_RATIO = 0.10
BOTTOM_PADDING_RATIO = 0.05
ROI_SIZE_PERCENTILE = 99.0

REGIONS = {"left": (0.00, 0.35), "right": (0.65, 1.00)}
PRESENCE_THRESHOLD = 0.9

progress_lock = Lock()


def update_progress(bar, n):
    with progress_lock:
        bar.update(n)


def region_of(box, width):
    x = (box[0] + box[2]) / 2 / width
    for name, (start, end) in REGIONS.items():
        if start <= x <= end:
            return name
    return None


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def hip_roi(box, scale, pose_result):
    """以 hip 11、12 算出原圖上的上半身 ROI（scale 為裁切圖的縮放比例）。"""
    keypoints = pose_result.keypoints.data
    if len(keypoints) == 0:
        return None
    left_hip, right_hip = keypoints[0][11].tolist(), keypoints[0][12].tolist()
    if min(left_hip[2], right_hip[2]) < KEYPOINT_CONF:
        return None

    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    bottom = y1 + max(left_hip[1], right_hip[1]) / scale + height * BOTTOM_PADDING_RATIO
    side = width * SIDE_PADDING_RATIO
    return x1 - side, y1, x2 + side, bottom


def square_roi(samples, width, height):
    """中位數中心 + 百分位邊長，回傳 (x1, y1, size)。"""
    s = np.clip(np.array(samples, dtype=float), 0, [width, height, width, height])
    cx = np.median((s[:, 0] + s[:, 2]) / 2)
    cy = np.median((s[:, 1] + s[:, 3]) / 2)
    sizes = 2 * np.max([cx - s[:, 0], s[:, 2] - cx, cy - s[:, 1], s[:, 3] - cy], axis=0)
    size = int(min(np.ceil(np.percentile(sizes, ROI_SIZE_PERCENTILE)), width, height))
    x1 = int(np.clip(round(cx - size / 2), 0, width - size))
    y1 = int(np.clip(round(cy - size / 2), 0, height - size))
    return x1, y1, size


def read_batch(cap):
    """每 FRAME_STRIDE 幀取 1 幀，跳過的幀只 grab 不轉色彩。回傳 (取樣幀, 實際讀過的幀數)。"""
    frames, read = [], 0
    while len(frames) < BATCH_SIZE and cap.grab():
        frames.append(cap.retrieve()[1])
        read += 1
        for _ in range(FRAME_STRIDE - 1):
            if not cap.grab():
                break
            read += 1
    return frames, read


def prefetch_batches(cap):
    """背景執行緒先解碼下一批，主執行緒同時做推論。"""
    queue = Queue(maxsize=PREFETCH)

    def reader():
        while True:
            batch = read_batch(cap)
            queue.put(batch)
            if not batch[0]:
                break

    Thread(target=reader, daemon=True).start()
    while True:
        frames, read = queue.get()
        if not frames:
            break
        yield frames, read


def analyze_video(video, person_model, pose_model, bar):
    cap = cv2.VideoCapture(str(video))
    bar.reset(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_count, sampled = 0, 0
    crops = {name: [] for name in REGIONS}  # 每區域：(裁切圖, 原圖框, 縮放比例)

    # 第一步：person 偵測，每個區域挑面積最大的人，先把裁切圖存起來
    for frames, read in prefetch_batches(cap):
        height, width = frames[0].shape[:2]
        results = person_model(frames, conf=PERSON_CONF, quantize=QUANTIZE, verbose=False)
        for frame, result in zip(frames, results):
            best = {}
            for box in result.boxes.xyxy.tolist():
                region = region_of(box, width)
                if region and (region not in best or box_area(box) > box_area(best[region])):
                    best[region] = box
            for region, box in best.items():
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                scale = min(1.0, POSE_MAX_SIDE / max(crop.shape[:2]))
                if scale < 1:
                    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                else:
                    crop = crop.copy()  # 不 copy 的話 view 會讓整張原圖留在記憶體
                crops[region].append((crop, (x1, y1, x2, y2), scale))

        sampled += len(frames)
        frame_count += read
        update_progress(bar, read)
    cap.release()

    ratios = {name: len(crops[name]) / sampled for name in REGIONS}
    region = max(ratios, key=ratios.get)
    if ratios[region] < PRESENCE_THRESHOLD:
        return frame_count, ratios, None

    # 第二步：只對勝出的區域跑 pose
    items = crops[region]
    samples = []
    for i in range(0, len(items), BATCH_SIZE):
        chunk = items[i:i + BATCH_SIZE]
        results = pose_model([crop for crop, _, _ in chunk], conf=POSE_CONF, quantize=QUANTIZE, verbose=False)
        for (_, box, scale), result in zip(chunk, results):
            roi = hip_roi(box, scale, result)
            if roi:
                samples.append(roi)

    if not samples:
        return frame_count, ratios, None
    return frame_count, ratios, square_roi(samples, width, height)


def crop_video(video, output, roi):
    x1, y1, size = roi
    cmd = [
        FFMPEG, "-y", "-loglevel", "error", *HWACCEL,
        "-i", str(video),
        "-vf", f"crop={size}:{size}:{x1}:{y1},scale={OUTPUT_SIZE}:{OUTPUT_SIZE}:flags=area",
        "-an", "-c:v", VIDEO_CODEC, *VIDEO_CODEC_ARGS, "-pix_fmt", "yuv420p",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def main():
    person_model = YOLO(PERSON_MODEL)
    pose_model = YOLO(POSE_MODEL)

    # 有失敗紀錄就只重跑紀錄裡的影片，否則跑整個 part
    if FAILED_LOG.exists():
        videos = [INPUT_DIR / line for line in FAILED_LOG.read_text(encoding="utf-8").splitlines() if line]
        tqdm.write(f"讀取失敗紀錄 {FAILED_LOG.name}，重跑 {len(videos)} 支影片")
    else:
        videos = sorted(INPUT_DIR.glob("*/*.mp4"))
        tqdm.write(f"處理 part {PART}，共 {len(videos)} 支影片")
    failed = []   # (影片, 錯誤訊息)
    futures = []  # (影片, 裁切 future)

    # 主進度條：完成的影片數（分析失敗、略過、裁切完成都算完成）
    # 副進度條：目前分析中影片的幀數
    with tqdm(total=len(videos), desc="全部影片", unit="video", position=0) as main_bar, \
            tqdm(unit="frame", position=1, leave=False) as sub_bar, \
            ThreadPoolExecutor(MAX_WORKERS) as pool:
        for video in videos:
            relative = video.relative_to(INPUT_DIR)
            sub_bar.set_description_str(relative.as_posix())
            try:
                frame_count, ratios, roi = analyze_video(video, person_model, pose_model, sub_bar)
            except Exception as exc:
                failed.append((relative, f"分析失敗：{exc}"))
                tqdm.write(f"{relative.as_posix()}：分析失敗，跳過")
                update_progress(main_bar, 1)
                continue
            tqdm.write(
                f"{relative.as_posix()}：{frame_count} frames，"
                + "、".join(f"{name}={ratio:.2%}" for name, ratio in ratios.items())
            )

            if roi is None:
                tqdm.write("  略過：存在率不足或無有效 hip ROI")
                update_progress(main_bar, 1)
                continue

            output = OUTPUT_DIR / relative.parent / f"{video.stem}_upper_body.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            future = pool.submit(crop_video, video, output, roi)
            future.add_done_callback(lambda _: update_progress(main_bar, 1))  # 裁切完成才算這支影片完成
            futures.append((relative, future))
        sub_bar.set_description_str("等待裁切完成")

    # 離開 with 時已等所有裁切完成，這裡收集裁切錯誤
    for relative, future in futures:
        if future.exception():
            failed.append((relative, f"裁切失敗：{future.exception()}"))

    tqdm.write("所有影片處理完成")

    # 更新失敗紀錄：有失敗就覆寫，全部成功就刪除
    if failed:
        FAILED_LOG.write_text(
            "".join(f"{relative.as_posix()}\n" for relative, _ in failed), encoding="utf-8"
        )
    else:
        FAILED_LOG.unlink(missing_ok=True)

    if failed:
        tqdm.write(f"\n發生錯誤的影片（{len(failed)} 支）：")
        for relative, error in failed:
            tqdm.write(f"  {relative.as_posix()}：{error}")


if __name__ == "__main__":
    main()
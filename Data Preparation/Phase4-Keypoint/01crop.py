from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


PART = "2"  # 要處理的 part：videos/part_videos/<PART>/

INPUT_DIR = Path("Data Preparation/videos/part_videos") / PART
OUTPUT_DIR = Path("Data Preparation/videos/keypoint_videos") / PART
FAILED_LOG = Path(__file__).parent / f"failed_part_{PART}.txt"

PERSON_MODEL = r"Data Preparation\models\Yolo_models\person.pt"
POSE_MODEL = r"Data Preparation\models\Yolo_models\yolo26n-pose.pt"

PERSON_CONF = 0.5
POSE_CONF = 0.5
KEYPOINT_CONF = 0.5

BATCH_SIZE = 32
MAX_WORKERS = 2

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


def hip_roi(box, pose_result):
    """以 hip 11、12 算出原圖上的上半身 ROI。"""
    keypoints = pose_result.keypoints.data
    if len(keypoints) == 0:
        return None
    left_hip, right_hip = keypoints[0][11].tolist(), keypoints[0][12].tolist()
    if min(left_hip[2], right_hip[2]) < KEYPOINT_CONF:
        return None

    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    bottom = y1 + max(left_hip[1], right_hip[1]) + height * BOTTOM_PADDING_RATIO
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


def analyze_video(video, person_model, pose_model, bar):
    cap = cv2.VideoCapture(str(video))
    frame_count = 0
    counts = {name: 0 for name in REGIONS}
    samples = {name: [] for name in REGIONS}

    while True:
        frames = []
        for _ in range(BATCH_SIZE):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            break
        height, width = frames[0].shape[:2]

        # 每個區域挑面積最大的人
        crops, metas = [], []
        for frame, result in zip(frames, person_model(frames, conf=PERSON_CONF, verbose=False)):
            best = {}
            for box in result.boxes.xyxy.tolist():
                region = region_of(box, width)
                if region and (region not in best or box_area(box) > box_area(best[region])):
                    best[region] = box
            for region, box in best.items():
                counts[region] += 1
                x1, y1, x2, y2 = map(int, box)
                crops.append(frame[y1:y2, x1:x2])
                metas.append((region, (x1, y1, x2, y2)))

        for i in range(0, len(crops), BATCH_SIZE):
            results = pose_model(crops[i:i + BATCH_SIZE], conf=POSE_CONF, verbose=False)
            for (region, box), result in zip(metas[i:i + BATCH_SIZE], results):
                roi = hip_roi(box, result)
                if roi:
                    samples[region].append(roi)

        frame_count += len(frames)
        update_progress(bar, len(frames))
    cap.release()

    ratios = {name: counts[name] / frame_count for name in REGIONS}
    region = max(ratios, key=ratios.get)
    if ratios[region] < PRESENCE_THRESHOLD or not samples[region]:
        return frame_count, ratios, None
    return frame_count, ratios, square_roi(samples[region], width, height)


def crop_video(video, output, roi, bar):
    x1, y1, size = roi
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        cap.get(cv2.CAP_PROP_FPS),
        (OUTPUT_SIZE, OUTPUT_SIZE),
    )
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y1:y1 + size, x1:x1 + size]
        writer.write(cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA))
        update_progress(bar, 1)
    cap.release()
    writer.release()


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
    total = sum(2 * int(cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FRAME_COUNT)) for v in videos)

    failed = []   # (影片, 錯誤訊息)
    futures = []  # (影片, 裁切 future)

    with tqdm(total=total, unit="step") as bar, ThreadPoolExecutor(MAX_WORKERS) as pool:
        for video in videos:
            relative = video.relative_to(INPUT_DIR)
            try:
                frame_count, ratios, roi = analyze_video(video, person_model, pose_model, bar)
            except Exception as exc:
                failed.append((relative, f"分析失敗：{exc}"))
                tqdm.write(f"{relative.as_posix()}：分析失敗，跳過")
                continue
            tqdm.write(
                f"{relative.as_posix()}：{frame_count} frames，"
                + "、".join(f"{name}={ratio:.2%}" for name, ratio in ratios.items())
            )

            if roi is None:
                update_progress(bar, frame_count)  # 補上不需裁切的步數
                tqdm.write("  略過：存在率不足或無有效 hip ROI")
                continue

            output = OUTPUT_DIR / relative.parent / f"{video.stem}_upper_body.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            futures.append((relative, pool.submit(crop_video, video, output, roi, bar)))

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
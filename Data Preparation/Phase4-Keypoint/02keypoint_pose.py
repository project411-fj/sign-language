import os
import time
import queue
import threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import cv2
import numpy as np
from ultralytics import YOLO

from tqdm import tqdm


# 要處理的 part 在執行時輸入：「3」、「1,2,5」或「all」(keypoint_videos 底下所有 part)
INPUT_ROOT = Path("Data Preparation") / ("videos") / "keypoint_videos"
OUTPUT_ROOT = Path("Data Preparation") / ("Phase4-Keypoint") / ("Keypoint_output") / "01pose"

MODEL_PATH = "yolo26n-pose.pt"
YOLO_DEVICE = "0"         

BATCH_SIZE = 32            
PREFETCH_BATCHES = 4       
GPU_WORKERS = 4            
CPU_WORKERS = 8            
USE_HALF = True            

PERSON_CONF = 0.5
KEYPOINT_CONF = 0.5
IMAGE_SIZE = 640

POSE_IDS = [0, 5, 6, 7, 8, 9, 10, 11, 12]

TARGET_CENTER = (0.5, 0.5)
MAX_TRACK_DISTANCE = 0.25

_pose_model = None
IS_CPU = YOLO_DEVICE.lower() == "cpu"

PRECISION_KWARGS = {"quantize": 16} if (USE_HALF and not IS_CPU) else {}


def init_worker():
    global _pose_model
    cv2.setNumThreads(1)

    import torch
    torch.set_num_threads(1)

    _pose_model = YOLO(MODEL_PATH)

    dummy = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    _pose_model.predict(
        source=[dummy], device=YOLO_DEVICE, imgsz=IMAGE_SIZE,
        verbose=False, **PRECISION_KWARGS,
    )


def parse_result(result, width, height, previous_center):
    output = np.zeros(28, dtype=np.float32)

    if result.boxes is None or result.keypoints is None:
        return output, previous_center

    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy()

    if len(boxes) == 0 or len(keypoints) == 0:
        return output, previous_center

    num_people = min(len(boxes), len(keypoints))

    centers = np.column_stack((
        (boxes[:num_people, 0] + boxes[:num_people, 2]) / (2 * width),
        (boxes[:num_people, 1] + boxes[:num_people, 3]) / (2 * height),
    ))

    reference = np.array(
        TARGET_CENTER if previous_center is None else previous_center,
        dtype=np.float32,
    )

    distances = np.linalg.norm(centers - reference, axis=1)
    person_idx = int(np.argmin(distances))

    if previous_center is not None and distances[person_idx] > MAX_TRACK_DISTANCE:
        return output, previous_center

    points = keypoints[person_idx, POSE_IDS].copy()

    if points.shape[1] < 3:
        raise ValueError("YOLO Pose 未輸出 Keypoint confidence")

    valid = points[:, 2] >= KEYPOINT_CONF
    points[~valid] = 0.0
    points[valid, 0] = np.clip(points[valid, 0] / width, 0.0, 1.0)
    points[valid, 1] = np.clip(points[valid, 1] / height, 0.0, 1.0)

    output[:27] = points.reshape(-1)
    output[27] = float(np.any(valid))

    return output, tuple(centers[person_idx])


def frame_reader(capture, q, stop_event):
    """背景執行緒：持續讀取影片並以批次放入 queue（I/O 與解碼和推論重疊）。"""
    try:
        batch = []
        while not stop_event.is_set():
            success, frame = capture.read()
            if not success:
                break
            batch.append(frame)
            if len(batch) == BATCH_SIZE:
                q.put(batch)
                batch = []
        if batch:
            q.put(batch)
    finally:
        q.put(None) 


def process_video(input_path, output_path):
    start_time = time.perf_counter()

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"無法開啟影片：{input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    q = queue.Queue(maxsize=PREFETCH_BATCHES)
    stop_event = threading.Event()
    reader = threading.Thread(
        target=frame_reader, args=(capture, q, stop_event), daemon=True
    )
    reader.start()

    all_frames = []
    previous_center = None

    try:
        while True:
            batch = q.get()
            if batch is None:
                break

            results = _pose_model.predict(
                source=batch,
                device=YOLO_DEVICE,
                imgsz=IMAGE_SIZE,
                conf=PERSON_CONF,
                verbose=False,
                **PRECISION_KWARGS,
            )

            for result in results:
                features, previous_center = parse_result(
                    result, width, height, previous_center
                )
                all_frames.append(features)
    finally:
        stop_event.set()
        # 清空 queue，避免 reader 卡在 put
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        reader.join(timeout=5)
        capture.release()

    if not all_frames:
        raise RuntimeError(f"影片沒有可讀取的 Frame：{input_path}")

    data = np.stack(all_frames).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data)

    elapsed = time.perf_counter() - start_time

    return {
        "video": f"{input_path.parent.name}/{input_path.name}",
        "shape": data.shape,
        "seconds": elapsed,
        "speed": len(all_frames) / elapsed if elapsed > 0 else 0.0,
        "pid": os.getpid(),
    }


def video_sort_key(path):
    # <子資料夾>/<start>_<end>_upper_body.mp4
    parts = path.stem.split("_")
    return int(path.parent.name), int(parts[0]), int(parts[1])


def run_part(part):
    input_path, output_root = INPUT_ROOT / part, OUTPUT_ROOT / part
    print(f"\n===== part {part}")
    video_paths = sorted(input_path.glob("*/*_upper_body.mp4"), key=video_sort_key)

    if not video_paths:
        print(f"找不到 MP4 影片：{input_path}")
        return

    tasks = []
    for video_path in video_paths:
        # 1/000600_001507_upper_body.mp4 → 1/000600_001507.npy
        output_name = video_path.stem.removesuffix("_upper_body") + ".npy"
        output_path = output_root / video_path.parent.name / output_name
        # 已處理過的跳過，方便中斷後續跑
        if output_path.exists():
            continue
        tasks.append((video_path, output_path))

    if not tasks:
        print("所有影片都已處理完成")
        return

    max_workers = CPU_WORKERS if IS_CPU else GPU_WORKERS
    workers = max(1, min(max_workers, os.cpu_count() or 1, len(tasks)))

    print(f"待處理影片: {len(tasks)}")
    print(f"使用 Process 數量: {workers}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"YOLO 裝置: {YOLO_DEVICE}")

    YOLO(MODEL_PATH)

    start_time = time.perf_counter()
    failures = 0

    progress = tqdm(total=len(tasks), desc=f"pose part {part}", unit="video")
    log = tqdm.write

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        futures = {
            executor.submit(process_video, video, output): video
            for video, output in tasks
        }

        for future in as_completed(futures):
            video = futures[future]
            try:
                result = future.result()
                log(
                    f"完成: {result['video']} | PID {result['pid']} | "
                    f"Shape {result['shape']} | {result['seconds']:.2f} 秒 | "
                    f"{result['speed']:.2f} Frames/s"
                )
            except Exception as exc:
                failures += 1
                log(f"失敗: {video.parent.name}/{video.name} | 原因: {exc}")
            finally:
                progress.set_postfix(失敗=failures)
                progress.update(1)

    progress.close()

    total_time = time.perf_counter() - start_time
    print(f"\n全部處理結束，總耗時: {total_time:.2f} 秒")
    print(f"成功: {len(tasks) - failures}，失敗: {failures}")


def parse_parts(answer, root):
    """「3」→ ["3"]；「1,2,5」→ ["1", "2", "5"]；「all」→ root 底下所有 part (依數字排序)。"""
    answer = answer.strip()
    if answer.lower() == "all":
        return sorted((p.name for p in root.iterdir() if p.is_dir()), key=lambda x: (len(x), x))
    return [p.strip() for p in answer.split(",") if p.strip()]


def main():
    parts = parse_parts(input("part (例如 3、1,2,5 或 all): "), INPUT_ROOT)
    if not parts:
        print("沒有輸入 part")
        return
    print(f"要處理的 part：{'、'.join(parts)}")
    for part in parts:
        run_part(part)


if __name__ == "__main__":
    freeze_support()
    main()

import cv2
import mediapipe as mp
import numpy as np
import os
import time

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

INPUT_PATH = Path("yolo_pose_mediapie") / "cropped_videos_v3"
OUTPUT_PATH = Path("yolo_pose_mediapie") / "hand_keypoints"
MODEL_PATH = Path("yolo_pose_mediapie") / "hand_landmarker.task"

NORMALIZE = True
MAX_WORKERS = 8


def normalize_hand(points):
    wrist = points[0]
    centered = points - wrist
    scale = np.linalg.norm(points[9] - wrist) + 1e-6
    return centered / scale


def extract_frame(result):
    hands = np.zeros((2, 21, 3), dtype=np.float32)
    present = np.zeros(2, dtype=np.float32)

    for landmarks, handedness in zip(
        result.hand_landmarks,
        result.handedness
    ):
        side = handedness[0].category_name

        if side not in ("Left", "Right"):
            continue

        hand_idx = 0 if side == "Left" else 1

        points = np.array(
            [[p.x, p.y, p.z] for p in landmarks],
            dtype=np.float32
        )

        if NORMALIZE:
            points = normalize_hand(points)

        hands[hand_idx] = points
        present[hand_idx] = 1.0

    return np.concatenate([
        hands.reshape(-1),
        present
    ])


def process_video(input_path, output_path):
    start_time = time.perf_counter()

    cv2.setNumThreads(1)

    capture = cv2.VideoCapture(str(input_path))
    fps = capture.get(cv2.CAP_PROP_FPS)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    all_frames = []
    frame_idx = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:
            success, frame = capture.read()

            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )

            timestamp_ms = int(frame_idx * 1000 / fps)

            result = landmarker.detect_for_video(
                image,
                timestamp_ms
            )

            all_frames.append(extract_frame(result))
            frame_idx += 1

    capture.release()

    data = np.stack(all_frames).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data)

    elapsed = time.perf_counter() - start_time

    return {
        "video": input_path.name,
        "shape": data.shape,
        "seconds": elapsed,
        "speed": frame_idx / elapsed,
        "pid": os.getpid()
    }


def video_sort_key(path):
    parts = path.stem.split("_")
    return int(parts[1]), int(parts[2])


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        INPUT_PATH.glob("*.mp4"),
        key=video_sort_key
    )

    tasks = []

    for video_path in video_paths:
        output_name = "_".join(
            video_path.stem.split("_")[:3]
        ) + ".npy"

        output_path = OUTPUT_PATH / output_name
        tasks.append((video_path, output_path))

    workers = min(MAX_WORKERS, os.cpu_count() or 1, len(tasks))

    print(f"待處理影片: {len(tasks)}")
    print(f"使用 Process 數量: {workers}")

    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_video, video, output): video
            for video, output in tasks
        }

        for future in as_completed(futures):
            result = future.result()

            print(
                f"\n完成: {result['video']}"
                f"\nPID: {result['pid']}"
                f"\nShape: {result['shape']}"
                f"\n時間: {result['seconds']:.2f} 秒"
                f"\n速度: {result['speed']:.2f} Frames/s"
            )

    total_time = time.perf_counter() - start_time
    print(f"\n全部處理結束，總耗時: {total_time:.2f} 秒")


if __name__ == "__main__":
    freeze_support()
    main()

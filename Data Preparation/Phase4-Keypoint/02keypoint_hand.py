import os

# 關閉 MediaPipe / TFLite 的 log（必須在 import mediapipe 之前設定）
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

PART = "12"  # 要處理的 part：videos/keypoint_videos/<PART>/

INPUT_PATH = Path("Data Preparation") / ("videos") / "keypoint_videos" / PART
OUTPUT_PATH = Path("Data Preparation") / ("Phase4-Keypoint") / ("Keypoint_output") / "01hand" / PART
MODEL_PATH = Path("Data Preparation") / ("models") / ("mediapipe_models") / "hand_landmarker.task"

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


def init_worker():
    """子 Process：把 stdout / stderr 導向 devnull，擋掉 C++ 層直接印出的訊息。"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)


def process_video(input_path, output_path):
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


def video_sort_key(path):
    # <子資料夾>/<start>_<end>_upper_body.mp4
    parts = path.stem.split("_")
    return int(path.parent.name), int(parts[0]), int(parts[1])


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        INPUT_PATH.glob("*/*_upper_body.mp4"),
        key=video_sort_key
    )

    tasks = []

    for video_path in video_paths:
        # 1/000600_001507_upper_body.mp4 → 1/000600_001507.npy
        output_name = video_path.stem.removesuffix("_upper_body") + ".npy"

        output_path = OUTPUT_PATH / video_path.parent.name / output_name
        tasks.append((video_path, output_path))

    workers = min(MAX_WORKERS, os.cpu_count() or 1, len(tasks))

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        futures = [
            executor.submit(process_video, video, output)
            for video, output in tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="hand", unit="video"):
            future.result()


if __name__ == "__main__":
    freeze_support()
    main()
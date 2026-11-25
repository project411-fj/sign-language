"""STT 版：MediaPipe 雙手 21 點 → Keypoint_output_STT/01hand (每格 128 維)

走 STT 路徑的影片，Phase1 切出來的 part_videos 本身就只有手語員 (等同 OCR 版 01cropV2.py 裁好的 keypoint_videos)，
所以不用裁切，直接讀 part_videos/<part>/<n>/<start>_<end>.mp4。
處理方式和 OCR/02keypoint_hand.py 完全一樣，只是輸入改成 part_videos、輸出改到 Keypoint_output_STT/，不會和 OCR 版混在一起。
"""
import os
import re

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

# 要處理的 part 在執行時輸入：「3」、「1,2,5」或「all」(part_videos 底下所有 part)
INPUT_ROOT = Path("Data Preparation") / ("videos") / "part_videos"
OUTPUT_ROOT = Path("Data Preparation") / ("Phase4-Keypoint") / ("Keypoint_output_STT") / "01hand"
SKIP_EXISTING = True   # 已經有輸出的影片就跳過 (輸入 all 時不會把舊的 part 重做一次)
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
    # <子資料夾>/<start>_<end>.mp4
    parts = path.stem.split("_")
    return int(path.parent.name), int(parts[0]), int(parts[1])


def list_videos(folder):
    """part_videos/<part>/<n>/<start>_<end>.mp4 (只取「數字_數字」的檔名，略過暫存檔)。"""
    return sorted((p for p in folder.glob("*/*.mp4") if re.fullmatch(r"\d+_\d+", p.stem)), key=video_sort_key)


def run_part(part):
    input_path, output_path_root = INPUT_ROOT / part, OUTPUT_ROOT / part
    if not input_path.is_dir():
        print(f"part {part}：找不到 {input_path}，跳過")
        return

    video_paths = list_videos(input_path)
    tasks = []
    for video_path in video_paths:
        # 1/000600_001507.mp4 → 1/000600_001507.npy
        output_name = video_path.stem + ".npy"
        output_path = output_path_root / video_path.parent.name / output_name
        if SKIP_EXISTING and output_path.exists():
            continue
        tasks.append((video_path, output_path))

    print(f"part {part}：共 {len(video_paths)} 支影片，這次要處理 {len(tasks)} 支")
    if not tasks:
        return

    workers = min(MAX_WORKERS, os.cpu_count() or 1, len(tasks))
    failed = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        futures = {executor.submit(process_video, video, output): video for video, output in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"hand part {part}", unit="video"):
            try:
                future.result()
            except Exception as exc:          # 一支失敗不影響其他影片
                failed.append((futures[future], exc))

    if failed:
        tqdm.write(f"\npart {part} 失敗 {len(failed)} 支 (下次執行會自動重做，因為沒有輸出檔)：")
        for video, exc in failed:
            tqdm.write(f"  {video.as_posix()}：{exc}")


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
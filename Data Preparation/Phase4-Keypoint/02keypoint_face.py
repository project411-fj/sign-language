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

# 要處理的 part 在執行時輸入：「3」、「1,2,5」或「all」(keypoint_videos 底下所有 part)
INPUT_ROOT = Path("Data Preparation") / ("videos") / "keypoint_videos"
OUTPUT_ROOT = Path("Data Preparation") / ("Phase4-Keypoint") / ("Keypoint_output") / "01face"
MODEL_PATH = Path("Data Preparation") / ("models") / ("mediapipe_models") / "face_landmarker.task"

MAX_WORKERS = 8
SKIP_EXISTING = True   # 已經有輸出的影片就跳過 (中斷後可以接著跑)

# ---------------------------------------------------------------- 輸出格式 (每格 136 維)
#   0:52    52 個表情分數 (blendshapes，0~1)：眉毛、眼睛、嘴巴形狀
#   52:132  嘴唇 40 點的 (x, y)：以兩邊嘴角的中點為原點、除以兩眼外眼角的距離
#   132:135 頭部轉向 (pitch 點頭、yaw 轉頭、roll 歪頭，單位：弧度)
#   135     是否偵測到臉 (0 / 1)
N_BLENDSHAPES = 52
LIPS = [
    # 外唇
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    # 內唇
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]
MOUTH_CORNERS = (61, 291)
EYE_OUTER = (33, 263)
FACE_DIM = N_BLENDSHAPES + len(LIPS) * 2 + 3 + 1   # 136


def head_angles(matrix):
    """從 4x4 臉部轉換矩陣算出 (pitch, yaw, roll)。"""
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    rot = rot / (np.linalg.norm(rot, axis=0, keepdims=True) + 1e-9)   # 去掉縮放
    pitch = np.arctan2(rot[2, 1], rot[2, 2])
    yaw = np.arctan2(-rot[2, 0], np.hypot(rot[2, 1], rot[2, 2]))
    roll = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([pitch, yaw, roll], dtype=np.float32)


def extract_frame(result, width, height):
    output = np.zeros(FACE_DIM, dtype=np.float32)
    if not result.face_landmarks:
        return output

    # 座標換成像素，避免畫面不是正方形時 x、y 比例不同
    points = np.array([[p.x * width, p.y * height] for p in result.face_landmarks[0]], dtype=np.float32)
    center = (points[MOUTH_CORNERS[0]] + points[MOUTH_CORNERS[1]]) / 2
    scale = np.linalg.norm(points[EYE_OUTER[0]] - points[EYE_OUTER[1]]) + 1e-6
    lips = (points[LIPS] - center) / scale

    if result.face_blendshapes:
        output[:N_BLENDSHAPES] = [c.score for c in result.face_blendshapes[0][:N_BLENDSHAPES]]
    output[N_BLENDSHAPES:N_BLENDSHAPES + len(LIPS) * 2] = lips.reshape(-1)
    if result.facial_transformation_matrixes:
        output[-4:-1] = head_angles(result.facial_transformation_matrixes[0])
    output[-1] = 1.0
    return output


def init_worker():
    """子 Process：把 stdout / stderr 導向 devnull，擋掉 C++ 層直接印出的訊息。"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)


def process_video(input_path, output_path):
    cv2.setNumThreads(1)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"無法開啟影片：{input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    all_frames = []
    frame_idx = 0
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = capture.read()
            if not success:
                break
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(image, int(frame_idx * 1000 / fps))
            all_frames.append(extract_frame(result, width, height))
            frame_idx += 1

    capture.release()
    if not all_frames:
        raise RuntimeError(f"影片沒有任何 frame：{input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.stack(all_frames).astype(np.float32))


def video_sort_key(path):
    # <子資料夾>/<start>_<end>_upper_body.mp4
    parts = path.stem.split("_")
    return int(path.parent.name), int(parts[0]), int(parts[1])


def run_part(part):
    input_path, output_root = INPUT_ROOT / part, OUTPUT_ROOT / part
    failed_log = Path(__file__).parent / f"failed_face_part_{part}.txt"
    video_paths = sorted(input_path.glob("*/*_upper_body.mp4"), key=video_sort_key)
    tasks = []
    for video_path in video_paths:
        # 1/000600_001507_upper_body.mp4 → 1/000600_001507.npy
        output_name = video_path.stem.removesuffix("_upper_body") + ".npy"
        output_path = output_root / video_path.parent.name / output_name
        if SKIP_EXISTING and output_path.exists():
            continue
        tasks.append((video_path, output_path))

    print(f"part {part}：共 {len(video_paths)} 支影片，這次要處理 {len(tasks)} 支")
    if not tasks:
        failed_log.unlink(missing_ok=True)
        return

    workers = min(MAX_WORKERS, os.cpu_count() or 1, len(tasks))
    failed = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        futures = {executor.submit(process_video, video, output): video for video, output in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"face part {part}", unit="video"):
            try:
                future.result()
            except Exception as exc:
                failed.append((futures[future], exc))

    # 有失敗就寫紀錄，全部成功就刪除紀錄
    if failed:
        failed_log.write_text("".join(f"{video.as_posix()}\n" for video, _ in failed), encoding="utf-8")
        tqdm.write(f"\npart {part} 失敗 {len(failed)} 個，已記錄到 {failed_log.name}：")
        for video, exc in failed:
            tqdm.write(f"  {video.as_posix()}：{exc}")
    else:
        failed_log.unlink(missing_ok=True)


def parse_parts(answer, root):
    """「3」→ ["3"]；「1,2,5」→ ["1", "2", "5"]；「all」→ root 底下所有 part (依數字排序)。"""
    answer = answer.strip()
    if answer.lower() == "all":
        return sorted((p.name for p in root.iterdir() if p.is_dir()), key=lambda x: (len(x), x))
    return [p.strip() for p in answer.split(",") if p.strip()]


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"找不到模型：{MODEL_PATH}\n請先下載 face_landmarker.task 放到這個位置")
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

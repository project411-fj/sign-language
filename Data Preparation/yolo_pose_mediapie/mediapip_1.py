
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


DEFAULT_INPUT_DIR = Path("yolo_pose_mediapie/cropped_videos_v3")
DEFAULT_OUTPUT_DIR = Path("yolo_pose_mediapie/pose_analysis_v3")
DEFAULT_POSE_MODEL = Path("yolo26n-pose.pt")
DEFAULT_HOLISTIC_MODEL = Path("mediapipe_models/holistic_landmarker.task")
HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

YOLO_CONFIDENCE = 0.50
MEDIAPIPE_DETECTION_CONFIDENCE = 0.50
MEDIAPIPE_TRACKING_CONFIDENCE = 0.50
MEDIAPIPE_POSE_VISIBILITY = 0.50

YOLO_COLOR = (0, 220, 0)
MEDIAPIPE_POSE_COLOR = (0, 165, 255)
LEFT_HAND_COLOR = (255, 160, 0)
RIGHT_HAND_COLOR = (40, 40, 255)

YOLO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

YOLO_CONNECTIONS = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

MEDIAPIPE_POSE_LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

MEDIAPIPE_POSE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)

MEDIAPIPE_HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze crop2.py output videos with MediaPipe Holistic and "
            "YOLO Pose, then save annotated videos and normalized coordinates."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input MP4 directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=DEFAULT_POSE_MODEL,
        help=f"YOLO pose model path (default: {DEFAULT_POSE_MODEL})",
    )
    parser.add_argument(
        "--holistic-model",
        type=Path,
        default=DEFAULT_HOLISTIC_MODEL,
        help=(
            "MediaPipe Tasks Holistic model path. It is downloaded "
            f"automatically if missing (default: {DEFAULT_HOLISTIC_MODEL})"
        ),
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=YOLO_CONFIDENCE,
        help=f"YOLO detection confidence (default: {YOLO_CONFIDENCE})",
    )
    return parser.parse_args()


def validate_confidence(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("--yolo-confidence must be between 0 and 1")


def ensure_holistic_model(model_path: Path) -> Path:
    """Download the official MediaPipe Tasks model when it is missing."""
    model_path = model_path.expanduser().resolve()
    if model_path.is_file():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".download")
    print("Downloading MediaPipe model / 下載 MediaPipe 模型:")
    print(HOLISTIC_MODEL_URL)
    try:
        urlretrieve(HOLISTIC_MODEL_URL, temporary_path)
        temporary_path.replace(model_path)
    except (OSError, URLError) as error:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(
            "Unable to download holistic_landmarker.task. Download it "
            f"manually from {HOLISTIC_MODEL_URL} and save it as "
            f"{model_path}"
        ) from error
    return model_path


class LegacyHolisticRunner:
    """Compatibility wrapper for MediaPipe 0.10.21 and older."""

    def __init__(self) -> None:
        self._holistic = None

    def __enter__(self):
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
        )
        return self

    def process(self, rgb_frame: np.ndarray, timestamp_ms: int):
        del timestamp_ms
        return self._holistic.process(rgb_frame)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._holistic is not None:
            self._holistic.close()


class TasksHolisticRunner:
    """MediaPipe 0.10.31+ wrapper using the supported Tasks API."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._landmarker = None

    def __enter__(self):
        vision = mp.tasks.vision
        options = vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self._model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_pose_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_pose_landmarks_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
            min_hand_landmarks_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(options)
        return self

    def process(self, rgb_frame: np.ndarray, timestamp_ms: int):
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        return self._landmarker.detect_for_video(image, timestamp_ms)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._landmarker is not None:
            self._landmarker.close()


def create_holistic_runner(model_path: Path):
    if (
        hasattr(mp, "solutions")
        and hasattr(mp.solutions, "holistic")
    ):
        print("MediaPipe API: legacy Solutions")
        return LegacyHolisticRunner()

    if (
        hasattr(mp, "tasks")
        and hasattr(mp.tasks, "vision")
        and hasattr(mp.tasks.vision, "HolisticLandmarker")
    ):
        print("MediaPipe API: Tasks HolisticLandmarker")
        return TasksHolisticRunner(ensure_holistic_model(model_path))

    raise RuntimeError(
        "This MediaPipe installation provides neither mp.solutions.holistic "
        "nor mp.tasks.vision.HolisticLandmarker. Upgrade MediaPipe with: "
        "python -m pip install --upgrade mediapipe"
    )


def safe_fps(cap: cv2.VideoCapture) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        return 30.0
    return fps


def normalized_side(name: str) -> str:
    if name.startswith("left_"):
        return "left"
    if name.startswith("right_"):
        return "right"
    return "center"


def in_frame(x: float, y: float, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x, y = position
    cv2.rectangle(
        frame,
        (x - 3, y - text_height - 4),
        (x + text_width + 3, y + baseline + 3),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_legend(frame: np.ndarray) -> None:
    entries = (
        ("YOLO Pose", YOLO_COLOR),
        ("MediaPipe Pose", MEDIAPIPE_POSE_COLOR),
        ("Left Hand", LEFT_HAND_COLOR),
        ("Right Hand", RIGHT_HAND_COLOR),
    )
    y = 24
    for label, color in entries:
        draw_text_with_background(frame, label, (10, y), color)
        y += 24


def select_largest_yolo_person(result) -> int | None:
    keypoints = result.keypoints
    if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
        return None

    person_count = len(keypoints.data)
    if result.boxes is not None and len(result.boxes) == person_count:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
            0.0, boxes[:, 3] - boxes[:, 1]
        )
        return int(np.argmax(areas))

    data = keypoints.data.detach().cpu().numpy()
    if data.shape[2] >= 3:
        return int(np.argmax(np.mean(data[:, :, 2], axis=1)))
    return 0


def write_row(
    text_file,
    frame_index: int,
    timestamp_sec: float,
    model: str,
    side: str,
    landmark_id: int,
    landmark_name: str,
    x_norm: float,
    y_norm: float,
    z_norm: float | None,
    confidence: float | None,
) -> None:
    z_text = "" if z_norm is None else f"{z_norm:.8f}"
    confidence_text = "" if confidence is None else f"{confidence:.8f}"
    text_file.write(
        f"{frame_index}\t{timestamp_sec:.6f}\t{model}\t{side}\t"
        f"{landmark_id}\t{landmark_name}\t{x_norm:.8f}\t{y_norm:.8f}\t"
        f"{z_text}\t{confidence_text}\n"
    )


def process_yolo_pose(
    frame: np.ndarray,
    result,
    text_file,
    frame_index: int,
    timestamp_sec: float,
    confidence_threshold: float,
) -> None:
    height, width = frame.shape[:2]
    person_index = select_largest_yolo_person(result)
    if person_index is None:
        return

    keypoint_data = (
        result.keypoints.data[person_index].detach().cpu().numpy()
    )
    points: dict[int, tuple[int, int]] = {}

    for landmark_id, values in enumerate(keypoint_data):
        if landmark_id >= len(YOLO_KEYPOINT_NAMES):
            break

        x = float(values[0])
        y = float(values[1])
        confidence = float(values[2]) if len(values) >= 3 else 1.0
        if not math.isfinite(x) or not math.isfinite(y):
            continue

        name = YOLO_KEYPOINT_NAMES[landmark_id]
        write_row(
            text_file,
            frame_index,
            timestamp_sec,
            "yolo_pose",
            normalized_side(name),
            landmark_id,
            name,
            float(np.clip(x / width, 0.0, 1.0)),
            float(np.clip(y / height, 0.0, 1.0)),
            None,
            confidence,
        )

        if confidence >= confidence_threshold and in_frame(x, y, width, height):
            points[landmark_id] = (int(round(x)), int(round(y)))

    for start, end in YOLO_CONNECTIONS:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], YOLO_COLOR, 2, cv2.LINE_AA)

    for point in points.values():
        cv2.circle(frame, point, 4, YOLO_COLOR, -1, cv2.LINE_AA)


def draw_normalized_landmarks(
    frame: np.ndarray,
    landmarks: Sequence,
    connections: Iterable[tuple[int, int]],
    color: tuple[int, int, int],
    visibility_threshold: float | None = None,
) -> None:
    height, width = frame.shape[:2]
    points: dict[int, tuple[int, int]] = {}

    for landmark_id, landmark in enumerate(landmarks):
        visibility = getattr(landmark, "visibility", None)
        if visibility is None:
            visibility = 1.0
        if visibility_threshold is not None and visibility < visibility_threshold:
            continue

        if landmark.x is None or landmark.y is None:
            continue
        x = float(landmark.x) * width
        y = float(landmark.y) * height
        if in_frame(x, y, width, height):
            points[landmark_id] = (int(round(x)), int(round(y)))

    for start, end in connections:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)

    for point in points.values():
        cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)


def write_mediapipe_landmarks(
    text_file,
    frame_index: int,
    timestamp_sec: float,
    model: str,
    side: str,
    landmark_names: Sequence[str],
    landmarks: Sequence,
    include_visibility: bool,
) -> None:
    for landmark_id, landmark in enumerate(landmarks):
        if landmark_id >= len(landmark_names):
            break
        if landmark.x is None or landmark.y is None:
            continue
        visibility = getattr(landmark, "visibility", None)
        confidence = (
            float(visibility)
            if include_visibility and visibility is not None
            else None
        )
        z_value = getattr(landmark, "z", None)
        write_row(
            text_file,
            frame_index,
            timestamp_sec,
            model,
            side,
            landmark_id,
            landmark_names[landmark_id],
            float(np.clip(landmark.x, 0.0, 1.0)),
            float(np.clip(landmark.y, 0.0, 1.0)),
            None if z_value is None else float(z_value),
            confidence,
        )


def get_landmark_sequence(value):
    """Accept legacy LandmarkList objects and Tasks API Python lists."""
    if value is None:
        return None
    if hasattr(value, "landmark"):
        value = value.landmark
    return value if len(value) > 0 else None


def process_mediapipe(
    frame: np.ndarray,
    results,
    text_file,
    frame_index: int,
    timestamp_sec: float,
    pose_landmark_names: Sequence[str],
    pose_connections: Iterable[tuple[int, int]],
    hand_connections: Iterable[tuple[int, int]],
) -> None:
    pose = get_landmark_sequence(results.pose_landmarks)
    if pose is not None:
        write_mediapipe_landmarks(
            text_file,
            frame_index,
            timestamp_sec,
            "mediapipe_pose",
            "body",
            pose_landmark_names,
            pose,
            include_visibility=True,
        )
        draw_normalized_landmarks(
            frame,
            pose,
            pose_connections,
            MEDIAPIPE_POSE_COLOR,
            visibility_threshold=MEDIAPIPE_POSE_VISIBILITY,
        )

    for side, hand, color in (
        (
            "left",
            get_landmark_sequence(results.left_hand_landmarks),
            LEFT_HAND_COLOR,
        ),
        (
            "right",
            get_landmark_sequence(results.right_hand_landmarks),
            RIGHT_HAND_COLOR,
        ),
    ):
        if hand is None:
            continue
        write_mediapipe_landmarks(
            text_file,
            frame_index,
            timestamp_sec,
            "mediapipe_hand",
            side,
            HAND_LANDMARK_NAMES,
            hand,
            include_visibility=False,
        )
        draw_normalized_landmarks(
            frame,
            hand,
            hand_connections,
            color,
        )


def process_video(
    input_video: Path,
    output_video: Path,
    output_text: Path,
    yolo_model: YOLO,
    yolo_confidence: float,
    holistic_model: Path,
) -> bool:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        print(f"Cannot open video / 無法開啟影片: {input_video}")
        return False

    fps = safe_fps(cap)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        print(f"Invalid video size / 無效影片尺寸: {input_video}")
        return False

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        print(f"Cannot create video / 無法建立影片: {output_video}")
        return False

    frame_index = 0
    with output_text.open("w", encoding="utf-8", newline="") as text_file:
        text_file.write(f"# source_video\t{input_video.resolve()}\n")
        text_file.write(f"# frame_width\t{width}\n")
        text_file.write(f"# frame_height\t{height}\n")
        text_file.write(f"# fps\t{fps:.8f}\n")
        text_file.write(
            "# x_norm and y_norm are normalized to the analyzed video frame; "
            "MediaPipe z_norm is relative depth, while YOLO z_norm is blank.\n"
        )
        text_file.write(
            "frame_index\ttimestamp_sec\tmodel\tside\tlandmark_id\t"
            "landmark_name\tx_norm\ty_norm\tz_norm\tconfidence\n"
        )

        with create_holistic_runner(holistic_model) as holistic:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_index += 1
                timestamp_sec = (frame_index - 1) / fps
                annotated = frame.copy()

                yolo_result = yolo_model.predict(
                    frame,
                    conf=yolo_confidence,
                    verbose=False,
                )[0]
                process_yolo_pose(
                    annotated,
                    yolo_result,
                    text_file,
                    frame_index,
                    timestamp_sec,
                    yolo_confidence,
                )

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                timestamp_ms = int(round(timestamp_sec * 1000.0))
                mediapipe_results = holistic.process(rgb, timestamp_ms)
                process_mediapipe(
                    annotated,
                    mediapipe_results,
                    text_file,
                    frame_index,
                    timestamp_sec,
                    MEDIAPIPE_POSE_LANDMARK_NAMES,
                    MEDIAPIPE_POSE_CONNECTIONS,
                    MEDIAPIPE_HAND_CONNECTIONS,
                )

                draw_legend(annotated)
                draw_text_with_background(
                    annotated,
                    f"Frame {frame_index}/{total_frames}",
                    (10, height - 12),
                    (255, 255, 255),
                )
                writer.write(annotated)
                print(
                    f"\rProcessing / 處理中: {frame_index}/{total_frames}",
                    end="",
                    flush=True,
                )

    cap.release()
    writer.release()
    print()
    print(f"Annotated video / 標註影片: {output_video}")
    print(f"Coordinates / 座標資料: {output_text}")
    return frame_index > 0


def main() -> None:
    args = parse_args()
    validate_confidence(args.yolo_confidence)

    if not args.input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist / 輸入資料夾不存在: "
            f"{args.input_dir.resolve()}"
        )

    video_files = sorted(args.input_dir.glob("*.mp4"))
    if not video_files:
        print(
            f"No MP4 files found / 找不到 MP4: {args.input_dir.resolve()}"
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    yolo_model = YOLO(str(args.pose_model))

    print(f"Videos found / 找到影片: {len(video_files)}")
    successful = 0
    for video_number, input_video in enumerate(video_files, start=1):
        stem = input_video.stem
        output_video = args.output_dir / f"{stem}_pose.mp4"
        output_text = args.output_dir / f"{stem}_coordinates.txt"
        print(
            f"\n[{video_number}/{len(video_files)}] "
            f"Analyzing / 分析: {input_video}"
        )
        if process_video(
            input_video,
            output_video,
            output_text,
            yolo_model,
            args.yolo_confidence,
            args.holistic_model,
        ):
            successful += 1

    print(
        f"\nFinished / 完成: {successful}/{len(video_files)} video(s)"
    )


if __name__ == "__main__":
    main()

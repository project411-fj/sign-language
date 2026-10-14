
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

SOURCE_DIRECTORY = Path("source_videos")
OUTPUT_DIRECTORY = Path("time_stamp_v2")

PERSON_MODEL = "person.pt"
POSE_MODEL = "yolo26n-pose.pt"

PERSON_CONF = 0.5
POSE_CONF = 0.5
KEYPOINT_CONF = 0.5

OUTPUT_SIZE = 640

BOX_ALPHA = 0.15
KEYPOINT_ALPHA = 0.15
ROI_ALPHA = 0.15

SIDE_PADDING_RATIO = 0.10
BOTTOM_PADDING_RATIO = 0.05

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

model = YOLO(PERSON_MODEL)
pose_model = YOLO(POSE_MODEL)


def ema(previous, current, alpha):
    if previous is None:
        return current

    return previous * (1 - alpha) + current * alpha


def smooth_box(previous_box, current_box, alpha):
    if previous_box is None:
        return current_box

    return tuple(
        ema(previous, current, alpha)
        for previous, current in zip(previous_box, current_box)
    )


def resize_with_padding(image, target_size=640):
    h, w = image.shape[:2]

    if h == 0 or w == 0:
        return None

    scale = min(target_size / w, target_size / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


cap = cv2.VideoCapture(str(SOURCE_DIRECTORY / "1.mp4"))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    str(OUTPUT_DIRECTORY / "1.mp4"),
    fourcc,
    fps,
    (OUTPUT_SIZE, OUTPUT_SIZE)
)

previous_person_box = None
previous_left_hip_y = None
previous_right_hip_y = None
previous_roi = None

with tqdm(total=total_frames) as pbar:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        pbar.update(1)

        original_frame = frame.copy()

        results = model(frame, conf=PERSON_CONF, verbose=False)

        current_person_box = None
        largest_area = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                area = (x2 - x1) * (y2 - y1)

                if area > largest_area:
                    largest_area = area
                    current_person_box = (x1, y1, x2, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f"Person {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        if current_person_box is not None:
            previous_person_box = smooth_box(
                previous_person_box,
                current_person_box,
                BOX_ALPHA
            )

        upper_body = frame

        if previous_person_box is not None:
            x1, y1, x2, y2 = previous_person_box

            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(width, x2))
            y2 = int(min(height, y2))

            if x2 > x1 and y2 > y1:
                person_crop = original_frame[y1:y2, x1:x2]

                if person_crop.size > 0:
                    person_h, person_w = person_crop.shape[:2]

                    pose_results = pose_model(
                        person_crop,
                        conf=POSE_CONF,
                        verbose=False
                    )

                    pose_success = False

                    if (
                        pose_results[0].keypoints is not None
                        and len(pose_results[0].keypoints.data) > 0
                    ):
                        keypoints = pose_results[0].keypoints.data[0].cpu().numpy()

                        left_hip = keypoints[11]
                        right_hip = keypoints[12]

                        left_conf = float(left_hip[2])
                        right_conf = float(right_hip[2])

                        if left_conf >= KEYPOINT_CONF and right_conf >= KEYPOINT_CONF:
                            left_hip_y = y1 + float(left_hip[1])
                            right_hip_y = y1 + float(right_hip[1])

                            previous_left_hip_y = ema(
                                previous_left_hip_y,
                                left_hip_y,
                                KEYPOINT_ALPHA
                            )

                            previous_right_hip_y = ema(
                                previous_right_hip_y,
                                right_hip_y,
                                KEYPOINT_ALPHA
                            )

                            pose_success = True

                    if (
                        pose_success
                        and previous_left_hip_y is not None
                        and previous_right_hip_y is not None
                    ):
                        hip_y = max(
                            previous_left_hip_y,
                            previous_right_hip_y
                        )

                        side_padding = person_w * SIDE_PADDING_RATIO
                        bottom_padding = person_h * BOTTOM_PADDING_RATIO

                        roi_x1 = max(0, x1 - side_padding)
                        roi_y1 = max(0, y1)
                        roi_x2 = min(width, x2 + side_padding)
                        roi_y2 = min(height, hip_y + bottom_padding)

                        current_roi = (
                            roi_x1,
                            roi_y1,
                            roi_x2,
                            roi_y2
                        )

                        previous_roi = smooth_box(
                            previous_roi,
                            current_roi,
                            ROI_ALPHA
                        )

                    if previous_roi is None:
                        upper_body = frame[y1:y2, x1:x2]

        if previous_roi is not None:
            rx1, ry1, rx2, ry2 = previous_roi

            rx1 = int(max(0, rx1))
            ry1 = int(max(0, ry1))
            rx2 = int(min(width, rx2))
            ry2 = int(min(height, ry2))

            if rx2 > rx1 and ry2 > ry1:
                upper_body = frame[ry1:ry2, rx1:rx2]

        output = resize_with_padding(upper_body, OUTPUT_SIZE)

        if output is None:
            output = resize_with_padding(frame, OUTPUT_SIZE)

        out.write(output)

cap.release()
out.release()

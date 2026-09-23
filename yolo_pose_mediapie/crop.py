import cv2
import numpy as np
from ultralytics import YOLO

INPUT_VIDEO = r"part_videos\part_1_1.mp4"
OUTPUT_VIDEO = r"yolo_pose_mediapie\output_upper_body.mp4"

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

person_model = YOLO(PERSON_MODEL)
pose_model = YOLO(POSE_MODEL)


def ema(previous, current, alpha):
    if previous is None:
        return current

    return previous * (1 - alpha) + current * alpha


def smooth_box(previous_box, current_box, alpha):
    if previous_box is None:
        return current_box

    px1, py1, px2, py2 = previous_box
    cx1, cy1, cx2, cy2 = current_box

    x1 = ema(px1, cx1, alpha)
    y1 = ema(py1, cy1, alpha)
    x2 = ema(px2, cx2, alpha)
    y2 = ema(py2, cy2, alpha)

    return (x1, y1, x2, y2)


def resize_with_padding(image, target_size=640):
    h, w = image.shape[:2]

    if h == 0 or w == 0:
        return None

    scale = min(target_size / w, target_size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((target_size, target_size, 3),dtype=np.uint8)

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    canvas[y_offset:y_offset + new_h,x_offset:x_offset + new_w] = resized

    return canvas


cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("原始尺寸:", frame_width, "x", frame_height)
print("總 Frame:", total_frames)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(OUTPUT_VIDEO,fourcc,fps,(OUTPUT_SIZE, OUTPUT_SIZE))

previous_person_box = None

previous_left_hip_y = None
previous_right_hip_y = None

previous_roi = None

frame_id = 0
written_frames = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    print(f"\rProcessing: {frame_id}/{total_frames}", end="")

    person_results = person_model(frame,conf=PERSON_CONF,verbose=False)

    current_person_box = None
    largest_area = 0

    if len(person_results[0].boxes) > 0:

        for box in person_results[0].boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            area = (x2 - x1) * (y2 - y1)

            if area > largest_area:
                largest_area = area

                current_person_box = (float(x1),float(y1),float(x2),float(y2))

    if current_person_box is not None:
        previous_person_box = smooth_box(previous_person_box,current_person_box,BOX_ALPHA)

    if previous_person_box is None:

        output = resize_with_padding(frame, OUTPUT_SIZE)

        writer.write(output)
        written_frames += 1

        continue

    x1, y1, x2, y2 = previous_person_box

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))

    x2 = int(min(frame_width, x2))
    y2 = int(min(frame_height, y2))

    if x2 <= x1 or y2 <= y1:

        output = resize_with_padding(frame, OUTPUT_SIZE)

        writer.write(output)
        written_frames += 1

        continue

    person_crop = frame[y1:y2, x1:x2]

    if person_crop.size == 0:

        output = resize_with_padding(frame, OUTPUT_SIZE)

        writer.write(output)
        written_frames += 1

        continue

    person_h, person_w = person_crop.shape[:2]

    pose_results = pose_model(person_crop,conf=POSE_CONF,verbose=False)

    pose_success = False

    if (
        pose_results[0].keypoints is not None
        and len(pose_results[0].keypoints.data) > 0
    ):

        keypoints = pose_results[0].keypoints.data[0].cpu().numpy()

        left_hip = keypoints[11]
        right_hip = keypoints[12]

        left_hip_y = float(left_hip[1])
        right_hip_y = float(right_hip[1])

        left_conf = float(left_hip[2])
        right_conf = float(right_hip[2])

        if left_conf >= KEYPOINT_CONF and right_conf >= KEYPOINT_CONF:

            previous_left_hip_y = ema(previous_left_hip_y,left_hip_y,KEYPOINT_ALPHA)

            previous_right_hip_y = ema(previous_right_hip_y,right_hip_y,KEYPOINT_ALPHA)

            pose_success = True

    if (
        pose_success
        and previous_left_hip_y is not None
        and previous_right_hip_y is not None
    ):

        hip_y = max(previous_left_hip_y, previous_right_hip_y)

        bottom_padding = person_h * BOTTOM_PADDING_RATIO

        crop_bottom = hip_y + bottom_padding
        crop_bottom = min(crop_bottom, person_h)

        side_padding = person_w * SIDE_PADDING_RATIO

        roi_x1 = x1 - side_padding
        roi_x2 = x2 + side_padding

        roi_y1 = y1
        roi_y2 = y1 + crop_bottom

        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)

        roi_x2 = min(frame_width, roi_x2)
        roi_y2 = min(frame_height, roi_y2)

        current_roi = (roi_x1,roi_y1,roi_x2,roi_y2)

        previous_roi = smooth_box(previous_roi,current_roi,ROI_ALPHA)

    if previous_roi is not None:

        rx1, ry1, rx2, ry2 = previous_roi

        rx1 = int(max(0, rx1))
        ry1 = int(max(0, ry1))

        rx2 = int(min(frame_width, rx2))
        ry2 = int(min(frame_height, ry2))

        if rx2 > rx1 and ry2 > ry1:
            upper_body = frame[ry1:ry2, rx1:rx2]
        else:
            upper_body = frame

    else:
        upper_body = person_crop

    output = resize_with_padding(upper_body, OUTPUT_SIZE)

    if output is not None:
        writer.write(output)
        written_frames += 1


cap.release()
writer.release()

print()
print("==============================")
print("處理完成")
print("==============================")

print("原始 Frame:", total_frames)
print("輸出 Frame:", written_frames)
print("輸出影片:", OUTPUT_VIDEO)
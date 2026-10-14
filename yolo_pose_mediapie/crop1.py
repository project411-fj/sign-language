import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO


INPUT_DIR = r"part_videos"
OUTPUT_DIR = r"yolo_pose_mediapipe/cropped_videos_v2"

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

# People whose center is in one of these horizontal regions are counted.
# Values are percentages of the video width.
PERSISTENCE_REGIONS = {
    "left": (0.00, 0.35),
    "right": (0.65, 1.00),
}
PRESENCE_THRESHOLD = 0.9


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

    return (
        ema(px1, cx1, alpha),
        ema(py1, cy1, alpha),
        ema(px2, cx2, alpha),
        ema(py2, cy2, alpha),
    )


def resize_with_padding(image, target_size=640):
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def detect_people(frame):
    """Return all detected person boxes as (x1, y1, x2, y2)."""
    results = person_model(frame, conf=PERSON_CONF, verbose=False)
    boxes = []

    if len(results[0].boxes) == 0:
        return boxes

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        boxes.append((float(x1), float(y1), float(x2), float(y2)))

    return boxes


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def horizontal_center_ratio(box, frame_width):
    x1, _, x2, _ = box
    return ((x1 + x2) / 2.0) / frame_width


def region_for_position(x_ratio):
    for region_name, (start_ratio, end_ratio) in PERSISTENCE_REGIONS.items():
        # Include 1.00 in the final region without double-counting boundaries.
        if start_ratio <= x_ratio < end_ratio or (
            end_ratio == 1.0 and x_ratio == 1.0
        ):
            return region_name
    return None


def analyze_region_presence(input_video):
    """
    First pass: count the frames in which each region contains at least one
    person. A region is counted at most once per frame.
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return 0, {}, {}

    frame_count = 0
    region_frame_counts = {name: 0 for name in PERSISTENCE_REGIONS}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_width = frame.shape[1]
        boxes = detect_people(frame)

        occupied_regions = set()
        for box in boxes:
            x_ratio = horizontal_center_ratio(box, frame_width)
            region_name = region_for_position(x_ratio)
            if region_name is not None:
                occupied_regions.add(region_name)

        for region_name in occupied_regions:
            region_frame_counts[region_name] += 1

        print(f"\rAnalyzing regions: {frame_count}", end="")

    cap.release()
    print()

    if frame_count == 0:
        return 0, region_frame_counts, {
            name: 0.0 for name in PERSISTENCE_REGIONS
        }

    region_ratios = {
        name: count / frame_count
        for name, count in region_frame_counts.items()
    }
    return frame_count, region_frame_counts, region_ratios


def choose_target_region(region_ratios):
    """Choose the most persistent region that reaches the 95% threshold."""
    qualified = {
        name: ratio
        for name, ratio in region_ratios.items()
        if ratio >= PRESENCE_THRESHOLD
    }
    if not qualified:
        return None
    return max(qualified, key=qualified.get)


def select_person_in_region(boxes, frame_width, target_region):
    """
    Select the largest detected person whose center lies in the target
    persistent region.
    """
    candidates = []
    for box in boxes:
        x_ratio = horizontal_center_ratio(box, frame_width)
        if region_for_position(x_ratio) == target_region:
            candidates.append(box)

    if not candidates:
        return None
    return max(candidates, key=box_area)


def crop_video(input_video, output_video, target_region):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("無法開啟影片：", input_video)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (OUTPUT_SIZE, OUTPUT_SIZE),
    )

    if not writer.isOpened():
        cap.release()
        print("無法建立輸出影片：", output_video)
        return False

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

        current_person_box = select_person_in_region(
            detect_people(frame),
            frame_width,
            target_region,
        )

        # Keep the last valid location through a short detector miss. This does
        # not introduce a non-qualified person because the initial location had
        # to come from the selected persistent region.
        if current_person_box is not None:
            previous_person_box = smooth_box(
                previous_person_box,
                current_person_box,
                BOX_ALPHA,
            )

        if previous_person_box is None:
            output = resize_with_padding(frame, OUTPUT_SIZE)
            if output is not None:
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
            if output is not None:
                writer.write(output)
                written_frames += 1
            continue

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            output = resize_with_padding(frame, OUTPUT_SIZE)
            if output is not None:
                writer.write(output)
                written_frames += 1
            continue

        person_h, person_w = person_crop.shape[:2]
        pose_results = pose_model(person_crop, conf=POSE_CONF, verbose=False)
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
                previous_left_hip_y = ema(
                    previous_left_hip_y,
                    left_hip_y,
                    KEYPOINT_ALPHA,
                )
                previous_right_hip_y = ema(
                    previous_right_hip_y,
                    right_hip_y,
                    KEYPOINT_ALPHA,
                )
                pose_success = True

        if (
            pose_success
            and previous_left_hip_y is not None
            and previous_right_hip_y is not None
        ):
            hip_y = max(previous_left_hip_y, previous_right_hip_y)
            crop_bottom = min(
                hip_y + person_h * BOTTOM_PADDING_RATIO,
                person_h,
            )
            side_padding = person_w * SIDE_PADDING_RATIO

            current_roi = (
                max(0, x1 - side_padding),
                max(0, y1),
                min(frame_width, x2 + side_padding),
                min(frame_height, y1 + crop_bottom),
            )
            previous_roi = smooth_box(previous_roi, current_roi, ROI_ALPHA)

        if previous_roi is not None:
            rx1, ry1, rx2, ry2 = previous_roi
            rx1 = int(max(0, rx1))
            ry1 = int(max(0, ry1))
            rx2 = int(min(frame_width, rx2))
            ry2 = int(min(frame_height, ry2))

            if rx2 > rx1 and ry2 > ry1:
                upper_body = frame[ry1:ry2, rx1:rx2]
            else:
                upper_body = person_crop
        else:
            upper_body = person_crop

        output = resize_with_padding(upper_body, OUTPUT_SIZE)
        if output is not None:
            writer.write(output)
            written_frames += 1

    cap.release()
    writer.release()
    print()
    print("原始 Frame：", total_frames)
    print("輸出 Frame：", written_frames)
    print("選定區域：", target_region)
    print("輸出影片：", output_video)
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    print("找到影片數量：", len(video_files))

    for video_index, input_video in enumerate(video_files, start=1):
        file_name = os.path.basename(input_video)
        file_stem = os.path.splitext(file_name)[0]
        output_video = os.path.join(
            OUTPUT_DIR,
            f"{file_stem}_upper_body.mp4",
        )

        print(f"\n[{video_index}/{len(video_files)}] 分析影片：{input_video}")
        frame_count, region_counts, region_ratios = analyze_region_presence(
            input_video
        )

        print("有效分析 Frame：", frame_count)
        for region_name in PERSISTENCE_REGIONS:
            start_ratio, end_ratio = PERSISTENCE_REGIONS[region_name]
            print(
                f"{region_name} "
                f"({start_ratio:.0%}-{end_ratio:.0%})："
                f"{region_counts.get(region_name, 0)}/{frame_count} "
                f"({region_ratios.get(region_name, 0.0):.2%})"
            )

        target_region = choose_target_region(region_ratios)
        if target_region is None:
            print(
                f"略過：沒有區域達到 {PRESENCE_THRESHOLD:.0%} 的人物存在率。"
            )
            continue

        print(
            f"開始裁切：{target_region} 區域，存在率 "
            f"{region_ratios[target_region]:.2%}"
        )
        crop_video(input_video, output_video, target_region)

    print("\n所有影片處理完成")


if __name__ == "__main__":
    main()

import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO


INPUT_DIR = r"part_videos"
OUTPUT_DIR = r"yolo_pose_mediapie/cropped_videos_v3"

PERSON_MODEL = "person.pt"
POSE_MODEL = "yolo26n-pose.pt"

PERSON_CONF = 0.5
POSE_CONF = 0.5
KEYPOINT_CONF = 0.5

OUTPUT_SIZE = 640

SIDE_PADDING_RATIO = 0.10
BOTTOM_PADDING_RATIO = 0.05

ROI_SIZE_PERCENTILE = 99.0

PERSISTENCE_REGIONS = {
    "left": (0.00, 0.35),
    "right": (0.65, 1.00),
}
PRESENCE_THRESHOLD = 0.9


person_model = YOLO(PERSON_MODEL)
pose_model = YOLO(POSE_MODEL)


def detect_people(frame):
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
        if start_ratio <= x_ratio < end_ratio or (
            end_ratio == 1.0 and x_ratio == 1.0
        ):
            return region_name
    return None


def choose_target_region(region_ratios):
    qualified = {
        name: ratio
        for name, ratio in region_ratios.items()
        if ratio >= PRESENCE_THRESHOLD
    }
    if not qualified:
        return None
    return max(qualified, key=qualified.get)


def select_person_in_region(boxes, frame_width, target_region):
    candidates = []
    for box in boxes:
        x_ratio = horizontal_center_ratio(box, frame_width)
        if region_for_position(x_ratio) == target_region:
            candidates.append(box)

    if not candidates:
        return None
    return max(candidates, key=box_area)


def upper_body_roi_from_pose(frame, person_box):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = person_box

    x1 = int(max(0, np.floor(x1)))
    y1 = int(max(0, np.floor(y1)))
    x2 = int(min(frame_width, np.ceil(x2)))
    y2 = int(min(frame_height, np.ceil(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        return None

    person_height, person_width = person_crop.shape[:2]
    pose_results = pose_model(person_crop, conf=POSE_CONF, verbose=False)

    if (
        pose_results[0].keypoints is None
        or len(pose_results[0].keypoints.data) == 0
    ):
        return None

    keypoints = pose_results[0].keypoints.data[0].cpu().numpy()
    if len(keypoints) <= 12:
        return None

    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_conf = float(left_hip[2])
    right_conf = float(right_hip[2])

    if left_conf < KEYPOINT_CONF or right_conf < KEYPOINT_CONF:
        return None

    hip_y_in_crop = max(float(left_hip[1]), float(right_hip[1]))
    bottom_y = y1 + hip_y_in_crop + person_height * BOTTOM_PADDING_RATIO
    side_padding = person_width * SIDE_PADDING_RATIO

    return (
        max(0.0, x1 - side_padding),
        float(y1),
        min(float(frame_width), x2 + side_padding),
        min(float(frame_height), bottom_y),
    )


def get_square_roi(frame_width, frame_height, center_x, center_y, crop_size):
    crop_size = int(max(1, min(crop_size, frame_width, frame_height)))

    x1 = int(round(center_x - crop_size / 2.0))
    y1 = int(round(center_y - crop_size / 2.0))

    x1 = max(0, min(x1, frame_width - crop_size))
    y1 = max(0, min(y1, frame_height - crop_size))

    return x1, y1, x1 + crop_size, y1 + crop_size


def calculate_fixed_square_roi(
    roi_samples,
    frame_width,
    frame_height,
    percentile=ROI_SIZE_PERCENTILE,
):
    """Calculate one robust fixed center and square size for the video."""
    if not roi_samples:
        return None

    samples = np.asarray(roi_samples, dtype=np.float64)
    centers_x = (samples[:, 0] + samples[:, 2]) / 2.0
    centers_y = (samples[:, 1] + samples[:, 3]) / 2.0

    requested_center_x = float(np.median(centers_x))
    requested_center_y = float(np.median(centers_y))

    required_sizes = 2.0 * np.maximum.reduce(
        [
            requested_center_x - samples[:, 0],
            samples[:, 2] - requested_center_x,
            requested_center_y - samples[:, 1],
            samples[:, 3] - requested_center_y,
        ]
    )
    required_sizes = required_sizes[np.isfinite(required_sizes)]
    required_sizes = required_sizes[required_sizes > 0]
    if len(required_sizes) == 0:
        return None

    crop_size = int(np.ceil(np.percentile(required_sizes, percentile)))
    crop_size = max(1, min(crop_size, frame_width, frame_height))

    x1, y1, x2, y2 = get_square_roi(
        frame_width,
        frame_height,
        requested_center_x,
        requested_center_y,
        crop_size,
    )

    fixed_center_x = (x1 + x2) / 2.0
    fixed_center_y = (y1 + y2) / 2.0

    return {
        "fixed_center_x": fixed_center_x,
        "fixed_center_y": fixed_center_y,
        "crop_size": crop_size,
        "roi": (x1, y1, x2, y2),
        "sample_count": len(roi_samples),
    }


def analyze_video(input_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("無法開啟影片：", input_video)
        return None

    frame_count = 0
    frame_width = 0
    frame_height = 0
    region_frame_counts = {name: 0 for name in PERSISTENCE_REGIONS}
    region_roi_samples = {name: [] for name in PERSISTENCE_REGIONS}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        boxes = detect_people(frame)

        occupied_regions = set()
        for box in boxes:
            region_name = region_for_position(
                horizontal_center_ratio(box, frame_width)
            )
            if region_name is not None:
                occupied_regions.add(region_name)

        for region_name in occupied_regions:
            region_frame_counts[region_name] += 1

            person_box = select_person_in_region(
                boxes,
                frame_width,
                region_name,
            )
            roi_sample = upper_body_roi_from_pose(frame, person_box)
            if roi_sample is not None:
                region_roi_samples[region_name].append(roi_sample)

        print(f"\rAnalyzing fixed ROI: {frame_count}", end="")

    cap.release()
    print()

    if frame_count == 0:
        return None

    region_ratios = {
        name: count / frame_count
        for name, count in region_frame_counts.items()
    }
    target_region = choose_target_region(region_ratios)

    analysis = {
        "frame_count": frame_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "region_counts": region_frame_counts,
        "region_ratios": region_ratios,
        "target_region": target_region,
        "fixed_roi": None,
    }

    if target_region is not None:
        analysis["fixed_roi"] = calculate_fixed_square_roi(
            region_roi_samples[target_region],
            frame_width,
            frame_height,
        )

    return analysis


def crop_video_with_fixed_roi(input_video, output_video, fixed_roi):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("無法開啟影片：", input_video)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x1, y1, x2, y2 = fixed_roi["roi"]
    crop_size = fixed_roi["crop_size"]

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

    frame_id = 0
    written_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        print(f"\rCropping: {frame_id}/{total_frames}", end="")

        square_crop = frame[y1:y2, x1:x2]
        if square_crop.shape[:2] != (crop_size, crop_size):
            print("\n錯誤：影片幀尺寸改變，無法維持固定正方形 ROI。")
            cap.release()
            writer.release()
            return False

        # The input crop is already square, so direct resize does not distort
        # its aspect ratio and no black padding is needed.
        output = cv2.resize(
            square_crop,
            (OUTPUT_SIZE, OUTPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        writer.write(output)
        written_frames += 1

    cap.release()
    writer.release()
    print()
    print("原始 Frame：", total_frames)
    print("輸出 Frame：", written_frames)
    print("固定中心：", fixed_roi["fixed_center_x"], fixed_roi["fixed_center_y"])
    print("固定 crop_size：", crop_size)
    print("固定 ROI：", fixed_roi["roi"])
    print("輸出影片：", output_video)
    return written_frames > 0


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
        analysis = analyze_video(input_video)
        if analysis is None:
            print("略過：影片無法讀取或沒有有效 Frame。")
            continue

        frame_count = analysis["frame_count"]
        print("有效分析 Frame：", frame_count)
        for region_name in PERSISTENCE_REGIONS:
            start_ratio, end_ratio = PERSISTENCE_REGIONS[region_name]
            count = analysis["region_counts"].get(region_name, 0)
            ratio = analysis["region_ratios"].get(region_name, 0.0)
            print(
                f"{region_name} "
                f"({start_ratio:.0%}-{end_ratio:.0%})："
                f"{count}/{frame_count} ({ratio:.2%})"
            )

        target_region = analysis["target_region"]
        if target_region is None:
            print(
                f"略過：沒有區域達到 {PRESENCE_THRESHOLD:.0%} 的人物存在率。"
            )
            continue

        fixed_roi = analysis["fixed_roi"]
        if fixed_roi is None:
            print("略過：目標人物沒有足夠且可信的 hip 11/12 ROI 樣本。")
            continue

        print(
            f"選定區域：{target_region}，存在率 "
            f"{analysis['region_ratios'][target_region]:.2%}"
        )
        print("ROI 樣本數：", fixed_roi["sample_count"])
        print(
            "固定中心：",
            fixed_roi["fixed_center_x"],
            fixed_roi["fixed_center_y"],
        )
        print("固定 crop_size：", fixed_roi["crop_size"])
        print("開始第二次掃描並使用固定 ROI 裁切")

        crop_video_with_fixed_roi(input_video, output_video, fixed_roi)

    print("\n所有影片處理完成")


if __name__ == "__main__":
    main()

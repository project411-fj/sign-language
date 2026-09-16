from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


SOURCE_DIRECTORY = Path("source_videos")
OUTPUT_DIRECTORY = Path("time_stamp")
CONFIDENCE_THRESHOLD = 0.25
TIE_CONFIDENCE_THRESHOLD = 0.25
TOLERANCE_SECONDS = 2.0
MINIMUM_INTERVAL_SECONDS = 15.0
MAX_CENTER_X_CHANGE_RATIO = 0.1


def person_geometry_matches(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    center_x_ratio = ((x1 + x2) / 2) / frame_width
    height_ratio = (y2 - y1) / frame_height
    return (
        0.15 <= center_x_ratio <= 0.25
        or 0.70 <= center_x_ratio <= 0.80
    ) and height_ratio >= 0.60


def tie_is_inside_person(tie_box, person_box):
    tx1, ty1, tx2, ty2 = tie_box
    px1, py1, px2, py2 = person_box
    tie_center_x = (tx1 + tx2) / 2
    tie_center_y = (ty1 + ty2) / 2
    return px1 <= tie_center_x <= px2 and py1 <= tie_center_y <= py2


def format_timestamp(seconds):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def process_video(video_path, output_path, person_model, tie_model):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"無法取得影片 FPS：{video_path}")

    tolerance_frames = round(fps * TOLERANCE_SECONDS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    matching_ranges = []
    range_start = None
    last_matching_end = None
    missed_frames = 0
    previous_center_x_ratio = None
    frame_index = 0

    progress_bar = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc=video_path.name,
        unit="frame",
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        person_result = person_model(frame, verbose=False)[0]
        tie_result = tie_model(frame, verbose=False)[0]
        frame_height, frame_width = frame.shape[:2]

        tie_boxes = []
        for tie_box in tie_result.boxes:
            if float(tie_box.conf[0]) < TIE_CONFIDENCE_THRESHOLD:
                continue
            coordinates = tie_box.xyxy[0].tolist()
            x1, y1, x2, y2 = coordinates
            if (y2 - y1) <= (x2 - x1):
                continue
            tie_boxes.append(coordinates)

        matching_center_x_ratios = []
        for person_box in person_result.boxes:
            if float(person_box.conf[0]) < CONFIDENCE_THRESHOLD:
                continue
            coordinates = person_box.xyxy[0].tolist()
            geometry_matches = person_geometry_matches(
                coordinates, frame_width, frame_height
            )
            has_tie = any(
                tie_is_inside_person(tie_box, coordinates)
                for tie_box in tie_boxes
            )
            if geometry_matches and has_tie:
                x1, _, x2, _ = coordinates
                matching_center_x_ratios.append(
                    ((x1 + x2) / 2) / frame_width
                )

        current_time = frame_index / fps
        if matching_center_x_ratios:
            if previous_center_x_ratio is None:
                current_center_x_ratio = min(matching_center_x_ratios)
            else:
                current_center_x_ratio = min(
                    matching_center_x_ratios,
                    key=lambda value: abs(value - previous_center_x_ratio),
                )

            if range_start is None:
                range_start = current_time
            elif (
                abs(current_center_x_ratio - previous_center_x_ratio)
                > MAX_CENTER_X_CHANGE_RATIO
            ):
                matching_ranges.append((range_start, last_matching_end))
                range_start = current_time

            last_matching_end = (frame_index + 1) / fps
            previous_center_x_ratio = current_center_x_ratio
            missed_frames = 0
        elif range_start is not None:
            missed_frames += 1
            if missed_frames > tolerance_frames:
                matching_ranges.append((range_start, last_matching_end))
                range_start = None
                last_matching_end = None
                previous_center_x_ratio = None
                missed_frames = 0

        frame_index += 1
        progress_bar.update(1)

    cap.release()
    progress_bar.close()

    if range_start is not None:
        matching_ranges.append((range_start, last_matching_end))

    with output_path.open("w", encoding="utf-8") as output_file:
        for start, end in matching_ranges:
            if end - start < MINIMUM_INTERVAL_SECONDS:
                continue
            output_file.write(
                f"{format_timestamp(start)} - {format_timestamp(end)}\n"
            )


def main():
    video_paths = sorted(SOURCE_DIRECTORY.glob("*.mp4"))
    if not video_paths:
        raise FileNotFoundError(
            f"{SOURCE_DIRECTORY} 目錄中找不到任何 .mp4 檔案。"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    person_model = YOLO("person.pt")
    tie_model = YOLO("ties.pt")

    for video_path in video_paths:
        output_path = OUTPUT_DIRECTORY / f"{video_path.stem}.txt"
        process_video(video_path, output_path, person_model, tie_model)


if __name__ == "__main__":
    main()

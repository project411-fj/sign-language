import cv2

import os

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

INPUT_PATH_DIR= Path("part_videos")
OUTPUT_ROOT = Path("rewrite_Phase2") / "frame_cropped"


def process_video(input_path):
    print(f"Processing {input_path}, PID: {os.getpid()}")

    frame_count = crop(input_path)

    return f"{input_path} done, {frame_count} frames"

if __name__ == "__main__":
    input_paths = sorted(INPUT_PATH_DIR.glob("*.mp4"))

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_video, input_paths))

    for result in results:
        print(result)

def crop(input_path):
    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        print(f"Failed to open video: {input_path}")
        return

    OUTPUT_PATH_DIR = OUTPUT_ROOT / str(input_path.stem+"_frames")
    OUTPUT_PATH_DIR.mkdir(parents=True, exist_ok=True)

    frame_number = 0

    while True:
        success, frame = capture.read()
        
        if not success: break

        frame_number += 1

        height, width = frame.shape[:2]

        x1 = int(width * 0.27)
        x2 = int(width)
        y1 = int(height * 0.805)
        y2 = int(height * 0.88)

        subtitle = frame[y1:y2, x1:x2]

        subtitle = cv2.resize(
            subtitle,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

        image_name = f"frame_{frame_number:06d}.jpg"
        image_path = OUTPUT_PATH_DIR / image_name

        cv2.imwrite(str(image_path), subtitle)

    capture.release()
    return frame_number
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

INPUT_DIR = Path("Data Preparation") / "videos" / "part_videos"          # part_videos/<part>/<n>/<start>_<end>.mp4
OUTPUT_ROOT = Path("Data Preparation") / "Phase3-OCR" / "01frames_crop"    # frames_crop/<part>/<n>/<start>_<end>_frames/


def crop(input_path, output_root):
    output_dir = output_root / f"{input_path.stem}_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    frame_number = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_number += 1
        h, w = frame.shape[:2]
        subtitle = frame[int(h * 0.805):int(h * 0.88), int(w * 0.27):w]
        subtitle = cv2.resize(subtitle, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(output_dir / f"frame_{frame_number:06d}.jpg"), subtitle)
    capture.release()


if __name__ == "__main__":
    num = input("part: ").strip()
    input_paths = sorted((INPUT_DIR / num).glob("*/*.mp4"), key=lambda p: (int(p.parent.name), p.stem))
    with ProcessPoolExecutor(max_workers=3) as executor:
        # 保留子目錄（1, 2, 3…），避免不同影片的片段檔名互相覆蓋
        futures = [executor.submit(crop, p, OUTPUT_ROOT / num / p.parent.name) for p in input_paths]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"part {num}", unit="video"):
            pass

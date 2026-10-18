import easyocr

from pathlib import Path

reader = easyocr.Reader(['ch_tra'], gpu=True)

def scan_frame(video_frame):
    results = reader.readtext(str(video_frame))

    for box, text, confidence in results:
        with open("rewrite_Phase2/EasOCR_text/" + video_frame.parent.stem + ".txt", "a", encoding="utf-8") as f:
            f.write(str(int(video_frame.stem.split("_")[-1])) + ": " + text + "\n")
        print(int(video_frame.stem.split("_")[-1]), text)

if __name__ == "__main__":
    INPUT_PATH_DIR = Path("rewrite_Phase2") / "frame_cropped"
    image_paths = sorted(INPUT_PATH_DIR.rglob("*.jpg"))

    temp = ""
    frame = 0

    for image_path in image_paths:
        scan_frame(image_path)


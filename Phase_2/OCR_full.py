import argparse
from difflib import SequenceMatcher
from pathlib import Path


def extract_ocr(
    video_path, output_path="result.txt", skip_frames=5, confidence=0.5,
    progress_position=4,
):
    import cv2
    import easyocr
    import torch
    from tqdm import tqdm

    video_path = str(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(["ch_tra"], gpu=torch.cuda.is_available())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid video FPS: {video_path}")

    frame_idx, start_frame = 0, None
    previous_texts = []
    try:
        with output_path.open("w", encoding="utf-8") as output, tqdm(
            total=total_frames, desc="Phase 2 OCR frames", unit="frame",
            position=progress_position, leave=False
        ) as progress:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % max(1, skip_frames) == 0:
                    results = reader.readtext(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    texts = [text for _, text, probability in results
                             if probability >= confidence and isinstance(text, str)]
                    if texts:
                        if not previous_texts:
                            previous_texts, start_frame = texts, frame_idx
                        elif SequenceMatcher(None, "\n".join(previous_texts),
                                             "\n".join(texts)).ratio() < 0.3:
                            output.write(f"{start_frame / fps:.2f}s --- {frame_idx / fps:.2f}s ---文字: {previous_texts}\n")
                            previous_texts, start_frame = texts, frame_idx
                frame_idx += 1
                progress.update(1)
            if previous_texts and start_frame is not None:
                output.write(f"{start_frame / fps:.2f}s --- {frame_idx / fps:.2f}s ---文字: {previous_texts}\n")
    finally:
        cap.release()
    return str(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract OCR text from a video")
    parser.add_argument("video_path")
    parser.add_argument("output_path", nargs="?", default="result.txt")
    parser.add_argument("--skip-frames", type=int, default=5)
    args = parser.parse_args(argv)
    print(extract_ocr(args.video_path, args.output_path, args.skip_frames))


if __name__ == "__main__":
    main()

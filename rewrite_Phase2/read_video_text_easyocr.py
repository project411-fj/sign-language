"""Read text from part_videos/part_1_1.mp4 with EasyOCR.

The program displays the video with OCR bounding boxes and shows each box's
left, top, right, and bottom coordinates as percentages of the video frame.
The same information is also printed in the terminal.

Controls:
    Q or Esc: quit
    Space: pause/resume
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent


def find_default_video() -> Path:
    """Support placing this script in the project root or a subdirectory."""
    candidates = (
        SCRIPT_DIR / "part_videos" / "part_1_1.mp4",
        SCRIPT_DIR / "yolo_pose_mediapie" / "part_videos" / "part_1_1.mp4",
        SCRIPT_DIR.parent / "part_videos" / "part_1_1.mp4",
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


DEFAULT_VIDEO = find_default_video()


@dataclass
class TextDetection:
    text: str
    confidence: float
    points: np.ndarray
    left_pct: float
    top_pct: float
    right_pct: float
    bottom_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use EasyOCR to display text and percentage coordinates in a video."
        )
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"Input video (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["ch_tra", "en"],
        help="EasyOCR languages (default: ch_tra en)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Minimum OCR confidence from 0 to 1 (default: 0.40)",
    )
    parser.add_argument(
        "--ocr-every",
        type=int,
        default=5,
        help="Run OCR every N video frames (default: 5)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use a CUDA GPU for EasyOCR when available",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="Optional TrueType/OpenType font used for video text",
    )
    parser.add_argument(
        "--min-left",
        type=float,
        default=27.5,
        help="Minimum left coordinate percentage (default: 27.5)",
    )
    parser.add_argument(
        "--min-top",
        type=float,
        default=79.0,
        help="Minimum top coordinate percentage (default: 79.0)",
    )
    parser.add_argument(
        "--max-bottom",
        type=float,
        default=88.0,
        help="Maximum bottom coordinate percentage (default: 88.0)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if args.ocr_every < 1:
        raise ValueError("--ocr-every must be at least 1")
    for option, value in (
        ("--min-left", args.min_left),
        ("--min-top", args.min_top),
        ("--max-bottom", args.max_bottom),
    ):
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{option} must be between 0 and 100")


def load_font(font_path: Path | None, size: int) -> ImageFont.ImageFont:
    candidates: list[Path] = []
    if font_path is not None:
        candidates.append(font_path.expanduser())
    candidates.extend(
        [
            Path("C:/Windows/Fonts/msjh.ttc"),
            Path("C:/Windows/Fonts/msjhbd.ttc"),
            Path("C:/Windows/Fonts/mingliu.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def parse_detections(
    raw_results: list,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
    min_left_pct: float,
    min_top_pct: float,
    max_bottom_pct: float,
) -> list[TextDetection]:
    detections: list[TextDetection] = []

    for box, text, confidence in raw_results:
        confidence = float(confidence)
        text = str(text).strip()
        if confidence < confidence_threshold or not text:
            continue

        points = np.asarray(box, dtype=np.float32)
        x_values = np.clip(points[:, 0], 0, frame_width - 1)
        y_values = np.clip(points[:, 1], 0, frame_height - 1)
        points[:, 0] = x_values
        points[:, 1] = y_values

        left_pct = float(x_values.min() / frame_width * 100.0)
        top_pct = float(y_values.min() / frame_height * 100.0)
        right_pct = float(x_values.max() / frame_width * 100.0)
        bottom_pct = float(y_values.max() / frame_height * 100.0)

        # Keep only boxes fully inside the requested subtitle area:
        # L >= 27.5%, T >= 79%, and B <= 88% by default.
        if not (
            left_pct >= min_left_pct
            and top_pct >= min_top_pct
            and bottom_pct <= max_bottom_pct
        ):
            continue

        detections.append(
            TextDetection(
                text=text,
                confidence=confidence,
                points=points,
                left_pct=left_pct,
                top_pct=top_pct,
                right_pct=right_pct,
                bottom_pct=bottom_pct,
            )
        )

    return detections


def print_detections(
    frame_index: int,
    timestamp_sec: float,
    detections: list[TextDetection],
) -> None:
    print(f"\nFrame {frame_index} | time={timestamp_sec:.3f}s")
    if not detections:
        print("  No text detected / 未偵測到文字")
        return

    for number, item in enumerate(detections, start=1):
        print(
            f"  [{number}] text={item.text!r} | "
            f"confidence={item.confidence:.3f} | "
            f"left={item.left_pct:.2f}% top={item.top_pct:.2f}% "
            f"right={item.right_pct:.2f}% bottom={item.bottom_pct:.2f}%"
        )


def draw_detections(
    frame: np.ndarray,
    detections: list[TextDetection],
    font: ImageFont.ImageFont,
) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(image)

    for item in detections:
        polygon = [tuple(map(float, point)) for point in item.points]
        draw.line(polygon + [polygon[0]], fill=(0, 255, 0), width=3)

        label = (
            f"{item.text}  "
            f"L:{item.left_pct:.1f}% T:{item.top_pct:.1f}% "
            f"R:{item.right_pct:.1f}% B:{item.bottom_pct:.1f}% "
            f"C:{item.confidence:.2f}"
        )
        text_x = int(item.points[:, 0].min())
        text_y = max(0, int(item.points[:, 1].min()) - 30)
        text_bbox = draw.textbbox((text_x, text_y), label, font=font)
        draw.rectangle(text_bbox, fill=(0, 0, 0))
        draw.text((text_x, text_y), label, font=font, fill=(0, 255, 0))

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    validate_args(args)
    video_path = args.video.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video does not exist / 影片不存在: {video_path}"
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video / 無法開啟影片: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    delay_ms = max(1, int(round(125.0 / fps)))

    print(f"Video / 影片: {video_path}")
    print(f"Frames / 幀數: {total_frames}")
    print(f"FPS: {fps:.3f}")
    print(f"Languages / 語言: {args.languages}")
    print(
        "OCR box filter / 文字框篩選: "
        f"L >= {args.min_left:.1f}%, "
        f"T >= {args.min_top:.1f}%, "
        f"B <= {args.max_bottom:.1f}%"
    )
    print("Loading EasyOCR / 正在載入 EasyOCR...")

    reader = easyocr.Reader(args.languages, gpu=True)
    font = load_font(args.font, size=20)
    frame_index = 0
    detections: list[TextDetection] = []
    paused = False
    displayed_frame: np.ndarray | None = None

    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_index += 1
                height, width = frame.shape[:2]

                if frame_index == 1 or (frame_index - 1) % args.ocr_every == 0:
                    raw_results = reader.readtext(
                        frame,
                        detail=1,
                        paragraph=False,
                    )
                    detections = parse_detections(
                        raw_results,
                        width,
                        height,
                        args.confidence,
                        args.min_left,
                        args.min_top,
                        args.max_bottom,
                    )
                    timestamp_sec = (frame_index - 1) / fps
                    print_detections(frame_index, timestamp_sec, detections)

                displayed_frame = draw_detections(frame, detections, font)
                cv2.putText(
                    displayed_frame,
                    f"Frame {frame_index}/{total_frames} | Q: quit | Space: pause",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if displayed_frame is None:
                break

            cv2.imshow("EasyOCR video text", displayed_frame)
            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

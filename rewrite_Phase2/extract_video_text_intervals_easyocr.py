from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import easyocr
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def find_default_input() -> Path:
    candidates = (
        SCRIPT_DIR / "part_videos",
        SCRIPT_DIR / "yolo_pose_mediapie" / "part_videos",
        SCRIPT_DIR.parent / "part_videos",
    )
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


DEFAULT_INPUT = find_default_input()


def find_project_root() -> Path:
    """Find the project root whether this script is at root or in a subfolder."""
    for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent):
        if (candidate / "rewrite_Phase2").is_dir():
            return candidate
    if SCRIPT_DIR.name.lower() == "yolo_pose_mediapie":
        return SCRIPT_DIR.parent
    return SCRIPT_DIR


DEFAULT_OUTPUT_DIR = find_project_root() / "rewrite_Phase2" / "ocr_text"


@dataclass
class FrameText:
    text: str
    normalized_text: str
    confidence: float
    left_pct: float
    top_pct: float
    right_pct: float
    bottom_pct: float


@dataclass
class TextInterval:
    start_frame: int
    last_seen_frame: int
    start_sec: float
    last_seen_sec: float
    text: str
    normalized_text: str
    best_confidence: float
    confidence_sum: float
    sample_count: int
    left_pct: float
    top_pct: float
    right_pct: float
    bottom_pct: float

    def update(self, frame_index: int, timestamp_sec: float, item: FrameText) -> None:
        self.last_seen_frame = frame_index
        self.last_seen_sec = timestamp_sec
        self.confidence_sum += item.confidence
        self.sample_count += 1
        self.left_pct = min(self.left_pct, item.left_pct)
        self.top_pct = min(self.top_pct, item.top_pct)
        self.right_pct = max(self.right_pct, item.right_pct)
        self.bottom_pct = max(self.bottom_pct, item.bottom_pct)

        # Keep the spelling from the highest-confidence OCR sample.
        if item.confidence > self.best_confidence:
            self.text = item.text
            self.normalized_text = item.normalized_text
            self.best_confidence = item.confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract EasyOCR text intervals without opening a window."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input MP4 file or directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=None,
        help=(
            "Directory for TSV-style TXT files "
            "named <video_stem>.txt (default: rewrite_Phase2/ocr_text)."
        ),
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
        help="Run OCR every N frames (default: 5)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use a CUDA GPU for EasyOCR",
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
        "--max-right",
        type=float,
        default=100.0,
        help="Maximum right coordinate percentage (default: 100.0)",
    )
    parser.add_argument(
        "--max-bottom",
        type=float,
        default=88.0,
        help="Maximum bottom coordinate percentage (default: 88.0)",
    )
    parser.add_argument(
        "--text-similarity",
        type=float,
        default=0.85,
        help=(
            "Similarity needed to merge OCR variants into one interval "
            "(default: 0.85)"
        ),
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.50,
        help=(
            "Maximum seconds without matching text before closing an interval "
            "(default: 0.50)"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if args.ocr_every < 1:
        raise ValueError("--ocr-every must be at least 1")
    if not 0.0 <= args.text_similarity <= 1.0:
        raise ValueError("--text-similarity must be between 0 and 1")
    if args.max_gap < 0.0:
        raise ValueError("--max-gap must be zero or greater")
    for option, value in (
        ("--min-left", args.min_left),
        ("--min-top", args.min_top),
        ("--max-right", args.max_right),
        ("--max-bottom", args.max_bottom),
    ):
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{option} must be between 0 and 100")
    if args.min_left >= args.max_right:
        raise ValueError("--min-left must be less than --max-right")
    if args.min_top >= args.max_bottom:
        raise ValueError("--min-top must be less than --max-bottom")


def normalize_text(text: str) -> str:
    """Normalize OCR text for stable interval matching and later comparison."""
    text = re.sub(r"\s+", "", text).strip().lower()
    return text


def text_similarity(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    return SequenceMatcher(None, first, second).ratio()


def crop_ocr_region(
    frame: np.ndarray,
    min_left_pct: float,
    min_top_pct: float,
    max_right_pct: float,
    max_bottom_pct: float,
) -> tuple[np.ndarray, int, int]:
    """Crop the configured subtitle area before running EasyOCR.

    The returned offsets allow OCR coordinates from the cropped image to be
    translated back to coordinates in the original video frame.
    """
    frame_height, frame_width = frame.shape[:2]
    left_px = int(np.ceil(frame_width * min_left_pct / 100.0))
    top_px = int(np.ceil(frame_height * min_top_pct / 100.0))
    right_px = int(np.floor(frame_width * max_right_pct / 100.0))
    bottom_px = int(np.floor(frame_height * max_bottom_pct / 100.0))

    left_px = min(max(left_px, 0), frame_width - 1)
    top_px = min(max(top_px, 0), frame_height - 1)
    right_px = min(max(right_px, left_px + 1), frame_width)
    bottom_px = min(max(bottom_px, top_px + 1), frame_height)

    return frame[top_px:bottom_px, left_px:right_px], left_px, top_px


def extract_frame_text(
    raw_results: list,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
    min_left_pct: float,
    min_top_pct: float,
    max_right_pct: float,
    max_bottom_pct: float,
    x_offset: int = 0,
    y_offset: int = 0,
) -> FrameText | None:
    accepted: list[tuple[float, float, str, float, float, float, float, float]] = []

    for box, text, confidence in raw_results:
        text = re.sub(r"\s+", " ", str(text)).strip()
        confidence = float(confidence)
        if not text or confidence < confidence_threshold:
            continue

        # EasyOCR coordinates are relative to the cropped subtitle image.
        # Translate them back so saved percentages still describe the full
        # video frame.
        points = np.asarray(box, dtype=np.float32).copy()
        points[:, 0] += x_offset
        points[:, 1] += y_offset
        x_values = np.clip(points[:, 0], 0, frame_width - 1)
        y_values = np.clip(points[:, 1], 0, frame_height - 1)

        left_pct = float(x_values.min() / frame_width * 100.0)
        top_pct = float(y_values.min() / frame_height * 100.0)
        right_pct = float(x_values.max() / frame_width * 100.0)
        bottom_pct = float(y_values.max() / frame_height * 100.0)

        if not (
            left_pct >= min_left_pct
            and top_pct >= min_top_pct
            and right_pct <= max_right_pct
            and bottom_pct <= max_bottom_pct
        ):
            continue

        accepted.append(
            (
                top_pct,
                left_pct,
                text,
                confidence,
                left_pct,
                top_pct,
                right_pct,
                bottom_pct,
            )
        )

    if not accepted:
        return None

    # Reading order: upper lines first, then left to right.
    accepted.sort(key=lambda item: (round(item[0] / 2.0), item[1]))
    combined_text = " ".join(item[2] for item in accepted)
    normalized = normalize_text(combined_text)
    if not normalized:
        return None

    return FrameText(
        text=combined_text,
        normalized_text=normalized,
        confidence=float(np.mean([item[3] for item in accepted])),
        left_pct=min(item[4] for item in accepted),
        top_pct=min(item[5] for item in accepted),
        right_pct=max(item[6] for item in accepted),
        bottom_pct=max(item[7] for item in accepted),
    )


def start_interval(
    frame_index: int,
    timestamp_sec: float,
    item: FrameText,
) -> TextInterval:
    return TextInterval(
        start_frame=frame_index,
        last_seen_frame=frame_index,
        start_sec=timestamp_sec,
        last_seen_sec=timestamp_sec,
        text=item.text,
        normalized_text=item.normalized_text,
        best_confidence=item.confidence,
        confidence_sum=item.confidence,
        sample_count=1,
        left_pct=item.left_pct,
        top_pct=item.top_pct,
        right_pct=item.right_pct,
        bottom_pct=item.bottom_pct,
    )


def interval_to_row(
    segment_id: int,
    source_video: Path,
    interval: TextInterval,
    sample_duration: float,
    video_duration: float,
    ocr_every: int,
    total_frames: int,
) -> dict[str, str | int]:
    end_sec = min(video_duration, interval.last_seen_sec + sample_duration)
    end_frame = interval.last_seen_frame + ocr_every - 1
    if total_frames > 0:
        end_frame = min(total_frames, end_frame)
    return {
        "segment_id": segment_id,
        "source_video": source_video.name,
        "start_frame": interval.start_frame,
        "end_frame": end_frame,
        "start_sec": f"{interval.start_sec:.6f}",
        "end_sec": f"{end_sec:.6f}",
        "duration_sec": f"{max(0.0, end_sec - interval.start_sec):.6f}",
        "text": interval.text.replace("\t", " ").replace("\r", " ").replace("\n", " "),
        "mean_confidence": f"{interval.confidence_sum / interval.sample_count:.6f}",
        "left_pct": f"{interval.left_pct:.3f}",
        "top_pct": f"{interval.top_pct:.3f}",
        "right_pct": f"{interval.right_pct:.3f}",
        "bottom_pct": f"{interval.bottom_pct:.3f}",
    }


def write_intervals(output_path: Path, rows: list[dict[str, str | int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        for row in rows:
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            text = str(row["text"])
            destination.write(f"{start_sec:.3f} - {end_sec:.3f}: {text}\n")


def find_video_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".mp4":
            raise ValueError(f"Not an MP4 file / 不是 MP4 檔案: {input_path}")
        return [input_path]

    if input_path.is_dir():
        videos = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4"
        )
        if not videos:
            raise FileNotFoundError(
                f"No MP4 files found / 目錄中找不到 MP4: {input_path}"
            )
        return videos

    raise FileNotFoundError(
        f"Input does not exist / 輸入檔案或目錄不存在: {input_path}"
    )


def process_video(
    video_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    reader,
    video_number: int,
    video_count: int,
) -> int:
    output_path = output_dir / f"{video_path.stem}.txt"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video / 無法開啟影片: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if total_frames > 0 else 0.0
    sample_duration = args.ocr_every / fps

    print()
    print(f"[{video_number}/{video_count}] Video / 影片: {video_path}")
    print(f"Output / 輸出: {output_path}")
    print(f"Frames / 幀數: {total_frames}")
    print(f"FPS: {fps:.3f}")
    completed: list[TextInterval] = []
    active: TextInterval | None = None
    frame_index = 0

    try:
        while True:
            # Advance every frame to keep timestamps accurate, but only
            # retrieve a full BGR image for frames that will actually be sent
            # to OCR. This avoids unnecessary image conversion/copying for
            # skipped frames.
            ok = capture.grab()
            if not ok:
                break
            frame_index += 1

            if frame_index != 1 and (frame_index - 1) % args.ocr_every != 0:
                continue

            ok, frame = capture.retrieve()
            if not ok:
                raise RuntimeError(
                    f"Cannot decode frame / 無法解碼影格: {frame_index}"
                )

            timestamp_sec = (frame_index - 1) / fps
            height, width = frame.shape[:2]
            ocr_frame, x_offset, y_offset = crop_ocr_region(
                frame,
                args.min_left,
                args.min_top,
                args.max_right,
                args.max_bottom,
            )
            raw_results = reader.readtext(
                ocr_frame,
                detail=1,
                paragraph=False,
            )
            item = extract_frame_text(
                raw_results,
                width,
                height,
                args.confidence,
                args.min_left,
                args.min_top,
                args.max_right,
                args.max_bottom,
                x_offset,
                y_offset,
            )

            if item is None:
                if active is not None and timestamp_sec - active.last_seen_sec > args.max_gap:
                    completed.append(active)
                    active = None
            elif active is None:
                active = start_interval(frame_index, timestamp_sec, item)
            else:
                similarity = text_similarity(
                    active.normalized_text,
                    item.normalized_text,
                )
                gap = timestamp_sec - active.last_seen_sec
                if similarity >= args.text_similarity and gap <= args.max_gap:
                    active.update(frame_index, timestamp_sec, item)
                else:
                    completed.append(active)
                    active = start_interval(frame_index, timestamp_sec, item)

            shown_text = item.text if item is not None else "-"
            print(
                f"\r[{video_number}/{video_count}] Processing / 處理中: "
                f"{frame_index}/{total_frames} "
                f"time={timestamp_sec:.2f}s text={shown_text[:40]!r}",
                end="",
                flush=True,
            )
    finally:
        capture.release()

    if active is not None:
        completed.append(active)

    if video_duration <= 0.0:
        video_duration = frame_index / fps

    rows = [
        interval_to_row(
            segment_id=index,
            source_video=video_path,
            interval=interval,
            sample_duration=sample_duration,
            video_duration=video_duration,
            ocr_every=args.ocr_every,
            total_frames=total_frames,
        )
        for index, interval in enumerate(completed, start=1)
    ]
    write_intervals(output_path, rows)

    print()
    print(f"Intervals / 文字區間: {len(rows)}")
    print(f"Saved / 已儲存: {output_path}")
    return len(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)

    input_path = args.input_path.expanduser().resolve()
    video_files = find_video_files(input_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input / 輸入: {input_path}")
    print(f"MP4 files / MP4 數量: {len(video_files)}")
    print(f"Output directory / 輸出目錄: {output_dir}")
    print(f"OCR every / OCR 間隔: {args.ocr_every} frame(s)")
    print(
        "OCR box filter / 文字框篩選: "
        f"L >= {args.min_left:.1f}%, "
        f"T >= {args.min_top:.1f}%, "
        f"R <= {args.max_right:.1f}%, "
        f"B <= {args.max_bottom:.1f}%"
    )
    print("Loading EasyOCR once / 正在載入 EasyOCR（僅一次）...")
    reader = easyocr.Reader(args.languages, gpu=args.gpu)

    successful = 0
    total_intervals = 0
    for video_number, video_path in enumerate(video_files, start=1):
        try:
            interval_count = process_video(
                video_path,
                output_dir,
                args,
                reader,
                video_number,
                len(video_files),
            )
        except (OSError, RuntimeError, ValueError) as error:
            print()
            print(f"Failed / 處理失敗: {video_path.name}: {error}")
            continue
        successful += 1
        total_intervals += interval_count

    print()
    print(
        f"Finished / 完成: {successful}/{len(video_files)} video(s), "
        f"{total_intervals} text interval(s)"
    )


if __name__ == "__main__":
    main()

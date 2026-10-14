from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LINE_PATTERN = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)\s*-\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*:\s*(?P<text>.*?)\s*$"
)
NATURAL_NUMBER_PATTERN = re.compile(r"(\d+)")


def find_project_root() -> Path:
    candidates = (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent)
    for candidate in candidates:
        if (candidate / "rewrite_Phase1").is_dir() or (
            candidate / "yolo_pose_mediapie"
        ).is_dir():
            return candidate
    return SCRIPT_DIR


PROJECT_ROOT = find_project_root()
DEFAULT_STT_DIR = PROJECT_ROOT / "rewrite_Phase1" / "STT"
DEFAULT_NPZ_DIR = (
    PROJECT_ROOT / "yolo_pose_mediapie" / "pose_analysis_v3"
)
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "rewrite_Phase3" / "labels.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build labels.json from timestamped STT TXT files."
    )
    parser.add_argument(
        "--stt-dir",
        type=Path,
        default=DEFAULT_STT_DIR,
        help=f"STT TXT directory (default: {DEFAULT_STT_DIR})",
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=DEFAULT_NPZ_DIR,
        help=f"NPZ directory used for validation (default: {DEFAULT_NPZ_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output labels file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--min-intervals",
        type=int,
        default=5,
        help=(
            "Minimum number of valid timestamped sentences required per TXT "
            "(default: 5)"
        ),
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> list[int | str]:
    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in NATURAL_NUMBER_PATTERN.split(path.name)
    ]


def read_sentences(txt_path: Path) -> list[str]:
    sentences: list[str] = []
    with txt_path.open("r", encoding="utf-8-sig") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = LINE_PATTERN.match(line)
            if match is None:
                raise ValueError(
                    f"Invalid line format / 格式錯誤: "
                    f"{txt_path}:{line_number}: {line}"
                )

            start_sec = float(match.group("start"))
            end_sec = float(match.group("end"))
            if end_sec < start_sec:
                raise ValueError(
                    f"End precedes start / 結束時間早於開始時間: "
                    f"{txt_path}:{line_number}"
                )

            sentence = re.sub(r"\s+", " ", match.group("text")).strip()
            if sentence:
                sentences.append(sentence)

    return sentences


def main() -> None:
    args = parse_args()
    if args.min_intervals < 1:
        raise ValueError("--min-intervals must be at least 1")

    stt_dir = args.stt_dir.expanduser().resolve()
    npz_dir = args.npz_dir.expanduser().resolve()
    output_file = args.output.expanduser().resolve()

    if not stt_dir.is_dir():
        raise FileNotFoundError(
            f"STT directory does not exist / STT 目錄不存在: {stt_dir}"
        )
    if not npz_dir.is_dir():
        raise FileNotFoundError(
            f"NPZ directory does not exist / NPZ 目錄不存在: {npz_dir}"
        )

    txt_files = sorted(stt_dir.glob("*.txt"), key=natural_sort_key)
    if not txt_files:
        raise FileNotFoundError(
            f"No TXT files found / 找不到 TXT: {stt_dir}"
        )

    labels: list[dict[str, str]] = []
    skipped_empty = 0
    skipped_too_few_intervals = 0
    skipped_missing_npz = 0

    for txt_path in txt_files:
        sentences = read_sentences(txt_path)
        if not sentences:
            skipped_empty += 1
            print(f"Skipped empty TXT / 跳過空白 TXT: {txt_path.name}")
            continue

        if len(sentences) < args.min_intervals:
            skipped_too_few_intervals += 1
            print(
                f"Skipped too few intervals / 時間區間不足，已跳過: "
                f"{txt_path.name} ({len(sentences)} < {args.min_intervals})"
            )
            continue

        npz_name = f"{txt_path.stem}_upper_body_coordinates.npz"
        npz_path = npz_dir / npz_name
        if not npz_path.is_file():
            skipped_missing_npz += 1
            print(f"Skipped missing NPZ / 找不到 NPZ，已跳過: {npz_name}")
            continue

        labels.append(
            {
                "npz_file": npz_name,
                "text": " ".join(sentences),
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as destination:
        for label in labels:
            destination.write(
                json.dumps(label, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

    print(f"STT files found / 找到 STT: {len(txt_files)}")
    print(f"Labels written / 已寫入標籤: {len(labels)}")
    print(f"Skipped empty / 空白檔案: {skipped_empty}")
    print(
        "Skipped too few intervals / 時間區間不足: "
        f"{skipped_too_few_intervals}"
    )
    print(f"Skipped missing NPZ / 缺少 NPZ: {skipped_missing_npz}")
    print(f"Saved / 已儲存: {output_file}")


if __name__ == "__main__":
    main()

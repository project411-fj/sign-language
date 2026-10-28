from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def get_default_input_dir() -> Path:
    direct_path = SCRIPT_DIR / "pose_analysis_v3"
    nested_path = SCRIPT_DIR / "yolo_pose_mediapie" / "pose_analysis_v3"

    if direct_path.is_dir():
        return direct_path
    if nested_path.is_dir():
        return nested_path

    if SCRIPT_DIR.name.lower() == "yolo_pose_mediapie":
        return direct_path
    return nested_path


DEFAULT_INPUT_DIR = get_default_input_dir()

REQUIRED_COLUMNS = {
    "frame_index",
    "model",
    "side",
    "landmark_id",
    "x_norm",
    "y_norm",
    "z_norm",
    "confidence",
}

HAND_INDEX = {"left": 0, "right": 1}
NUM_HANDS = 2
NUM_HAND_LANDMARKS = 21

YOLO_POSE_INDEX = {
    0: 0,
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 5,
    10: 6,
    11: 7,
    12: 8,
}
NUM_POSE_LANDMARKS = len(YOLO_POSE_INDEX)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pose coordinate TXT files to compressed NPZ files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"TXT input directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="NPZ output directory; defaults to the TXT input directory.",
    )
    return parser.parse_args()


def parse_metadata_value(key: str, value: str) -> Any:
    if key in {
        "frame_width",
        "frame_height",
        "frame_count",
        "num_frames",
        "total_frames",
        "video_frame_count",
    }:
        try:
            return int(value)
        except ValueError:
            return value
    if key == "fps":
        try:
            return float(value)
        except ValueError:
            return value
    return value


def parse_optional_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


def read_pose_txt(txt_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metadata: dict[str, Any] = {}
    data_lines: list[str] = []

    with txt_path.open("r", encoding="utf-8-sig", newline="") as source:
        for raw_line in source:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("#"):
                comment = line[1:].strip()
                if "\t" in comment:
                    key, value = comment.split("\t", 1)
                    key = key.strip()
                    metadata[key] = parse_metadata_value(key, value.strip())
                continue
            data_lines.append(line)

    if not data_lines:
        raise ValueError("The file contains no coordinate table")

    reader = csv.DictReader(data_lines, delimiter="\t")
    missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")
    return metadata, list(reader)


def infer_num_frames(metadata: dict[str, Any], rows: list[dict[str, str]]) -> int:
    max_source_frame = 0
    for row in rows:
        try:
            source_frame = int(row["frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if source_frame < 1:
            raise ValueError(
                f"Expected one-based frame_index, but found {source_frame}"
            )
        max_source_frame = max(max_source_frame, source_frame)

    for key in ("frame_count", "num_frames", "total_frames", "video_frame_count"):
        value = metadata.get(key)
        if isinstance(value, int) and value >= max_source_frame:
            return value

    return max_source_frame


def build_training_arrays(
    metadata: dict[str, Any], rows: list[dict[str, str]]
) -> dict[str, np.ndarray]:
    num_frames = infer_num_frames(metadata, rows)
    if num_frames <= 0:
        raise ValueError("Could not infer a positive frame count")

    hands = np.zeros((num_frames, 2, 21, 3), dtype=np.float32)
    pose = np.zeros((num_frames, 9, 2), dtype=np.float32)
    hand_mask = np.zeros((num_frames, 2, 21), dtype=np.uint8)
    pose_mask = np.zeros((num_frames, 9), dtype=np.uint8)
    pose_confidence = np.zeros((num_frames, 9), dtype=np.float32)

    for line_number, row in enumerate(rows, start=2):
        try:
            source_frame = int(row["frame_index"])
            frame_index = source_frame - 1
            model = row["model"].strip().lower()
            landmark_id = int(row["landmark_id"])

            if not 0 <= frame_index < num_frames:
                raise ValueError(
                    f"source frame {source_frame} maps outside output range"
                )

            if model == "mediapipe_hand":
                side = row["side"].strip().lower()
                if side not in HAND_INDEX or not 0 <= landmark_id < 21:
                    continue
                hand_index = HAND_INDEX[side]
                hands[frame_index, hand_index, landmark_id] = (
                    float(row["x_norm"]),
                    float(row["y_norm"]),
                    parse_optional_float(row["z_norm"]),
                )
                hand_mask[frame_index, hand_index, landmark_id] = 1

            elif model == "yolo_pose":
                pose_index = YOLO_POSE_INDEX.get(landmark_id)
                if pose_index is None:
                    continue
                pose[frame_index, pose_index] = (
                    float(row["x_norm"]),
                    float(row["y_norm"]),
                )
                pose_confidence[frame_index, pose_index] = parse_optional_float(
                    row["confidence"]
                )
                pose_mask[frame_index, pose_index] = 1

        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid coordinate data near table row {line_number}: {error}"
            ) from error

    return {
        "hands": hands,
        "pose": pose,
        "hand_mask": hand_mask,
        "pose_mask": pose_mask,
        "pose_confidence": pose_confidence,
    }


def convert_file(txt_path: Path, npz_path: Path) -> dict[str, tuple[int, ...]]:
    metadata, rows = read_pose_txt(txt_path)
    arrays = build_training_arrays(metadata, rows)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    return {name: array.shape for name, array in arrays.items()}


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No TXT files found / 找不到 TXT: {input_dir}")
        return

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    for txt_path in txt_files:
        npz_path = output_dir / f"{txt_path.stem}.npz"
        try:
            shapes = convert_file(txt_path, npz_path)
        except (OSError, ValueError) as error:
            print(f"Failed / 轉換失敗: {txt_path.name}: {error}")
            continue
        successful += 1
        print(f"Created / 已建立: {npz_path}")
        for name, shape in shapes.items():
            print(f"  {name}: {shape}")

    print(f"Finished / 完成: {successful}/{len(txt_files)} file(s)")


if __name__ == "__main__":
    main()

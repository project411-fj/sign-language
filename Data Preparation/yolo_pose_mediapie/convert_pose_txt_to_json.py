from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("yolo_pose_mediapie/pose_analysis_v3")

REQUIRED_COLUMNS = {
    "frame_index",
    "timestamp_sec",
    "model",
    "side",
    "landmark_id",
    "landmark_name",
    "x_norm",
    "y_norm",
    "z_norm",
    "confidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert pose_analysis_v3 coordinate TXT files to structured JSON."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing coordinate TXT files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="JSON output directory. By default, files are saved beside the TXT files.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of indented, human-readable JSON.",
    )
    return parser.parse_args()


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def parse_metadata_value(key: str, value: str) -> Any:
    if key in {"frame_width", "frame_height"}:
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
                    metadata[key.strip()] = parse_metadata_value(
                        key.strip(), value.strip()
                    )
                elif comment:
                    metadata.setdefault("notes", []).append(comment)
                continue

            data_lines.append(line)

    if not data_lines:
        raise ValueError("The file contains no coordinate table")

    reader = csv.DictReader(data_lines, delimiter="\t")
    columns = set(reader.fieldnames or [])
    missing_columns = REQUIRED_COLUMNS - columns
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    return metadata, list(reader)


def new_frame(frame_index: int, timestamp_sec: float) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "yolo_pose": [],
        "mediapipe_pose": [],
        "mediapipe_hands": {
            "left": [],
            "right": [],
            "unknown": [],
        },
    }


def landmark_from_row(row: dict[str, str], include_side: bool) -> dict[str, Any]:
    landmark = {
        "landmark_id": int(row["landmark_id"]),
        "landmark_name": row["landmark_name"],
        "x_norm": float(row["x_norm"]),
        "y_norm": float(row["y_norm"]),
        "z_norm": parse_optional_float(row["z_norm"]),
        "confidence": parse_optional_float(row["confidence"]),
    }
    if include_side:
        landmark["side"] = row["side"]
    return landmark


def build_json_document(
    txt_path: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    frames: dict[int, dict[str, Any]] = {}

    for line_number, row in enumerate(rows, start=2):
        try:
            frame_index = int(row["frame_index"])
            timestamp_sec = float(row["timestamp_sec"])
            model = row["model"].strip().lower()
            side = row["side"].strip().lower()

            frame = frames.setdefault(
                frame_index,
                new_frame(frame_index, timestamp_sec),
            )

            if model == "yolo_pose":
                frame["yolo_pose"].append(
                    landmark_from_row(row, include_side=True)
                )
            elif model == "mediapipe_pose":
                frame["mediapipe_pose"].append(
                    landmark_from_row(row, include_side=False)
                )
            elif model == "mediapipe_hand":
                hand_side = side if side in {"left", "right"} else "unknown"
                frame["mediapipe_hands"][hand_side].append(
                    landmark_from_row(row, include_side=False)
                )
            else:
                frame.setdefault("other_models", {}).setdefault(model, []).append(
                    landmark_from_row(row, include_side=True)
                )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid coordinate data near table row {line_number}: {error}"
            ) from error

    ordered_frames = [frames[index] for index in sorted(frames)]
    for frame in ordered_frames:
        for key in ("yolo_pose", "mediapipe_pose"):
            frame[key].sort(key=lambda point: point["landmark_id"])
        for hand_points in frame["mediapipe_hands"].values():
            hand_points.sort(key=lambda point: point["landmark_id"])

    return {
        "format_version": 1,
        "source_txt": str(txt_path.resolve()),
        "metadata": metadata,
        "coordinate_system": {
            "x_norm": "0.0 is the left edge and 1.0 is the right edge",
            "y_norm": "0.0 is the top edge and 1.0 is the bottom edge",
            "z_norm": (
                "MediaPipe relative depth; null for YOLO pose landmarks"
            ),
        },
        "frame_count_with_landmarks": len(ordered_frames),
        "frames": ordered_frames,
    }


def convert_file(txt_path: Path, json_path: Path, compact: bool) -> None:
    metadata, rows = read_pose_txt(txt_path)
    document = build_json_document(txt_path, metadata, rows)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8", newline="\n") as destination:
        if compact:
            json.dump(
                document,
                destination,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            json.dump(document, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist / 輸入資料夾不存在: {input_dir}"
        )

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
    print(f"TXT files found / 找到 TXT: {len(txt_files)}")
    for txt_path in txt_files:
        json_path = output_dir / f"{txt_path.stem}.json"
        try:
            convert_file(txt_path, json_path, args.compact)
        except (OSError, ValueError) as error:
            print(f"Failed / 轉換失敗: {txt_path.name}: {error}")
            continue

        successful += 1
        print(f"Created / 已建立: {json_path}")

    print(
        f"Finished / 完成: {successful}/{len(txt_files)} file(s) converted"
    )


if __name__ == "__main__":
    main()

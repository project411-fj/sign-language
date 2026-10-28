from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def get_default_npz_path() -> Path:
    direct_path = SCRIPT_DIR / "pose_analysis_v3"
    nested_path = SCRIPT_DIR / "yolo_pose_mediapie" / "pose_analysis_v3"

    if direct_path.is_dir():
        return direct_path
    if nested_path.is_dir():
        return nested_path

    if SCRIPT_DIR.name.lower() == "yolo_pose_mediapie":
        return direct_path
    return nested_path


DEFAULT_NPZ_PATH = get_default_npz_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect every frame in one NPZ file or every NPZ in a directory."
    )
    parser.add_argument(
        "npz_path",
        nargs="?",
        type=Path,
        default=DEFAULT_NPZ_PATH,
        help=(
            "NPZ file or directory to inspect "
            f"(default: {DEFAULT_NPZ_PATH})"
        ),
    )
    return parser.parse_args()


def inspect_npz(npz_path: Path) -> None:
    with np.load(npz_path) as data:
        required = {
            "hands",
            "pose",
            "hand_mask",
            "pose_mask",
            "pose_confidence",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Missing NPZ array(s): {', '.join(sorted(missing))}")

        hands = data["hands"]
        pose = data["pose"]
        hand_mask = data["hand_mask"]
        pose_mask = data["pose_mask"]
        pose_confidence = data["pose_confidence"]

        num_frames = hands.shape[0]
        if not all(array.shape[0] == num_frames for array in (
            pose,
            hand_mask,
            pose_mask,
            pose_confidence,
        )):
            raise ValueError("NPZ arrays do not have the same number of frames")

        print(f"File / 檔案: {npz_path}")
        print(f"Frames / 幀數: {num_frames}")
        print(f"Arrays / 陣列: {data.files}")
        for key in data.files:
            array = data[key]
            print(
                f"  {key}: shape={array.shape}, dtype={array.dtype}, "
                f"nonzero={np.count_nonzero(array)}"
            )

        for frame_index in range(num_frames):
            source_frame = frame_index + 1
            print("\n" + "=" * 80)
            print(
                f"NPZ index {frame_index} | "
                f"source video frame {source_frame} / 原始影片第 {source_frame} 幀"
            )
            print("-" * 80)
            print("Left hand / 左手:")
            print(hands[frame_index, 0])
            print("Left-hand mask / 左手 mask:")
            print(hand_mask[frame_index, 0])
            print("Right hand / 右手:")
            print(hands[frame_index, 1])
            print("Right-hand mask / 右手 mask:")
            print(hand_mask[frame_index, 1])
            print("YOLO pose:")
            print(pose[frame_index])
            print("Pose mask:")
            print(pose_mask[frame_index])
            print("Pose confidence:")
            print(pose_confidence[frame_index])


def find_npz_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Not an NPZ file / 不是 NPZ 檔案: {path}")
        return [path]

    if path.is_dir():
        npz_files = sorted(path.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(
                f"No NPZ files found / 目錄中找不到 NPZ: {path}"
            )
        return npz_files

    raise FileNotFoundError(
        f"File or directory does not exist / 檔案或目錄不存在: {path}"
    )


def main() -> None:
    args = parse_args()
    npz_path = args.npz_path.expanduser().resolve()
    npz_files = find_npz_files(npz_path)

    np.set_printoptions(precision=6, suppress=True, linewidth=160)
    print(f"NPZ files found / 找到 NPZ: {len(npz_files)}")

    for file_number, file_path in enumerate(npz_files, start=1):
        print("\n" + "#" * 100)
        print(f"File {file_number}/{len(npz_files)} / 檔案 {file_number}/{len(npz_files)}")
        print("#" * 100)
        inspect_npz(file_path)


if __name__ == "__main__":
    main()

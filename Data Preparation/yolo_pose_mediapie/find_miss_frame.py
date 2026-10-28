
import numpy as np
from pathlib import Path

INPUT_PATH = Path("yolo_pose_mediapie") / "hand_keypoints" / "part_1_1.npy"

FPS = 30
TOP_N = 10


def find_missing_ranges(missing):
    missing = np.asarray(missing, dtype=bool)

    padded = np.pad(missing.astype(np.int8), (1, 1))
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    return [
        (int(start), int(end), int(end - start + 1))
        for start, end in zip(starts, ends)
    ]


def print_statistics(name, missing, fps):
    ranges = find_missing_ranges(missing)

    total_missing = int(np.sum(missing))
    total_frames = len(missing)

    short = [r for r in ranges if r[2] <= 2]
    medium = [r for r in ranges if 3 <= r[2] <= 5]
    long = [r for r in ranges if r[2] >= 6]

    print(f"\n========== {name} ==========")

    print(f"總 Frames: {total_frames}")
    print(f"未偵測 Frames: {total_missing}")
    print(f"未偵測比例: {total_missing / total_frames * 100:.2f}%")

    print(f"\n連續未偵測區間數: {len(ranges)}")

    print(f"1~2 Frames: {len(short)} 次")
    print(f"3~5 Frames: {len(medium)} 次")
    print(f"6 Frames 以上: {len(long)} 次")

    if not ranges:
        print("\n全部 Frames 都有偵測到")
        return

    longest = max(ranges, key=lambda r: r[2])

    print(f"\n最長連續未偵測: {longest[2]} Frames")
    print(f"發生範圍: Frame {longest[0]} ~ {longest[1]}")
    print(f"持續時間: {longest[2] / fps:.3f} 秒")

    ranges.sort(key=lambda r: r[2], reverse=True)

    print(f"\n最長的前 {TOP_N} 個未偵測區間:")

    for start, end, length in ranges[:TOP_N]:
        print(
            f"Frame {start} ~ {end} | "
            f"{length} Frames | "
            f"{start / fps:.2f}s ~ {(end + 1) / fps:.2f}s"
        )


def main():
    data = np.load(INPUT_PATH)

    if data.ndim != 2 or data.shape[1] != 128:
        raise ValueError(f"資料格式錯誤: {data.shape}")

    left = data[:, 126] == 1
    right = data[:, 127] == 1

    print("資料形狀:", data.shape)

    print_statistics("左手未偵測", ~left, FPS)
    print_statistics("右手未偵測", ~right, FPS)
    print_statistics("雙手都未偵測", ~left & ~right, FPS)


if __name__ == "__main__":
    main()

"""
split_segments.py
依 find_sign_pauses.py 輸出的 .txt 區間，把完整的 .npy 切成一句一段。

輸入 (<part> 執行時輸入，<子資料夾> 為 1、2、3…)：
    .npy : 02combined/<part>/<子資料夾>/000600_001507.npy
    .txt : 02pause/<part>/<子資料夾>/000600_001507_frames.txt   每行「34-204: 字幕文字」
輸出：
    分段 : 03segments/<part>/<子資料夾>/000600_001507_s00.npy、_s01.npy …
    標籤 : 04labels/<part>/<子資料夾>/000600_001507.txt        每行「000600_001507_s00: 字幕文字」

執行時只顯示進度條；出錯的檔案記錄在 failed_segments_part_<part>.txt，
下次執行同一個 part 時只重跑紀錄裡的檔案，全部成功後紀錄檔會刪除。
"""
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ================================================================ 路徑設定 (後面會再接 /<part>)
NPY_DIR = r"Data Preparation\Phase4-Keypoint\Keypoint_output\02combined"   # 完整的 .npy
TXT_DIR = r"Data Preparation\Phase5-Pause\output\02pause"                  # find_sign_pauses.py 的 OUT_DIR
SEG_DIR = r"Data Preparation\Phase5-Pause\output\03segments"               # 輸出分段 .npy
LABEL_DIR = r"Data Preparation\Phase5-Pause\output\03labels"               # 輸出「分段檔名: 文字」的 .txt

# ================================================================ 參數設定
TXT_SUFFIX = "_frames"     # 000600_001507.npy -> 000600_001507_frames.txt
FRAME_OFFSET = 0           # .txt 的 frame 編號 + FRAME_OFFSET = .npy 的列索引 (跟 find_sign_pauses.py 一樣)

# .txt 每行：「34-204: 字幕文字」
LINE_RE = re.compile(r"^\s*(\d+)\s*[~～\-–—]\s*(\d+)\s*[:：]?\s*(.*?)\s*$")


def read_ranges(path):
    """回傳 [(start, end, text), ...]，順序跟 .txt 的行順序一樣。"""
    ranges = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = LINE_RE.match(line)
        if m:
            s, e = sorted((int(m[1]) + FRAME_OFFSET, int(m[2]) + FRAME_OFFSET))
            ranges.append((s, e, m[3]))
    return ranges


def split_file(npy_path, txt_path, seg_dir, label_path):
    """切一個 .npy，回傳切出的段數。"""
    arr = np.load(npy_path)
    T = len(arr)
    ranges = read_ranges(txt_path)
    digits = max(2, len(str(len(ranges) - 1)))      # 超過 100 句時自動變成 3 位數

    seg_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (s, e, text) in enumerate(ranges):
        if s > T - 1:                               # 整句都在 .npy 之外
            continue
        name = f"{npy_path.stem}_s{i:0{digits}d}"
        np.save(seg_dir / f"{name}.npy", arr[max(s, 0): min(e, T - 1) + 1])
        lines.append(f"{name}: {text}\n")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("".join(lines), encoding="utf-8")
    return len(lines)


def main():
    part = input("part: ").strip()
    npy_dir, txt_dir = Path(NPY_DIR) / part, Path(TXT_DIR) / part
    seg_dir, label_dir = Path(SEG_DIR) / part, Path(LABEL_DIR) / part
    failed_log = Path(__file__).parent / f"failed_segments_part_{part}.txt"

    # 有失敗紀錄就只跑紀錄裡的檔案，否則跑整個 part (名稱如 1/000600_001507)
    if failed_log.exists():
        names = [line for line in failed_log.read_text(encoding="utf-8").splitlines() if line]
    else:
        files = sorted(
            npy_dir.glob("*/*.npy"),
            key=lambda p: (int(p.parent.name), [int(x) for x in p.stem.split("_")]),
        )
        names = [p.relative_to(npy_dir).with_suffix("").as_posix() for p in files]

    failed = []
    for name in tqdm(names, desc="segments", unit="file"):
        rel = Path(name)
        try:
            split_file(
                npy_dir / f"{name}.npy",
                txt_dir / rel.parent / f"{rel.name}{TXT_SUFFIX}.txt",
                seg_dir / rel.parent,
                label_dir / f"{name}.txt",
            )
        except Exception as exc:
            failed.append((name, f"{type(exc).__name__}: {exc}"))

    # 有失敗就覆寫紀錄，全部成功就刪除紀錄
    if failed:
        failed_log.write_text("".join(f"{name}\n" for name, _ in failed), encoding="utf-8")
        tqdm.write(f"\n失敗 {len(failed)} 個，已記錄到 {failed_log.name}：")
        for name, error in failed:
            tqdm.write(f"  {name}：{error}")
    else:
        failed_log.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

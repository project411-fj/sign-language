"""STT 版第 3 步：依 02find_sign_pauses.py 找到的切點，把關鍵點 (02combined，292 維) 切成一句一個 .npy。

和 OCR/03split_npy.py 一樣，輸出也在 output/ (Data Training V2 只讀 output/)，標籤是語音轉出來的文字。
只處理 STT 路徑的 part (part_videos 裡有範圍資料夾的 part)；OCR 的 part 請用 OCR/03split_npy.py。
執行後輸入 part：「13」、「13,14」或「all」(context_breeze 底下 STT 路徑的 part)。
"""
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ================================================================ 路徑設定 (後面會再接 /<part>)
# 讀含臉部的 292 維 (Phase4 的 03keypoint_combine.py 的輸出)
NPY_DIR = "Data Preparation/Phase4-Keypoint/Keypoint_output/02combined"   # 完整的 .npy
TXT_DIR = "Data Preparation/Phase5-Pause/output/02pause"                  # STT/02find_sign_pauses.py 的 OUT_DIR
SEG_DIR = "Data Preparation/Phase5-Pause/output/03segments"               # 輸出分段 .npy (292 維)
LABEL_DIR = "Data Preparation/Phase5-Pause/output/03labels"               # 輸出「分段檔名: 語音文字」的 .txt
STT_DIR = "Data Preparation/Phase2-STT/context_breeze"                   # all：這裡有的 part 才處理
PART_VIDEOS_DIR = "Data Preparation/videos/part_videos"                  # 用來判斷是不是 STT 路徑的 part
BOX_RE = re.compile(r"\d+(\.\d+)?(_\d+(\.\d+)?){3}")                  # STT 路徑的範圍資料夾，例如 0.20_0.73_0.10_0.33
FEAT_DIM = 292                 # 手 128 + 身體 28 + 臉 136

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
    if arr.ndim != 2 or arr.shape[1] != FEAT_DIM:
        raise ValueError(f"應該是 {FEAT_DIM} 維 (含臉部)，收到 {arr.shape}；請用 Phase4 的 03keypoint_combine.py 重新產生")
    T = len(arr)
    ranges = read_ranges(txt_path)
    digits = max(2, len(str(len(ranges) - 1)))      # 超過 100 句時自動變成 3 位數

    # 先刪掉這支影片上次切出來的分段，避免句數變少時留下舊檔
    seg_dir.mkdir(parents=True, exist_ok=True)
    for old in seg_dir.glob(f"{npy_path.stem}_s*.npy"):
        old.unlink()

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


def parse_parts(answer, npy_root):
    """「3」→ [3]；「1,2,5」→ [1, 2, 5]；「all」→ NPY_DIR 底下所有 part。"""
    answer = answer.strip()
    if answer.lower() == "all":
        return sorted((p.name for p in npy_root.iterdir() if p.is_dir()), key=lambda x: (len(x), x))
    return [p.strip() for p in answer.split(",") if p.strip()]


def is_stt_part(part):
    """STT 路徑的 part：part_videos/<part>/<n>/ 底下有 <TOP>_<BOTTOM>_<LEFT>_<RIGHT>/ 範圍資料夾
    (Phase1 03clipV2_13~.py 的輸出)，或有 crop_boxes.csv (Phase1 03clipV3.py 的輸出)。"""
    root = (Path(PART_VIDEOS_DIR) / part)
    return any((d.is_dir() and BOX_RE.fullmatch(d.name)) or d.name == "crop_boxes.csv" for d in root.glob("*/*"))


def run_part(part):
    if not is_stt_part(part):
        print(f"part {part}：part_videos 裡沒有範圍資料夾或 crop_boxes.csv，不是 STT 路徑的 part (OCR 的 part 請用 OCR/03split_npy.py)，跳過")
        return
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

    # 沒有切點檔的片段 (沒有語音句子，02 跳過的) 不算失敗
    no_speech = [n for n in names if not (txt_dir / Path(n).parent / f"{Path(n).name}{TXT_SUFFIX}.txt").exists()]
    skip = set(no_speech)
    names = [n for n in names if n not in skip]
    if no_speech:
        print(f"part {part}：{len(no_speech)} 支片段沒有切點檔 (沒有語音句子)，跳過")

    failed = []
    for name in tqdm(names, desc=f"segments part {part}", unit="file"):
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


def main():
    print(f"讀取 {Path(NPY_DIR).name} ({FEAT_DIM} 維)，輸出到 {Path(SEG_DIR).name}、{Path(LABEL_DIR).name}")
    answer = input("part (例如 13、13,14 或 all): ")
    parts = parse_parts(answer, Path(STT_DIR))
    if answer.strip().lower() == "all":
        parts = [part for part in parts if is_stt_part(part)]   # all 只取 STT 路徑的 part
    for part in parts:
        run_part(part)


if __name__ == "__main__":
    main()

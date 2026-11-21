import csv
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ================================================================ 路徑設定 (後面會再接 /<part>)
NPY_DIR = r"Data Preparation\Phase4-Keypoint\Keypoint_output\02combined"   # 完整的 .npy
TXT_DIR = r"Data Preparation\Phase5-Pause\output\02pause"                  # find_sign_pauses.py 的 OUT_DIR
SEG_DIR = r"Data Preparation\Phase5-Pause\output\03segments"               # 輸出分段 .npy
LABEL_DIR = r"Data Preparation\Phase5-Pause\output\03labels"               # 輸出「分段檔名: 文字」的 .txt
FPS_CSV = r"Data Preparation\Phase5-Pause\output\01fps"                    # 裡面是 fps_<part>.csv (MARGIN > 0 才會用到)

# ================================================================ 參數設定
TXT_SUFFIX = "_frames"     # 000600_001507.npy -> 000600_001507_frames.txt
FRAME_OFFSET = 0           # .txt 的 frame 編號 + FRAME_OFFSET = .npy 的列索引 (跟 find_sign_pauses.py 一樣)

# 實驗 3：每句前後各多切幾秒 (寬鬆切法)，0 = 照原本的切點切
#   MARGIN > 0 時會輸出到 03segments_m<MARGIN>、03labels_m<MARGIN>，不會覆蓋原本的結果
#   相鄰句子會互相重疊，這是刻意的：目的是確保每句的手語一定完整包含在片段裡
MARGIN = 0.0               # 寬鬆切法的實驗已經做完 (跟原本沒有差異)，改回 0

# 特徵版本：""  = 原本的 156 維 (02combined)
#          "_face" = 加上臉部的 292 維 (02combined_face，由 Phase4 的 03keypoint_combine_face.py 產生)
#   會從 NPY_DIR + FEATURE_TAG 讀取，輸出到 03segments<FEATURE_TAG>、03labels<FEATURE_TAG>，不會覆蓋原本的結果
#   切點一樣用 02pause 的結果，所以句子跟原本一一對應
FEATURE_TAG = ""

# .txt 每行：「34-204: 字幕文字」
LINE_RE = re.compile(r"^\s*(\d+)\s*[~～\-–—]\s*(\d+)\s*[:：]?\s*(.*?)\s*$")


def output_tag():
    """輸出資料夾的後綴，例如 ""、"_face"、"_m0.5"、"_face_m0.5"。"""
    return FEATURE_TAG + (f"_m{MARGIN:g}" if MARGIN > 0 else "")


def read_ranges(path):
    """回傳 [(start, end, text), ...]，順序跟 .txt 的行順序一樣。"""
    ranges = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = LINE_RE.match(line)
        if m:
            s, e = sorted((int(m[1]) + FRAME_OFFSET, int(m[2]) + FRAME_OFFSET))
            ranges.append((s, e, m[3]))
    return ranges


def load_fps_table(csv_path):
    """{子資料夾/檔名: fps}，1/000600_001507_upper_body.mp4 -> 1/000600_001507"""
    with csv_path.open(encoding="utf-8-sig") as f:
        return {
            Path(row["filename"]).with_suffix("").as_posix().removesuffix("_upper_body"): float(row["fps"])
            for row in csv.DictReader(f)
        }


def split_file(npy_path, txt_path, seg_dir, label_path, pad=0):
    """切一個 .npy，回傳切出的段數。pad = 每句前後多切幾個 frame。"""
    arr = np.load(npy_path)
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
        np.save(seg_dir / f"{name}.npy", arr[max(s - pad, 0): min(e + pad, T - 1) + 1])
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


def run_part(part):
    tag = output_tag()
    npy_dir, txt_dir = Path(NPY_DIR + FEATURE_TAG) / part, Path(TXT_DIR) / part
    seg_dir, label_dir = Path(SEG_DIR + tag) / part, Path(LABEL_DIR + tag) / part
    failed_log = Path(__file__).parent / f"failed_segments{tag}_part_{part}.txt"
    fps_table = load_fps_table(Path(FPS_CSV) / f"fps_{part}.csv") if MARGIN > 0 else {}

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
    for name in tqdm(names, desc=f"segments part {part}{tag}", unit="file"):
        rel = Path(name)
        try:
            pad = round(MARGIN * fps_table[name]) if MARGIN > 0 else 0
            split_file(
                npy_dir / f"{name}.npy",
                txt_dir / rel.parent / f"{rel.name}{TXT_SUFFIX}.txt",
                seg_dir / rel.parent,
                label_dir / f"{name}.txt",
                pad,
            )
        except KeyError:
            failed.append((name, f"fps_{part}.csv 裡找不到這支影片的 fps"))
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
    if MARGIN > 0:
        print(f"寬鬆切法：每句前後各多切 {MARGIN:g} 秒")
    print(f"讀取 {Path(NPY_DIR + FEATURE_TAG).name}，輸出到 {Path(SEG_DIR + output_tag()).name}、{Path(LABEL_DIR + output_tag()).name}")
    for part in parse_parts(input("part (例如 3、1,2,5 或 all): "), Path(NPY_DIR + FEATURE_TAG)):
        run_part(part)


if __name__ == "__main__":
    main()

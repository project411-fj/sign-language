import re
from difflib import SequenceMatcher
from pathlib import Path
from tqdm import tqdm

MERGED_DIR = Path(r"Data Preparation\Phase3-OCR\OCR_output\02EasOCR_merged")   # merge_frames.py 的輸出（直接覆蓋）
THRESHOLD = 0.8                # 相似度門檻
MIN_FRAMES = 10                # 短於此幀數的區間視為 OCR 雜訊，直接丟掉


def similarity(a, b):
    return SequenceMatcher(None, a.replace(" ", ""), b.replace(" ", "")).ratio()


def smooth(segments):
    out = []
    for s, e, t in segments:
        if e - s + 1 < MIN_FRAMES:
            continue
        if out and similarity(t, out[-1][2]) >= THRESHOLD:
            ps, pe, pt = out[-1]
            out[-1] = (ps, e, pt if pe - ps >= e - s else t)   # 文字取較長區間的版本
        else:
            out.append((s, e, t))
    return out


num = input("part: ").strip()
for path in tqdm(sorted((MERGED_DIR / num).glob("*.txt")), desc=f"part {num}", unit="file"):
    segments = [(int(s), int(e), t) for s, e, t in
                re.findall(r"(\d+)-(\d+):\s*(.+)", path.read_text(encoding="utf-8"))]
    path.write_text("".join(f"{s}-{e}: {t}\n" for s, e, t in smooth(segments)), encoding="utf-8")
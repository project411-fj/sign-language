import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path(r"Data Preparation\Phase3-OCR\OCR_output\01EasOCR_text")      # 01EasOCR_text/<part>/<n>/<start>_<end>_frames.txt
OUTPUT_DIR = Path(r"Data Preparation\Phase3-OCR\OCR_output\02EasOCR_merged")    # 02EasOCR_merged/<part>/<n>/<start>_<end>_frames.txt
THRESHOLD = 0.8                # 相似度門檻
MAX_GAP = float("inf")                    # frame 編號差距 <= 此值才算相鄰


def similarity(a, b):
    return SequenceMatcher(None, a.replace(" ", ""), b.replace(" ", "")).ratio()


def read_frames(path):
    frames = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d+):\s*(.+)", line)
        if m:
            bucket = frames.setdefault(int(m.group(1)), [])
            if m.group(2) not in bucket:
                bucket.append(m.group(2))
    return {k: " ".join(frames[k]) for k in sorted(frames)}


def merge_segments(frames):
    segments = []
    for frame, text in frames.items():
        if segments:
            seg = segments[-1]
            rep = seg["counter"].most_common(1)[0][0]
            if frame - seg["end"] <= MAX_GAP and max(similarity(text, seg["last"]), similarity(text, rep)) >= THRESHOLD:
                seg["end"], seg["last"] = frame, text
                seg["counter"][text] += 1
                continue
        segments.append({"start": frame, "end": frame, "last": text, "counter": Counter([text])})
    return [(s["start"], s["end"], s["counter"].most_common(1)[0][0]) for s in segments]

num = input("part: ").strip()
INPUT_DIR = INPUT_DIR / num
OUTPUT_DIR = OUTPUT_DIR / num
srcs = sorted(INPUT_DIR.glob("*/*.txt"), key=lambda p: (int(p.parent.name), p.name))
for src in tqdm(srcs, desc=f"part {num}", unit="file"):
    segments = merge_segments(read_frames(src))
    out_dir = OUTPUT_DIR / src.parent.name   # 保留子目錄（1, 2, 3…），避免不同影片的片段檔名互相覆蓋
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / src.name).write_text(
        "".join(f"{s}-{e}: {t}\n" for s, e, t in segments), encoding="utf-8")
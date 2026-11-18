import re
from pathlib import Path

from tqdm import tqdm

# ========== 設定 ==========
BASE = Path("Data Preparation/Phase3-OCR/OCR_output")
INPUT_DIR = BASE / "04GeminiOCR_context"
OUTPUT_DIR = BASE / "05GeminiOCR_merged"

num = input("part: ").strip()
files = sorted((INPUT_DIR / num).glob("*/*_frames.txt"), key=lambda p: (int(p.parent.name), p.name))

before = after = 0
for src in tqdm(files, desc=f"part {num}", unit="file"):
    segments = []   # [起始幀, 結束幀, 文字]
    for s, e, t in re.findall(r"(\d+)-(\d+):[ \t]*(.*)", src.read_text(encoding="utf-8")):
        if segments and t.replace(" ", "") == segments[-1][2].replace(" ", ""):
            segments[-1][1] = int(e)            # 和上一個文字相同 → 延長上一個區間
        else:
            segments.append([int(s), int(e), t])
        before += 1
    after += len(segments)

    out = OUTPUT_DIR / num / src.parent.name / src.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{s}-{e}: {t}\n" for s, e, t in segments), encoding="utf-8")

print(f"part {num}：{len(files)} 個檔案，區間 {before} → {after}（合併掉 {before - after} 個）")

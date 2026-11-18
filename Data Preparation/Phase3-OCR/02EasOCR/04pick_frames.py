import re
import shutil
from pathlib import Path
from tqdm import tqdm

MERGED_DIR = Path(r"Data Preparation\Phase3-OCR\OCR_output\02EasOCR_merged")   # 02EasOCR_merged/<part>/<n>/<start>_<end>_frames.txt
FRAME_DIR = Path(r"Data Preparation\Phase3-OCR\01frames_crop")                       # frames_crop/<part>/<n>/<start>_<end>_frames/*.jpg（crop_text.py 的輸出）
OUTPUT_DIR = Path(r"Data Preparation\Phase3-OCR\OCR_output\03EasOCR_frame")     # 03EasOCR_frame/<part>/<n>/<start>_<end>_sXX.jpg

num = input("part: ").strip()
missing = []
srcs = sorted((MERGED_DIR / num).glob("*/*.txt"), key=lambda p: (int(p.parent.name), p.name))
for src in tqdm(srcs, desc=f"part {num}", unit="file"):
    n = src.parent.name                                             # 子目錄 1, 2, 3…
    name = src.stem.removesuffix("_frames")                      # 021794_022876_frames -> 021794_022876
    out_dir = OUTPUT_DIR / num / n
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines()):
        s, e = map(int, re.match(r"(\d+)-(\d+):", line).groups())
        frame = (s + e) // 2                                        # 取區間中間的 frame
        img = FRAME_DIR / num / n / src.stem / f"frame_{frame:06d}.jpg"
        if not img.exists():
            missing.append(img)
            continue
        shutil.copy(img, out_dir / f"{name}_s{i:02d}.jpg")

print(f"\n找不到的圖片共 {len(missing)} 張：")
for img in missing:
    print(img)
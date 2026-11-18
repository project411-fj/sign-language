"""讀取 output/style_result_*.csv，把 style 為 target 的圖片從 03EasOCR_frame/<num>/<n>/ 複製到 Gemini_OCR/send/<num>/<n>/。
CSV 的 filename 欄位需為相對於 03EasOCR_frame/<num>/ 的路徑，例如 1/021794_022876_s00.jpg。"""

import csv
import shutil
from pathlib import Path

# ========== 設定 ==========
CSV_DIR = Path("Data Preparation/Phase3-OCR/style/output")
IMAGE_DIR = Path("Data Preparation/Phase3-OCR/EasOCR_output/03EasOCR_frame")   # 03EasOCR_frame/<num>/<n>/*.jpg
SEND_DIR = Path("Data Preparation/Phase3-OCR/Gemini_OCR/send")                 # send/<num>/<n>/*.jpg

for csv_path in sorted(CSV_DIR.glob("style_result_*.csv")):
    num = csv_path.stem.removeprefix("style_result_")
    dst = SEND_DIR / num

    count = 0
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["style"] == "target":
                rel = Path(row["filename"])                     # 1/021794_022876_s00.jpg
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(IMAGE_DIR / num / rel, dst / rel)
                count += 1

    print(f"{csv_path.name}：複製 {count} 張到 {dst}")
from pathlib import Path

import json

INPUT_PATH_DIR = Path("rewrite_Phase2") / "merge_text_no_timestamp"

flie_paths = sorted(INPUT_PATH_DIR.glob("*.txt"))

labels = []

for file_path in flie_paths:
    title = file_path.name.removesuffix("_merged.txt")
    title_check = Path("rewrite_Phase3_train") / "features" / f"{title}_upper_body_coordinates.npz"
    if title_check.is_file():
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()
            record = {
                "npz_file": f"{title}_upper_body_coordinates.npz",
                "text": f"{text}"
            }
            labels.append(record)
    else:
        print(f"缺少文件：{title_check}")

with open(r"rewrite_Phase3_train\Labels.json", "w", encoding="utf-8") as file:
    json.dump(labels, file, ensure_ascii=False, indent=4)
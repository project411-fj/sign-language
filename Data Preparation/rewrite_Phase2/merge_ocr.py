from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

INPUT_PATH_DIR = Path("rewrite_Phase2") / "EasOCR_merged"
OUTPUT_PATH_DIR = Path("rewrite_Phase2") / "EasOCR_filtered"

OUTPUT_PATH_DIR.mkdir(parents=True, exist_ok=True)

file_paths = sorted(INPUT_PATH_DIR.glob("*.txt"))

for file_path in file_paths:
    groups = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            content = line.strip().split(":", 1)[-1].strip()

            if not content:
                continue

            if not groups:
                groups.append([content])
                continue

            similarity = SequenceMatcher(None, content, groups[-1][-1]).ratio()

            if similarity >= 0.8:
                groups[-1].append(content)
            else:
                groups.append([content])

    contents = []

    for group in groups:
        most_common = Counter(group).most_common(1)[0][0]
        contents.append(most_common)

    output_path = OUTPUT_PATH_DIR / file_path.name

    with open(output_path, "w", encoding="utf-8") as file:
        for content in contents:
            file.write(content + "\n")

    print(f"{file_path.name}: {len(contents)} 筆文字")
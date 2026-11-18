from pathlib import Path

DIR = Path(r"Data Preparation\rewrite_Phase2\EasOCR_output\1\EasOCR_select")

count = 0
MAX = 0
MAX_FILE = None

for file in sorted(DIR.glob("*.txt")):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

        count += len(lines)

        if len(lines) > MAX:
            MAX = len(lines)
            MAX_FILE = file.name

print(f"Total number of lines in all text files: {count}", f"Maximum number of lines in a single text file: {MAX}", f"File with maximum lines: {MAX_FILE}")
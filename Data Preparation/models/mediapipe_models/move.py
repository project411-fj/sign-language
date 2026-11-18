from pathlib import Path
import shutil

INPUT_DIR = Path("Data\\1")
OUTPUT_DIR = Path("Data\\1\\npy")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for txt_file in sorted(INPUT_DIR.glob("*.npy")):
    shutil.copy2(txt_file, OUTPUT_DIR / txt_file.name)
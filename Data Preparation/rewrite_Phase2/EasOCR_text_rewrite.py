from pathlib import Path

INPUT_PATH_DIR = Path("rewrite_Phase2") / "EasOCR_text"
OUTPUT_PATH_DIR = Path("rewrite_Phase2") / "EasOCR_merged"

OUTPUT_PATH_DIR.mkdir(parents=True, exist_ok=True)

file_paths = sorted(INPUT_PATH_DIR.glob("*.txt"))

for file_path in file_paths:
    frames = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            frame, text = line.split(":", 1)

            frame = frame.strip()
            text = text.strip()

            if frame not in frames:
                frames[frame] = []

            if text:
                frames[frame].append(text)

    output_path = OUTPUT_PATH_DIR / file_path.name

    with open(output_path, "w", encoding="utf-8") as f:
        for frame, texts in frames.items():
            merged_text = " ".join(texts)
            f.write(f"{frame}: {merged_text}\n")

    print(f"完成：{file_path.name}")
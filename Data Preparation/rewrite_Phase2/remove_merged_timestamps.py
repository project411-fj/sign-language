import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "merge_text"
OUTPUT_DIR = BASE_DIR / "merge_text_no_timestamp"

if not INPUT_DIR.is_dir():
    raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")
if INPUT_DIR.resolve() == OUTPUT_DIR.resolve():
    raise ValueError("Input and output directories must be different.")

input_files = sorted(INPUT_DIR.glob("*.txt"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp_pattern = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*:\s*(.*?)\s*$"
)
converted = 0
failed = 0

for input_path in input_files:
    try:
        text_parts = []
        for line_number, line in enumerate(
            input_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            match = timestamp_pattern.fullmatch(line)
            if match is None:
                raise ValueError(f"Invalid timestamp format at line {line_number}")
            text = match.group(1).strip()
            if text:
                text_parts.append(text)

        output_text = " ".join(text_parts)
        output_path = OUTPUT_DIR / input_path.name
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(output_text + "\n" if output_text else "")
        print(f"OK: {input_path.name}")
        converted += 1
    except (OSError, UnicodeError, ValueError) as error:
        print(f"ERROR: {input_path.name}: {error}")
        failed += 1

print(f"Converted: {converted}; failed: {failed}; output: {OUTPUT_DIR}")
if failed:
    raise SystemExit(1)

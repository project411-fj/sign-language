import json
import re
from pathlib import Path


STT_LINE = re.compile(
    r"\[\d+(?:\.\d+)?s\s*->\s*\d+(?:\.\d+)?s\]\s*(.+)"
)


def parse_stt_txt_to_json(txt_file_path, json_file_path):
    """Convert Whisper timestamp lines into a JSON list of utterances."""
    texts = []
    with open(txt_file_path, "r", encoding="utf-8") as source:
        for line in source:
            match = STT_LINE.search(line.strip())
            if match:
                texts.append(match.group(1).strip())

    output = Path(json_file_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        json.dump(texts, target, ensure_ascii=False, indent=4)
    return texts


if __name__ == "__main__":
    parse_stt_txt_to_json("stt_output.txt", "stt_result.json")

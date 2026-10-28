import ast
import json
import re
from pathlib import Path


OCR_LINE = re.compile(
    r"(\d+(?:\.\d+)?)s\s*---\s*(\d+(?:\.\d+)?)s\s*---(?:文字|æ–‡å­—):\s*(\[.*\])"
)


def parse_ocr_txt_to_json(txt_file_path, json_file_path):
    """Convert OCR time ranges into one JSON record per covered second."""
    results = []
    with open(txt_file_path, "r", encoding="utf-8") as source:
        for line in source:
            match = OCR_LINE.search(line.strip())
            if not match:
                continue
            start_second = int(float(match.group(1)))
            end_second = int(float(match.group(2)))
            try:
                parsed = ast.literal_eval(match.group(3))
            except (SyntaxError, ValueError):
                parsed = re.findall(r"['\"](.*?)['\"]", match.group(3))
            if not isinstance(parsed, (list, tuple)):
                parsed = [parsed]
            texts = [str(text) for text in parsed if str(text).strip()]
            for second in range(start_second, end_second + 1):
                results.append({"second": second, "texts": texts})

    output = Path(json_file_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        json.dump(results, target, ensure_ascii=False, indent=4)
    return results


if __name__ == "__main__":
    parse_ocr_txt_to_json("result.txt", "ocr_result.json")

"""Merge matching STT/filtered EasyOCR files. Only main() is defined.

Phase2 wins spelling/substitution conflicts. Phase1-only occurrences are kept
in sequence, even when the same words appeared earlier. Output keeps the
original Phase1 intervals; these are segment times, not word-level alignment.

Run from the project, or place this script in rewrite_Phase2:
    python merge_stt_easyocr.py
    python merge_stt_easyocr.py --file part_1_1 --overwrite

Uses only the Python standard library. Existing outputs are skipped unless
--overwrite is supplied. Neither input directory is modified.
"""

import argparse
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--stt-dir", type=Path)
    parser.add_argument("--ocr-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--file", help="Process one name, e.g. part_1_1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root
    if root is None:
        for candidate in (*Path(__file__).resolve().parents, Path.cwd()):
            if (candidate / "rewrite_Phase1" / "STT").is_dir():
                root = candidate
                break
    if root is None and (args.stt_dir is None or args.ocr_dir is None):
        parser.error("Project not found. Supply --project-root or both input directories.")
    root = (root or Path.cwd()).resolve()
    stt_dir = (args.stt_dir or root / "rewrite_Phase1" / "STT").resolve()
    ocr_dir = (args.ocr_dir or root / "rewrite_Phase2" / "EasOCR_filtered").resolve()
    output_dir = (args.output_dir or root / "rewrite_Phase2" / "merge_text").resolve()
    if not stt_dir.is_dir() or not ocr_dir.is_dir():
        parser.error("Both input directories must exist.")
    for input_dir in (stt_dir, ocr_dir):
        if output_dir == input_dir or output_dir.is_relative_to(input_dir):
            parser.error("Output must be outside the input directories.")

    # part_1_1.txt <-> part_1_1_frames.txt (also accepts a same-name OCR file).
    ocr_files = {}
    for path in sorted(ocr_dir.glob("*.txt")):
        key = path.stem.removesuffix("_frames")
        if key in ocr_files:
            parser.error(f"Ambiguous Phase2 filenames: {ocr_files[key].name}, {path.name}")
        ocr_files[key] = path
    stt_files = {path.stem: path for path in sorted(stt_dir.glob("*.txt"))}
    names = sorted(stt_files.keys() | ocr_files.keys())
    if args.file:
        requested = Path(args.file).stem.removesuffix("_frames")
        if requested not in names:
            parser.error(f"No input found for {requested}")
        names = [requested]

    pattern = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+?)\s*$")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = skipped = failed = 0
    for name in names:
        if name not in stt_files or name not in ocr_files:
            missing = "Phase1" if name not in stt_files else "Phase2"
            print(f"SKIP {name}: missing {missing} counterpart")
            skipped += 1
            continue
        output_path = output_dir / f"{name}_merged.txt"
        if output_path.exists() and not args.overwrite:
            print(f"SKIP {name}: output exists (use --overwrite)")
            skipped += 1
            continue

        try:
            segments = []
            for number, line in enumerate(stt_files[name].read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                match = pattern.fullmatch(line)
                if match is None:
                    raise ValueError(f"Invalid Phase1 line {number}")
                start, end, text = match.groups()
                if float(end) < float(start):
                    raise ValueError(f"Reversed Phase1 interval at line {number}")
                segments.append((start, end, text))
            if not segments:
                raise ValueError("Phase1 is empty")
            ocr_text = " ".join(line.strip() for line in ocr_files[name].read_text(encoding="utf-8-sig").splitlines() if line.strip())

            # Normalize only for matching; retain raw characters and spaces in
            # output. Keep occurrence positions so repeated words stay distinct.
            sequences = []
            positions = []
            pieces = []
            for texts in ([segment[2] for segment in segments], [ocr_text]):
                normalized = []
                character_positions = []
                raw_pieces = []
                for segment_id, text in enumerate(texts):
                    local_positions = []
                    for offset, character in enumerate(text):
                        value = unicodedata.normalize("NFKC", character).casefold().replace("臺", "台")
                        for normalized_character in value:
                            if normalized_character.isspace() or unicodedata.category(normalized_character).startswith("P"):
                                continue
                            normalized.append(normalized_character)
                            character_positions.append(segment_id)
                            local_positions.append(offset)
                    for index, offset in enumerate(local_positions):
                        if index and offset == local_positions[index - 1]:
                            raw_pieces.append("")
                            continue
                        next_index = index + 1
                        while next_index < len(local_positions) and local_positions[next_index] == offset:
                            next_index += 1
                        end = local_positions[next_index] if next_index < len(local_positions) else len(text)
                        raw_pieces.append(text[0 if index == 0 else offset:end])
                sequences.append("".join(normalized))
                positions.append(character_positions)
                pieces.append(raw_pieces)

            stt_sequence, ocr_sequence = sequences
            alignment = []
            for tag, a0, a1, b0, b1 in SequenceMatcher(None, stt_sequence, ocr_sequence, autojunk=False).get_opcodes():
                if tag == "equal":
                    alignment.extend(zip(range(a0, a1), range(b0, b1)))
                elif tag == "delete":
                    alignment.extend((index, None) for index in range(a0, a1))
                elif tag == "insert":
                    alignment.extend((None, index) for index in range(b0, b1))
                else:
                    # Refine replacement blocks: substitutions use Phase2;
                    # surplus Phase1 characters remain as extra occurrences.
                    left, right = stt_sequence[a0:a1], ocr_sequence[b0:b1]
                    costs = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
                    for i in range(len(left), -1, -1):
                        for j in range(len(right), -1, -1):
                            if i == len(left):
                                costs[i][j] = len(right) - j
                            elif j == len(right):
                                costs[i][j] = len(left) - i
                            else:
                                costs[i][j] = min(costs[i + 1][j + 1] + (left[i] != right[j]), costs[i + 1][j] + 1, costs[i][j + 1] + 1)
                    i = j = 0
                    # Resolve ties from left to right: 位臺南市 -> 味 should
                    # correct 位 and preserve the following 臺南市 occurrence.
                    while i < len(left) or j < len(right):
                        if i < len(left) and j < len(right) and costs[i][j] == costs[i + 1][j + 1] + (left[i] != right[j]):
                            alignment.append((a0 + i, b0 + j))
                            i += 1
                            j += 1
                        elif i < len(left) and costs[i][j] == costs[i + 1][j] + 1:
                            alignment.append((a0 + i, None))
                            i += 1
                        else:
                            alignment.append((None, b0 + j))
                            j += 1

            result = [[] for _ in segments]
            source_index = 0
            emitted_ocr = []
            for stt_index, ocr_index in alignment:
                if stt_index is not None:
                    source_index = stt_index
                segment_id = positions[0][source_index] if positions[0] else 0
                if ocr_index is not None:
                    piece = pieces[1][ocr_index]
                    result[segment_id].append(piece)
                    emitted_ocr.append(piece)
                else:
                    result[segment_id].append(pieces[0][stt_index])
            if ocr_sequence and "".join(emitted_ocr) != ocr_text:
                raise ValueError("Phase2 coverage check failed")

            lines = []
            for segment, text_parts in zip(segments, result):
                start, end, original = segment
                text = "".join(text_parts).strip() or original
                lines.append(f"{start} - {end}: {text}")
            # If Phase2 contains only punctuation, there is no linguistic text
            # to align. Keep it explicitly rather than silently dropping it.
            if ocr_text and not ocr_sequence:
                lines.append(f"{segments[-1][0]} - {segments[-1][1]}: {ocr_text}")
            payload = "\n".join(lines) + "\n"
            with output_path.open("w" if args.overwrite else "x", encoding="utf-8", newline="\n") as destination:
                destination.write(payload)
            print(f"OK {name}: {output_path.name}")
            merged += 1
        except (OSError, UnicodeError, ValueError) as error:
            print(f"ERROR {name}: {error}")
            failed += 1

    print(f"Merged: {merged}; skipped: {skipped}; failed: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

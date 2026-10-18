"""Merge matching STT and OCR timestamp-text files without losing unique STT.

Input/output line format:
    1.200 - 2.880: text

All OCR intervals are retained. When an STT interval overlaps OCR, only text
already represented by OCR is removed. Unique STT prefixes or suffixes are
kept as adjusted intervals. Nearby full-text duplicates are still discarded.
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LINE_PATTERN = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)\s*-\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*:\s*(?P<text>.*?)\s*$"
)


def find_project_root() -> Path:
    for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent):
        if (candidate / "rewrite_Phase1").is_dir() or (
            candidate / "rewrite_Phase2"
        ).is_dir():
            return candidate
    return SCRIPT_DIR


PROJECT_ROOT = find_project_root()
DEFAULT_STT_DIR = PROJECT_ROOT / "rewrite_Phase1" / "STT"
DEFAULT_OCR_DIR = PROJECT_ROOT / "rewrite_Phase2" / "ocr_text"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "rewrite_Phase2" / "merge_text"


@dataclass(frozen=True)
class Segment:
    start_sec: float
    end_sec: float
    text: str
    source: str
    source_line: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge same-named STT/OCR files while giving OCR priority."
    )
    parser.add_argument(
        "--stt-dir",
        type=Path,
        default=DEFAULT_STT_DIR,
        help=f"STT TXT directory (default: {DEFAULT_STT_DIR})",
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=DEFAULT_OCR_DIR,
        help=f"OCR TXT directory (default: {DEFAULT_OCR_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Merged TXT directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=0.25,
        help=(
            "Seconds added around OCR intervals when checking STT coverage "
            "(default: 0.25)"
        ),
    )
    parser.add_argument(
        "--text-similarity",
        type=float,
        default=0.85,
        help="Nearby-text duplicate threshold from 0 to 1 (default: 0.85)",
    )
    parser.add_argument(
        "--text-window",
        type=float,
        default=2.0,
        help=(
            "Maximum interval gap in seconds for text duplicate checking "
            "(default: 2.0)"
        ),
    )
    parser.add_argument(
        "--min-fragment-chars",
        type=int,
        default=2,
        help=(
            "Minimum normalized characters required for a remaining STT "
            "fragment (default: 2)"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.time_tolerance < 0:
        raise ValueError("--time-tolerance must be zero or greater")
    if not 0.0 <= args.text_similarity <= 1.0:
        raise ValueError("--text-similarity must be between 0 and 1")
    if args.text_window < 0:
        raise ValueError("--text-window must be zero or greater")
    if args.min_fragment_chars < 1:
        raise ValueError("--min-fragment-chars must be at least 1")


def read_segments(path: Path, source: str) -> list[Segment]:
    segments: list[Segment] = []
    with path.open("r", encoding="utf-8-sig") as source_file:
        for line_number, raw_line in enumerate(source_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = LINE_PATTERN.match(line)
            if match is None:
                raise ValueError(
                    f"Invalid line format / 格式錯誤: {path}:{line_number}: {line}"
                )

            start_sec = float(match.group("start"))
            end_sec = float(match.group("end"))
            text = match.group("text").strip()
            if end_sec < start_sec:
                raise ValueError(
                    f"End precedes start / 結束時間早於開始時間: "
                    f"{path}:{line_number}"
                )
            if not text:
                continue

            segments.append(
                Segment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=text,
                    source=source,
                    source_line=line_number,
                )
            )
    return segments


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("臺", "台")
    return "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith(("P", "Z"))
    )


def similarity(first: str, second: str) -> float:
    first = normalize_text(first)
    second = normalize_text(second)
    if not first or not second:
        return 0.0
    return SequenceMatcher(None, first, second).ratio()


def normalize_text_with_positions(text: str) -> tuple[str, list[int]]:
    """Normalize text while retaining a map to original character indexes."""
    normalized_characters: list[str] = []
    original_positions: list[int] = []
    for original_index, character in enumerate(text):
        normalized_piece = (
            unicodedata.normalize("NFKC", character)
            .lower()
            .replace("臺", "台")
        )
        for normalized_character in normalized_piece:
            if unicodedata.category(normalized_character).startswith(("P", "Z")):
                continue
            normalized_characters.append(normalized_character)
            original_positions.append(original_index)
    return "".join(normalized_characters), original_positions


def interval_gap(first: Segment, second: Segment) -> float:
    if first.end_sec < second.start_sec:
        return second.start_sec - first.end_sec
    if second.end_sec < first.start_sec:
        return first.start_sec - second.end_sec
    return 0.0


def overlaps_with_tolerance(
    first: Segment,
    second: Segment,
    tolerance: float,
) -> bool:
    return (
        first.start_sec <= second.end_sec + tolerance
        and second.start_sec <= first.end_sec + tolerance
    )


def clean_fragment_text(text: str) -> str:
    return text.strip(" \t\r\n,，.。:：;；!?！？-–—")


def subtract_ocr_text(
    stt_segment: Segment,
    ocr_segment: Segment,
    text_similarity: float,
    min_fragment_chars: int,
) -> list[Segment]:
    """Remove only the portion of STT text already represented by OCR."""
    stt_text, stt_positions = normalize_text_with_positions(stt_segment.text)
    ocr_text = normalize_text(ocr_segment.text)
    if not stt_text or not ocr_text:
        return [stt_segment]

    matcher = SequenceMatcher(None, stt_text, ocr_text)
    matching_blocks = [
        block for block in matcher.get_matching_blocks() if block.size > 0
    ]
    if not matching_blocks:
        return [stt_segment]

    # If nearly all STT characters already occur in OCR, OCR represents the
    # whole STT segment. This also absorbs small OCR/STT spelling differences.
    matched_characters = sum(block.size for block in matching_blocks)
    if matched_characters / len(stt_text) >= text_similarity:
        return []

    # For partial duplicates, require a meaningful contiguous common phrase.
    # This prevents incidental one- or two-character matches from trimming STT.
    longest_index, longest_block = max(
        enumerate(matching_blocks),
        key=lambda indexed_block: indexed_block[1].size,
    )
    minimum_partial_size = max(
        3,
        math.ceil(min(len(stt_text), len(ocr_text)) * 0.40),
    )
    if longest_block.size < minimum_partial_size:
        return [stt_segment]

    # Extend the main matching phrase across neighboring matches separated by
    # at most two characters. Those small gaps are usually OCR/STT spelling
    # differences (for example 貨/獲 or 位/味), not unique spoken content.
    match_start = longest_block.a
    match_end = longest_block.a + longest_block.size
    ocr_match_start = longest_block.b
    ocr_match_end = longest_block.b + longest_block.size
    for block in reversed(matching_blocks[:longest_index]):
        stt_gap = match_start - (block.a + block.size)
        ocr_gap = ocr_match_start - (block.b + block.size)
        if stt_gap > 2 or ocr_gap > 2:
            break
        match_start = block.a
        ocr_match_start = block.b
    for block in matching_blocks[longest_index + 1:]:
        stt_gap = block.a - match_end
        ocr_gap = block.b - ocr_match_end
        if stt_gap > 2 or ocr_gap > 2:
            break
        match_end = block.a + block.size
        ocr_match_end = block.b + block.size

    duplicate_start = stt_positions[match_start]
    duplicate_end = stt_positions[
        match_end - 1
    ] + 1
    prefix_text = clean_fragment_text(stt_segment.text[:duplicate_start])
    suffix_text = clean_fragment_text(stt_segment.text[duplicate_end:])
    duration = stt_segment.end_sec - stt_segment.start_sec

    proportional_prefix_end = stt_segment.start_sec + duration * (
        match_start / len(stt_text)
    )
    proportional_suffix_start = stt_segment.start_sec + duration * (
        match_end / len(stt_text)
    )

    # When the OCR boundary lies inside STT, use it as the clean split point.
    # Otherwise fall back to a character-proportional timestamp estimate.
    prefix_end = (
        ocr_segment.start_sec
        if stt_segment.start_sec < ocr_segment.start_sec < stt_segment.end_sec
        else proportional_prefix_end
    )
    suffix_start = (
        ocr_segment.end_sec
        if stt_segment.start_sec < ocr_segment.end_sec < stt_segment.end_sec
        else proportional_suffix_start
    )

    fragments: list[Segment] = []
    if (
        len(normalize_text(prefix_text)) >= min_fragment_chars
        and prefix_end > stt_segment.start_sec
    ):
        fragments.append(
            Segment(
                start_sec=stt_segment.start_sec,
                end_sec=min(prefix_end, stt_segment.end_sec),
                text=prefix_text,
                source=stt_segment.source,
                source_line=stt_segment.source_line,
            )
        )
    if (
        len(normalize_text(suffix_text)) >= min_fragment_chars
        and suffix_start < stt_segment.end_sec
    ):
        fragments.append(
            Segment(
                start_sec=max(suffix_start, stt_segment.start_sec),
                end_sec=stt_segment.end_sec,
                text=suffix_text,
                source=stt_segment.source,
                source_line=stt_segment.source_line,
            )
        )
    return fragments


def merge_stt_segment(
    stt_segment: Segment,
    ocr_segments: list[Segment],
    time_tolerance: float,
    text_similarity: float,
    text_window: float,
    min_fragment_chars: int,
) -> list[Segment]:
    fragments = [stt_segment]
    for ocr_segment in ocr_segments:
        updated_fragments: list[Segment] = []
        for fragment in fragments:
            if overlaps_with_tolerance(fragment, ocr_segment, time_tolerance):
                updated_fragments.extend(
                    subtract_ocr_text(
                        fragment,
                        ocr_segment,
                        text_similarity,
                        min_fragment_chars,
                    )
                )
                continue

            # Catch the same sentence when OCR/STT timestamps differ slightly.
            if interval_gap(fragment, ocr_segment) <= text_window and similarity(
                fragment.text,
                ocr_segment.text,
            ) >= text_similarity:
                continue

            updated_fragments.append(fragment)
        fragments = updated_fragments
        if not fragments:
            break

    return fragments


def write_segments(path: Path, segments: list[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        for segment in segments:
            destination.write(
                f"{segment.start_sec:.3f} - {segment.end_sec:.3f}: "
                f"{segment.text}\n"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)

    stt_dir = args.stt_dir.expanduser().resolve()
    ocr_dir = args.ocr_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not stt_dir.is_dir():
        raise FileNotFoundError(f"STT directory does not exist / STT 目錄不存在: {stt_dir}")
    if not ocr_dir.is_dir():
        raise FileNotFoundError(f"OCR directory does not exist / OCR 目錄不存在: {ocr_dir}")

    stt_files = {path.name: path for path in stt_dir.glob("*.txt")}
    ocr_files = {path.name: path for path in ocr_dir.glob("*.txt")}
    matching_names = sorted(stt_files.keys() & ocr_files.keys())
    if not matching_names:
        raise FileNotFoundError(
            "No same-named TXT files found in STT and OCR directories / "
            "STT 與 OCR 目錄中找不到同名 TXT"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Matching files / 同名檔案: {len(matching_names)}")
    print(f"Output directory / 輸出目錄: {output_dir}")

    for file_number, name in enumerate(matching_names, start=1):
        ocr_segments = read_segments(ocr_files[name], source="OCR")
        stt_segments = read_segments(stt_files[name], source="STT")

        accepted_stt: list[Segment] = []
        partially_kept = 0
        fully_skipped = 0
        for segment in stt_segments:
            fragments = merge_stt_segment(
                segment,
                ocr_segments,
                args.time_tolerance,
                args.text_similarity,
                args.text_window,
                args.min_fragment_chars,
            )
            if not fragments:
                fully_skipped += 1
            elif fragments != [segment]:
                partially_kept += 1
            accepted_stt.extend(fragments)

        merged = sorted(
            [*ocr_segments, *accepted_stt],
            key=lambda segment: (
                segment.start_sec,
                segment.end_sec,
                0 if segment.source == "OCR" else 1,
            ),
        )

        output_path = output_dir / f"{Path(name).stem}merge.txt"
        write_segments(output_path, merged)
        print(
            f"[{file_number}/{len(matching_names)}] {name}: "
            f"OCR kept={len(ocr_segments)}, "
            f"STT fragments added={len(accepted_stt)}, "
            f"STT partially kept={partially_kept}, "
            f"STT fully skipped={fully_skipped} -> "
            f"{output_path.name}"
        )


if __name__ == "__main__":
    main()

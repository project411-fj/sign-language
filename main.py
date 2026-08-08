import argparse
import json
import os
import re
from pathlib import Path

from tqdm import tqdm

from Phase_1 import cropping
from Phase_2 import OCR_full, voice
from Phase_3 import ocr_compare_stt, ocr_txt_to_json, tts_txt_to_json


BASE_DIR = Path(__file__).resolve().parent


def find_videos(target_dir):
    """Return the MP4 files in natural filename order."""
    target_dir = BASE_DIR / target_dir
    key = lambda path: [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]
    return sorted(
        (path for path in target_dir.iterdir() if path.suffix.casefold() == ".mp4"),
        key=key,
    )


def run_phase_1(source_video):
    """Filter the source video and pass its result to Phase 2."""
    original_cwd = Path.cwd()
    try:
        os.chdir(BASE_DIR)
        filtered_video = BASE_DIR / cropping.crop(str(source_video))
    finally:
        os.chdir(original_cwd)

    return {
        "source_video": source_video,
        "filtered_video": filtered_video,
    }


def run_phase_2(phase_1_result, output_dir, progress):
    """Create OCR and STT text from the Phase 1 result."""
    source_video = phase_1_result["source_video"]
    stem = source_video.stem
    ocr_text = output_dir / f"{stem}_ocr.txt"
    stt_text = output_dir / f"{stem}_stt.txt"

    OCR_full.extract_ocr(phase_1_result["filtered_video"], ocr_text)
    progress.update()

    # The filtered AVI has no audio; Phase 1 carries the source video for STT.
    voice.transcribe_video(source_video, stt_text)
    progress.update()

    return {
        **phase_1_result,
        "ocr_text": ocr_text,
        "stt_text": stt_text,
    }


def run_phase_3(phase_2_result, output_dir, threshold, progress):
    """Convert and match the OCR/STT data returned by Phase 2."""
    stem = phase_2_result["source_video"].stem
    ocr_json = output_dir / f"{stem}_ocr.json"
    stt_json = output_dir / f"{stem}_stt.json"
    final_json = output_dir / f"{stem}_final_subtitles.json"

    ocr_data = ocr_txt_to_json.parse_ocr_txt_to_json(
        phase_2_result["ocr_text"], ocr_json
    )
    progress.update()

    stt_data = tts_txt_to_json.parse_stt_txt_to_json(
        phase_2_result["stt_text"], stt_json
    )
    progress.update()

    final_data = ocr_compare_stt.process_video_subtitles(
        ocr_data, stt_data, threshold
    )
    final_json.write_text(
        json.dumps(final_data, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    progress.update()

    return final_json


def run_pipeline(source_video, output_dir, threshold):
    """Run Phase 1 -> Phase 2 -> Phase 3 for one video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "cropvideos").mkdir(exist_ok=True)

    with tqdm(total=1, desc="Phase 1", unit="step", position=1, leave=False) as bar:
        phase_1_result = run_phase_1(source_video)
        bar.update()

    with tqdm(total=2, desc="Phase 2", unit="step", position=2, leave=False) as bar:
        phase_2_result = run_phase_2(phase_1_result, output_dir, bar)

    with tqdm(total=3, desc="Phase 3", unit="step", position=3, leave=False) as bar:
        return run_phase_3(phase_2_result, output_dir, threshold, bar)


def run_all(target_dir="target", output_dir="results", threshold=80):
    """Process every target MP4 sequentially."""
    videos = find_videos(target_dir)
    results_dir = BASE_DIR / output_dir
    results = {}

    with tqdm(videos, desc="All videos", unit="video", position=0) as overall:
        for video in overall:
            overall.set_postfix_str(video.name)
            results[video.name] = run_pipeline(
                video, results_dir / video.stem, threshold
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", default="target")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--threshold", type=float, default=80)
    args = parser.parse_args()

    for video, final_json in run_all(
        args.target_dir, args.output_dir, args.threshold
    ).items():
        print(f"{video}: {final_json}")


if __name__ == "__main__":
    main()

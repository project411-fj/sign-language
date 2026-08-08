import argparse
from pathlib import Path


def transcribe_video(video_path, output_path, model_size="large-v3", device="auto"):
    from faster_whisper import WhisperModel

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(model_size, device=device)
    segments, _ = model.transcribe(
        str(video_path), language="zh", initial_prompt="這是一段繁體中文的影片。"
    )
    with output_path.open("w", encoding="utf-8") as output:
        output.write("=== Time stamp ===\n")
        for segment in segments:
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
            output.write(line)
    return str(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transcribe a video's speech")
    parser.add_argument("video_path")
    parser.add_argument("output_path")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    print(f"Saved at: {transcribe_video(args.video_path, args.output_path, args.model, args.device)}")


if __name__ == "__main__":
    main()

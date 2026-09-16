from faster_whisper import WhisperModel
from pathlib import Path
from opencc import OpenCC

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float32"
)

cc = OpenCC("s2twp")

video_file = Path(r"part_videos\part_1_1.mp4")

segments, info = model.transcribe(
    str(video_file),
    language="zh",
    beam_size=5,
    vad_filter=True,
    word_timestamps=True
)

with open(fr"STT\{video_file.stem}_single.txt", "w", encoding="utf-8") as file:
    for segment in segments:
        file.write(
            f"{segment.start:.3f} - "
            f"{segment.end:.3f}: "
            f"{cc.convert(segment.text.strip())}\n"
        )
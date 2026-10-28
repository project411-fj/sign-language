from faster_whisper import WhisperModel
from pathlib import Path
from tqdm import tqdm
from opencc import OpenCC
from moviepy import VideoFileClip

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
)

cc = OpenCC("s2twp")

input_directory = Path("part_videos")
output_directory = Path("STT")

for video_file in tqdm(sorted(input_directory.glob("*.mp4"),key=lambda video_file: tuple(map(int, video_file.stem.split("_")[1:])))):

    with VideoFileClip(video_file) as clip:
        if clip.audio is None: continue

    segments, info = model.transcribe(
        str(video_file),
        language = "zh",
        beam_size = 5,
        vad_filter = True,
        word_timestamps = True
    )

    with open(fr"{output_directory}/{video_file.stem}.txt","w",encoding="utf-8") as file:
        for segment in segments:
            file.write(
                f"{segment.start:.3f} - "
                f"{segment.end:.3f}: "
                f"{cc.convert(segment.text.strip())}\n"
            )
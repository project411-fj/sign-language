from pathlib import Path
from moviepy import VideoFileClip


directory = Path(r"time_stamp")


def timestamp_to_seconds(timestamp):
    hours, minutes, seconds = timestamp.split(":")
    return round(int(hours) * 3600 + int(minutes) * 60 + float(seconds),3)

for timestamp_file in directory.iterdir():

    timestamp_ranges = []

    with timestamp_file.open("r", encoding="utf-8-sig") as file:
        for line in file:
            start_text, end_text = line.strip().split(" - ")
            start = timestamp_to_seconds(start_text)
            end = timestamp_to_seconds(end_text)
            timestamp_ranges.append((start, end))

    print(timestamp_ranges)

    video = VideoFileClip(fr"source_videos\{timestamp_file.stem}.mp4")
    Path("part_videos").mkdir(parents=True, exist_ok=True)

    index = 1
    for start, end in timestamp_ranges:
        part = video.subclipped(start, end)
        part.write_videofile(f"part_videos\part_{timestamp_file.stem}_{index}.mp4", codec="libx264", audio_codec="aac")
        index += 1

    video.close()

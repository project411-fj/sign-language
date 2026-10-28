from pathlib import Path
import subprocess
import json

INPUT_PATH = Path("Data Preparation\\part_videos")

total_seconds = 0

for video_path in INPUT_PATH.glob("*.mp4"):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path)
        ],
        capture_output=True,
        text=True,
        check=True
    )

    info = json.loads(result.stdout)
    duration = float(info["format"]["duration"])

    total_seconds += duration

hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"總時長：{int(hours)} 小時 {int(minutes)} 分 {seconds:.2f} 秒")
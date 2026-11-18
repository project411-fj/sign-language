from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
import csv
import av
from faster_whisper import WhisperModel
from opencc import OpenCC
from tqdm import tqdm

PART = "1"
WORKERS = 4
input_directory = Path("Data Preparation/videos/part_videos") / PART      # part_videos/<PART>/<n>/<start>_<end>.mp4
output_directory = Path("Data Preparation/Phase2-STT/context") / PART     # context/<PART>/<n>/<start>_<end>.csv

model = WhisperModel("large-v3", device="cuda", compute_type="float16", num_workers=WORKERS)
cc = OpenCC("s2twp")

videos = {}
for video_file in sorted(input_directory.glob("*/*.mp4"), key=lambda p: (int(p.parent.name), p.stem)):
    with av.open(str(video_file)) as container:
        if container.streams.audio: videos[video_file] = container.duration / 1e6

slots = Queue()
for i in range(1, WORKERS + 1): slots.put(i)
main_bar = tqdm(total=len(videos), desc="總進度", unit="支", position=0)

def transcribe(item):
    video_file, duration = item
    slot = slots.get()
    out_dir = output_directory / video_file.parent.name   # 保留子目錄（1, 2, 3…），避免不同影片的片段檔名互相覆蓋
    out_dir.mkdir(parents=True, exist_ok=True)
    segments, _ = model.transcribe(str(video_file), language="zh", beam_size=5, vad_filter=True)
    done = 0
    with tqdm(total=round(duration, 1), desc=f"{video_file.parent.name}/{video_file.stem}", unit="s", position=slot, leave=False) as bar, \
         open(out_dir / f"{video_file.stem}.csv", "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["start", "end", "text"])
        for s in segments:
            writer.writerow([f"{s.start:.3f}", f"{s.end:.3f}", cc.convert(s.text.strip())])
            bar.update(round(s.end - done, 1))
            done = s.end
    slots.put(slot)
    main_bar.update()

with ThreadPoolExecutor(WORKERS) as executor:
    list(executor.map(transcribe, videos.items()))
main_bar.close()

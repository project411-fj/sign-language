"""語音轉文字：用 Whisper large-v3 轉出 part_videos 每個片段的主播語音 (轉成繁體)。

執行後輸入 part：「3」、「1,2,5」或「all」(part_videos 底下所有 part)
輸入可以是 part_videos/<part>/<n>/<start>_<end>.mp4，或 STT 路徑的 part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/<start>_<end>.mp4。
輸出 context/<part>/<n>/<start>_<end>.csv (欄位 start, end, text)；已經轉過的片段會跳過。
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
import csv
import re

import av
from faster_whisper import WhisperModel
from opencc import OpenCC
from tqdm import tqdm

INPUT_ROOT = Path("Data Preparation/videos/part_videos")     # part_videos/<part>/<n>/<start>_<end>.mp4
OUTPUT_ROOT = Path("Data Preparation/Phase2-STT/context")    # context/<part>/<n>/<start>_<end>.csv
WORKERS = 8
SKIP_EXISTING = True   # 已經有 .csv 的片段就跳過 (輸入 all 時不會把舊的 part 重做一次)；要重做就改成 False


BOX_RE = re.compile(r"\d+(\.\d+)?(_\d+(\.\d+)?){3}")   # STT 路徑的範圍資料夾，例如 0.20_0.73_0.10_0.33


def clip_number(video_file):
    """片段屬於哪支影片 (<n>)：part_videos/<part>/<n>/<start>_<end>.mp4，
    或 STT 路徑的 part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/<start>_<end>.mp4。"""
    parent = video_file.parent
    return parent.parent.name if BOX_RE.fullmatch(parent.name) else parent.name


def list_clips(input_directory):
    """兩種結構的片段都找出來 (略過暫存檔)，依 <n>、檔名排序。"""
    clips = [p for p in input_directory.glob("*/*.mp4")] + \
            [p for p in input_directory.glob("*/*/*.mp4") if BOX_RE.fullmatch(p.parent.name)]
    clips = [p for p in clips if re.fullmatch(r"\d+_\d+", p.stem)]
    return sorted(clips, key=lambda p: (int(clip_number(p)), p.stem))


def parse_parts(answer, root):
    """「3」→ ["3"]；「1,2,5」→ ["1", "2", "5"]；「all」→ root 底下所有 part (依數字排序)。"""
    answer = answer.strip()
    if answer.lower() == "all":
        return sorted((p.name for p in root.iterdir() if p.is_dir()), key=lambda x: (len(x), x))
    return [p.strip() for p in answer.split(",") if p.strip()]


def run_part(part, model, cc):
    input_directory, output_directory = INPUT_ROOT / part, OUTPUT_ROOT / part
    if not input_directory.is_dir():
        print(f"part {part}：找不到 {input_directory}，跳過")
        return

    def output_of(video_file):
        # 保留子目錄（1, 2, 3…），避免不同影片的片段檔名互相覆蓋
        return output_directory / clip_number(video_file) / f"{video_file.stem}.csv"

    all_videos = list_clips(input_directory)
    videos, skipped, no_audio = {}, 0, 0
    for video_file in all_videos:
        if SKIP_EXISTING and output_of(video_file).exists():
            skipped += 1
            continue
        with av.open(str(video_file)) as container:
            if container.streams.audio:
                videos[video_file] = container.duration / 1e6
            else:
                no_audio += 1
    print(f"part {part}：共 {len(all_videos)} 支片段，"
          + (f"{skipped} 支已經轉過 (跳過)，" if skipped else "")
          + (f"{no_audio} 支沒有音軌，" if no_audio else "")
          + f"這次處理 {len(videos)} 支")
    if not videos:
        return

    slots = Queue()
    for i in range(1, WORKERS + 1):
        slots.put(i)
    failed = []
    main_bar = tqdm(total=len(videos), desc=f"part {part}", unit="支", position=0)

    def transcribe(item):
        video_file, duration = item
        slot = slots.get()
        output = output_of(video_file)
        temp = output.with_name(output.stem + ".tmp.csv")   # 寫完才改名，中斷不會留下只轉一半的 .csv
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            segments, _ = model.transcribe(str(video_file), language="zh", beam_size=5, vad_filter=True)
            done = 0
            with tqdm(total=round(duration, 1), desc=f"{clip_number(video_file)}/{video_file.stem}", unit="s",
                      position=slot, leave=False) as bar, \
                    open(temp, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["start", "end", "text"])
                for s in segments:
                    writer.writerow([f"{s.start:.3f}", f"{s.end:.3f}", cc.convert(s.text.strip())])
                    bar.update(round(s.end - done, 1))
                    done = s.end
            temp.replace(output)
        except Exception as exc:                            # 一支失敗不影響其他片段
            temp.unlink(missing_ok=True)
            failed.append((video_file, exc))
        finally:
            slots.put(slot)
            main_bar.update()

    with ThreadPoolExecutor(WORKERS) as executor:
        list(executor.map(transcribe, videos.items()))
    main_bar.close()

    if failed:
        tqdm.write(f"\npart {part} 失敗 {len(failed)} 支 (下次執行會自動重做，因為沒有輸出檔)：")
        for video_file, exc in failed:
            tqdm.write(f"  {video_file.as_posix()}：{exc}")


def main():
    parts = parse_parts(input("part (例如 3、1,2,5 或 all): "), INPUT_ROOT)
    if not parts:
        print("沒有輸入 part")
        return
    print(f"要處理的 part：{'、'.join(parts)}")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16", num_workers=WORKERS)  # 只載入一次
    cc = OpenCC("s2twp")
    for part in parts:
        run_part(part, model, cc)


if __name__ == "__main__":
    main()

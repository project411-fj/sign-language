"""語音轉文字：用 Breeze-ASR-25 (聯發科創新基地，針對台灣華語與中英夾雜微調的 Whisper) 轉出 part_videos 每個片段的主播語音。

執行後輸入 part：「3」、「1,2,5」或「all」(part_videos 底下所有 part)
輸出 context_breeze/<part>/<n>/<start>_<end>.csv (欄位 start, end, text)；已經轉過的片段會跳過。
輸入與輸出格式和 voice.py (faster-whisper large-v3) 相同，只是輸出到另一個資料夾，方便兩者比較。

需要的套件：torch、transformers、av、opencc、tqdm、numpy (第一次執行會從 Hugging Face 下載模型，約 3 GB)
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv

import av
import numpy as np
import torch
from opencc import OpenCC
from tqdm import tqdm
from transformers import pipeline

INPUT_ROOT = Path("Data Preparation/videos/part_videos")            # part_videos/<part>/<n>/<start>_<end>.mp4
OUTPUT_ROOT = Path("Data Preparation/Phase2-STT/context_breeze")    # context_breeze/<part>/<n>/<start>_<end>.csv
MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
SAMPLE_RATE = 16_000   # Whisper 系列模型固定吃 16 kHz 單聲道
DECODE_WORKERS = 4     # 在背景解碼音訊的執行緒數；GPU 一次只轉一支，解碼和轉錄同時進行
PREFETCH = 8           # 最多先解碼好幾支等著轉 (避免整個 part 的音訊同時放在記憶體裡)
SKIP_EXISTING = True   # 已經有 .csv 的片段就跳過 (輸入 all 時不會把舊的 part 重做一次)；要重做就改成 False


def parse_parts(answer, root):
    """「3」→ ["3"]；「1,2,5」→ ["1", "2", "5"]；「all」→ root 底下所有 part (依數字排序)。"""
    answer = answer.strip()
    if answer.lower() == "all":
        return sorted((p.name for p in root.iterdir() if p.is_dir()), key=lambda x: (len(x), x))
    return [p.strip() for p in answer.split(",") if p.strip()]


def load_audio(video_file):
    """用 PyAV 把影片的音軌解碼成 16 kHz 單聲道 float32 (numpy)。"""
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(str(video_file)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):               # 把 resampler 裡剩下的樣本也取出來
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def prefetched(executor, items, func, size):
    """依序回傳 (item, func(item))，背景最多先算好 size 個。"""
    items, pending = iter(items), deque()
    for item in items:
        pending.append((item, executor.submit(func, item)))
        if len(pending) >= size:
            break
    while pending:
        item, future = pending.popleft()
        next_item = next(items, None)
        if next_item is not None:
            pending.append((next_item, executor.submit(func, next_item)))
        try:
            yield item, future.result(), None
        except Exception as exc:
            yield item, None, exc


def run_part(part, asr, cc, executor):
    input_directory, output_directory = INPUT_ROOT / part, OUTPUT_ROOT / part
    if not input_directory.is_dir():
        print(f"part {part}：找不到 {input_directory}，跳過")
        return

    def output_of(video_file):
        # 保留子目錄（1, 2, 3…），避免不同影片的片段檔名互相覆蓋
        return output_directory / video_file.parent.name / f"{video_file.stem}.csv"

    all_videos = sorted(input_directory.glob("*/*.mp4"), key=lambda p: (int(p.parent.name), p.stem))
    videos, skipped, no_audio, failed = [], 0, 0, []
    for video_file in all_videos:
        if SKIP_EXISTING and output_of(video_file).exists():
            skipped += 1
            continue
        try:
            with av.open(str(video_file)) as container:
                if container.streams.audio:
                    videos.append(video_file)
                else:
                    no_audio += 1
        except Exception as exc:                            # 壞掉的影片檔：記下來，最後一起列出
            failed.append((video_file, exc))
    print(f"part {part}：共 {len(all_videos)} 支片段，"
          + (f"{skipped} 支已經轉過 (跳過)，" if skipped else "")
          + (f"{no_audio} 支沒有音軌，" if no_audio else "")
          + (f"{len(failed)} 支打不開，" if failed else "")
          + f"這次處理 {len(videos)} 支")

    for video_file, audio, exc in tqdm(prefetched(executor, videos, load_audio, PREFETCH),
                                       total=len(videos), desc=f"part {part}", unit="支", disable=not videos):
        output = output_of(video_file)
        temp = output.with_name(output.stem + ".tmp.csv")   # 寫完才改名，中斷不會留下只轉一半的 .csv
        try:
            if exc is not None:
                raise exc
            output.parent.mkdir(parents=True, exist_ok=True)
            duration = len(audio) / SAMPLE_RATE
            rows = []
            if len(audio) > 0:
                # return_timestamps=True：超過 30 秒的片段會自動分段接續轉錄，並回傳每一句的起訖時間
                result = asr(audio.copy(), return_timestamps=True,
                             generate_kwargs={"language": "zh", "task": "transcribe"})
                for chunk in result.get("chunks", []):
                    text = cc.convert(chunk["text"].strip())
                    if not text:
                        continue
                    start, end = chunk["timestamp"]
                    start = start or 0.0
                    end = duration if end is None else end    # 最後一句有時沒有結束時間
                    rows.append([f"{start:.3f}", f"{end:.3f}", text])
            with open(temp, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["start", "end", "text"])
                writer.writerows(rows)
            temp.replace(output)
        except Exception as exc:                            # 一支失敗不影響其他片段
            temp.unlink(missing_ok=True)
            failed.append((video_file, exc))
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("找不到 GPU，改用 CPU (會非常慢)")
    asr = pipeline(                                         # 模型只載入一次
        "automatic-speech-recognition",
        model=MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
    )
    cc = OpenCC("s2twp")   # Breeze 本來就輸出繁體；保險起見仍轉一次台灣用字，和 voice.py 一致
    with ThreadPoolExecutor(DECODE_WORKERS) as executor:
        for part in parts:
            run_part(part, asr, cc, executor)


if __name__ == "__main__":
    main()

"""語音轉文字：用 Breeze-ASR-25 (聯發科創新基地，針對台灣華語與中英夾雜微調的 Whisper) 轉出 part_videos 每個片段的主播語音。

執行後輸入 part：「3」、「1,2,5」或「all」(part_videos 底下所有 part)
輸入可以是 part_videos/<part>/<n>/<start>_<end>.mp4，或 STT 路徑的 part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/<start>_<end>.mp4。
輸出 context_breeze/<part>/<n>/<start>_<end>.csv (欄位 start, end, text)；已經轉過的片段會跳過。
輸入與輸出格式和 voice.py (faster-whisper large-v3) 相同，只是輸出到另一個資料夾，方便兩者比較。

時間戳記：Whisper 系列模型的句子時間是一句接一句排下去的 (上一句的 end = 下一句的 start)，
句子之間的停頓會被算進前後句裡。所以這裡另外用 Silero VAD 找出「實際有人聲」的區間，
把每一句的 start / end 對齊到最近的停頓邊緣 (詳見 refine_timestamps)，句子之間就會留下真正的停頓。

需要的套件：torch、transformers、av、opencc、tqdm、numpy、silero-vad
(第一次執行會從 Hugging Face 下載 Breeze 模型，約 3 GB；Silero VAD 的模型包含在 pip 套件裡)
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import re
import threading

import av
import numpy as np
import torch
from opencc import OpenCC
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm
from transformers import pipeline

INPUT_ROOT = Path("Data Preparation/videos/part_videos")            # part_videos/<part>/<n>/<start>_<end>.mp4
OUTPUT_ROOT = Path("Data Preparation/Phase2-STT/context_breeze")    # context_breeze/<part>/<n>/<start>_<end>.csv
MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
SAMPLE_RATE = 16_000   # Whisper 系列模型固定吃 16 kHz 單聲道
DECODE_WORKERS = 4     # 在背景解碼音訊的執行緒數；GPU 一次只轉一支，解碼和轉錄同時進行
PREFETCH = 8           # 最多先解碼好幾支等著轉 (避免整個 part 的音訊同時放在記憶體裡)
SKIP_EXISTING = True   # 已經有 .csv 的片段就跳過 (輸入 all 時不會把舊的 part 重做一次)；要重做就改成 False

# ---- 用 VAD 修正時間戳記 ----
VAD_REFINE = True          # False：直接輸出模型給的時間 (句子會首尾相接)
VAD_MIN_SILENCE_MS = 150   # 至少停頓這麼久才算句子間的空白 (太小會把字和字之間的空隙也當成停頓)
VAD_SPEECH_PAD_MS = 30     # 人聲區間前後各多留一點，避免切到字頭字尾
SNAP_TOLERANCE = 0.6       # 句子邊界離停頓多近 (秒) 才對齊過去；模型的時間通常誤差在半秒內


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


_vad = threading.local()   # Silero VAD 有內部狀態，每個解碼執行緒各用一個


def detect_speech(audio):
    """用 Silero VAD 找出有人聲的區間，回傳 [(start 秒, end 秒), ...]。"""
    if not hasattr(_vad, "model"):
        _vad.model = load_silero_vad()
    stamps = get_speech_timestamps(torch.from_numpy(audio), _vad.model, sampling_rate=SAMPLE_RATE,
                                   min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                                   speech_pad_ms=VAD_SPEECH_PAD_MS)   # 回傳樣本數，比 return_seconds 精準
    return [(t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE) for t in stamps]


def load_audio_and_speech(video_file):
    """背景執行緒：解碼音訊並做 VAD (都在 CPU 上，和 GPU 轉錄同時進行)。"""
    audio = load_audio(video_file)
    speech = detect_speech(audio) if VAD_REFINE and len(audio) else []
    return audio, speech


def refine_timestamps(rows, speech, duration):
    """把模型給的句子時間對齊到 VAD 找到的停頓邊緣。

    rows：[(start, end, text), ...]，speech：有人聲的區間。
    停頓 (gap) = 沒有人聲的區間，包含開頭和結尾的靜音。
    - start 落在停頓裡，或離某段停頓不到 SNAP_TOLERANCE 秒 → 改成那段停頓的結束 (人聲開始)
    - end   落在停頓裡，或離某段停頓不到 SNAP_TOLERANCE 秒 → 改成那段停頓的開始 (人聲結束)
    上一句的 end 和下一句的 start 原本是同一個點，會找到同一段停頓，於是被拆成停頓的兩端。
    兩句之間沒有停頓 (連著講) 就維持原本的邊界。
    整句都落在沒有人聲的地方 (可能是模型幻覺，或 VAD 沒偵測到) 就保留原本的時間。
    """
    if not speech:
        return rows
    gaps, previous = [], 0.0
    for a, b in speech:
        if a > previous:
            gaps.append((previous, a))
        previous = max(previous, b)
    if duration > previous:
        gaps.append((previous, duration))
    if not gaps:
        return rows

    def nearest_gap(t):
        distance, gap = min(((0.0 if a <= t <= b else min(abs(t - a), abs(t - b))), (a, b)) for a, b in gaps)
        return gap if distance <= SNAP_TOLERANCE else None

    refined, previous_end = [], 0.0
    for start, end, text in rows:
        gap = nearest_gap(start)
        new_start = max(gap[1] if gap else start, previous_end)
        gap = nearest_gap(end)
        new_end = gap[0] if gap else end
        if new_end - new_start < 0.05:                      # 對齊後沒剩下人聲：保留原本的時間
            new_start, new_end = max(start, previous_end), max(end, previous_end)
        refined.append((new_start, new_end, text))
        previous_end = new_end
    return refined


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
        return output_directory / clip_number(video_file) / f"{video_file.stem}.csv"

    all_videos = list_clips(input_directory)
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

    for video_file, loaded, exc in tqdm(prefetched(executor, videos, load_audio_and_speech, PREFETCH),
                                       total=len(videos), desc=f"part {part}", unit="支", disable=not videos):
        output = output_of(video_file)
        temp = output.with_name(output.stem + ".tmp.csv")   # 寫完才改名，中斷不會留下只轉一半的 .csv
        try:
            if exc is not None:
                raise exc
            audio, speech = loaded
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
                    rows.append((start, end, text))
                if VAD_REFINE:
                    rows = refine_timestamps(rows, speech, duration)
            with open(temp, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["start", "end", "text"])
                writer.writerows([f"{start:.3f}", f"{end:.3f}", text] for start, end, text in rows)
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

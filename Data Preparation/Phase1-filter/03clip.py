from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager
from pathlib import Path

from moviepy import VideoFileClip
from tqdm import tqdm

# ========== 設定 ==========
BASE = Path(__file__).resolve().parent
INTERVAL_DIR = BASE / "02intervals"
VIDEO_DIR = BASE.parent / "videos" / "source_videos"   # Data Preparation/videos/source_videos
OUTPUT = BASE.parent / "videos" / "part_videos"

FRAME_START = 1      # frame_000001.jpg 是影片的第 0 幀；如果檔名從 0 開始就改成 0
WORKERS = 4          # 同時處理幾支影片（每支一個程序）
THREADS = 2          # 每個程序給 ffmpeg 編碼的執行緒數；WORKERS × THREADS ≈ CPU 核心數
CODEC = "libx264"    # 有 NVIDIA 顯卡可改成 "h264_nvenc"
PRESET = "medium"  # MoviePy 預設是 medium，越快檔案越大、畫質幾乎一樣


def cut_video(num, n, intervals, q):
    """剪一支影片的所有片段，透過 q 通知主程式：開始（片段數）、每剪完一段、結束。"""
    q.put(("start", n, len(intervals)))
    out_dir = OUTPUT / num / n
    out_dir.mkdir(parents=True, exist_ok=True)

    with VideoFileClip(str(VIDEO_DIR / num / f"{n}.mp4")) as video:
        fps = video.fps
        for s, e in intervals:   # 區間由前往後，讀取器只需往後跳，不會重新從頭開
            t0 = (s - FRAME_START) / fps
            t1 = min((e - FRAME_START + 1) / fps, video.duration)   # +1：包含結束那一幀
            video.subclipped(t0, t1).write_videofile(
                str(out_dir / f"{s:06d}_{e:06d}.mp4"),
                fps=fps, codec=CODEC, audio_codec="aac", preset=PRESET, threads=THREADS,
                temp_audiofile_path=str(out_dir), logger=None,
            )
            q.put(("tick", n, 1))
    q.put(("end", n, 0))


if __name__ == "__main__":   # Windows 開多程序必須包在這裡面
    num = input("part: ").strip()
    txts = sorted((INTERVAL_DIR / num).glob("*_frames.txt"), key=lambda p: int(p.stem.split("_")[0]))

    # 先讀完所有區間，才知道總片段數
    jobs = []
    for txt in txts:
        n = txt.stem.split("_")[0]   # 1_frames.txt → 1 → 對應 1.mp4
        intervals = [tuple(map(int, line.split())) for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if intervals:
            jobs.append((n, intervals))

    # 主進度條：影片數；副進度條：每個同時在剪的影片各一條，顯示該影片剪了幾個片段
    main_bar = tqdm(total=len(jobs), desc=f"part {num}", unit="video", position=0)
    sub_bars = [tqdm(total=1, desc="  等待中", unit="clip", position=i + 1) for i in range(WORKERS)]
    free = list(range(WORKERS))   # 目前空著的副進度條
    using = {}                    # 影片編號 → 它用的副進度條

    def handle(msg):
        kind, n, val = msg
        if kind == "start":
            i = free.pop(0)
            using[n] = i
            sub_bars[i].reset(total=val)
            sub_bars[i].set_description(f"  {n}.mp4")
        elif kind == "tick":
            sub_bars[using[n]].update(val)
        else:
            free.append(using.pop(n))
            free.sort()

    with Manager() as manager, ProcessPoolExecutor(WORKERS) as executor:
        q = manager.Queue()
        pending = {executor.submit(cut_video, num, n, iv, q) for n, iv in jobs}
        while pending:
            done, pending = wait(pending, timeout=0.2)
            while not q.empty():
                handle(q.get())
            for f in done:
                f.result()   # 某支影片失敗時把錯誤拋出來
                main_bar.update()
        while not q.empty():
            handle(q.get())

    for bar in sub_bars:
        bar.close()
    main_bar.close()
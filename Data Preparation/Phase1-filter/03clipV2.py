import subprocess
from concurrent.futures import ThreadPoolExecutor, wait
from fractions import Fraction
from pathlib import Path
from queue import Queue

from tqdm import tqdm

# ========== 設定 ==========
BASE = Path(__file__).resolve().parent
INTERVAL_DIR = BASE / "02intervals"
VIDEO_DIR = BASE.parent / "videos" / "source_videos"   # Data Preparation/videos/source_videos
OUTPUT = BASE.parent / "videos" / "part_videos"

FFMPEG = "ffmpeg"    # 不在 PATH 就改成 ffmpeg.exe 完整路徑
FFPROBE = "ffprobe"  # 和 ffmpeg 同一個資料夾

FRAME_START = 1      # frame_000001.jpg 是影片的第 0 幀；如果檔名從 0 開始就改成 0
WORKERS = 4          # 同時處理幾支影片
THREADS = 2          # 每個 ffmpeg 編碼的執行緒數；WORKERS × THREADS ≈ CPU 核心數

# GPU：解碼、編碼都在 NVIDIA 顯卡上，畫面不搬回 CPU
CODEC = "h264_nvenc"
CODEC_ARGS = ["-preset", "p4"]
HWACCEL = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
PIX_FMT = []                       # 畫面留在 GPU 上，不能再指定 pix_fmt
# 改回 CPU（與原本 MoviePy 設定相同）
# CODEC = "libx264"
# CODEC_ARGS = ["-preset", "medium"]
# HWACCEL = []
# PIX_FMT = ["-pix_fmt", "yuv420p"]


def video_fps(path):
    """用 ffprobe 讀影片 fps，回傳精確分數（例如 30000/1001）。"""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return Fraction(out)


def cut_video(num, n, intervals, q):
    """剪一支影片的所有片段，透過 q 通知主程式：開始（片段數）、每剪完一段、結束。"""
    q.put(("start", n, len(intervals)))
    out_dir = OUTPUT / num / n
    out_dir.mkdir(parents=True, exist_ok=True)

    src = VIDEO_DIR / num / f"{n}.mp4"
    fps = video_fps(src)
    for s, e in intervals:
        t0 = (s - FRAME_START) / fps
        duration = (e - s + 1) / fps   # +1：包含結束那一幀
        cmd = [
            FFMPEG, "-y", "-loglevel", "error", *HWACCEL,
            "-ss", f"{float(t0):.6f}", "-i", str(src), "-t", f"{float(duration):.6f}",
            "-c:v", CODEC, *CODEC_ARGS, "-threads", str(THREADS), *PIX_FMT,
            "-c:a", "aac",
            str(out_dir / f"{s:06d}_{e:06d}.mp4"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"{n}.mp4 {s}-{e}：{result.stderr.strip()}")
        q.put(("tick", n, 1))
    q.put(("end", n, 0))


if __name__ == "__main__":
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

    # 實際工作在 ffmpeg 子程序裡做，用執行緒就夠了，不需要多程序和 Manager
    q = Queue()
    with ThreadPoolExecutor(WORKERS) as executor:
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
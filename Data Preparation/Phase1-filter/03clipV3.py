"""依 02intervals 的區間剪片段 (03clipV2.py 的新版，同樣用 NVIDIA 顯卡解碼、編碼)。

    02intervals/<part>/<n>_frames.txt + source_videos/<part>/<n>.mp4 → part_videos/<part>/<n>/...

.txt 每一行可以是兩種格式 (同一個檔案裡可以混用)：
    「起始格 結束格」                (02style.py 的輸出)        → part_videos/<part>/<n>/<start>_<end>.mp4  (跟 03clipV2.py 相同)
    「起始格 結束格 範圍」            (02signer_track.py 的輸出) → part_videos/<part>/<n>/<範圍>/<start>_<end>.mp4
範圍 = 手語員的 TOP_BOTTOM_LEFT_RIGHT (例如 0.00_0.92_0.00_0.34)，Phase4 STT/01cropV2.py 會依這個資料夾名稱裁出手語員；
手語員中途換位置時，同一支影片會有好幾個範圍資料夾。

跟 03clipV2.py 不同的地方：
    - 看得懂第三欄範圍
    - 已經剪好的片段會跳過 (SKIP_EXISTING)；先寫成 .tmp.mp4 再改名，中斷不會留下剪一半的檔案
    - 一個片段失敗不會停掉整支影片，最後列出失敗的片段 (再執行一次會只重剪沒剪好的)
"""
import re
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
SKIP_EXISTING = True # 已經剪好的片段跳過；要全部重剪就改成 False

# GPU：解碼、編碼都在 NVIDIA 顯卡上，畫面不搬回 CPU (跟 03clipV2.py 相同)
CODEC = "h264_nvenc"
CODEC_ARGS = ["-preset", "p4"]
HWACCEL = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
PIX_FMT = []                       # 畫面留在 GPU 上，不能再指定 pix_fmt
# 改回 CPU
# CODEC = "libx264"
# CODEC_ARGS = ["-preset", "medium"]
# HWACCEL = []
# PIX_FMT = ["-pix_fmt", "yuv420p"]

BOX_RE = re.compile(r"\d+(\.\d+)?(_\d+(\.\d+)?){3}")


def video_fps(path):
    """用 ffprobe 讀影片 fps，回傳精確分數（例如 30000/1001）。"""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return Fraction(out)


def read_intervals(txt):
    """02intervals 的 .txt → [(起, 迄, 範圍或 None), ...]；兩欄、三欄都可以。"""
    out = []
    for no, line in enumerate(txt.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) not in (2, 3):
            raise ValueError(f"{txt} 第 {no} 行格式不對 (要「起始格 結束格」或「起始格 結束格 範圍」)：{line}")
        s, e = int(parts[0]), int(parts[1])
        box = parts[2] if len(parts) == 3 else None
        if box is not None and not BOX_RE.fullmatch(box):
            raise ValueError(f"{txt} 第 {no} 行的第三欄 {box} 不是範圍 (TOP_BOTTOM_LEFT_RIGHT)")
        if e < s:
            raise ValueError(f"{txt} 第 {no} 行結束格比起始格小：{line}")
        out.append((s, e, box))
    return out


def output_of(num, n, s, e, box):
    folder = OUTPUT / num / n if box is None else OUTPUT / num / n / box
    return folder / f"{s:06d}_{e:06d}.mp4"


def cut_video(num, n, intervals, q):
    """剪一支影片的所有片段，透過 q 通知主程式：開始（片段數）、每剪完一段、結束。回傳失敗的片段。"""
    q.put(("start", n, len(intervals)))
    src = VIDEO_DIR / num / f"{n}.mp4"
    fps = video_fps(src)
    failed = []
    for s, e, box in intervals:
        output = output_of(num, n, s, e, box)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(output.stem + ".tmp.mp4")
        t0 = (s - FRAME_START) / fps
        duration = (e - s + 1) / fps   # +1：包含結束那一幀
        cmd = [
            FFMPEG, "-y", "-loglevel", "error", *HWACCEL,
            "-ss", f"{float(t0):.6f}", "-i", str(src), "-t", f"{float(duration):.6f}",
            "-c:v", CODEC, *CODEC_ARGS, "-threads", str(THREADS), *PIX_FMT,
            "-c:a", "aac",
            str(temp),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode == 0:
            temp.replace(output)
        else:
            temp.unlink(missing_ok=True)
            failed.append(f"{num}/{n}.mp4 {s}-{e}：{result.stderr.strip()}")
        q.put(("tick", n, 1))
    q.put(("end", n, 0))
    return failed


if __name__ == "__main__":
    num = input("part: ").strip()
    txts = sorted((INTERVAL_DIR / num).glob("*_frames.txt"), key=lambda p: int(p.stem.split("_")[0]))
    if not txts:
        raise SystemExit(f"{INTERVAL_DIR / num} 裡沒有 *_frames.txt")

    # 先讀完所有區間，才知道總片段數；同時檢查影片存在、格式正確，避免剪到一半才失敗
    jobs, problems, skipped = [], [], 0
    for txt in txts:
        n = txt.stem.split("_")[0]   # 1_frames.txt → 1 → 對應 1.mp4
        try:
            intervals = read_intervals(txt)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if intervals and not (VIDEO_DIR / num / f"{n}.mp4").is_file():
            problems.append(f"找不到影片 {VIDEO_DIR / num / f'{n}.mp4'}")
            continue
        todo = [iv for iv in intervals if not (SKIP_EXISTING and output_of(num, n, *iv).exists())]
        skipped += len(intervals) - len(todo)
        if todo:
            jobs.append((n, todo))
            boxes = sorted({b for _, _, b in todo if b})
            print(f"{n}.mp4：{len(todo)} 個片段" + (f"，範圍 {'、'.join(boxes)}" if boxes else ""))
    if problems:
        raise SystemExit("\n".join(problems))
    total = sum(len(iv) for _, iv in jobs)
    print(f"共 {total} 個片段要剪" + (f" ({skipped} 個已經剪好，跳過)" if skipped else ""))
    if not jobs:
        raise SystemExit(0)

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

    # 實際工作在 ffmpeg 子程序裡做，用執行緒就夠了
    q = Queue()
    failed = []
    with ThreadPoolExecutor(WORKERS) as executor:
        pending = {executor.submit(cut_video, num, n, iv, q) for n, iv in jobs}
        while pending:
            done, pending = wait(pending, timeout=0.2)
            while not q.empty():
                handle(q.get())
            for f in done:
                failed += f.result()   # 整支影片失敗 (例如讀不到 fps) 時把錯誤拋出來
                main_bar.update()
        while not q.empty():
            handle(q.get())

    for bar in sub_bars:
        bar.close()
    main_bar.close()
    if failed:
        print(f"\n失敗 {len(failed)} 個片段 (再執行一次會只重剪這些)：")
        for line in failed:
            print(f"  {line}")
    else:
        print("\n全部完成")

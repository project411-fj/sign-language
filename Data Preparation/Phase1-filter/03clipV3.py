"""依 02intervals 的區間剪片段，同時輸出「全幅片段」和「裁好的手語員片段」(用 NVIDIA 顯卡解碼、編碼)。

    02intervals/<part>/<n>_frames.txt + source_videos/<part>/<n>.mp4
        → part_videos/<part>/<n>/<start>_<end>.mp4                    全幅，跟 03clipV2.py 一樣
        → keypoint_videos/<part>/<n>/<start>_<end>_upper_body.mp4     依範圍裁出手語員，640×640
                                                                     (跟 Phase4 STT/01cropV2.py 的輸出相同，可以直接跑 02keypoint_*.py)

.txt 每一行可以是兩種格式 (同一個檔案裡可以混用)：
    「起始格 結束格 範圍」 (02signer_track.py 的輸出) → 全幅 + 裁切兩個都輸出
    「起始格 結束格」      (02style.py 的輸出)        → 只輸出全幅 (沒有範圍可以裁；手語員請用 Phase4 OCR/01cropV2.py 找)
範圍 = 手語員的 TOP_BOTTOM_LEFT_RIGHT (佔畫面高 / 寬的比例，例如 0.00_0.92_0.00_0.34)。
兩個輸出都直接從原影片的同一段時間剪，格數相同 (裁切版不是從全幅片段再轉一次，畫質不會多損失一次)。

每支影片會寫一個 part_videos/<part>/<n>/crop_boxes.csv (片段, 範圍)，記錄每個片段用哪個範圍裁切；
Phase5 STT/ 的程式靠這個檔 (或範圍資料夾) 判斷這個 part 是 STT 路徑。

其他：已經剪好的片段跳過 (SKIP_EXISTING)；先寫成 .tmp.mp4 再改名；一個片段失敗不影響其他片段，最後列出失敗的片段。
"""
import csv
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
VIDEO_DIR = BASE.parent / "videos" / "source_videos"       # Data Preparation/videos/source_videos
OUTPUT = BASE.parent / "videos" / "part_videos"            # 全幅片段
KEYPOINT_OUTPUT = BASE.parent / "videos" / "keypoint_videos"   # 裁好的手語員片段

FFMPEG = "ffmpeg"    # 不在 PATH 就改成 ffmpeg.exe 完整路徑
FFPROBE = "ffprobe"  # 和 ffmpeg 同一個資料夾

FRAME_START = 1      # frame_000001.jpg 是影片的第 0 幀；如果檔名從 0 開始就改成 0
WORKERS = 4          # 同時處理幾支影片 (GPU 模式下每支佔一個 NVENC 編碼器，一般顯卡同時最多 5~8 個)
THREADS = 2          # 每個 ffmpeg 編碼的執行緒數；WORKERS × THREADS ≈ CPU 核心數
SKIP_EXISTING = True # 已經剪好的片段跳過；要全部重剪就改成 False

# 裁切版 (跟 Phase4 STT/01cropV2.py 相同)
OUTPUT_SIZE = 640
PAD_TO_SQUARE = True   # True：等比例縮放到最長邊 640 後補黑邊成 640×640；False：只縮放不補邊

GPU = True           # True：解碼、裁切、縮放、編碼都在 NVIDIA 顯卡上 (裁切版補黑邊在 CPU)；False：全部用 CPU
GPU_FULL_ARGS = ["-c:v", "h264_nvenc", "-preset", "p4"]                                     # 全幅 (跟 03clipV2.py 相同)
GPU_CROP_ARGS = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", "0"]   # 裁切版 (跟 01cropV2.py 相同)
CPU_FULL_ARGS = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p"]
CPU_CROP_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]

BOX_RE = re.compile(r"\d+(\.\d+)?(_\d+(\.\d+)?){3}")
BOX_LOG = "crop_boxes.csv"


# ---------------------------------------------------------------- 影片資訊
def video_info(path):
    """(編碼格式, 寬, 高, fps 精確分數)。"""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,r_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"讀不到 {path} 的影片資訊：{out.stderr.strip()}")
    codec, w, h, rate = out.stdout.strip().split(",")[:4]
    return codec, int(w), int(h), Fraction(rate)


# ---------------------------------------------------------------- 讀區間
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
        if box is not None:
            if not BOX_RE.fullmatch(box):
                raise ValueError(f"{txt} 第 {no} 行的第三欄 {box} 不是範圍 (TOP_BOTTOM_LEFT_RIGHT)")
            top, bottom, left, right = map(float, box.split("_"))
            if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
                raise ValueError(f"{txt} 第 {no} 行的範圍 {box} 不合理 (需要 0 <= 上 < 下 <= 1、0 <= 左 < 右 <= 1)")
        if e < s:
            raise ValueError(f"{txt} 第 {no} 行結束格比起始格小：{line}")
        out.append((s, e, box))
    return out


def full_output(num, n, s, e):
    return OUTPUT / num / n / f"{s:06d}_{e:06d}.mp4"


def crop_output(num, n, s, e):
    return KEYPOINT_OUTPUT / num / n / f"{s:06d}_{e:06d}_upper_body.mp4"


def todo_of(num, n, s, e, box):
    """這個區間還要做哪些輸出：{"full", "crop"} 的子集合。"""
    need = set()
    if not (SKIP_EXISTING and full_output(num, n, s, e).exists()):
        need.add("full")
    if box is not None and not (SKIP_EXISTING and crop_output(num, n, s, e).exists()):
        need.add("crop")
    return need


# ---------------------------------------------------------------- ffmpeg 指令
def full_command(src, t0, duration, out):
    """全幅片段 (跟 03clipV2.py 相同)。"""
    hw = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] if GPU else []
    return [FFMPEG, "-nostdin", "-y", "-loglevel", "error", *hw,
            "-ss", f"{float(t0):.6f}", "-i", str(src), "-t", f"{float(duration):.6f}",
            *(GPU_FULL_ARGS if GPU else CPU_FULL_ARGS), "-threads", str(THREADS),
            "-c:a", "aac", str(out)]


def crop_command(src, info, box, t0, duration, out):
    """裁好的手語員片段 (跟 Phase4 STT/01cropV2.py 相同：裁切 → 等比例縮到 640 → 補黑邊)。"""
    codec, width, height, _ = info
    top, bottom, left, right = map(float, box.split("_"))
    x1, y1 = int(width * left) // 2 * 2, int(height * top) // 2 * 2     # cuvid、yuv420 都要偶數
    x2, y2 = int(width * right) // 2 * 2, int(height * bottom) // 2 * 2
    cw, ch = x2 - x1, y2 - y1
    if cw < 2 or ch < 2:
        raise ValueError(f"裁切範圍太小：{cw}×{ch}")
    ratio = OUTPUT_SIZE / max(cw, ch)
    sw, sh = max(2, round(cw * ratio / 2) * 2), max(2, round(ch * ratio / 2) * 2)
    pad = (f",pad={OUTPUT_SIZE}:{OUTPUT_SIZE}:{(OUTPUT_SIZE - sw) // 4 * 2}:{(OUTPUT_SIZE - sh) // 4 * 2}:color=black"
           if PAD_TO_SQUARE else "")
    if GPU:
        decode = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                  "-c:v", f"{codec}_cuvid",                             # h264_cuvid、hevc_cuvid、vp9_cuvid…
                  "-crop", f"{y1}x{height - y2}x{x1}x{width - x2}"]     # 解碼時直接裁掉 上x下x左x右 的像素
        vf = f"scale_cuda=w={sw}:h={sh}:interp_algo=bicubic,hwdownload,format=nv12" + pad
        encode = GPU_CROP_ARGS
    else:
        decode = []
        vf = f"crop={cw}:{ch}:{x1}:{y1},scale={sw}:{sh}:flags=area" + pad
        encode = CPU_CROP_ARGS
    return [FFMPEG, "-nostdin", "-y", "-loglevel", "error", *decode,
            "-ss", f"{float(t0):.6f}", "-i", str(src), "-t", f"{float(duration):.6f}",
            "-vf", vf, "-an", *encode, "-threads", str(THREADS), str(out)]


def run(cmd, output):
    """執行 ffmpeg 寫到暫存檔，成功才改成正式檔名；失敗回傳錯誤訊息。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + ".tmp.mp4")
    cmd = cmd[:-1] + [str(temp)]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode == 0:
        temp.replace(output)
        return None
    temp.unlink(missing_ok=True)
    return result.stderr.strip()


def update_box_log(num, n, intervals):
    """part_videos/<part>/<n>/crop_boxes.csv：每個片段用哪個範圍裁切 (保留舊的紀錄，同一個片段以這次為準)。"""
    path = OUTPUT / num / n / BOX_LOG
    rows = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = {r["clip"]: r["box"] for r in csv.DictReader(f)}
    for s, e, box in intervals:
        if box is not None:
            rows[f"{s:06d}_{e:06d}"] = box
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["clip", "box"])
            writer.writerows(sorted(rows.items()))


# ---------------------------------------------------------------- 剪一支影片
def cut_video(num, n, intervals, q):
    """剪一支影片的所有片段，透過 q 通知主程式：開始 (片段數)、每剪完一段、結束。回傳失敗的片段。"""
    q.put(("start", n, len(intervals)))
    src = VIDEO_DIR / num / f"{n}.mp4"
    info = video_info(src)
    fps = info[3]
    failed = []
    for s, e, box in intervals:
        t0 = (s - FRAME_START) / fps
        duration = (e - s + 1) / fps   # +1：包含結束那一幀
        need = todo_of(num, n, s, e, box)
        if "full" in need:
            out = full_output(num, n, s, e)
            err = run(full_command(src, t0, duration, out), out)
            if err:
                failed.append(f"{num}/{n}.mp4 {s}-{e} 全幅：{err}")
        if "crop" in need:
            out = crop_output(num, n, s, e)
            try:
                err = run(crop_command(src, info, box, t0, duration, out), out)
            except ValueError as exc:
                err = str(exc)
            if err:
                failed.append(f"{num}/{n}.mp4 {s}-{e} 裁切 ({box})：{err}")
        q.put(("tick", n, 1))
    q.put(("end", n, 0))
    return failed


if __name__ == "__main__":
    num = input("part: ").strip()
    txts = sorted((INTERVAL_DIR / num).glob("*_frames.txt"), key=lambda p: int(p.stem.split("_")[0]))
    if not txts:
        raise SystemExit(f"{INTERVAL_DIR / num} 裡沒有 *_frames.txt")

    # 先讀完所有區間，才知道總片段數；同時檢查影片存在、格式正確，避免剪到一半才失敗
    jobs, problems, skipped, no_box = [], [], 0, 0
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
        no_box += sum(box is None for _, _, box in intervals)
        todo = [iv for iv in intervals if todo_of(num, n, *iv)]
        skipped += len(intervals) - len(todo)
        if todo or any(box for _, _, box in intervals):
            jobs.append((n, todo, intervals))
        if todo:
            boxes = sorted({b for _, _, b in todo if b})
            print(f"{n}.mp4：{len(todo)} 個片段" + (f"，範圍 {'、'.join(boxes)}" if boxes else ""))
    if problems:
        raise SystemExit("\n".join(problems))
    total = sum(len(todo) for _, todo, _ in jobs)
    print(f"共 {total} 個片段要剪" + (f" ({skipped} 個已經剪好，跳過)" if skipped else ""))
    if no_box:
        print(f"注意：{no_box} 個區間沒有範圍 (只有兩欄)，只輸出全幅片段；手語員要用 Phase4 OCR/01cropV2.py 裁")
    if not total:
        for n, _, intervals in jobs:          # 片段都剪好了，還是把範圍紀錄補齊
            update_box_log(num, n, intervals)
        raise SystemExit(0)

    # 主進度條：影片數；副進度條：每個同時在剪的影片各一條，顯示該影片剪了幾個片段
    work = [(n, todo, intervals) for n, todo, intervals in jobs if todo]
    main_bar = tqdm(total=len(work), desc=f"part {num}", unit="video", position=0)
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
        pending = {executor.submit(cut_video, num, n, todo, q) for n, todo, _ in work}
        while pending:
            done, pending = wait(pending, timeout=0.2)
            while not q.empty():
                handle(q.get())
            for f in done:
                failed += f.result()   # 整支影片失敗 (例如讀不到影片資訊) 時把錯誤拋出來
                main_bar.update()
        while not q.empty():
            handle(q.get())
    for n, _, intervals in jobs:          # 每個片段用哪個範圍裁切 (Phase5 STT/ 靠這個判斷 STT 路徑)
        update_box_log(num, n, intervals)

    for bar in sub_bars:
        bar.close()
    main_bar.close()
    if failed:
        print(f"\n失敗 {len(failed)} 個 (再執行一次會只重做這些)：")
        for line in failed:
            print(f"  {line}")
    else:
        print(f"\n全部完成：全幅片段在 {OUTPUT / num}，裁好的手語員片段在 {KEYPOINT_OUTPUT / num}")

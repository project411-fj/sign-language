import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ========== 設定 ==========
INPUT_DIR = Path("Data Preparation") / "videos" / "source_videos"
OUTPUT_ROOT = Path("Data Preparation") / "Phase1-filter" / "01frames_crop"

WORKERS = 1       # 同時處理幾支影片（ffmpeg 本身也是多執行緒）
JPEG_Q = 10       # ffmpeg JPEG 品質，2 最好、31 最差
GPU = True        # True：解碼、裁切、放大都在 NVIDIA 顯卡上；False：全部用 CPU

# 裁切範圍：四個邊各自設定，都是佔畫面寬 / 高的比例 (0 = 最上 / 最左，1 = 最下 / 最右)
#   TOP ~ BOTTOM：保留的高度範圍；LEFT ~ RIGHT：保留的寬度範圍
#   例：TOP, BOTTOM, LEFT, RIGHT = 0.805, 0.88, 0.27, 1.0 → 高 80.5%~88%、寬 27%~100% (跟 01crop_text.py 相同)
TOP, BOTTOM, LEFT, RIGHT = 0.805, 0.88, 0.27, 1.0
SCALE = 2         # 裁切後放大幾倍

# 個別影片用不同的範圍 (檔名不含 .mp4 → (TOP, BOTTOM, LEFT, RIGHT))；沒列出的影片用上面的預設值
#   例 (note.md 記錄的 part 13，寬 10%~33%、高 20%~73% → (0.20, 0.73, 0.10, 0.33))：
#   CROP_BY_VIDEO = {
#       "1": (0.20, 0.73, 0.10, 0.33),
#       "2": (0.65, 1.00, 0.77, 1.00),
#       "3": (0.20, 0.75, 0.63, 1.00),
#   }
CROP_BY_VIDEO = {}

# 每個 worker 佔一行子進度條（第 0 行留給主進度條）
slots = queue.Queue()
for i in range(WORKERS):
    slots.put(i + 1)


def video_info(path):
    """回傳 (編碼格式, 寬, 高)，例如 ("h264", 1920, 1080)。"""
    codec, w, h = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    return codec, int(w), int(h)


def count_frames(path):
    # 只解封裝、不解碼，數影像串流的封包數 = 幀數（和 passthrough 輸出張數一致）
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip())


def crop_box(stem):
    """這支影片要用的 (TOP, BOTTOM, LEFT, RIGHT)，並檢查範圍是否合理。"""
    top, bottom, left, right = CROP_BY_VIDEO.get(stem, (TOP, BOTTOM, LEFT, RIGHT))
    if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
        raise ValueError(f"{stem}.mp4 的裁切範圍不合理：上 {top}、下 {bottom}、左 {left}、右 {right} "
                         "(需要 0 <= 上 < 下 <= 1、0 <= 左 < 右 <= 1)")
    return top, bottom, left, right


def crop(input_path, output_root):
    pos = slots.get()
    try:
        output_dir = output_root / f"{input_path.stem}_frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        total = count_frames(input_path)

        codec, w, h = video_info(input_path)
        top, bottom, left, right = crop_box(input_path.stem)
        y1, y2 = int(h * top), int(h * bottom)
        x1, x2 = int(w * left), int(w * right)

        if GPU:
            # cuvid 解碼器在 GPU 上直接裁出字幕區，只有這一小條搬回 CPU
            # GPU 的 NV12 格式要求偶數座標，所以邊界往外擴到偶數（可能比 CPU 版多 1 pixel）
            y1, x1 = y1 // 2 * 2, x1 // 2 * 2
            y2, x2 = min(h, (y2 + 1) // 2 * 2), min(w, (x2 + 1) // 2 * 2)
            decode = [
                "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                "-c:v", f"{codec}_cuvid",                 # 依影片格式選解碼器：h264_cuvid、hevc_cuvid…
                "-crop", f"{y1}x{h - y2}x{x1}x{w - x2}",  # 上x下x左x右 要裁掉的像素
            ]
            vf = f"scale_cuda=w=iw*{SCALE}:h=ih*{SCALE}:interp_algo=bicubic,hwdownload,format=nv12"
        else:
            decode = []
            vf = f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1}:exact=1,scale=iw*{SCALE}:ih*{SCALE}:flags=bicubic"

        cmd = [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
            *decode,
            "-i", str(input_path), "-an",
            "-vf", vf,
            "-fps_mode", "passthrough",  # 每個解碼出來的幀都輸出一張，不補幀也不丟幀
            "-q:v", str(JPEG_Q),
            str(output_dir / "frame_%06d.jpg"),
        ]

        with tqdm(total=total, desc=input_path.stem, position=pos, leave=False, unit="frame") as bar:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
            done = 0
            for line in proc.stdout:
                if line.startswith("frame="):
                    n = int(line.split("=")[1])
                    bar.update(n - done)
                    done = n
            if proc.wait() != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
    finally:
        slots.put(pos)


if __name__ == "__main__":
    num = input("part: ").strip()
    input_paths = sorted((INPUT_DIR / num).glob("*.mp4"))
    with ThreadPoolExecutor(WORKERS) as executor:
        futures = [executor.submit(crop, p, OUTPUT_ROOT / num) for p in input_paths]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"part {num}", unit="video", position=0):
            f.result()  # 某支影片失敗時把錯誤拋出來
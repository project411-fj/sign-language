import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ========== 設定 ==========
INPUT_DIR = Path("Data Preparation") / "videos" / "source_videos"
OUTPUT_ROOT = Path("Data Preparation") / "Phase1-filter" / "01frames_crop"

WORKERS = 1       # 同時處理幾支影片（ffmpeg 本身也是多執行緒）
JPEG_Q = 2        # ffmpeg JPEG 品質，2 最好、31 最差
HWACCEL = True   # True = 用 NVIDIA GPU 解碼

# 和原本 OpenCV 版相同的裁切範圍：高 80.5%~88%、寬 27%~100%，再放大 2 倍
VF = (
    "crop=w='iw-trunc(iw*0.27)':h='trunc(ih*0.88)-trunc(ih*0.805)'"
    ":x='trunc(iw*0.27)':y='trunc(ih*0.805)':exact=1,"
    "scale=iw*2:ih*2:flags=bicubic"
)


def crop(input_path, output_root):
    output_dir = output_root / f"{input_path.stem}_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
    if HWACCEL:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-i", str(input_path),
        "-vf", VF,
        "-fps_mode", "passthrough",  # 每個解碼出來的幀都輸出一張，不補幀也不丟幀
        "-q:v", str(JPEG_Q),
        str(output_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


num = input("part: ").strip()
input_paths = sorted((INPUT_DIR / num).glob("*.mp4"))
with ThreadPoolExecutor(WORKERS) as executor:
    futures = [executor.submit(crop, p, OUTPUT_ROOT / num) for p in input_paths]
    for f in tqdm(as_completed(futures), total=len(futures), desc=f"part {num}", unit="video"):
        f.result()  # 某支影片失敗時把錯誤拋出來
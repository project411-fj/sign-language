import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

INPUT_DIR = Path("Data Preparation") / "videos" / "part_videos"          # part_videos/<part>/<n>/<start>_<end>.mp4
OUTPUT_ROOT = Path("Data Preparation") / "Phase3-OCR" / "01frames_crop"    # frames_crop/<part>/<n>/<start>_<end>_frames/

FFMPEG = "ffmpeg"    # 不在 PATH 就改成 ffmpeg.exe 完整路徑
FFPROBE = "ffprobe"
WORKERS = 3          # 同時處理幾支影片（ffmpeg 本身就是多執行緒）
GPU = True           # True：解碼、裁切、放大都在 NVIDIA 顯卡上；False：全部用 CPU

# 字幕區域（比例），和原本 frame[int(h*0.805):int(h*0.88), int(w*0.27):w] 相同
TOP, BOTTOM, LEFT = 0.805, 0.88, 0.27
SCALE = 2            # 放大倍數；設 1 就不放大，速度更快、檔案更小
JPEG_Q = 2           # ffmpeg JPEG 品質，2～31，越小越好（2 約等於 OpenCV 預設的 95）


def video_size(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return map(int, out.split(","))


def crop(input_path, output_root):
    output_dir = output_root / f"{input_path.stem}_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    w, h = video_size(input_path)
    y1, y2, x1 = int(h * TOP), int(h * BOTTOM), int(w * LEFT)

    if GPU:
        # 解碼器 h264_cuvid 在 GPU 上直接裁出字幕區，只有這一小條搬回 CPU
        # GPU 的 NV12 格式要求偶數座標，所以邊界往外擴到偶數（可能比 CPU 版多 1 pixel）
        y1, x1 = y1 // 2 * 2, x1 // 2 * 2
        y2 = (y2 + 1) // 2 * 2
        decode = [
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-c:v", "h264_cuvid", "-crop", f"{y1}x{h - y2}x{x1}x0",   # 上x下x左x右 要裁掉的像素
        ]
        vf = f"scale_cuda=w=iw*{SCALE}:h=ih*{SCALE}:interp_algo=bicubic,hwdownload,format=nv12"
    else:
        decode = []
        vf = f"crop={w - x1}:{y2 - y1}:{x1}:{y1}:exact=1,scale=iw*{SCALE}:ih*{SCALE}:flags=bicubic"

    cmd = [
        FFMPEG, "-y", "-loglevel", "error", *decode,
        "-i", str(input_path), "-an", "-vf", vf,
        "-fps_mode", "passthrough",   # 每一幀都輸出，不補幀也不丟幀
        "-q:v", str(JPEG_Q), "-start_number", "1",
        str(output_dir / "frame_%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"{input_path}：{result.stderr.strip()}")


if __name__ == "__main__":
    num = input("part: ").strip()
    input_paths = sorted((INPUT_DIR / num).glob("*/*.mp4"), key=lambda p: (int(p.parent.name), p.stem))
    # 實際工作在 ffmpeg 子程序裡做，用執行緒就夠了
    with ThreadPoolExecutor(WORKERS) as executor:
        # 保留子目錄（1, 2, 3…），避免不同影片的片段檔名互相覆蓋
        futures = [executor.submit(crop, p, OUTPUT_ROOT / num / p.parent.name) for p in input_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"part {num}", unit="video"):
            future.result()   # 某支影片失敗時把錯誤拋出來
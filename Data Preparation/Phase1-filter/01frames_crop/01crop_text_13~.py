import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ========== 設定 ==========
# 要處理的影片：直接寫 .mp4 的路徑 → 裁切範圍 (TOP, BOTTOM, LEFT, RIGHT)；範圍寫 None 就用下面的預設值
#   輸出到 OUTPUT_ROOT/<影片所在資料夾名稱>/<檔名>_frames/，例如 source_videos/13/1.mp4 → 01frames_crop/13/1_frames/
#   (02style.py 依這個結構找對應的影片，所以影片請放在 source_videos/<part>/ 底下)
VIDEOS = {
    r"Data Preparation\videos\source_videos\13\3.mp4": (0.20, 0.73,0.10, 0.33),
    r"Data Preparation\videos\source_videos\13\6.mp4": (0.65, 1.00, 0.77, 1.00),
    r"Data Preparation\videos\source_videos\13\4.mp4": (0.20, 0.75, 0.63, 1.00)
}
OUTPUT_ROOT = Path("Data Preparation") / "Phase1-filter" / "01frames_crop"

WORKERS = 1       # 同時處理幾支影片（ffmpeg 本身也是多執行緒）
JPEG_Q = 10       # ffmpeg JPEG 品質，2 最好、31 最差
GPU = True        # True：解碼、裁切、放大都在 NVIDIA 顯卡上；False：全部用 CPU

# 預設裁切範圍 (VIDEOS 裡範圍寫 None 的影片用這個)：四個邊都是佔畫面寬 / 高的比例
#   TOP ~ BOTTOM：保留的高度範圍 (0 = 最上，1 = 最下)；LEFT ~ RIGHT：保留的寬度範圍 (0 = 最左，1 = 最右)
#   0.805, 0.88, 0.27, 1.0 → 高 80.5%~88%、寬 27%~100% (跟 01crop_text.py 相同)
TOP, BOTTOM, LEFT, RIGHT = 0.805, 0.88, 0.27, 1.0
SCALE = 2         # 裁切後放大幾倍

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


def crop_box(path, box):
    """這支影片要用的 (TOP, BOTTOM, LEFT, RIGHT)，並檢查範圍是否合理。"""
    top, bottom, left, right = box if box is not None else (TOP, BOTTOM, LEFT, RIGHT)
    if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
        raise ValueError(f"{path} 的裁切範圍不合理：上 {top}、下 {bottom}、左 {left}、右 {right} "
                         "(需要 0 <= 上 < 下 <= 1、0 <= 左 < 右 <= 1)")
    return top, bottom, left, right


def output_dir_of(input_path):
    """source_videos/14/3.mp4 → 01frames_crop/14/3_frames"""
    return OUTPUT_ROOT / input_path.parent.name / f"{input_path.stem}_frames"


def crop(input_path, box):
    pos = slots.get()
    try:
        output_dir = output_dir_of(input_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for old in output_dir.glob("frame_*.jpg"):     # 先清掉上次的輸出，避免殘留的舊圖片混在一起
            old.unlink()
        total = count_frames(input_path)

        codec, w, h = video_info(input_path)
        top, bottom, left, right = crop_box(input_path, box)
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
            # 有些影片 (例如 VP9) 的色彩資訊標錯 (YUV 畫面卻標成 RGB 的 gbr、其他欄位是 reserved)，
            # 新版 ffmpeg 轉成 JPEG 時會報 "Unsupported input" (-129)。下載回 CPU 後把標記改成一般影片的 BT.709，
            # 只改標記、不改畫面內容；本來就標對的影片不受影響
            vf = (f"scale_cuda=w=iw*{SCALE}:h=ih*{SCALE}:interp_algo=bicubic,hwdownload,format=nv12,"
                  "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709")
        else:
            decode = []
            vf = f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1}:exact=1,scale=iw*{SCALE}:ih*{SCALE}:flags=bicubic"

        # 有些影片 (例如 VP9) 有時間戳記重複的畫面，輸出圖片時會出現
        # "non monotonically increasing dts" 並可能丟格，造成後面的圖片編號跟影片時間對不上。
        # 最後用 setpts=N 依畫面順序重新編時間戳記 (0, 1, 2…)，並沿用原影片的時間單位，確保每一格都依序輸出
        vf += ",setpts=N"

        cmd = [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
            *decode,
            "-i", str(input_path), "-an",
            "-vf", vf,
            "-fps_mode", "passthrough",  # 每個解碼出來的幀都輸出一張，不補幀也不丟幀
            "-enc_time_base", "demux",
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

        # 確認每一格都有輸出：張數不對的話，圖片編號會跟影片時間錯位
        written = sum(1 for _ in output_dir.glob("frame_*.jpg"))
        if written != total:
            raise RuntimeError(f"{input_path}：影片有 {total} 格，卻輸出了 {written} 張圖片 "
                               f"(差 {total - written} 張)，圖片編號會跟影片時間對不上")
        tqdm.write(f"{input_path}：完成，{written} 張 = 影片總格數")
    finally:
        slots.put(pos)


if __name__ == "__main__":
    videos = {Path(p): box for p, box in VIDEOS.items()}
    missing = [p for p in videos if not p.is_file()]
    if missing:
        raise FileNotFoundError("找不到這些影片：\n" + "\n".join(f"  {p}" for p in missing))
    for p, box in videos.items():          # 先檢查全部的範圍，避免跑到一半才發現設定錯誤
        crop_box(p, box)
        print(f"{p} → {output_dir_of(p)}")

    with ThreadPoolExecutor(WORKERS) as executor:
        futures = [executor.submit(crop, p, box) for p, box in videos.items()]
        for f in tqdm(as_completed(futures), total=len(futures), desc="影片", unit="video", position=0):
            f.result()  # 某支影片失敗時把錯誤拋出來
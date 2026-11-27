"""STT 路徑的 01cropV2：依資料夾名稱的範圍裁出手語員，輸出和 OCR 版相同的 keypoint_videos。

    part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/<start>_<end>.mp4
        → keypoint_videos/<part>/<n>/<start>_<end>_upper_body.mp4   (640×640，frame 數不變)

<TOP>_<BOTTOM>_<LEFT>_<RIGHT> 是 Phase1 01crop_text_13~.py 或 02signer_track.py 建立的資料夾 (例如 0.20_0.73_0.10_0.33)，
四個數字是要保留的範圍佔畫面高 / 寬的比例：高 20%~73%、寬 10%~33%。
手語員中途換位置的影片會有好幾個範圍資料夾，每個片段用自己所在資料夾的範圍裁切；
不同範圍資料夾裡的片段時間重疊時 (通常是舊範圍留下的片段)，這支影片會先跳過並提醒。
OCR 版 (OCR/01cropV2.py) 是用 YOLO 自動找手語員；STT 版範圍已經知道，所以直接照範圍裁切，不需要模型。
預設全程用 NVIDIA 顯卡 (GPU = True)：解碼時裁切、顯卡上縮放、NVENC 編碼，只有補黑邊在 CPU 上做。
輸出和 OCR 版放在同一個 keypoint_videos，後面的 02keypoint_*.py、03keypoint_combine.py 兩條路徑共用。

執行後輸入 part：「13」、「13,14」或「all」(part_videos 底下有範圍資料夾的 part)。
"""
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

INPUT_ROOT = Path("Data Preparation/videos/part_videos")       # part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/<start>_<end>.mp4
OUTPUT_ROOT = Path("Data Preparation/videos/keypoint_videos")  # keypoint_videos/<part>/<n>/<start>_<end>_upper_body.mp4
SKIP_EXISTING = True   # 已經裁切過的影片就跳過；要重做就改成 False

OUTPUT_SIZE = 640
PAD_TO_SQUARE = True   # True：範圍不是正方形時，等比例縮放後上下或左右補黑邊成 640×640 (和 OCR 版尺寸一致、畫面不變形)
                       # False：只等比例縮放到最長邊 640，不補邊 (輸出不一定是正方形)

# 裁切輸出用 ffmpeg / ffprobe（需在 PATH 中）
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
GPU = True             # True：解碼 + 裁切 (cuvid)、縮放 (scale_cuda)、編碼 (NVENC) 都在 NVIDIA 顯卡上，只有補黑邊在 CPU 上做
                       # False：全部用 CPU (libx264)，沒有 NVIDIA 顯卡時用這個
MAX_WORKERS = 2        # 同時處理幾支影片；GPU 模式下每支佔一個 NVENC 編碼器，一般顯卡同時最多 5~8 個

GPU_CODEC_ARGS = ["-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", "0"]   # NVENC 固定品質 (cq 越小品質越好)
CPU_CODEC_ARGS = ["-preset", "veryfast", "-crf", "18"]                       # libx264 固定品質

BOX_RE = re.compile(r"\d+(\.\d+)?(_\d+(\.\d+)?){3}")
CLIP_RE = re.compile(r"\d+_\d+")


def parse_box(name):
    """"0.20_0.73_0.10_0.33" → (0.2, 0.73, 0.1, 0.33)，並檢查範圍是否合理。"""
    top, bottom, left, right = map(float, name.split("_"))
    if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
        raise ValueError(f"範圍資料夾 {name} 不合理 (需要 0 <= 上 < 下 <= 1、0 <= 左 < 右 <= 1)")
    return top, bottom, left, right


def find_jobs(part_dir):
    """回傳 ([(影片, 範圍資料夾名稱), ...], [問題說明, ...])。"""
    jobs, problems = [], []
    for n_dir in sorted((d for d in part_dir.iterdir() if d.is_dir()), key=lambda d: (len(d.name), d.name)):
        boxes = [d for d in n_dir.iterdir() if d.is_dir() and BOX_RE.fullmatch(d.name)]
        if not boxes:
            continue
        try:
            for box_dir in boxes:
                parse_box(box_dir.name)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        clips = sorted(((p, d.name) for d in boxes for p in d.glob("*.mp4") if CLIP_RE.fullmatch(p.stem)),
                       key=lambda c: [int(x) for x in c[0].stem.split("_")])
        # 好幾個範圍資料夾時，片段的時間不能重疊 (重疊表示有舊範圍留下的片段，輸出檔名也會撞在一起)
        overlap = [(a, b) for a, b in zip(clips, clips[1:])
                   if int(b[0].stem.split("_")[0]) <= int(a[0].stem.split("_")[1])]
        if overlap:
            (a, abox), (b, bbox) = overlap[0]
            problems.append(f"{n_dir.as_posix()}：{abox}/{a.name} 和 {bbox}/{b.name} 時間重疊 "
                            f"(共 {len(overlap)} 處，通常是舊範圍留下的片段)，請刪掉舊的，這支影片先跳過")
            continue
        jobs.extend(clips)
    return jobs, problems


def video_info(path):
    """回傳 (編碼格式, 寬, 高)，例如 ("h264", 1920, 1080)。"""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"讀不到影片資訊：{out.stderr.strip()}")
    codec, width, height = out.stdout.strip().split(",")[:3]
    return codec, int(width), int(height)


def crop_rect(width, height, box):
    """範圍換成像素 (x1, y1, x2, y2)，四個邊都取偶數 (cuvid 和 yuv420 都要求偶數)。"""
    top, bottom, left, right = box
    x1, y1 = int(width * left) // 2 * 2, int(height * top) // 2 * 2
    x2, y2 = int(width * right) // 2 * 2, int(height * bottom) // 2 * 2
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError(f"裁切範圍太小：{x2 - x1}×{y2 - y1}")
    return x1, y1, x2, y2


def scaled_size(w, h):
    """等比例縮放到最長邊 OUTPUT_SIZE，寬高取偶數。"""
    ratio = OUTPUT_SIZE / max(w, h)
    return max(2, round(w * ratio / 2) * 2), max(2, round(h * ratio / 2) * 2)


def pad_filter(sw, sh):
    """在 CPU 上補黑邊成 OUTPUT_SIZE×OUTPUT_SIZE (置中，位移取偶數)；PAD_TO_SQUARE = False 時不補。"""
    if not PAD_TO_SQUARE:
        return ""
    px, py = (OUTPUT_SIZE - sw) // 4 * 2, (OUTPUT_SIZE - sh) // 4 * 2
    return f",pad={OUTPUT_SIZE}:{OUTPUT_SIZE}:{px}:{py}:color=black"


def ffmpeg_command(video, temp, codec, width, height, box):
    x1, y1, x2, y2 = crop_rect(width, height, box)
    sw, sh = scaled_size(x2 - x1, y2 - y1)
    if GPU:
        decode = [
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-c:v", f"{codec}_cuvid",                                 # 依影片格式選解碼器：h264_cuvid、hevc_cuvid…
            "-crop", f"{y1}x{height - y2}x{x1}x{width - x2}",         # 解碼時直接裁掉 上x下x左x右 的像素
        ]
        # 在顯卡上縮放，下載回 CPU 補黑邊，再交給 NVENC 編碼
        vf = f"scale_cuda=w={sw}:h={sh}:interp_algo=bicubic,hwdownload,format=nv12" + pad_filter(sw, sh)
        encode = ["-c:v", "h264_nvenc", *GPU_CODEC_ARGS]            # NVENC 直接吃 nv12，不用再轉格式
    else:
        decode = []
        vf = f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1},scale={sw}:{sh}:flags=area" + pad_filter(sw, sh)
        encode = ["-c:v", "libx264", *CPU_CODEC_ARGS, "-pix_fmt", "yuv420p"]
    return [
        FFMPEG, "-nostdin", "-y", "-loglevel", "error",
        *decode, "-i", str(video),
        "-vf", vf,
        "-fps_mode", "passthrough",          # 每個解碼出來的幀都輸出，不補幀也不丟幀 (frame 數要和原片段一樣)
        "-an", *encode,
        str(temp),
    ]


def crop_video(video, output, box_name):
    # 先寫到暫存檔，成功後才改成正式檔名：中途中斷不會留下只寫一半的影片
    codec, width, height = video_info(video)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + ".tmp.mp4")
    cmd = ffmpeg_command(video, temp, codec, width, height, parse_box(box_name))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        temp.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip())
    temp.replace(output)


def run_part(part):
    input_dir, output_dir = INPUT_ROOT / part, OUTPUT_ROOT / part
    failed_log = Path(__file__).parent / f"failed_part_{part}.txt"
    if not input_dir.is_dir():
        tqdm.write(f"part {part}：找不到 {input_dir}，跳過")
        return

    def output_of(video):
        # part_videos/13/1/0.20_0.73_0.10_0.33/000600_001507.mp4 → keypoint_videos/13/1/000600_001507_upper_body.mp4
        return output_dir / video.parent.parent.name / f"{video.stem}_upper_body.mp4"

    jobs, problems = find_jobs(input_dir)
    for problem in problems:
        tqdm.write(f"part {part}：{problem}")
    if not jobs:
        tqdm.write(f"part {part}：沒有 <n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/ 結構的片段 (請先跑 Phase1 的 "
                   "01crop_text_13~.py、03clipV2_13~.py；OCR 路徑的 part 請用 OCR/01cropV2.py)，跳過")
        return

    # 有失敗紀錄就只重跑紀錄裡的影片，否則跑整個 part
    if failed_log.exists():
        retry = {line for line in failed_log.read_text(encoding="utf-8").splitlines() if line}
        jobs = [(v, b) for v, b in jobs if v.relative_to(input_dir).as_posix() in retry]
        tqdm.write(f"part {part}：讀取失敗紀錄 {failed_log.name}，重跑 {len(jobs)} 支影片")
    else:
        total = len(jobs)
        jobs = [(v, b) for v, b in jobs if not (SKIP_EXISTING and output_of(v).exists())]
        tqdm.write(f"part {part}：共 {total} 支影片，"
                   + (f"{total - len(jobs)} 支已經裁切過 (跳過)，" if total > len(jobs) else "")
                   + f"這次處理 {len(jobs)} 支")
    if not jobs:
        failed_log.unlink(missing_ok=True)
        return

    failed = []
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {pool.submit(crop_video, video, output_of(video), box): video for video, box in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"part {part}", unit="video"):
            try:
                future.result()
            except Exception as exc:                    # 一支失敗不影響其他影片
                failed.append((futures[future].relative_to(input_dir), exc))

    # 有失敗就覆寫紀錄，全部成功就刪除紀錄
    if failed:
        failed_log.write_text("".join(f"{relative.as_posix()}\n" for relative, _ in failed), encoding="utf-8")
        tqdm.write(f"\npart {part} 失敗 {len(failed)} 支，已記錄到 {failed_log.name}：")
        for relative, exc in failed:
            tqdm.write(f"  {relative.as_posix()}：{exc}")
    else:
        failed_log.unlink(missing_ok=True)
        tqdm.write(f"part {part}：處理完成")


def parse_parts(answer):
    """「13」→ ["13"]；「13,14」→ ["13", "14"]；「all」→ part_videos 底下所有 part (依數字排序)。"""
    answer = answer.strip()
    if answer.lower() == "all":
        return sorted((p.name for p in INPUT_ROOT.iterdir() if p.is_dir()), key=lambda x: (len(x), x))
    return [p.strip() for p in answer.split(",") if p.strip()]


def main():
    parts = parse_parts(input("part (例如 13、13,14 或 all): "))
    if not parts:
        print("沒有輸入 part")
        return
    print(f"要處理的 part：{'、'.join(parts)}")
    for part in parts:
        run_part(part)


if __name__ == "__main__":
    main()

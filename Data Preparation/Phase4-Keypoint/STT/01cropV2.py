"""STT 路徑的 01cropV2：依資料夾名稱的範圍裁出手語員，輸出和 OCR 版相同的 keypoint_videos。

    part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/<start>_<end>.mp4
        → keypoint_videos/<part>/<n>/<start>_<end>_upper_body.mp4   (640×640，frame 數不變)

<TOP>_<BOTTOM>_<LEFT>_<RIGHT> 是 Phase1 01crop_text_13~.py 建立的資料夾 (例如 0.20_0.73_0.10_0.33)，
四個數字是要保留的範圍佔畫面高 / 寬的比例：高 20%~73%、寬 10%~33%。
OCR 版 (OCR/01cropV2.py) 是用 YOLO 自動找手語員；STT 版範圍已經知道，所以直接照範圍裁切，不需要模型。
輸出和 OCR 版放在同一個 keypoint_videos，後面的 02keypoint_*.py、03keypoint_combine.py 兩條路徑共用。

執行後輸入 part：「13」、「13,14」或「all」(part_videos 底下有範圍資料夾的 part)。
"""
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm

INPUT_ROOT = Path("Data Preparation/videos/part_videos")       # part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/<start>_<end>.mp4
OUTPUT_ROOT = Path("Data Preparation/videos/keypoint_videos")  # keypoint_videos/<part>/<n>/<start>_<end>_upper_body.mp4
SKIP_EXISTING = True   # 已經裁切過的影片就跳過；要重做就改成 False

OUTPUT_SIZE = 640
PAD_TO_SQUARE = True   # True：範圍不是正方形時，等比例縮放後上下或左右補黑邊成 640×640 (和 OCR 版尺寸一致、畫面不變形)
                       # False：只等比例縮放到最長邊 640，不補邊 (輸出不一定是正方形)

# 裁切輸出用 ffmpeg（需在 PATH 中）
MAX_WORKERS = 2
FFMPEG = "ffmpeg"
VIDEO_CODEC = "libx264"
VIDEO_CODEC_ARGS = ["-preset", "veryfast", "-crf", "18"]
# 有 NVIDIA 顯卡可改用硬體編碼：
# VIDEO_CODEC = "h264_nvenc"
# VIDEO_CODEC_ARGS = ["-preset", "p4", "-cq", "19"]

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
        if len(boxes) > 1:
            problems.append(f"{n_dir.as_posix()} 底下有 {len(boxes)} 個範圍資料夾 "
                            f"({'、'.join(d.name for d in boxes)})，請只留下正確的那個，這支影片先跳過")
            continue
        box_dir = boxes[0]
        try:
            parse_box(box_dir.name)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        clips = sorted((p for p in box_dir.glob("*.mp4") if CLIP_RE.fullmatch(p.stem)),
                       key=lambda p: [int(x) for x in p.stem.split("_")])
        jobs.extend((clip, box_dir.name) for clip in clips)
    return jobs, problems


def crop_filter(width, height, box):
    top, bottom, left, right = box
    x1, x2 = int(width * left), int(width * right)
    y1, y2 = int(height * top), int(height * bottom)
    w, h = (x2 - x1) // 2 * 2, (y2 - y1) // 2 * 2          # yuv420p 需要偶數寬高
    vf = f"crop={w}:{h}:{x1}:{y1}"
    if PAD_TO_SQUARE:
        vf += (f",scale={OUTPUT_SIZE}:{OUTPUT_SIZE}:force_original_aspect_ratio=decrease:flags=area"
               f",pad={OUTPUT_SIZE}:{OUTPUT_SIZE}:(ow-iw)/2:(oh-ih)/2:color=black")
    else:
        vf += f",scale='if(gt(iw,ih),{OUTPUT_SIZE},-2)':'if(gt(iw,ih),-2,{OUTPUT_SIZE})':flags=area"
    return vf


def crop_video(video, output, box_name):
    # 先寫到暫存檔，成功後才改成正式檔名：中途中斷不會留下只寫一半的影片
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not width or not height:
        raise RuntimeError("讀不到影片尺寸")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + ".tmp.mp4")
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", str(video),
        "-vf", crop_filter(width, height, parse_box(box_name)),
        "-an", "-c:v", VIDEO_CODEC, *VIDEO_CODEC_ARGS, "-pix_fmt", "yuv420p",
        str(temp),
    ]
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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import easyocr
from tqdm import tqdm

# ========== 設定 ==========
INPUT_DIR = Path("Data Preparation") / "Phase3-OCR" / "01frames_crop"                    # frames_crop/<part>/<n>/<start>_<end>_frames/*.jpg
OUTPUT_DIR = Path("Data Preparation") / "Phase3-OCR" / "OCR_output" / "01EasOCR_text"  # 01EasOCR_text/<part>/<n>/<start>_<end>_frames.txt
STEP = 15            # 粗掃取樣間隔（幀）
CHUNK = 16           # 一次送進文字偵測的幀數（記憶體不足就調小，最小 1）
BATCH_SIZE = 256     # 文字辨識階段的 batch size（顯存不足就調小）
IO_THREADS = 8       # 讀圖的執行緒數

reader = easyocr.Reader(["ch_tra"], gpu=True)


def frame_no(path):
    return int(path.stem.split("_")[-1])


def ocr(paths, idx, pool, bar, phase):
    """OCR paths 中第 idx 幾張，回傳每張的文字 tuple，並更新副進度條。"""
    bar.reset(total=len(idx))
    bar.set_description(f"  {phase}")
    out = []
    for c in range(0, len(idx), CHUNK):
        images = list(pool.map(lambda i: cv2.imread(str(paths[i])), idx[c:c + CHUNK]))
        for res in reader.readtext_batched(images, batch_size=BATCH_SIZE):
            out.append(tuple(t.strip() for _, t, _ in res if t.strip()))
        bar.update(len(images))
    return out


def process_video(frame_dir, out_path, pool, bar):
    paths = sorted(frame_dir.glob("*.jpg"), key=frame_no)
    labels = [None] * len(paths)

    # 1. 粗掃：每 STEP 幀 OCR 一次（最後一幀一定測）
    samples = list(range(0, len(paths), STEP))
    if samples[-1] != len(paths) - 1:
        samples.append(len(paths) - 1)
    for i, t in zip(samples, ocr(paths, samples, pool, bar, "粗掃")):
        labels[i] = t

    # 2. 細掃：前後兩個取樣點文字不同 → 中間逐幀 OCR，找出真實邊界
    gaps = [i for a, b in zip(samples, samples[1:]) if labels[a] != labels[b] for i in range(a + 1, b)]
    for i, t in zip(gaps, ocr(paths, gaps, pool, bar, "細掃")):
        labels[i] = t

    # 3. 其餘沒測的幀：前後取樣點文字完全相同，直接沿用
    for i in range(len(paths)):
        if labels[i] is None:
            labels[i] = labels[i - 1]

    with open(out_path, "w", encoding="utf-8") as f:
        for p, texts in zip(paths, labels):
            for text in texts:
                f.write(f"{frame_no(p)}: {text}\n")
    return len(samples) + len(gaps), len(paths)


if __name__ == "__main__":
    num = input("part: ").strip()
    frame_dirs = sorted(
        (p for p in (INPUT_DIR / num).glob("*/*_frames") if p.is_dir()),
        key=lambda p: (int(p.parent.name), p.name),
    )
    main_bar = tqdm(total=len(frame_dirs), desc=f"part {num}", unit="clip", position=0)
    sub_bar = tqdm(total=1, desc="  粗掃", unit="frame", position=1)
    total_ocr = total_frames = 0

    with ThreadPoolExecutor(IO_THREADS) as pool:
        for frame_dir in frame_dirs:
            main_bar.set_postfix_str(f"{frame_dir.parent.name}/{frame_dir.name}  已 OCR {total_ocr}/{total_frames} 幀")
            out_dir = OUTPUT_DIR / num / frame_dir.parent.name   # 保留子目錄（1, 2, 3…），避免不同影片的片段檔名互相覆蓋
            out_dir.mkdir(parents=True, exist_ok=True)
            c, n = process_video(frame_dir, out_dir / f"{frame_dir.name}.txt", pool, sub_bar)
            total_ocr += c
            total_frames += n
            main_bar.update()

    main_bar.set_postfix_str(f"已 OCR {total_ocr}/{total_frames} 幀")
    sub_bar.close()
    main_bar.close()
"""列出每個 part 在關鍵點與切句流程各步驟的狀態，找出哪一步還沒跑完 (OCR 路徑：Keypoint_output/、Phase5 output/)。

    01hand           02keypoint_hand.py 的輸出
    01face           02keypoint_face.py 的輸出
    02combined       03keypoint_combine.py 的輸出 (應該是 292 維：手 + 身體 + 臉)
    03segments       03split_npy.py 切出來的分段 (應該是 292 維)
"""
from pathlib import Path

import numpy as np

# face_pipeline_check.py 在 Data Preparation\tools\，往上一層就是 Data Preparation
ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "Phase4-Keypoint" / "Keypoint_output"
OUT = ROOT / "Phase5-Pause" / "output"
FEAT_DIM = 292

STEPS = [
    ("01hand", KEY / "01hand", "*/*.npy"),
    ("01face", KEY / "01face", "*/*.npy"),
    ("02combined", KEY / "02combined", "*/*.npy"),
    ("03segments", OUT / "03segments", "*/*.npy"),
]


def first_dim(folder):
    """folder 裡第一個 .npy 的維度 (沒有檔案就是 None)。"""
    first = next(folder.glob("*/*.npy"), None)
    return None if first is None else np.load(first, mmap_mode="r").shape[1]


parts = sorted({p.name for _, d, _ in STEPS if d.exists() for p in d.iterdir() if p.is_dir()},
               key=lambda x: (len(x), x))
print(f"{'part':<6}" + "".join(f"{name:>12}" for name, _, _ in STEPS) + f"{'合併維度':>10}{'分段維度':>10}   該做的事")
for part in parts:
    counts = [len(list((d / part).glob(pattern))) if (d / part).exists() else 0 for _, d, pattern in STEPS]
    hand, face, combined, seg = counts
    combined_dim = first_dim(KEY / "02combined" / part)
    seg_dim = first_dim(OUT / "03segments" / part)
    if face < hand:
        todo = f"OCR/02keypoint_face.py (part 輸入 {part})"
    elif combined < hand or combined_dim != FEAT_DIM:
        todo = f"OCR/03keypoint_combine.py (part 輸入 {part})" + ("：目前是舊的 156 維" if combined_dim == 156 else "")
    elif seg == 0 or seg_dim != FEAT_DIM:
        todo = "OCR/03split_npy.py (part 輸入 all)" + ("：目前是舊的 156 維" if seg_dim == 156 else "")
    else:
        todo = "完成"
    dims = "".join(f"{'-' if d is None else d:>12}" for d in (combined_dim, seg_dim))
    print(f"{part:<6}" + "".join(f"{n:>12}" for n in counts) + f"{dims}   {todo}")

# Phase4、Phase5 的程式都分成 OCR/、STT/ 兩版，失敗紀錄會寫在各自的資料夾裡 (上一層的是搬移前留下的舊紀錄)
logs = sorted(p.relative_to(ROOT).as_posix()
              for phase in ("Phase4-Keypoint", "Phase5-Pause")
              for folder in (ROOT / phase, ROOT / phase / "OCR", ROOT / phase / "STT")
              for pattern in ("failed_*face*.txt", "failed_combine_part_*.txt", "failed_segments_part_*.txt")
              for p in folder.glob(pattern))
if logs:
    print("\n有失敗紀錄 (裡面列的檔案沒有處理成功)：")
    for name in logs:
        print(f"  {name}")

"""列出每個 part 在關鍵點與切句流程各步驟的狀態，找出哪一步還沒跑完。

    02combined       手 + 身體 156 維 (03keypoint_combine.py)
    01face           02keypoint_face.py 的輸出
    02combined_face  含臉部的 292 維 (03keypoint_combine.py，WITH_FACE = True)
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
    ("02combined", KEY / "02combined", "*/*.npy"),
    ("01face", KEY / "01face", "*/*.npy"),
    ("02combined_face", KEY / "02combined_face", "*/*.npy"),
    ("03segments", OUT / "03segments", "*/*.npy"),
]


def segment_dim(part):
    """03segments/<part> 裡第一個檔案的維度 (沒有檔案就是 None)。"""
    first = next((OUT / "03segments" / part).glob("*/*.npy"), None)
    return None if first is None else np.load(first, mmap_mode="r").shape[1]


parts = sorted({p.name for _, d, _ in STEPS if d.exists() for p in d.iterdir() if p.is_dir()},
               key=lambda x: (len(x), x))
print(f"{'part':<6}" + "".join(f"{name:>17}" for name, _, _ in STEPS) + f"{'分段維度':>10}   該做的事")
for part in parts:
    counts = [len(list((d / part).glob(pattern))) if (d / part).exists() else 0 for _, d, pattern in STEPS]
    base, face, combined, seg = counts
    dim = segment_dim(part)
    if face < base:
        todo = f"02keypoint_face.py (PART = \"{part}\")"
    elif combined < base:
        todo = f"03keypoint_combine.py (PART = \"{part}\"，WITH_FACE = True)"
    elif seg == 0 or dim != FEAT_DIM:
        todo = "03split_npy.py (part 輸入 all)" + ("：目前是舊的 156 維" if dim == 156 else "")
    else:
        todo = "完成"
    print(f"{part:<6}" + "".join(f"{n:>17}" for n in counts) + f"{'-' if dim is None else dim:>12}   {todo}")

logs = sorted(p.name for folder in (ROOT / "Phase4-Keypoint", ROOT / "Phase5-Pause")
              for pattern in ("failed_*face*.txt", "failed_combine_part_*.txt", "failed_segments_part_*.txt")
              for p in folder.glob(pattern))
if logs:
    print("\n有失敗紀錄 (裡面列的檔案沒有處理成功)：")
    for name in logs:
        print(f"  {name}")

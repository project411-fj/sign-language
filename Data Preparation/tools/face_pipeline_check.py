"""列出每個 part 在臉部流程各步驟的檔案數，找出哪一步還沒跑完。

    02combined       原本的 156 維 (手 + pose)，臉部流程的起點
    01face           02keypoint_face.py 的輸出
    02combined_face  03keypoint_combine.py 的輸出 (WITH_FACE = True)
    03segments_face  03split_npy.py (FEATURE_TAG = "_face") 切出來的分段
    03segments       原本的分段，用來對照 03segments_face 應該要有幾個
"""
from pathlib import Path

# face_pipeline_check.py 在 Data Preparation\tools\，往上一層就是 Data Preparation
ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "Phase4-Keypoint" / "Keypoint_output"
OUT = ROOT / "Phase5-Pause" / "output"

STEPS = [
    ("02combined", KEY / "02combined", "*/*.npy"),
    ("01face", KEY / "01face", "*/*.npy"),
    ("02combined_face", KEY / "02combined_face", "*/*.npy"),
    ("03segments", OUT / "03segments", "*/*.npy"),
    ("03segments_face", OUT / "03segments_face", "*/*.npy"),
]

parts = sorted({p.name for _, d, _ in STEPS if d.exists() for p in d.iterdir() if p.is_dir()},
               key=lambda x: (len(x), x))
print(f"{'part':<6}" + "".join(f"{name:>17}" for name, _, _ in STEPS) + "   該做的事")
for part in parts:
    counts = [len(list((d / part).glob(pattern))) if (d / part).exists() else 0 for _, d, pattern in STEPS]
    base, face, combined, seg, seg_face = counts
    if face < base:
        todo = f"02keypoint_face.py (PART = \"{part}\")"
    elif combined < base:
        todo = f"03keypoint_combine.py (PART = \"{part}\"，WITH_FACE = True)"
    elif seg_face < seg:
        todo = "03split_npy.py (FEATURE_TAG = \"_face\"，part 輸入 all)"
    else:
        todo = "完成"
    print(f"{part:<6}" + "".join(f"{n:>17}" for n in counts) + f"   {todo}")

logs = sorted(p.name for folder in (ROOT / "Phase4-Keypoint", ROOT / "Phase5-Pause")
              for pattern in ("failed_*face*.txt", "failed_combine_part_*.txt") for p in folder.glob(pattern))
if logs:
    print("\n有失敗紀錄 (裡面列的檔案沒有處理成功)：")
    for name in logs:
        print(f"  {name}")

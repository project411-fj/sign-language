"""檢查 02keypoint_face.py 的輸出：臉偵測率、嘴巴有沒有在動。

用法：先用 02keypoint_face.py 跑完一個 part，把下面的 PART 改成同一個，再執行這支。
"""
from pathlib import Path

import numpy as np

PART = "1"

# face_check.py 在 Data Preparation\tools\，往上一層就是 Data Preparation
ROOT = Path(__file__).resolve().parents[1]
FACE_DIR = ROOT / "Phase4-Keypoint" / "Keypoint_output" / "01face" / PART
HAND_DIR = ROOT / "Phase4-Keypoint" / "Keypoint_output" / "02combined" / PART

# 02keypoint_face.py 的輸出格式
JAW_OPEN, MOUTH_CLOSE, MOUTH_FUNNEL, MOUTH_PUCKER, BROW_INNER_UP = 25, 27, 32, 38, 3
LIPS_START = 52
INNER_TOP, INNER_BOTTOM = 35, 25          # LIPS 清單裡內唇上緣 (13)、下緣 (14) 的位置
FLAG = 135

files = sorted(FACE_DIR.glob("*/*.npy"))
print(f"在 {FACE_DIR} 找到 {len(files)} 個 .npy")
if not files:
    raise SystemExit("找不到檔案，請確認 PART 跟 02keypoint_face.py 一樣，而且已經跑完")

per_file, all_frames, hand_active = [], [], []
for f in files:
    a = np.load(f)
    per_file.append((f.relative_to(FACE_DIR).as_posix(), a[:, FLAG].mean(), len(a)))
    all_frames.append(a)
    base = HAND_DIR / f.relative_to(FACE_DIR)
    if base.exists():
        b = np.load(base, mmap_mode="r")
        n = min(len(a), len(b))
        hand_active.append(np.pad((b[:n, 126] > 0.5) | (b[:n, 127] > 0.5), (0, len(a) - n)))
    else:
        hand_active.append(np.zeros(len(a), dtype=bool))

a = np.concatenate(all_frames)
hands = np.concatenate(hand_active)
face = a[:, FLAG] > 0.5
print(f"\n共 {len(a)} 格，偵測到臉：{face.mean():.1%}")

low = [(name, r, n) for name, r, n in per_file if r < 0.8]
print(f"臉偵測率低於 80% 的影片：{len(low)} / {len(per_file)} 支")
for name, r, n in sorted(low, key=lambda x: x[1])[:10]:
    print(f"  {name}：{r:.0%} ({n} 格)")

f = a[face]
opening = np.abs(f[:, LIPS_START + 2 * INNER_TOP + 1] - f[:, LIPS_START + 2 * INNER_BOTTOM + 1])
print("\n嘴巴開合 (內唇上下距離 ÷ 兩眼距離)：")
print(f"  平均 {opening.mean():.3f}，標準差 {opening.std():.3f}，"
      f"張開 (> 0.05) 的格數比例 {(opening > 0.05).mean():.1%}")
for name, idx in (("jawOpen 張嘴", JAW_OPEN), ("mouthFunnel 嘟成圓形", MOUTH_FUNNEL),
                  ("mouthPucker 噘嘴", MOUTH_PUCKER), ("browInnerUp 揚眉", BROW_INNER_UP)):
    print(f"  {name}：平均 {f[:, idx].mean():.3f}，標準差 {f[:, idx].std():.3f}")

# 比劃中 (有偵測到手) vs 沒在比劃：如果有嘴型，比劃時嘴巴應該動得比較多
active, idle = face & hands, face & ~hands
if active.sum() and idle.sum():
    def open_of(mask):
        x = a[mask]
        return np.abs(x[:, LIPS_START + 2 * INNER_TOP + 1] - x[:, LIPS_START + 2 * INNER_BOTTOM + 1])
    print(f"\n比劃中 (有手) 的嘴巴開合平均 {open_of(active).mean():.3f} ({active.sum()} 格)；"
          f"沒在比劃 {open_of(idle).mean():.3f} ({idle.sum()} 格)")

print("\n怎麼看：")
print("  臉偵測率最好在 90% 以上；太低的話，模型大部分時間拿不到臉部資訊")
print("  嘴巴開合的標準差、jawOpen 的標準差越大，代表嘴巴動得越多，嘴型資訊越豐富")

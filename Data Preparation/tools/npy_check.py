import random
from pathlib import Path

import numpy as np

# npy_check.py 在 Data Preparation\tools\，往上一層就是 Data Preparation
ROOT = Path(__file__).resolve().parents[1]
SEG_DIR = ROOT / "Phase5-Pause" / "output" / "03segments"
SAMPLE = 2000

files = sorted(SEG_DIR.glob("*/*/*.npy"))
print(f"在 {SEG_DIR} 找到 {len(files)} 個 .npy")
if not files:
    raise SystemExit("找不到檔案，請確認上面的路徑是否正確")

picked = random.Random(0).sample(files, min(SAMPLE, len(files)))   # 隨機抽樣，涵蓋所有 part
a = np.concatenate([np.load(f) for f in picked])
left, right, pose = a[:, 126] > 0.5, a[:, 127] > 0.5, a[:, 155] > 0.5

print(f"\n[版本 2] 共 {len(a)} 格 (隨機抽 {len(picked)} 個檔案)")
print(f"左手缺漏：{1 - left.mean():.1%}")
print(f"右手缺漏：{1 - right.mean():.1%}")
print(f"雙手都缺漏：{(~left & ~right).mean():.1%}")
print(f"pose 缺漏：{1 - pose.mean():.1%}")


# ---------------------------------------------------------------- 左手缺漏是真的放下，還是漏偵測？
# pose 9 點 (x, y, 信心值)：3 左肘、4 右肘、5 左腕、6 右腕；y 越小越上面
def raised(wrist, elbow):
    wy, wc = a[:, 128 + 3 * wrist + 1], a[:, 128 + 3 * wrist + 2]
    ey, ec = a[:, 128 + 3 * elbow + 1], a[:, 128 + 3 * elbow + 2]
    return (wc > 0.5) & (ec > 0.5) & (wy < ey)


def ratio(part, whole):
    return part.sum() / max(whole.sum(), 1)


both_up = raised(5, 3) & raised(6, 4) & pose
miss_left_only = ~left & right & pose
both_hands = left & right

print(f"\n只缺左手的格數：{miss_left_only.sum()}")
print(f"  其中 pose 看到兩隻手腕都舉起來：{ratio(miss_left_only & both_up, miss_left_only):.1%}")
print(f"  (對照) 雙手都偵測到的格中，兩隻手腕都舉起來：{ratio(both_hands & both_up, both_hands):.1%}")

# ---------------------------------------------------------------- 依 part 分開看
print("\n各 part 的左手缺漏：")
parts = np.concatenate([[f.relative_to(SEG_DIR).parts[0]] * len(np.load(f, mmap_mode="r")) for f in picked])
for p in sorted(set(parts), key=lambda x: (len(x), x)):
    m = parts == p
    print(f"  part {p}：{1 - left[m].mean():.1%} ({m.sum()} 格)")
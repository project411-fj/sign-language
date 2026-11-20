from pathlib import Path

import numpy as np
from tqdm import tqdm

PART = "7"  # 要合併的 part

ROOT = Path("Data Preparation/Phase4-Keypoint/Keypoint_output")
HAND_DIR = ROOT / "01hand" / PART
POSE_DIR = ROOT / "01pose" / PART
OUTPUT_DIR = ROOT / "02combined" / PART
FAILED_LOG = Path(__file__).parent / f"failed_combine_part_{PART}.txt"

# 有失敗紀錄就只合併紀錄裡的檔案，否則合併整個 part
if FAILED_LOG.exists():
    names = [Path(line) for line in FAILED_LOG.read_text(encoding="utf-8").splitlines() if line]
else:
    names = sorted(path.relative_to(HAND_DIR) for path in HAND_DIR.glob("*/*.npy"))

failed = []
for name in tqdm(names, desc="combine", unit="file"):
    try:
        hands = np.load(HAND_DIR / name)
        pose = np.load(POSE_DIR / name)
        if hands.shape[1:] != (128,) or pose.shape[1:] != (28,):
            raise ValueError(f"格式錯誤 hands={hands.shape} pose={pose.shape}")
        if len(hands) != len(pose):
            raise ValueError(f"Frame 數不一致 hands={len(hands)} pose={len(pose)}")

        output = OUTPUT_DIR / name
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.concatenate([hands, pose], axis=1).astype(np.float32))
    except Exception as exc:
        failed.append((name, exc))

# 有失敗就覆寫紀錄，全部成功就刪除紀錄
if failed:
    FAILED_LOG.write_text("".join(f"{name.as_posix()}\n" for name, _ in failed), encoding="utf-8")
    tqdm.write(f"\n合併失敗（{len(failed)} 個），已記錄到 {FAILED_LOG.name}：")
    for name, exc in failed:
        tqdm.write(f"  {name.as_posix()}：{exc}")
else:
    FAILED_LOG.unlink(missing_ok=True)
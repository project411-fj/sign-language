"""合併關鍵點：手 + 身體 + 臉 → 02combined (每格 292 維)。

輸入 (Keypoint_output/)：
    01hand/<PART>/   02keypoint_hand.py 的輸出，每格 128 維 (雙手 126 + 是否偵測到 2)
    01pose/<PART>/   02keypoint_pose.py 的輸出，每格 28 維 (9 點 x,y,信心值 + 是否偵測到)
    01face/<PART>/   02keypoint_face.py 的輸出，每格 136 維

輸出 02combined/<PART>/，每格 292 維：
    0:128    手 (左手 0:63、右手 63:126、是否偵測到 126/127)
    128:156  身體 9 點 (x, y, 信心值) + 是否偵測到 (155)
    156:208  52 個表情分數
    208:288  嘴唇 40 點 (x, y)
    288:291  頭部轉向 (pitch, yaw, roll)
    291      是否偵測到臉
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

PART = "12"  # 要合併的 part

ROOT = Path("Data Preparation/Phase4-Keypoint/Keypoint_output")
HAND_DIR = ROOT / "01hand" / PART
POSE_DIR = ROOT / "01pose" / PART
FACE_DIR = ROOT / "01face" / PART
OUTPUT_DIR = ROOT / "02combined" / PART
FAILED_LOG = Path(__file__).parent / f"failed_combine_part_{PART}.txt"

HAND_DIM, POSE_DIM, FACE_DIM = 128, 28, 136

for folder, script in ((HAND_DIR, "02keypoint_hand.py"), (POSE_DIR, "02keypoint_pose.py"), (FACE_DIR, "02keypoint_face.py")):
    if not folder.is_dir():
        raise SystemExit(f"找不到 {folder}：請先跑 {script} (PART = \"{PART}\")")

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
        face = np.load(FACE_DIR / name)
        if hands.shape[1:] != (HAND_DIM,) or pose.shape[1:] != (POSE_DIM,) or face.shape[1:] != (FACE_DIM,):
            raise ValueError(f"格式錯誤 hands={hands.shape} pose={pose.shape} face={face.shape}")
        if not len(hands) == len(pose) == len(face):
            raise ValueError(f"Frame 數不一致 hands={len(hands)} pose={len(pose)} face={len(face)}")

        output = OUTPUT_DIR / name
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.concatenate([hands, pose, face], axis=1).astype(np.float32))
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
    print(f"完成：{len(names)} 個檔案 → {OUTPUT_DIR} (每格 {HAND_DIM + POSE_DIM + FACE_DIM} 維)")

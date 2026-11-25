"""合併關鍵點：手 + 身體 → 02combined (156 維)，再接上臉部 → 02combined_face (292 維)。

(原本的 03keypoint_combine.py 與 03keypoint_combine_face.py 合併成這一支)

輸入 (Keypoint_output/)：
    01hand/<PART>/   02keypoint_hand.py 的輸出，每格 128 維 (雙手 126 + 是否偵測到 2)
    01pose/<PART>/   02keypoint_pose.py 的輸出，每格 28 維 (9 點 x,y,信心值 + 是否偵測到)
    01face/<PART>/   02keypoint_face.py 的輸出，每格 136 維 (WITH_FACE = True 時才需要)

輸出：
    02combined/<PART>/        0:156    手 128 + 身體 28
    02combined_face/<PART>/   0:156    跟 02combined 完全一樣
                              156:208  52 個表情分數
                              208:288  嘴唇 40 點 (x, y)
                              288:291  頭部轉向 (pitch, yaw, roll)
                              291      是否偵測到臉
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

PART = "12"        # 要合併的 part
WITH_FACE = True   # True = 同時輸出 02combined_face (需要先跑 02keypoint_face.py)；False = 只輸出 02combined

ROOT = Path("Data Preparation/Phase4-Keypoint/Keypoint_output")
HAND_DIR = ROOT / "01hand" / PART
POSE_DIR = ROOT / "01pose" / PART
FACE_DIR = ROOT / "01face" / PART
OUTPUT_DIR = ROOT / "02combined" / PART
OUTPUT_FACE_DIR = ROOT / "02combined_face" / PART
FAILED_LOG = Path(__file__).parent / f"failed_combine_part_{PART}.txt"

HAND_DIM, POSE_DIM, FACE_DIM = 128, 28, 136

if not HAND_DIR.is_dir():
    raise SystemExit(f"找不到 {HAND_DIR}：請先跑 02keypoint_hand.py (PART = \"{PART}\")")
if WITH_FACE and not FACE_DIR.is_dir():
    raise SystemExit(f"找不到 {FACE_DIR}：請先跑 02keypoint_face.py (PART = \"{PART}\")，"
                     "或把 WITH_FACE 改成 False 只合併手與身體")

# 有失敗紀錄就只合併紀錄裡的檔案，否則合併整個 part
if FAILED_LOG.exists():
    names = [Path(line) for line in FAILED_LOG.read_text(encoding="utf-8").splitlines() if line]
else:
    names = sorted(path.relative_to(HAND_DIR) for path in HAND_DIR.glob("*/*.npy"))

failed = []
for name in tqdm(names, desc="combine" + (" + face" if WITH_FACE else ""), unit="file"):
    try:
        hands = np.load(HAND_DIR / name)
        pose = np.load(POSE_DIR / name)
        if hands.shape[1:] != (HAND_DIM,) or pose.shape[1:] != (POSE_DIM,):
            raise ValueError(f"格式錯誤 hands={hands.shape} pose={pose.shape}")
        if len(hands) != len(pose):
            raise ValueError(f"Frame 數不一致 hands={len(hands)} pose={len(pose)}")
        base = np.concatenate([hands, pose], axis=1).astype(np.float32)
        output = OUTPUT_DIR / name
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, base)                          # 手 + 身體正常就先寫，臉部有問題也不影響 02combined

        if WITH_FACE:
            face = np.load(FACE_DIR / name)
            if face.shape[1:] != (FACE_DIM,):
                raise ValueError(f"臉部格式錯誤 face={face.shape} (02combined 已寫入，02combined_face 沒有)")
            if len(face) != len(base):
                raise ValueError(f"臉部 Frame 數不一致 hands/pose={len(base)} face={len(face)} "
                                 "(02combined 已寫入，02combined_face 沒有)")
            output = OUTPUT_FACE_DIR / name
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, np.concatenate([base, face], axis=1).astype(np.float32))
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
    print(f"完成：{len(names)} 個檔案 → {OUTPUT_DIR}" + (f"、{OUTPUT_FACE_DIR}" if WITH_FACE else ""))

"""把臉部特徵 (01face) 接在原本的 156 維 (02combined) 後面，輸出到 02combined_face。

輸出每格 292 維：
    0:156    跟原本 02combined 完全一樣 (手 128 + pose 28)
    156:208  52 個表情分數
    208:288  嘴唇 40 點 (x, y)
    288:291  頭部轉向 (pitch, yaw, roll)
    291      是否偵測到臉
原本的 02combined 不會被修改。
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

PART = "10"  # 要合併的 part

ROOT = Path("Data Preparation/Phase4-Keypoint/Keypoint_output")
BASE_DIR = ROOT / "02combined" / PART
FACE_DIR = ROOT / "01face" / PART
OUTPUT_DIR = ROOT / "02combined_face" / PART
FAILED_LOG = Path(__file__).parent / f"failed_combine_face_part_{PART}.txt"

BASE_DIM, FACE_DIM = 156, 136

# 有失敗紀錄就只合併紀錄裡的檔案，否則合併整個 part
if FAILED_LOG.exists():
    names = [Path(line) for line in FAILED_LOG.read_text(encoding="utf-8").splitlines() if line]
else:
    names = sorted(path.relative_to(BASE_DIR) for path in BASE_DIR.glob("*/*.npy"))

failed = []
for name in tqdm(names, desc="combine face", unit="file"):
    try:
        base = np.load(BASE_DIR / name)
        face = np.load(FACE_DIR / name)
        if base.shape[1:] != (BASE_DIM,) or face.shape[1:] != (FACE_DIM,):
            raise ValueError(f"格式錯誤 base={base.shape} face={face.shape}")
        if len(base) != len(face):
            raise ValueError(f"Frame 數不一致 base={len(base)} face={len(face)}")

        output = OUTPUT_DIR / name
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
    print(f"完成：{len(names)} 個檔案 → {OUTPUT_DIR}")

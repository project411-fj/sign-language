"""合併關鍵點：手 + 身體 + 臉 → 02combined (每格 292 維)。

輸入 (Keypoint_output/)：
    01hand/<part>/   02keypoint_hand.py 的輸出，每格 128 維 (雙手 126 + 是否偵測到 2)
    01pose/<part>/   02keypoint_pose.py 的輸出，每格 28 維 (9 點 x,y,信心值 + 是否偵測到)
    01face/<part>/   02keypoint_face.py 的輸出，每格 136 維

輸出 02combined/<part>/，每格 292 維：
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

# 要合併的 part 在執行時輸入：「3」、「1,2,5」或「all」(01hand 底下所有 part)
ROOT = Path("Data Preparation/Phase4-Keypoint/Keypoint_output")
INPUT_ROOT = ROOT / "01hand"

HAND_DIM, POSE_DIM, FACE_DIM = 128, 28, 136


def run_part(part):
    hand_dir, pose_dir, face_dir = ROOT / "01hand" / part, ROOT / "01pose" / part, ROOT / "01face" / part
    output_dir = ROOT / "02combined" / part
    failed_log = Path(__file__).parent / f"failed_combine_part_{part}.txt"

    for folder, script in ((hand_dir, "02keypoint_hand.py"), (pose_dir, "02keypoint_pose.py"),
                           (face_dir, "02keypoint_face.py")):
        if not folder.is_dir():
            print(f"part {part}：找不到 {folder}，請先跑 {script}，跳過這個 part")
            return

    # 有失敗紀錄就只合併紀錄裡的檔案，否則合併整個 part
    if failed_log.exists():
        names = [Path(line) for line in failed_log.read_text(encoding="utf-8").splitlines() if line]
    else:
        names = sorted(path.relative_to(hand_dir) for path in hand_dir.glob("*/*.npy"))

    failed = []
    for name in tqdm(names, desc=f"combine part {part}", unit="file"):
        try:
            hands = np.load(hand_dir / name)
            pose = np.load(pose_dir / name)
            face = np.load(face_dir / name)
            if hands.shape[1:] != (HAND_DIM,) or pose.shape[1:] != (POSE_DIM,) or face.shape[1:] != (FACE_DIM,):
                raise ValueError(f"格式錯誤 hands={hands.shape} pose={pose.shape} face={face.shape}")
            if not len(hands) == len(pose) == len(face):
                raise ValueError(f"Frame 數不一致 hands={len(hands)} pose={len(pose)} face={len(face)}")

            output = output_dir / name
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, np.concatenate([hands, pose, face], axis=1).astype(np.float32))
        except Exception as exc:
            failed.append((name, exc))

    # 有失敗就覆寫紀錄，全部成功就刪除紀錄
    if failed:
        failed_log.write_text("".join(f"{name.as_posix()}\n" for name, _ in failed), encoding="utf-8")
        tqdm.write(f"\npart {part} 合併失敗（{len(failed)} 個），已記錄到 {failed_log.name}：")
        for name, exc in failed:
            tqdm.write(f"  {name.as_posix()}：{exc}")
    else:
        failed_log.unlink(missing_ok=True)
        print(f"part {part} 完成：{len(names)} 個檔案 → {output_dir} (每格 {HAND_DIM + POSE_DIM + FACE_DIM} 維)")


def parse_parts(answer, root):
    """「3」→ ["3"]；「1,2,5」→ ["1", "2", "5"]；「all」→ root 底下所有 part (依數字排序)。"""
    answer = answer.strip()
    if answer.lower() == "all":
        return sorted((p.name for p in root.iterdir() if p.is_dir()), key=lambda x: (len(x), x))
    return [p.strip() for p in answer.split(",") if p.strip()]


def main():
    parts = parse_parts(input("part (例如 3、1,2,5 或 all): "), INPUT_ROOT)
    if not parts:
        print("沒有輸入 part")
        return
    print(f"要處理的 part：{'、'.join(parts)}")
    for part in parts:
        run_part(part)


if __name__ == "__main__":
    main()

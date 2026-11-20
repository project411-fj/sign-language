import re
from difflib import SequenceMatcher
from pathlib import Path

from tqdm import tqdm

# ========== 設定 ==========
BASE = Path("Data Preparation/Phase3-OCR/OCR_output")
INPUT_DIR = BASE / "05GeminiOCR_merged"
OUTPUT_DIR = BASE / "06GeminiOCR_merged"
SIMILARITY = 0.8   # 去掉空白後相似度 >= 這個值就視為同一句


def best(votes):
    return max(votes, key=votes.get)


def similar(a, b):
    a, b = a.replace(" ", ""), b.replace(" ", "")
    return a == b or SequenceMatcher(None, a, b).ratio() >= SIMILARITY


num = input("part: ").strip()
files = sorted((INPUT_DIR / num).glob("*/*_frames.txt"), key=lambda p: (int(p.parent.name), p.name))
log_path = OUTPUT_DIR / num / "merge_log.txt"

before = after = changed = 0
log = []
for src in tqdm(files, desc=f"part {num}", unit="file"):
    segments = []   # [起始幀, 結束幀, {文字: 幀數}, 原始行列表]
    for s, e, t in re.findall(r"(\d+)-(\d+):[ \t]*(.*)", src.read_text(encoding="utf-8")):
        s, e = int(s), int(e)
        if segments and similar(t, best(segments[-1][2])):
            segments[-1][1] = e                                   # 相似 → 延長上一個區間
            segments[-1][2][t] = segments[-1][2].get(t, 0) + e - s + 1
            segments[-1][3].append((s, e, t))
        else:
            segments.append([s, e, {t: e - s + 1}, [(s, e, t)]])
        before += 1
    after += len(segments)

    # 記錄有不同文字被合併的區間
    file_log = []
    for s, e, votes, originals in segments:
        if len(votes) > 1:
            changed += 1
            file_log.append(f"[{s}-{e}]")
            file_log.append("  改動前：")
            file_log += [f"    {os_}-{oe}: {ot}" for os_, oe, ot in originals]
            file_log.append(f"  改動後：{s}-{e}: {best(votes)}")
            file_log.append("  各版本幀數：" + "、".join(f"「{t}」{n}" for t, n in sorted(votes.items(), key=lambda x: -x[1])))
            file_log.append("")
    if file_log:
        log.append(f"===== {src.parent.name}/{src.name} =====")
        log += file_log

    out = OUTPUT_DIR / num / src.parent.name / src.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{s}-{e}: {best(v)}\n" for s, e, v, _ in segments), encoding="utf-8")

log_path.parent.mkdir(parents=True, exist_ok=True)
log_path.write_text(f"part {num}：共 {changed} 個區間有文字改動\n\n" + "\n".join(log), encoding="utf-8")

print(f"part {num}：{len(files)} 個檔案，區間 {before} → {after}（合併掉 {before - after} 個）")
print(f"有文字改動的區間 {changed} 個，詳見 {log_path}")
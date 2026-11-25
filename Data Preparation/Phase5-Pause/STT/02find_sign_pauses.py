"""STT 版第 2 步：在語音句子的邊界附近找手語員動作停下來的地方，調整每句的起訖格。

和 OCR/02find_sign_pauses.py 的演算法完全一樣，只是錨點改用 01stt_frames.py 轉出來的語音句子時間，
輸入、輸出都在 output_STT/ 底下，不會蓋掉 OCR 版的結果。
執行後輸入 part：「3」、「1,2,5」或「all」(output_STT/01frames 底下所有 part)。
"""
import csv
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks
from tqdm import tqdm

warnings.filterwarnings("ignore")   # 執行時只顯示進度條

# ================================================================ 路徑設定 (後面會再接 /<part>)
NPY_DIR = "Data Preparation/Phase4-Keypoint/Keypoint_output/02combined"
TXT_DIR = "Data Preparation/Phase5-Pause/output_STT/01frames"       # 01stt_frames.py 的輸出
OUT_DIR = "Data Preparation/Phase5-Pause/output_STT/02pause"
FPS_CSV = "Data Preparation/Phase5-Pause/output_STT/01fps"          # 裡面是 fps_<part>.csv (01stt_frames.py 寫的)
PLOT_DIR = "Data Preparation/Phase5-Pause/output_STT/02check_plots"  # 不想畫圖就改成 None

# ================================================================ 參數設定
TXT_SUFFIX = "_frames"     # 000600_001507.npy -> 000600_001507_frames.txt
# 參數沿用 OCR 版；手語比語音慢多少和比字幕慢多少不一定一樣，可以看 02check_plots 再調整
SEARCH_BEFORE = 0.5        # 往語音句子邊界之前找幾秒
SEARCH_AFTER = 2.0         # 往語音句子邊界之後找幾秒
EXPECTED_LAG = 0.5         # 預期手語比語音慢幾秒
MIN_LEN = 0.3              # 每句至少幾秒
MERGE_GAP = 0.5            # 兩句間隔小於幾秒就共用一個切點
FRAME_OFFSET = 0           # .txt 的 frame 編號 + FRAME_OFFSET = .npy 的列索引

# ---------------------------------------------------------------- 特徵位置
HANDS = slice(0, 126)          # 左手 0:63、右手 63:126，各 21 點 (x, y, z)
HAND_FLAGS = slice(126, 128)   # 左手、右手是否存在
POSE = slice(128, 155)         # 9 點 (x, y, conf)：鼻、左右肩、左右肘、左右腕、左右髖
POSE_FLAG = 155

POSE_SHOULDERS = (1, 2)
POSE_ELBOWS = (3, 4)
POSE_WRISTS = (5, 6)

CONF_THR = 0.3          # pose 點信心值低於這個就用內插
ELBOW_WEIGHT = 0.5      # 手肘速度相對於手腕的權重
SHAPE_WEIGHT = 0.5      # 手形變化相對於手臂移動的權重
LAG_WEIGHT = 0.5        # 「離預期延遲多遠」在分數裡的權重
ORDER_PENALTY = 1e3     # 違反順序 / 最短長度的懲罰

# .txt 每行：「1-194: 字幕文字」
LINE_RE = re.compile(r"^\s*(\d+)\s*[~～\-–—]\s*(\d+)\s*[:：]?\s*(.*?)\s*$")


# ---------------------------------------------------------------- 讀寫 txt
@dataclass
class Segment:
    start: int
    end: int
    text: str


def read_segments(path):
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp950"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    segs = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if m:
            s, e = sorted((int(m[1]) + FRAME_OFFSET, int(m[2]) + FRAME_OFFSET))
            segs.append(Segment(s, e, m[3]))
    return sorted(segs, key=lambda s: s.start)


def write_segments(path, segs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{s.start - FRAME_OFFSET}-{s.end - FRAME_OFFSET}: {s.text}\n" for s in segs),
        encoding="utf-8",
    )


def load_fps_table(csv_path):
    """{子資料夾/檔名: fps}，1/000600_001507_upper_body.mp4 -> 1/000600_001507"""
    with csv_path.open(encoding="utf-8-sig") as f:
        return {
            Path(row["filename"]).with_suffix("").as_posix().removesuffix("_upper_body"): float(row["fps"])
            for row in csv.DictReader(f)
        }


# ---------------------------------------------------------------- 動作能量
def interp_missing(xy, ok):
    """沒偵測到的 frame 用前後的值線性內插。"""
    idx = np.arange(len(xy))
    return np.stack([np.interp(idx, idx[ok], xy[ok, c]) for c in range(xy.shape[1])], axis=1)


def fill_short_gaps(x, max_gap):
    """NaN 的短缺口用內插補，長缺口 (手不在畫面) 補 0。"""
    nan = np.isnan(x)
    if nan.all():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[~nan], x[~nan])
    out = x.copy()
    edges = np.diff(np.r_[0, nan.astype(np.int8), 0])
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        short = e - s <= max_gap and s > 0 and e < len(x)
        out[s:e] = filled[s:e] if short else 0.0
    return out


def robust_scale(x):
    pos = x[x > 0]
    return float(np.median(pos)) if pos.size else 1.0


def motion_energy(arr, fps):
    """長度 T 的平滑動作能量，已除以中位數 (1.0 = 一般的動作量)。"""
    T = len(arr)
    pose = arr[:, POSE].reshape(T, 9, 3)
    pose_ok = arr[:, POSE_FLAG] > 0.5

    # 手腕 / 手肘速度 (YOLO Pose)，用肩寬正規化
    arm = np.zeros(T)
    sh = np.linalg.norm(pose[pose_ok, POSE_SHOULDERS[0], :2] - pose[pose_ok, POSE_SHOULDERS[1], :2], axis=1)
    scale = robust_scale(sh)
    for joints, w in ((POSE_WRISTS, 1.0), (POSE_ELBOWS, ELBOW_WEIGHT)):
        for k in joints:
            ok = pose_ok & (pose[:, k, 2] > CONF_THR)
            if ok.sum() < 2:
                continue
            xy = median_filter(interp_missing(pose[:, k, :2].astype(np.float64), ok), size=(3, 1), mode="nearest")
            arm[1:] += w * np.linalg.norm(np.diff(xy, axis=0), axis=1) / scale
    arm[0] = arm[1]

    # 手形變化速度 (MediaPipe Hands)，前後兩個 frame 都有手才算
    hands = arr[:, HANDS].reshape(T, 2, 21, 3).astype(np.float64)
    present = arr[:, HAND_FLAGS] > 0.5
    shape = np.zeros(T)
    for h in range(2):
        d = np.full(T, np.nan)
        both = present[1:, h] & present[:-1, h]
        d[1:][both] = np.linalg.norm(hands[1:, h] - hands[:-1, h], axis=-1).mean(-1)[both]
        if np.isfinite(d).sum() >= 5:
            d = np.minimum(d, np.nanpercentile(d, 95))   # 去掉左右手誤判的尖峰
        shape += fill_short_gaps(d, max(1, round(0.2 * fps)))

    energy = arm / robust_scale(arm) + SHAPE_WEIGHT * shape / robust_scale(shape)
    energy = gaussian_filter1d(energy, sigma=max(1.0, 0.1 * fps), mode="nearest")
    return energy / robust_scale(energy)


# ---------------------------------------------------------------- 找停頓點
@dataclass
class Boundary:
    anchor: int        # STT 給的 frame
    kind: str          # "start" / "end" / "cut" (前一句結尾兼下一句開頭)
    seg_end: int       # 是哪一句的結尾 (-1 = 沒有)
    seg_start: int     # 是哪一句的開頭 (-1 = 沒有)
    frame: int = -1


def build_boundaries(segs, merge_gap):
    """相鄰兩句幾乎相連 → 共用一個切點；中間有空白 → 各找各的。"""
    bs = [Boundary(segs[0].start, "start", -1, 0)]
    for i in range(len(segs) - 1):
        if segs[i + 1].start - segs[i].end <= merge_gap:
            bs.append(Boundary(segs[i].end, "cut", i, i + 1))
        else:
            bs.append(Boundary(segs[i].end, "end", i, -1))
            bs.append(Boundary(segs[i + 1].start, "start", -1, i + 1))
    bs.append(Boundary(segs[-1].end, "end", len(segs) - 1, -1))
    return bs


def low_run(energy, t, lo, hi):
    """以 t 為中心，能量跟 t 差不多低的連續範圍。"""
    thr = energy[t] + 0.15
    a = b = t
    while a > lo and energy[a - 1] <= thr:
        a -= 1
    while b < hi and energy[b + 1] <= thr:
        b += 1
    return a, b


def find_pauses(energy, bounds, fps):
    T = len(energy)
    before, after = round(SEARCH_BEFORE * fps), round(SEARCH_AFTER * fps)
    lag, span = round(EXPECTED_LAG * fps), max(1, before + after)
    min_len, hold = max(1, round(MIN_LEN * fps)), max(1, round(0.3 * fps))
    minima, _ = find_peaks(-energy)

    # 每個邊界的候選點與分數
    cands, costs, windows = [], [], []
    for b in bounds:
        lo, hi = int(np.clip(b.anchor - before, 0, T - 1)), int(np.clip(b.anchor + after, 0, T - 1))
        c = set(minima[(minima >= lo) & (minima <= hi)].tolist())
        c.add(lo + int(np.argmin(energy[lo:hi + 1])))
        c = np.array(sorted(c))
        cands.append(c)
        costs.append(energy[c] + LAG_WEIGHT * np.abs(c - b.anchor - lag) / span)
        windows.append((lo, hi))

    # 動態規劃：總分最低且順序合理
    best, back = [costs[0]], []
    for i in range(1, len(bounds)):
        need = min_len if bounds[i].seg_end >= 0 else 1
        trans = np.where(cands[i][None, :] - cands[i - 1][:, None] >= need, 0.0, ORDER_PENALTY)
        total = best[-1][:, None] + trans
        back.append(total.argmin(0))
        best.append(total.min(0) + costs[i])
    j = int(best[-1].argmin())
    picks = [0] * len(bounds)
    for i in range(len(bounds) - 1, -1, -1):
        picks[i] = int(cands[i][j])
        if i > 0:
            j = int(back[i - 1][j])

    # 微調：停頓是一段時間
    for b, t, (lo, hi) in zip(bounds, picks, windows):
        a, z = low_run(energy, t, lo, hi)
        mid = (a + z) // 2
        if b.kind == "end":          # 句尾：靜止開始後不久
            b.frame = min(mid, a + hold)
        elif b.kind == "start":      # 句首：靜止快結束時
            b.frame = max(mid, z - hold)
        else:                        # 兩句之間：停頓正中間
            b.frame = mid

    # 保證嚴格遞增
    for i in range(1, len(bounds)):
        if bounds[i].frame <= bounds[i - 1].frame:
            bounds[i].frame = min(T - 1, bounds[i - 1].frame + 1)


# ---------------------------------------------------------------- 單一檔案
def align_file(npy_path, txt_path, out_path, fps, plot_path):
    arr = np.nan_to_num(np.load(npy_path).astype(np.float32))
    T = len(arr)
    segs = [s for s in read_segments(txt_path) if s.start <= T - 1]   # 超出 .npy 的句子略過
    if not segs:
        raise ValueError(f"沒有可用的語音句子 (T={T})")

    energy = motion_energy(arr, fps)
    bounds = build_boundaries(segs, round(MERGE_GAP * fps))
    find_pauses(energy, bounds, fps)

    new = [Segment(s.start, s.end, s.text) for s in segs]
    for b in bounds:
        if b.seg_end >= 0:
            new[b.seg_end].end = b.frame
        if b.seg_start >= 0:
            new[b.seg_start].start = b.frame + 1 if b.kind == "cut" else b.frame
    for s in new:
        s.start = int(np.clip(s.start, 0, T - 1))
        s.end = int(np.clip(max(s.end, s.start), 0, T - 1))

    write_segments(out_path, new)
    if plot_path:
        save_plot(plot_path, energy, segs, new)


def save_plot(path, energy, old, new):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(12, len(energy) / 70), 4))
    ax.plot(energy, color="#444", lw=1)
    for i, s in enumerate(new):
        ax.axvspan(s.start, s.end, color=("#2a9d8f", "#e9c46a")[i % 2], alpha=0.18)
    for s in old:
        ax.axvline(s.start, color="#d62828", ls="--", lw=1)
        ax.axvline(s.end, color="#d62828", ls="--", lw=1)
    for s in new:
        ax.axvline(s.start, color="#1d6f42", lw=1.6)
        ax.axvline(s.end, color="#1d6f42", lw=1.6)
    ax.plot([], [], color="#d62828", ls="--", label="STT boundary")
    ax.plot([], [], color="#1d6f42", lw=1.6, label="sign pause (output)")
    ax.set(xlim=(0, len(energy)), ylim=(0, float(np.percentile(energy, 99)) * 1.3),
           xlabel="frame", ylabel="motion energy (1 = median)", title=path.stem)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- 主程式
def run_part(part):
    npy_dir = Path(NPY_DIR) / part
    txt_dir = Path(TXT_DIR) / part
    out_dir = Path(OUT_DIR) / part
    plot_dir = Path(PLOT_DIR) / part if PLOT_DIR else None
    failed_log = Path(__file__).parent / f"failed_pause_part_{part}.txt"
    fps_csv = Path(FPS_CSV) / f"fps_{part}.csv"
    if not fps_csv.exists():
        print(f"part {part}：找不到 {fps_csv}，請先跑 01stt_frames.py，跳過")
        return
    fps_table = load_fps_table(fps_csv)

    # 有失敗紀錄就只跑紀錄裡的檔案，否則跑整個 part (名稱如 1/000600_001507)
    if failed_log.exists():
        names = [line for line in failed_log.read_text(encoding="utf-8").splitlines() if line]
    else:
        files = sorted(
            npy_dir.glob("*/*.npy"),
            key=lambda p: (int(p.parent.name), [int(x) for x in p.stem.split("_")]),
        )
        names = [p.relative_to(npy_dir).with_suffix("").as_posix() for p in files]

    # 沒有語音句子的片段 (沒有 .txt 或是空的) 直接跳過，不算失敗
    def has_speech(name):
        txt = txt_dir / Path(name).parent / f"{Path(name).name}{TXT_SUFFIX}.txt"
        return txt.exists() and txt.stat().st_size > 0
    no_speech = [name for name in names if not has_speech(name)]
    names = [name for name in names if has_speech(name)]
    if no_speech:
        print(f"part {part}：{len(no_speech)} 支片段沒有語音句子，跳過")

    failed = []
    for name in tqdm(names, desc=f"pause part {part}", unit="file"):
        rel = Path(name)
        txt_name = f"{rel.name}{TXT_SUFFIX}.txt"
        try:
            align_file(
                npy_dir / f"{name}.npy",
                txt_dir / rel.parent / txt_name,
                out_dir / rel.parent / txt_name,
                fps_table[name],
                plot_dir / f"{name}.png" if plot_dir else None,
            )
        except Exception as exc:
            failed.append((name, f"{type(exc).__name__}: {exc}"))

    # 有失敗就覆寫紀錄，全部成功就刪除紀錄
    if failed:
        failed_log.write_text("".join(f"{name}\n" for name, _ in failed), encoding="utf-8")
        tqdm.write(f"\n失敗 {len(failed)} 個，已記錄到 {failed_log.name}：")
        for name, error in failed:
            tqdm.write(f"  {name}：{error}")
    else:
        failed_log.unlink(missing_ok=True)


def parse_parts(answer, root):
    """「3」→ ["3"]；「1,2,5」→ ["1", "2", "5"]；「all」→ root 底下所有 part (依數字排序)。"""
    answer = answer.strip()
    if answer.lower() == "all":
        return sorted((p.name for p in root.iterdir() if p.is_dir()), key=lambda x: (len(x), x))
    return [p.strip() for p in answer.split(",") if p.strip()]


def main():
    parts = parse_parts(input("part (例如 3、1,2,5 或 all): "), Path(TXT_DIR))
    if not parts:
        print("沒有輸入 part")
        return
    print(f"要處理的 part：{'、'.join(parts)}")
    for part in parts:
        run_part(part)


if __name__ == "__main__":
    main()

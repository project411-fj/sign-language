"""訓練 02style.py 用的字幕條樣式分類模型 (YOLO 分類，兩類：target / other)。

用法：
    1. 把「目標節目」的字幕條圖片放進 target/，其他節目 / 非字幕畫面的圖片放進 other/
       (可以直接放 01frames_crop 裁出來的 frame_xxxxxx.jpg；可以再分子資料夾，例如 target/13_1/、target/14_3/)
    2. 在專案根目錄執行：python "Data Preparation/Phase1-filter/02style_train/train_style.py"
    3. 訓練完的模型存成 02style_train/best.pt
       02style.py 讀的是 Phase1-filter/best.pt，要使用新模型就把它複製過去 (或修改 02style.py 的 WEIGHTS)

圖片會先經過跟 02style.py 完全相同的前處理 (只取字幕條左邊 3 倍高度的寬度，再縮成正方形)，
訓練時看到的畫面才會跟實際使用時一樣。
"""
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ========== 設定 ==========
BASE = Path(__file__).resolve().parent            # 02style_train/
CLASSES = ["target", "other"]                     # 資料夾名稱 = 類別名稱 (02style.py 依名稱找 "target")
DATASET = BASE / "_dataset"                       # 自動產生的訓練 / 驗證資料 (每次執行會重建)
RUNS = BASE / "runs"                              # YOLO 的訓練紀錄
OUTPUT = BASE / "best.pt"                         # 訓練完的模型

BASE_MODEL = "yolo26n-cls.pt"   # 預訓練的分類模型 (第一次會自動下載)；下載不到可改成 "yolo11n-cls.pt"
CROP_SIZE = 320                 # 必須跟 02style.py 的 CROP_SIZE 相同
EPOCHS = 50
PATIENCE = 10                   # 驗證集準確率連續幾個 epoch 沒進步就提早停止
BATCH = 64
VAL_RATIO = 0.2                 # 拿多少比例當驗證集
SEED = 42
TARGET_THRESHOLD = 0.9          # 跟 02style.py 相同：target 機率 >= 這個值才算 target (只用在最後的評估)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WORKERS = 8                     # 前處理圖片的執行緒數


def crop_subtitle(img):
    """只取字幕條左邊 3 倍高度的寬度，再縮成正方形 (跟 02style.py 完全相同)。"""
    img = img.convert("RGB")
    w, h = img.size
    return img.crop((0, 0, min(w, 3 * h), h)).resize((CROP_SIZE, CROP_SIZE))


def collect(cls):
    """回傳 {組別: [圖片, ...]}。放在子資料夾裡的圖片以子資料夾為一組，直接放在類別資料夾的每張自成一組。"""
    root = BASE / cls
    groups = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXT and p.is_file():
            rel = p.relative_to(root)
            key = rel.parts[0] if len(rel.parts) > 1 else rel.as_posix()
            groups.setdefault(key, []).append(p)
    return groups


def split(groups):
    """以組為單位分訓練 / 驗證：同一支影片的連續畫面幾乎一樣，拆開放會讓驗證準確率虛高。"""
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    if len(keys) == 1:                                  # 只有一組 (例如全部放在同一個子資料夾)：改成逐張分
        files = groups[keys[0]][:]
        random.Random(SEED).shuffle(files)
        n_val = max(1, round(len(files) * VAL_RATIO))
        return files[n_val:], files[:n_val]
    total = sum(len(groups[k]) for k in keys)
    val, train, n_val = [], [], 0
    for k in keys:
        if n_val < total * VAL_RATIO:
            val += groups[k]
            n_val += len(groups[k])
        else:
            train += groups[k]
    if not train:                                       # 組數太少全分到驗證集時，至少留一組訓練
        train, val = val[len(groups[keys[0]]):], val[:len(groups[keys[0]])]
    return train, val


def build_dataset():
    """依 YOLO 分類的格式建立 _dataset/{train,val}/{target,other}/，圖片先經過 crop_subtitle。"""
    if DATASET.exists():
        shutil.rmtree(DATASET)
    jobs, counts = [], {}
    for cls in CLASSES:
        groups = collect(cls)
        n = sum(len(v) for v in groups.values())
        if n < 2:
            raise SystemExit(f"{BASE / cls} 裡只有 {n} 張圖片，每一類至少需要 2 張 (建議數百張以上)")
        train, val = split(groups)
        counts[cls] = (len(train), len(val), len(groups))
        for part, files in (("train", train), ("val", val)):
            out = DATASET / part / cls
            out.mkdir(parents=True, exist_ok=True)
            jobs += [(p, out / f"{i:06d}.jpg") for i, p in enumerate(files)]

    def convert(job):
        src, dst = job
        with Image.open(src) as img:
            crop_subtitle(img).save(dst, quality=95)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(tqdm(pool.map(convert, jobs), total=len(jobs), desc="前處理圖片", unit="張"))

    print("\n資料：")
    for cls, (n_train, n_val, n_groups) in counts.items():
        print(f"  {cls:<7} 訓練 {n_train:>6} 張、驗證 {n_val:>6} 張 ({n_groups} 組)")
    return counts


def evaluate(model):
    """用跟 02style.py 相同的判斷方式 (target 機率 >= TARGET_THRESHOLD)，算驗證集的結果。"""
    target_idx = [k for k, v in model.names.items() if v == "target"][0]
    stats = {}
    for cls in CLASSES:
        paths = sorted((DATASET / "val" / cls).glob("*.jpg"))
        hits = 0
        for b in range(0, len(paths), BATCH):
            results = model.predict([str(p) for p in paths[b:b + BATCH]], imgsz=CROP_SIZE, verbose=False)
            hits += sum(float(r.probs.data[target_idx]) >= TARGET_THRESHOLD for r in results)
        stats[cls] = (hits, len(paths))
    (tp, n_t), (fp, n_o) = stats["target"], stats["other"]
    print(f"\n驗證集 (target 機率 >= {TARGET_THRESHOLD} 才算 target，跟 02style.py 相同)：")
    print(f"  target 被正確認出：{tp}/{n_t} ({tp / max(n_t, 1):.1%})")
    print(f"  other 被誤認成 target：{fp}/{n_o} ({fp / max(n_o, 1):.1%})")


def main():
    import torch
    from ultralytics import YOLO

    build_dataset()
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(BASE / BASE_MODEL))   # 預訓練模型下載到 02style_train/，不會散落在執行的資料夾
    model.train(
        data=str(DATASET), imgsz=CROP_SIZE, epochs=EPOCHS, patience=PATIENCE, batch=BATCH,
        device=device, seed=SEED, project=str(RUNS), name="train", exist_ok=True,
        fliplr=0.0,              # 字幕條有固定的左右版面，不做左右翻轉
        plots=True,
    )

    best = RUNS / "train" / "weights" / "best.pt"
    shutil.copy2(best, OUTPUT)
    print(f"\n模型已存成 {OUTPUT}")
    evaluate(YOLO(str(OUTPUT)))
    print(f"\n要讓 02style.py 使用這個模型：把 {OUTPUT.name} 複製成 {BASE.parent / 'best.pt'} (舊的請先備份)")


if __name__ == "__main__":
    main()

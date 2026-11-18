"""用 Gemini 重新辨識 03EasOCR_frame 的字幕圖片，取代 02EasOCR_merged 每個區間的文字，輸出到 04GeminiOCR_context。

輸入圖片：03EasOCR_frame/<num>/<n>/<name>_sXX.jpg   （XX = 02EasOCR_merged 檔案內第幾個區間，從 00 開始）
對應字幕：02EasOCR_merged/<num>/<n>/<name>_frames.txt 的第 XX 行
輸出字幕：04GeminiOCR_context/<num>/<n>/<name>_frames.txt（frame 區間不變，只換文字）

文字標記：NONE = 還沒有 Gemini 結果（失敗或尚未處理）、NOWORD = Gemini 判斷沒有字、UNFOUND = 03 找不到對應圖片。
再次執行同一個 part 時，只會把 04GeminiOCR_context 裡顯示為 NONE 的區間所對應的圖片送給 Gemini。
"""

import json
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

# ========== 設定 ==========
BASE = Path("Data Preparation/Phase3-OCR/OCR_output")
FRAME_DIR = BASE / "03EasOCR_frame"
MERGED_DIR = BASE / "02EasOCR_merged"
OUTPUT_DIR = BASE / "04GeminiOCR_context"

MODEL_NAME = "gemini-3.5-flash-lite"
BATCH_SIZE = 32       # 每次請求送幾張圖
MAX_RETRY = 3         # 單一批次失敗時最多重試幾次
SLEEP_BETWEEN = 1.0   # 每次請求之間停幾秒

PROMPT = """你會收到數張編號的字幕圖片。
請逐字輸出每張圖片中的字幕文字，包含中文、英文、數字與標點。
不要翻譯、不要改寫、不要補字、不要加任何說明。
若某張圖片中沒有文字，text 填 NOWORD。

回傳 JSON 陣列，每張圖片一個物件：{"index": 圖片編號, "text": 文字}
每個編號都必須出現一次，不可遺漏。"""

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {"index": {"type": "INTEGER"}, "text": {"type": "STRING"}},
        "required": ["index", "text"],
        "propertyOrdering": ["index", "text"],
    },
}

client = genai.Client()  # 讀取環境變數 GEMINI_API_KEY


def ocr_batch(paths):
    contents = [PROMPT]
    for i, path in enumerate(paths, start=1):
        contents.append(f"【圖片 {i}】")
        contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg"))

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0,
        ),
    )
    results = {int(item["index"]): " ".join(item["text"].split()) or "NOWORD" for item in json.loads(response.text)}
    return results, response.usage_metadata


def save(out):
    prefixes, texts = files[out]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{p}: {t}\n" for p, t in zip(prefixes, texts)), encoding="utf-8")


num = input("part: ").strip()
frame_root = FRAME_DIR / num
out_root = OUTPUT_DIR / num

# ===== 1. 讀 02 的區間和 04 目前的文字，找出要送 Gemini 的圖片 =====
files = {}       # 輸出檔 -> (區間列表, 文字列表)
todo = []        # (輸出檔, 第幾個區間, 圖片路徑)
no_image = []    # 是 NONE 但 03 裡沒有對應圖片的區間（改寫成 UNFOUND）
merged_files = sorted((MERGED_DIR / num).glob("*/*_frames.txt"), key=lambda p: (int(p.parent.name), p.name))
for src in merged_files:
    n = src.parent.name
    name = src.stem.removesuffix("_frames")       # 021794_022876_frames -> 021794_022876
    prefixes = [re.match(r"(\d+-\d+):", line).group(1) for line in src.read_text(encoding="utf-8").splitlines()]

    out = out_root / n / src.name
    texts = ["NONE"] * len(prefixes)
    if out.exists():
        old = [re.match(r"\d+-\d+:\s*(.*)", line).group(1) for line in out.read_text(encoding="utf-8").splitlines()]
        if len(old) == len(prefixes):             # 區間數對不上（02 被改過）就整個檔案重跑
            texts = old
    files[out] = (prefixes, texts)

    for i, text in enumerate(texts):
        if text == "NONE":
            img = frame_root / n / f"{name}_s{i:02d}.jpg"
            if img.exists():
                todo.append((out, i, img))
            else:
                texts[i] = "UNFOUND"
                no_image.append(img)
    save(out)   # 先把目前狀態寫出來，中斷後重跑也能從 NONE 接續

total = sum(len(p) for p, _ in files.values())
print(f"===== part {num}：共 {total} 個區間，本次送出 {len(todo)} 張 =====")

# ===== 2. 送 Gemini，每批完成就把有更新的檔案寫回 =====
total_in = total_out = 0
failed = []
for start in range(0, len(todo), BATCH_SIZE):
    batch = todo[start:start + BATCH_SIZE]

    results = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            results, usage = ocr_batch([img for _, _, img in batch])
            break
        except Exception as e:
            print(f"\n  第 {attempt} 次失敗：{e}")
            time.sleep(2 ** attempt)

    if results is None:
        failed.extend(img for _, _, img in batch)
    else:
        total_in += usage.prompt_token_count or 0
        total_out += usage.candidates_token_count or 0
        for k, (out, i, img) in enumerate(batch, start=1):
            if k in results:
                files[out][1][i] = results[k]
            else:
                failed.append(img)  # Gemini 漏回這張，保持 NONE，下次執行會重跑
        for out in {out for out, _, _ in batch}:
            save(out)

    done_n = min(start + BATCH_SIZE, len(todo))
    bar = "█" * (30 * done_n // len(todo))
    print(f"\r[{bar:<30}] {done_n}/{len(todo)} ({100 * done_n // len(todo)}%)", end="", flush=True)
    time.sleep(SLEEP_BETWEEN)
print()

print("\n--- Token 消耗統計 ---")
print(f"輸入 Token: {total_in}")
print(f"輸出 Token: {total_out}")
print(f"總消耗 Token: {total_in + total_out}")

if failed:
    print(f"\n有 {len(failed)} 張 Gemini 沒有成功（已寫 NONE），重新執行程式即可只重跑這些：")
    for img in failed:
        print(f"  {img}")

if no_image:
    print(f"\n有 {len(no_image)} 個區間在 03EasOCR_frame 找不到圖片（已寫 UNFOUND）：")
    for img in no_image:
        print(f"  {img}")
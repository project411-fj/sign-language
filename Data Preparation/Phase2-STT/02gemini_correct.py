"""用 Gemini 修正語音轉文字 (01voice.py / Breeze) 的繁體中文錯字。

輸入：context_breeze/<part>/<n>/<start>_<end>.csv   (01voice.py 的輸出，欄位 start, end, text)
輸出：context_breeze_gemini/<part>/<n>/<start>_<end>.csv
      欄位 start, end, text, original：text 是修正後的句子，original 是原本的句子 (方便對照)；
      start、end 不變，句子數不變，所以 Phase5 STT/ 的程式可以直接讀 (把 STT_DIR 改成 context_breeze_gemini)。

執行後輸入 part：「13」、「13,14」或「all」(context_breeze 底下所有 part)。
已經修正過的檔案 (輸出檔已存在) 會跳過；中斷後重新執行，會從還沒做完的檔案繼續。
"""
import csv
import difflib
import json
import time
from pathlib import Path

from google import genai
from google.genai import types

# ========== 設定 ==========
API_KEY = ""          # 在這裡貼上 Gemini API key；留空就讀環境變數 GEMINI_API_KEY
                      # 注意：填了 key 之後不要 commit / push 到 GitHub (公開的 key 會被盜用)

INPUT_ROOT = Path("Data Preparation/Phase2-STT/context_breeze")            # 01voice.py 的輸出
OUTPUT_ROOT = Path("Data Preparation/Phase2-STT/context_breeze_gemini")    # 修正後的輸出

MODEL_NAME = "gemini-3.5-flash-lite"
BATCH_SIZE = 40        # 每次請求送幾句 (同一支片段裡連續的句子，讓 Gemini 看得到上下文)
MAX_RETRY = 3          # 單一批次失敗時最多重試幾次
SLEEP_BETWEEN = 1.0    # 每次請求之間停幾秒
MIN_SIMILARITY = 0.5   # 修正後跟原句的相似度 (0~1) 低於這個，就當作 Gemini 改太多，保留原句

# 影片的主題、會出現的人名 / 地名 / 專有名詞，寫在這裡可以幫 Gemini 改對 (可以留空)
TOPIC = "台灣的電視新聞與政見辯論會"
TERMS = ""             # 例如 "盧秀燕、臺中市、公辦電視政見辯論會"

PROMPT = """你會收到一段語音辨識 (STT) 的逐句結果，內容是{topic}，每句有編號，順序就是說話的順序。
語音辨識常把字聽成同音或近音的錯字，請只修正這類錯字，輸出正確的繁體中文 (台灣用字)。
規則：
- 只改錯字與簡體字；不要改寫句子、不要換成同義詞、不要補字或刪字、不要加標點
- 不確定的地方保持原樣
- 數字、英文保持原樣
- 一句對一句：不要合併或拆開句子，每個編號都要回傳一次
{terms}
回傳 JSON 陣列，每句一個物件：{{"index": 編號, "text": 修正後的句子}}"""

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {"index": {"type": "INTEGER"}, "text": {"type": "STRING"}},
        "required": ["index", "text"],
        "propertyOrdering": ["index", "text"],
    },
}

client = genai.Client(api_key=API_KEY or None)   # API_KEY 留空 → 讀環境變數 GEMINI_API_KEY


def correct_batch(sentences, context_before):
    """送一批句子給 Gemini → ({編號: 修正後的句子}, usage)。context_before 是前面幾句 (只當上下文，不用修正)。"""
    terms = f"- 這些專有名詞請用這個寫法：{TERMS}\n" if TERMS.strip() else ""
    contents = [PROMPT.format(topic=TOPIC, terms=terms)]
    if context_before:
        contents.append("【前文，只供參考，不用回傳】\n" + "\n".join(context_before))
    contents.append("【要修正的句子】\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1)))

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0,
        ),
    )
    results = {int(item["index"]): " ".join(item["text"].split()) for item in json.loads(response.text)}
    return results, response.usage_metadata


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{"start": r["start"], "end": r["end"], "text": (r.get("text") or "").strip()} for r in csv.DictReader(f)]


def write_rows(path, rows):
    """先寫暫存檔再改名，中斷不會留下只寫一半的檔案。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.csv")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "text", "original"])
        for r in rows:
            writer.writerow([r["start"], r["end"], r["text"], r["original"]])
    temp.replace(path)


def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def parse_parts(answer, root):
    """「3」→ ["3"]；「1,2,5」→ ["1", "2", "5"]；「all」→ root 底下所有 part (依數字排序)。"""
    answer = answer.strip()
    if answer.lower() == "all":
        return sorted((p.name for p in root.iterdir() if p.is_dir()), key=lambda x: (len(x), x))
    return [p.strip() for p in answer.split(",") if p.strip()]


def run_part(part, stats):
    in_dir, out_dir = INPUT_ROOT / part, OUTPUT_ROOT / part
    files = sorted(in_dir.glob("*/*.csv"), key=lambda p: (int(p.parent.name), p.stem))
    files = [p for p in files if not p.stem.endswith(".tmp")]
    todo = [p for p in files if not (out_dir / p.relative_to(in_dir)).exists()]
    print(f"\n===== part {part}：共 {len(files)} 個檔案，{len(files) - len(todo)} 個已經修正過 (跳過)，這次處理 {len(todo)} 個 =====")

    for k, src in enumerate(todo, 1):
        rel = src.relative_to(in_dir)
        rows = read_rows(src)
        for r in rows:
            r["original"] = r["text"]
        idx = [i for i, r in enumerate(rows) if r["text"]]          # 空白的句子不用送
        n_batches = (len(idx) + BATCH_SIZE - 1) // BATCH_SIZE
        ok = True
        for b in range(n_batches):
            batch = idx[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            before = [rows[i]["original"] for i in idx[max(0, b * BATCH_SIZE - 3):b * BATCH_SIZE]]   # 前 3 句當上下文
            results = None
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    results, usage = correct_batch([rows[i]["original"] for i in batch], before)
                    break
                except Exception as e:
                    print(f"\n  {part}/{rel.as_posix()} 第 {b + 1} 批，第 {attempt} 次失敗：{e}")
                    time.sleep(2 ** attempt)
            if results is None:
                ok = False
                break
            stats["in"] += usage.prompt_token_count or 0
            stats["out"] += usage.candidates_token_count or 0
            for j, i in enumerate(batch, start=1):
                new = results.get(j)
                if not new:
                    stats["missing"] += 1                           # Gemini 漏回這句：保留原句
                elif similarity(new, rows[i]["original"]) < MIN_SIMILARITY:
                    stats["rejected"].append((f"{part}/{rel.as_posix()}", rows[i]["original"], new))   # 改太多：保留原句
                else:
                    rows[i]["text"] = new
                    stats["changed"] += new != rows[i]["original"]
            stats["sentences"] += len(batch)
            print(f"\r  [{k}/{len(todo)}] {part}/{rel.as_posix()}：第 {b + 1}/{n_batches} 批", end="", flush=True)
            time.sleep(SLEEP_BETWEEN)

        if ok:
            write_rows(out_dir / rel, rows)                        # 整個檔案都成功才寫出 (失敗的下次重跑)
        else:
            stats["failed"].append(f"{part}/{rel.as_posix()}")
    print()


def main():
    parts = parse_parts(input("part (例如 13、13,14 或 all): "), INPUT_ROOT)
    if not parts:
        print("沒有輸入 part")
        return
    stats = {"in": 0, "out": 0, "sentences": 0, "changed": 0, "missing": 0, "rejected": [], "failed": []}
    for part in parts:
        if not (INPUT_ROOT / part).is_dir():
            print(f"part {part}：找不到 {INPUT_ROOT / part}，跳過")
            continue
        run_part(part, stats)

    print("\n--- 結果 ---")
    print(f"送出 {stats['sentences']} 句，修改了 {stats['changed']} 句")
    if stats["missing"]:
        print(f"Gemini 漏回 {stats['missing']} 句 (保留原句)")
    print(f"輸入 Token: {stats['in']}，輸出 Token: {stats['out']}，總消耗 Token: {stats['in'] + stats['out']}")
    if stats["rejected"]:
        print(f"\n有 {len(stats['rejected'])} 句 Gemini 改太多 (相似度 < {MIN_SIMILARITY})，保留原句：")
        for name, old, new in stats["rejected"][:20]:
            print(f"  {name}：{old} → {new}")
        if len(stats["rejected"]) > 20:
            print(f"  ...還有 {len(stats['rejected']) - 20} 句")
    if stats["failed"]:
        print(f"\n有 {len(stats['failed'])} 個檔案失敗 (沒有寫出)，重新執行程式即可只重跑這些：")
        for name in stats["failed"]:
            print(f"  {name}")


if __name__ == "__main__":
    main()

"""用 Gemini 修正語音轉文字 (01voice.py / Breeze) 的繁體中文錯字。

輸入：context_breeze/<part>/<n>/<start>_<end>.csv   (01voice.py 的輸出，欄位 start, end, text)
輸出：context_breeze_gemini/<part>/<n>/<start>_<end>.csv
      欄位 start, end, text, original：text 是修正後的句子，original 是原本的句子 (方便對照)；
      start、end 不變，句子數不變，所以 Phase5 STT/ 的程式可以直接讀 (把 STT_DIR 改成 context_breeze_gemini)。

一次請求送 N_PER_REQUEST 個 <n> (一支影片 = 一個 <n>，裡面所有片段的句子依時間順序排好一起送)，
Gemini 看得到整支影片的上下文。一次送太多、Gemini 失敗或漏回太多句時，會自動把這次的句子切成兩半重送。

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
N_PER_REQUEST = 1      # 一次請求送幾個 <n> (一個 <n> 大約 450 句、6,500 字)
MAX_RETRY = 3          # 單一請求失敗時最多重試幾次 (都失敗就切成兩半重送)
MAX_MISSING = 0.05     # Gemini 漏回超過這個比例的句子，就當作這次失敗，切成兩半重送
MIN_SPLIT = 20         # 句子少於這個數就不再切半 (失敗的檔案下次執行重做)
SLEEP_BETWEEN = 1.0    # 每次請求之間停幾秒
MIN_SIMILARITY = 0.5   # 修正後跟原句的相似度 (0~1) 低於這個，就當作 Gemini 改太多，保留原句

# 影片的主題、會出現的人名 / 地名 / 專有名詞，寫在這裡可以幫 Gemini 改對 (可以留空)
TOPIC = "台灣的電視新聞與政見辯論會"   # 可以留空 (留空就不告訴 Gemini 主題)
TERMS = ""             # 例如 "盧秀燕、臺中市、公辦電視政見辯論會"

PROMPT = """你會收到語音辨識 (STT) 的逐句結果{topic}。
句子依片段分組，同一支影片的片段依時間順序排列，每句有編號，順序就是說話的順序。
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


# ---------------------------------------------------------------- Gemini
def ask_gemini(items):
    """items = [(片段名稱, 句子), ...] (依順序) → ({編號(從 1 開始): 修正後的句子}, usage)。"""
    terms = f"- 這些專有名詞請用這個寫法：{TERMS}\n" if TERMS.strip() else ""
    lines, current = [], None
    for i, (clip, text) in enumerate(items, start=1):
        if clip != current:                       # 換片段就加一個標題
            lines.append(f"【片段 {clip}】")
            current = clip
        lines.append(f"{i}. {text}")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[PROMPT.format(topic=f"，內容是{TOPIC.strip()}" if TOPIC.strip() else "", terms=terms),
                  "\n".join(lines)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0,
        ),
    )
    results = {int(item["index"]): " ".join(item["text"].split()) for item in json.loads(response.text)}
    return results, response.usage_metadata


def correct(items, stats, label):
    """修正 items，回傳跟 items 一一對應的 list：修正後的句子、None (Gemini 漏回這句) 或 False (請求失敗)。
    失敗或漏回太多時切成兩半各自重送。"""
    results = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            results, usage = ask_gemini(items)
            stats["requests"] += 1
            stats["in"] += usage.prompt_token_count or 0
            stats["out"] += usage.candidates_token_count or 0
            missing = sum(1 for i in range(1, len(items) + 1) if not results.get(i))
            if missing > MAX_MISSING * len(items) and len(items) >= MIN_SPLIT:
                print(f"\n  {label}：{len(items)} 句漏回 {missing} 句，切成兩半重送")
                results = None
            break
        except Exception as e:
            print(f"\n  {label}：{len(items)} 句，第 {attempt} 次失敗：{e}")
            time.sleep(2 ** attempt)
    time.sleep(SLEEP_BETWEEN)
    if results is not None:
        return [results.get(i) for i in range(1, len(items) + 1)]
    if len(items) < MIN_SPLIT:
        return [False] * len(items)
    half = len(items) // 2
    return correct(items[:half], stats, label) + correct(items[half:], stats, label)


# ---------------------------------------------------------------- 檔案
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
    files = sorted((p for p in in_dir.glob("*/*.csv") if not p.stem.endswith(".tmp")),
                   key=lambda p: (int(p.parent.name), [int(x) for x in p.stem.split("_") if x.isdigit()]))
    todo = [p for p in files if not (out_dir / p.relative_to(in_dir)).exists()]
    groups = {}                                   # <n> → 還沒修正的檔案 (依時間順序)
    for p in todo:
        groups.setdefault(p.parent.name, []).append(p)
    ns = sorted(groups, key=int)
    print(f"\n===== part {part}：共 {len(files)} 個檔案，{len(files) - len(todo)} 個已經修正過 (跳過)，"
          f"這次處理 {len(todo)} 個 ({len(ns)} 個 <n>，每次送 {N_PER_REQUEST} 個) =====")

    for k in range(0, len(ns), N_PER_REQUEST):
        chunk = ns[k:k + N_PER_REQUEST]
        label = f"part {part} <n> {'、'.join(chunk)}"
        # 這次要送的所有句子 (依 <n>、片段、句子順序)
        tables, items, where = {}, [], []         # where[j] = (檔案, 第幾列)
        for n in chunk:
            for src in groups[n]:
                rows = read_rows(src)
                for r in rows:
                    r["original"] = r["text"]
                tables[src] = rows
                for i, r in enumerate(rows):
                    if r["text"]:
                        items.append((f"{part}/{n}/{src.stem}", r["text"]))
                        where.append((src, i))
        print(f"[{min(k + N_PER_REQUEST, len(ns))}/{len(ns)}] {label}：{len(tables)} 個檔案、{len(items)} 句", flush=True)

        news = correct(items, stats, label) if items else []
        failed_files = set()
        for (src, i), new in zip(where, news):
            rows = tables[src]
            if new is False:
                failed_files.add(src)                 # 請求失敗：這個檔案不寫出，下次重做
            elif not new:
                stats["missing"] += 1                 # Gemini 漏回這句：保留原句
            elif similarity(new, rows[i]["original"]) < MIN_SIMILARITY:
                stats["rejected"].append((f"{part}/{src.parent.name}/{src.name}", rows[i]["original"], new))
            else:
                rows[i]["text"] = new
                stats["changed"] += new != rows[i]["original"]
        stats["sentences"] += len(items)
        for src, rows in tables.items():
            if src in failed_files:
                stats["failed"].append(f"{part}/{src.parent.name}/{src.name}")
            else:
                write_rows(out_dir / src.relative_to(in_dir), rows)


def main():
    parts = parse_parts(input("part (例如 13、13,14 或 all): "), INPUT_ROOT)
    if not parts:
        print("沒有輸入 part")
        return
    stats = {"requests": 0, "missing": 0, "in": 0, "out": 0, "sentences": 0, "changed": 0, "rejected": [], "failed": []}
    for part in parts:
        if not (INPUT_ROOT / part).is_dir():
            print(f"part {part}：找不到 {INPUT_ROOT / part}，跳過")
            continue
        run_part(part, stats)

    print("\n--- 結果 ---")
    print(f"送出 {stats['sentences']} 句 (請求 {stats['requests']} 次)，修改了 {stats['changed']} 句")
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
        print(f"\n有 {len(stats['failed'])} 個檔案沒有全部修正成功 (沒有寫出)，重新執行程式即可只重跑這些：")
        for name in stats["failed"]:
            print(f"  {name}")


if __name__ == "__main__":
    main()

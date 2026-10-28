import re
import json # 新增這行來讀取 JSON 檔案

def clean_text(text):
    """
    第一步：忽略所有標點符號與空白。
    利用正則表達式，只保留中英文字元與數字。
    """
    if not text:
        return ""
    return re.sub(r'[^\w\u4e00-\u9fff]', '', text)

def compare_by_pinyin(ocr_text, stt_text, threshold=80):
    from rapidfuzz import fuzz
    from pypinyin import lazy_pinyin
    """
    全拼音比對策略：
    先清除標點符號，再將兩者都轉為拼音字串進行相似度比對。
    """
    # 1. 清理標點符號
    ocr_clean = clean_text(ocr_text)
    stt_clean = clean_text(stt_text)

    # 處理極端情況（例如整句只有標點符號被清空後變成空字串）
    if not ocr_clean or not stt_clean:
        return {"is_match": False, "final_text": None, "score": 0}

    # 2. 轉換為拼音字串 (例如："yin wei jin tian...")
    # 使用空白將每個字的拼音隔開，這有助於 fuzz 演算法精準比對
    ocr_pinyin = " ".join(lazy_pinyin(ocr_clean))
    stt_pinyin = " ".join(lazy_pinyin(stt_clean))

    # 3. 計算拼音字串相似度
    score = fuzz.ratio(ocr_pinyin, stt_pinyin)
    is_match = score >= threshold

    return {
        "is_match": is_match,
        # 只要發音吻合度達標，我們就保留原汁原味（含標點）的 OCR 字幕
        "final_text": ocr_text if is_match else None,
        "score": round(score, 2),
        "ocr_pinyin_debug": ocr_pinyin, # 印出拼音方便你們除錯觀察
        "stt_pinyin_debug": stt_pinyin
    }

def process_video_subtitles(ocr_results_per_second, stt_segments, threshold=80):
    """
    ocr_results_per_second: 每一秒的 OCR 結果清單。
    格式範例: [{ "second": 1, "texts": ["TVBS新聞", "因為今天天氣非常晴朗", "LIVE"] }, ...]
    
    stt_segments: STT 辨識出來的句子清單。
    格式範例: ["音為今天天氣非常情郎", ...]
    """
    final_dataset = []

    for stt_text in stt_segments:
        current_match_text = None
        start_time = None
        end_time = None

        # 逐秒檢查 OCR 結果，尋找與這句 STT 吻合的畫面
        for frame in ocr_results_per_second:
            sec = frame["second"]
            frame_texts = frame["texts"] # 這一秒畫面上所有的文字區塊

            # 找出畫面中，與 STT 最吻合的那句文字
            best_match_result = None
            highest_score = 0
            
            for text_block in frame_texts:
                result = compare_by_pinyin(text_block, stt_text, threshold)
                if result["is_match"] and result["score"] > highest_score:
                    highest_score = result["score"]
                    best_match_result = text_block

            # 如果這一秒有找到匹配的字幕
            if best_match_result:
                if start_time is None:
                    # 這是這句話第一次出現
                    start_time = sec
                    current_match_text = best_match_result
                
                # 只要還有匹配到，就持續更新結束時間
                end_time = sec 

        # 當整部影片掃完，如果有找到這句 STT 對應的畫面，就存入資料庫
        if current_match_text and start_time is not None:
            final_dataset.append({
                "ocr_ground_truth": current_match_text,
                "stt_original": stt_text,
                "start_second": start_time,
                "end_second": end_time,
                "duration": end_time - start_time + 1
            })

    return final_dataset


# ================= 實際執行區塊 =================
if __name__ == "__main__":
    
    # 1. 定義檔案路徑 (請替換成專題電腦上實際的檔案位置)
    ocr_file_path = "ocr_result.json"
    stt_file_path = "stt_result.json"
    output_file_path = "final_clean_subtitles.json" # 比對成功後輸出的檔案

    try:
        # 2. 讀取 OCR 與 STT 檔案
        # 使用 utf-8 編碼確保中文字不會變成亂碼
        with open(ocr_file_path, 'r', encoding='utf-8') as f:
            real_ocr_data = json.load(f)
            
        with open(stt_file_path, 'r', encoding='utf-8') as f:
            real_stt_data = json.load(f)
            
        print("✅ 檔案讀取成功，開始進行雙重比對...")

        # 3. 丟入我們寫好的核心處理函數
        # 可以視情況微調 threshold (例如 80 或 85)
        results = process_video_subtitles(real_ocr_data, real_stt_data, threshold=80)
        
        # 4. 將最終純淨的資料輸出成新的 JSON 檔，供後續模型使用
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 讓存出來的檔案直接顯示中文，而不是 Unicode 編碼
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        print(f"🎉 比對完成！總共萃取出 {len(results)} 筆有效字幕。")
        print(f"檔案已儲存至：{output_file_path}")

    except FileNotFoundError as e:
        print(f"❌ 找不到檔案，請確認路徑是否正確：{e}")
    except Exception as e:
        print(f"❌ 發生錯誤：{e}")


# # ================= 測試區塊 =================
# if __name__ == "__main__":
#     # 模擬 1 到 5 秒的 OCR 辨識結果 (包含背景雜訊文字)
#     mock_ocr = [
#         {"second": 1, "texts": ["新聞快報", "接下來為您播報", "LIVE"]},
#         {"second": 2, "texts": ["新聞快報", "接下來為您播報", "LIVE"]},
#         {"second": 3, "texts": ["新聞快報", "國際新聞", "LIVE"]},
#         {"second": 4, "texts": ["新聞快報", "國際新聞", "LIVE"]},
#         {"second": 5, "texts": ["新聞快報", "國際新聞", "LIVE"]},
#     ]
    
#     # STT 辨識結果 (帶有錯字與贅字)
#     mock_stt = ["那這下來為您播報", "過季新聞"]

#     results = process_video_subtitles(mock_ocr, mock_stt, threshold=80)
    
#     print("=== 最終萃取出的純淨資料集 ===")
#     for res in results:
#         print(f"[{res['start_second']}秒 ~ {res['end_second']}秒] 擷取字幕: {res['ocr_ground_truth']}")


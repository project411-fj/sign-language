# 手語新聞關鍵詞辨識

從台灣手語新聞影片中，擷取手語翻譯員的關鍵點 (手、身體、臉)，對齊新聞字幕，訓練一個模型：
**輸入一句手語的關鍵點序列 (`.npy`)，輸出這句話裡出現的關鍵詞**，例如

```
字幕：看完報紙重點 繼續來關心
有把握  ：報紙 (1.00)、繼續 (0.99)、關心 (0.99)、重點 (0.92)
其他可能：立委 (0.14)
```

```text
新聞影片 ─▶ Phase1 篩選節目片段 ─▶ Phase2 語音轉文字 (參考用)
                                ─▶ Phase3 字幕 OCR ───────────────┐
                                ─▶ Phase4 手語員關鍵點 (手/身體/臉) ┤
                                                                  ▼
                                            Phase5 依手語停頓切成一句一句的 .npy
                                                                  ▼
                                   Data Training V2：訓練關鍵詞模型 → keyword_predict.py
```

## 目錄

| 位置 | 內容 |
|---|---|
| `Data Preparation/` | 資料前處理 Phase1～5 與小工具 |
| `Data Training V2/` | **目前使用的訓練與推論**：`data_train.ipynb`、`keyword_predict.py` |
| `Data Training/` | 之前的實驗 notebook (翻譯模型、各種比較實驗)，保留作為紀錄 |

影片 (`*.mp4`)、關鍵點 (`*.npy`)、模型權重 (`*.pt`、`*.task`)、字幕文字 (`*.txt`) 都被 `.gitignore` 排除，只存在本機。

---

## 環境

- Python 3.11、NVIDIA GPU (YOLO、Whisper、模型訓練)
- `ffmpeg` 需要在 PATH 中 (Phase1、Phase4 裁切影片)
- 主要套件：`torch`、`ultralytics`、`mediapipe`、`opencv-python`、`faster-whisper`、`opencc`、`av`、`easyocr`、`google-genai`、`numpy`、`scipy`、`tqdm`、`jieba`、`matplotlib`
- Phase3 的 Gemini OCR 需要環境變數 `GEMINI_API_KEY`

需要的模型檔 (放在本機，不在 repo 中)：

| 檔案 | 用途 |
|---|---|
| `Data Preparation/Phase1-filter/best.pt` | 辨識目標新聞節目的字幕條樣式 |
| `Data Preparation/models/Yolo_models/person.pt`、`yolo26n-pose.pt` | 找出手語員並裁切上半身、身體關鍵點 |
| `Data Preparation/models/mediapipe_models/hand_landmarker.task` | 手部關鍵點 |
| `Data Preparation/models/mediapipe_models/face_landmarker.task` | 臉部關鍵點 ([下載](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)) |

**所有程式都在專案根目錄 (`sign-language/`) 執行**。影片以 part 為單位處理：
大部分程式執行後會詢問 `part:`，Phase4 的程式則是改檔案開頭的 `PART = "..."`。

---

## 資料前處理 (`Data Preparation/`)

影片資料夾：

```text
Data Preparation/videos/
├─ source_videos/<part>/<n>.mp4                          原始新聞影片
├─ part_videos/<part>/<n>/<start>_<end>.mp4              Phase1 切出的節目片段
└─ keypoint_videos/<part>/<n>/<start>_<end>_upper_body.mp4  Phase4 裁出的手語員上半身 (640×640)
```

### Phase1：篩選節目片段 (`Phase1-filter/`)

| 順序 | 程式 | 做什麼 | 輸出 |
|---|---|---|---|
| 1 | `01frames_crop/01crop_text.py` | 裁出畫面下方的字幕條 | `01frames_crop/<part>/` |
| 2 | `02style.py` | 用 `best.pt` 判斷字幕條是不是目標節目的樣式，找出節目區間 | `02intervals/<part>/` |
| 3 | `03clipV2.py` (或 `03clip.py`) | 依區間把原始影片切成片段 | `videos/part_videos/<part>/` |

### Phase2：語音轉文字 (`Phase2-STT/voice.py`)

用 Whisper large-v3 轉出主播語音的文字 (轉成繁體)，輸出 `context/<part>/<n>/<start>_<end>.csv`
(欄位 `start, end, text`)。目前只作為參考，後面的步驟用的是 Phase3 的字幕。

### Phase3：字幕 OCR (`Phase3-OCR/`)

| 順序 | 程式 | 做什麼 | 輸出 (`OCR_output/`) |
|---|---|---|---|
| 1 | `01frames_crop/crop_text.py` | 逐格裁出字幕區域 | `01frames_crop/<part>/` |
| 2 | `02EasOCR/01EasOCR.py` | EasyOCR 辨識每一格 | `01EasOCR_text/` |
| 3 | `02EasOCR/02merge_frames.py` | 相似的連續格合併成一句 | `02EasOCR_merged/` |
| 4 | `02EasOCR/03smooth_merged.py` | 去掉太短的雜訊區間 | (覆蓋 `02EasOCR_merged/`) |
| 5 | `02EasOCR/04pick_frames.py` | 每句挑代表性的畫面 | `03EasOCR_frame/` |
| 6 | `03Gemini_OCR/01gemini_ocr_batch.py` | 用 Gemini 重新辨識，修正 EasyOCR 的錯字 | `04GeminiOCR_context/` |
| 7 | `03Gemini_OCR/02merge_frames.py` | 合併 | `05GeminiOCR_merged/` |
| 8 | `03Gemini_OCR/03merge_frames.py` | 再合併相似的句子 | `06GeminiOCR_merged/` |

最終字幕格式 (`06GeminiOCR_merged/<part>/<n>/<start>_<end>_frames.txt`)：每行 `起始格-結束格: 字幕`。

### Phase4：手語員關鍵點 (`Phase4-Keypoint/`)

| 順序 | 程式 | 做什麼 | 輸出 (`Keypoint_output/`) |
|---|---|---|---|
| 1 | `01cropV2.py` | 找出手語員，裁出上半身並縮成 640×640 (frame 數不變) | `videos/keypoint_videos/` |
| 2 | `02keypoint_hand.py` | MediaPipe 雙手 21 點 (以手腕為原點、依手掌大小縮放) | `01hand/` |
| 2 | `02keypoint_pose.py` | YOLO 身體 9 點：鼻、肩、肘、腕、髖 | `01pose/` |
| 2 | `02keypoint_face.py` | MediaPipe 臉部：52 個表情分數、嘴唇 40 點、頭部轉向 | `01face/` |
| 3 | `03keypoint_combine.py` | 合併手 + 身體 → **156 維** | `02combined/` |
| 4 | `03keypoint_combine_face.py` | 再接上臉部 → **292 維** | `02combined_face/` |

每一格的特徵：

| 維度 | 內容 |
|---|---|
| 0:63 / 63:126 | 左手 / 右手 21 點 (x, y, z) |
| 126 / 127 | 左手 / 右手是否偵測到 |
| 128:155 | 身體 9 點 (x, y, 信心值) |
| 155 | 身體是否偵測到 |
| 156:208 | 52 個表情分數 (眉毛、眼睛、臉頰、下巴、嘴巴) |
| 208:288 | 嘴唇 40 點 (x, y) |
| 288:291 | 頭部轉向 (點頭、轉頭、歪頭) |
| 291 | 臉是否偵測到 |

### Phase5：依手語停頓切句 (`Phase5-Pause/`)

字幕出現的時間通常比手語早一點，所以不直接用字幕的時間切，而是在字幕邊界附近找手語員動作停下來的地方。

| 順序 | 程式 | 做什麼 | 輸出 (`output/`) |
|---|---|---|---|
| 1 | `01get_fps.py` | 讀每個片段的 fps | `01fps/fps_<part>.csv` |
| 2 | `02find_sign_pauses.py` | 依動作能量找停頓點，調整每句的起訖格 | `02pause/<part>/`、`02check_plots/` |
| 3 | `03split_npy.py` | 依切點把關鍵點切成一句一個 `.npy` | `03segments<TAG>/`、`03labels<TAG>/` |

`03split_npy.py` 的設定：`FEATURE_TAG = "_face"` 讀 `02combined_face` (292 維)，輸出到 `03segments_face/`；
`""` 則是 156 維的 `03segments/`。**目前的訓練使用 `03segments_face`**。

### 小工具 (`tools/`)

| 程式 | 用途 |
|---|---|
| `face_pipeline_check.py` | 列出每個 part 在臉部流程各步驟的檔案數，告訴你下一步要跑什麼 |
| `face_check.py` | 檢查臉偵測率、嘴巴與各部位的動作量 |
| `npy_check.py` | 檢查手、身體的缺漏比例 |
| `calculation.py`、`frame_count.py`、`video_sort.py` | 影片總長度、OCR 行數統計、影片重新命名 |

---

## 訓練與推論 (`Data Training V2/`)

### `data_train.ipynb`：從資料切分到展示設定，一次完成

從上到下執行一次即可 (約 60～80 分鐘)，所有輸出放在 `keyword_model/`：

| 步驟 | 做什麼 |
|---|---|
| 0. 準備資料 | 以影片為單位切成訓練 / 驗證 / 測試 (8:1:1)；檢查有沒有還沒用到的新影片 |
| 1. 選詞 | 用 jieba (繁體詞典 `dict.txt.big`，自動下載) 斷詞，訓練集出現 ≥ `MIN_WORD_COUNT` 次的詞都列為候選 |
| 2. 第一階段訓練 | 用所有候選詞訓練一次，判斷哪些詞學得起來 |
| 3. 篩選詞彙 | 保留**驗證集** AP ≥ `SELECT_MIN_VAL_AP` 的詞 |
| 4. 第二階段訓練 | 用篩選後的詞 + CTC 輔助損失訓練最終模型；最終分數 = 兩個階段模型的平均 |
| 5. 評估與挑門檻 | 在驗證集挑門檻，讓「有把握」的詞有 `TARGET_PRECISION` 是對的；報告測試集效果，存成 `demo_config.json` |
| 6. 示範 | 確認 `keyword_predict.py` 跟訓練時一致，並實際示範 |

模型：Transformer encoder (4 層) 讀整句的關鍵點序列，對每個關鍵詞輸出「有出現」的機率。
缺漏的手、身體點會被遮罩成 0，訓練時有旋轉、縮放、平移、變速、時間遮罩等資料增強。
臉部特徵 (`USE_FACE`) 在實驗中沒有幫助，預設不使用。

中途中斷的話重新執行即可，已完成的階段會自動略過。

### 常用設定 (notebook 最上面)

| 設定 | 預設 | 說明 |
|---|---|---|
| `REBUILD` | `False` | **加入新影片時才改成 `True`**，見下一節 |
| `MIN_WORD_COUNT` | 30 | 候選詞的最少出現次數；資料越多，候選詞自動越多 |
| `SELECT_MIN_VAL_AP` | 0.1 | 驗證集 AP 低於這個值的詞不保留 |
| `TARGET_PRECISION` | 0.7 | 「有把握」的詞在驗證集要達到的準確率 (測試集通常低幾個百分點) |
| `SHOW_WORDS` | 5 | 每句顯示幾個詞 (有把握的不夠時，用「其他可能」補滿) |
| `ENSEMBLE_STAGE1` | `True` | 最終分數是否平均兩個階段的模型 |

只改 `TARGET_PRECISION`、`MAX_WORDS`、`SHOW_WORDS` 的話，從第 5 步重新執行即可，不用重新訓練。

### 加入新影片

1. 新影片跑完 Phase1～5 與臉部流程，執行 `python "Data Preparation/tools/face_pipeline_check.py"`，確認每個 part 都顯示「完成」
2. `data_train.ipynb` 設 `REBUILD = True`，從頭執行：舊的資料切分與模型會搬到 `backup_<時間>/`，再用全部影片重新切分、訓練
3. **跑完改回 `REBUILD = False`**，否則每次執行都會重新訓練

不確定有沒有新影片時，用 `False` 執行即可，第 0 步會提醒有幾支影片還沒被用到。

### `keyword_predict.py`：對任何一句 `.npy` 做預測

需要 `data_train.ipynb` 產生的 `keyword_model/demo_config.json`、`keyword_model/stage1/best.pt`、`keyword_model/stage2/best.pt`
與 `data/norm_masked_face.npz` (模型權重不在 repo 中，要在訓練的電腦上使用)。

命令列 (在 `Data Training V2/` 執行)：

```bash
python keyword_predict.py "..\Data Preparation\Phase5-Pause\output\03segments_face\2\10\000641_001738_s00.npy"
```

```text
門檻 0.69，每句顯示 5 個詞
...\000641_001738_s00.npy
  有把握：天氣 (0.83)、陣雨 (0.71)
  其他可能：氣象 (0.32)、北部 (0.28)、局部 (0.21)
```

程式中：

```python
from keyword_predict import KeywordPredictor

predictor = KeywordPredictor()
predictor.predict("某句.npy")                    # 只回傳有把握的詞：[("天氣", 0.83), ("陣雨", 0.71)]
predictor.predict_with_candidates("某句.npy")    # (有把握的詞, 其他可能的詞)
```

輸入可以是 292 維 (`03segments_face`) 或 156 維 (`03segments`) 的 `.npy`。

---

## 目前效果

最近一次訓練 (訓練集 25,276 句、驗證集 3,036 句、測試集 2,775 句；保留 285 個詞)，測試集：

| 指標 | 數值 |
|---|---|
| 「有把握」的詞的準確率 | **67.4%** |
| 有輸出「有把握」詞的句子 | 24.0% (其他句子回答「無法判斷」) |
| 有回答的句子中，至少答對 1 個詞 | 65.6% |
| 訓練時沒出現過的句子，「有把握」的準確率 | 43.2% |
| 「其他可能」的準確率 | 5.7% (只當參考) |

使用上要知道的限制：

- **模型只能說出詞彙清單裡的詞**。字幕平均每句 4.7 個詞，其中在清單裡的平均只有 1.1 個，這是每句能答對的上限
- 模型最可靠的是**固定說法**：天氣預報 (陣雨、短暫、局部…)、讀報單元 (報紙、聯合報…)、節目轉場 (接著、繼續、關心…)；全新內容的句子準確率明顯較低
- **增加資料是最有效的改善方式**：資料從 20,191 句增加到 25,276 句時，詞彙從 256 個增加到 285 個，準確率與能回答的句子比例都同時提升

## 實驗紀錄 (`Data Training/`)

在改成關鍵詞辨識之前做過的實驗，結論如下：

| notebook | 實驗 | 結論 |
|---|---|---|
| `sign_transformer*.ipynb` | 直接把手語翻譯成整句中文 | 新句的字元錯誤率一直在 0.99～1.03，資料量不足以做逐字翻譯 |
| `sign_transformer_copy.ipynb` (寬鬆切法) | 每句前後多切 0.5 秒 | 跟原本沒有差異，切點不是瓶頸 |
| `sign_transformer_ctc*.ipynb` | CTC 輔助損失、缺漏遮罩、資料增強 | CTC 有小幅幫助，其他沒有明顯差異 |
| `sign_transformer_learning_curve.ipynb` | 用 25% / 50% / 75% / 100% 的資料訓練 | 資料越多越好，曲線還沒變平 |
| `sign_transformer_face.ipynb` | 加入臉部特徵 (表情、嘴型) | 翻譯與關鍵詞辨識都沒有明顯幫助 |
| `sign_keyword*.ipynb` | 改成關鍵詞辨識 | 第一次明確學到句子層級的內容，之後整合成 `Data Training V2/data_train.ipynb` |

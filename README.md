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
| `Data Training V1/` | 整句翻譯模型 (`sign_transformer_fixed.ipynb`)，有自己的資料切分與預測結果 |
| `Data Training/` | 之前的實驗 notebook (翻譯模型、各種比較實驗)，保留作為紀錄 |

GitHub 上的工作分支是 **`main`**。

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
| `Data Preparation/Phase1-filter/best~12.pt` | 辨識目標新聞節目的字幕條樣式 (`02style.py` 讀這個檔名；可以用 `02style_train/` 重新訓練) |
| `Data Preparation/Phase1-filter/style_model/best<part>-<n>.pt` | `02style_13~.py` 每支影片各自的字幕條樣式模型，例如 `source_videos/13/3.mp4` 用 `style_model/best13-3.pt` |
| `Data Preparation/models/Yolo_models/person.pt`、`yolo26n-pose.pt` | 找出手語員並裁切上半身、身體關鍵點 |
| `Data Preparation/models/mediapipe_models/hand_landmarker.task` | 手部關鍵點 |
| `Data Preparation/models/mediapipe_models/face_landmarker.task` | 臉部關鍵點 ([下載](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)) |

**所有程式都在專案根目錄 (`sign-language/`) 執行**。影片以 part 為單位處理：
大部分程式執行後會詢問 `part:`；`Phase2-STT/` 的程式、Phase4 全部的程式 (含 `OCR/`、`STT/01cropV2.py`)、`Phase5-Pause/OCR/03split_npy.py` 與 `Phase5-Pause/STT/` 全部的程式可以輸入 `3`、`1,2,5` 或 `all` (已經處理過的影片會跳過)。
檔名有 `_13~` 的程式 (part 13 之後使用) 則是在檔案開頭直接列出要處理的影片。

---

## 資料前處理 (`Data Preparation/`)

影片資料夾：

```text
Data Preparation/videos/
├─ source_videos/<part>/<n>.mp4                          原始新聞影片
├─ part_videos/<part>/<n>/<start>_<end>.mp4              Phase1 切出的節目片段 (OCR 路徑，part 1~12)
├─ part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/<start>_<end>.mp4
│                                                        Phase1 切出的節目片段 (STT 路徑，part 13 之後；資料夾名稱是手語員的範圍)
└─ keypoint_videos/<part>/<n>/<start>_<end>_upper_body.mp4  Phase4 裁出的手語員上半身 (640×640，兩條路徑共用)
```

### Phase1：篩選節目片段 (`Phase1-filter/`)

| 順序 | 程式 | 做什麼 | 輸出 |
|---|---|---|---|
| 1 | `01frames_crop/01crop_text.py` | 逐格裁出畫面下方的字幕條 (固定範圍：高 80.5%~88%、寬 27%~100%) | `01frames_crop/<part>/<n>_frames/` |
| 2 | `02style.py` | 用分類模型判斷每一格字幕條是不是目標節目的樣式，把連續的目標畫面合併成區間 | `02intervals/<part>/<n>_frames.txt` |
| 3 | `03clipV2.py` (或 `03clip.py`) | 依區間把原始影片切成片段 | `videos/part_videos/<part>/` |

**part 13 之後的影片** (STT 路徑，字幕條位置不固定) 改用 `_13~` 版本：

| 程式 | 跟原版的差別 |
|---|---|
| `01frames_crop/01crop_text_13~.py` | 在 `VIDEOS` 直接列出影片，每支各自設定裁切範圍 `(上, 下, 左, 右)` (佔畫面的比例，`None` = 預設範圍)。會清掉舊圖片，並檢查輸出張數 = 影片總格數 |
| `01frames_crop/01crop_text_13~.py` (續) | 同時建立 `part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/` (例如 `13/1/0.20_0.73_0.10_0.33/`)。範圍改過時，舊的空資料夾會刪掉；舊資料夾裡有片段的話只會提醒，請自己確認後刪除 |
| `02style_13~.py` | 在 `VIDEOS` 直接列出影片 (通常跟上面相同)；每支影片用 `style_model/` 裡自己的模型 `best<part>-<n>.pt` (例如 `14/5.mp4` → `style_model/best14-5.pt`)。還沒裁切或找不到模型的影片會列出並略過 |
| `02signer_track.py` (可取代上面兩步) | 不用裁字幕條、不用訓練樣式模型。執行後輸入 part (`13` 或 `13,14`，處理 `source_videos/<part>/` 底下所有 .mp4；直接按 Enter 只處理 `VIDEOS` 列出的影片)。**手語員的位置可以中途改變**：① 每 1 秒取一格整個畫面，用 YOLO Pose (`yolo26n-pose.pt`) 找看得到手臂的人，依位置分群，累積出現 ≥ `MIN_PRESENCE_SEC` 秒、而且曾經連續出現 ≥ `MIN_RUN_SEC` 秒的位置才算手語員的位置 (最多 `MAX_POSITIONS` 個；新聞畫面裡的人鏡頭一換就變，不會被算進去)，每個位置用不同顏色畫在 `02intervals/<part>/<n>_box.jpg`；② 粗掃 + 細掃算出每一格每個位置有沒有人；③ 整支影片一起決定每一格手語員在哪個位置 (換位置要付 `SWITCH_SEC` 的代價，新聞畫面裡剛好有人站在別的位置幾秒鐘不會被當成換位置)。輸出 `02intervals/<part>/<n>_frames.txt` 每行「起始格 結束格 範圍」，並建好每個位置的範圍資料夾。自動找錯時可以在 `VIDEOS` 手動指定 `(上, 下, 左, 右)` 或多個位置的 list。`GPU = True` 時影片用 ffmpeg 在顯卡上解碼、縮小，YOLO 也在顯卡上。`02intervals` 已經有結果時預設跳過 (`OVERWRITE = True` 才覆蓋) |
| `03clipV2_13~.py` | 依 `02intervals` 剪片段到 `part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/`：有第三欄範圍 (`02signer_track.py` 的輸出) 就剪到該範圍的資料夾 (一支影片可以有好幾個)；只有兩欄 (`02style_13~.py` 的輸出) 就剪到上面建好的唯一範圍資料夾，開始前會先檢查 |

`VIDEOS` 的範圍是**先高再寬**，例如寬 10%~33%、高 20%~73% 寫成 `(0.20, 0.73, 0.10, 0.33)`；100% 要寫 `1.00` (不是 `0.100`)。
各影片的範圍記錄在根目錄的 `note.md`。

**訓練字幕條樣式模型** (`02style_train/`，`02-1style_train/` 是同樣的另一份，可以訓練另一個版本)：

1. 目標節目的字幕條圖片放進 `target/`，其他圖片放進 `other/` (可以直接放 `01frames_crop` 的圖片，建議依影片分子資料夾)
2. 執行 `python "Data Preparation/Phase1-filter/02style_train/train_style.py"`
3. 模型存成 `02style_train/best.pt`；要讓 `02style.py` 使用，把它複製到 `Phase1-filter/` 並修改 `WEIGHTS`

`train_style.py` 的前處理跟 `02style.py` 相同 (取字幕條左邊 3 倍高度的寬度，縮成 320×320)，只使用自己所在的資料夾，
整個資料夾複製到其他地方也能執行。

### Phase2：語音轉文字 (`Phase2-STT/`)

| 程式 | 模型 | 輸出 |
|---|---|---|
| `voice.py` | Whisper large-v3 (faster-whisper，轉成繁體) | `context/<part>/<n>/<start>_<end>.csv` |
| `01voice.py` | [Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) (針對台灣華語、中英夾雜微調；transformers) | `context_breeze/<part>/<n>/<start>_<end>.csv` |

兩支程式的 part 都可輸入 `all`，已經轉過的片段會跳過；輸出欄位都是 `start, end, text`。
兩種 `part_videos` 結構都讀得到 (STT 路徑多一層範圍資料夾)，輸出都是 `<part>/<n>/<start>_<end>.csv`。目前只作為參考，後面的步驟用的是 Phase3 的字幕。

Whisper 系列模型給的句子時間是首尾相接的 (上一句的 end = 下一句的 start，停頓被算進句子裡)。`01voice.py` 另外用
[Silero VAD](https://github.com/snakers4/silero-vad) (`pip install silero-vad`) 找出實際有人聲的區間，把每句的 start / end
對齊到最近的停頓邊緣，句子之間就會留下真正的停頓；兩句連著講、中間沒有停頓時維持相接。設定在檔案開頭的 `VAD_*`、`SNAP_TOLERANCE`。

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
| 1 | `OCR/01cropV2.py` | OCR 路徑：用 YOLO 找出手語員，裁出上半身並縮成 640×640 (frame 數不變)；已裁切過的影片會跳過 | `videos/keypoint_videos/` |
| 1 | `STT/01cropV2.py` | STT 路徑：讀 `part_videos/<part>/<n>/<TOP>_<BOTTOM>_<LEFT>_<RIGHT>/`，照資料夾名稱的範圍裁切 (不用模型；手語員中途換位置的影片有好幾個範圍資料夾，每個片段用自己資料夾的範圍，不同資料夾的片段時間重疊時這支影片會跳過並提醒)，等比例縮放後補黑邊成 640×640 (frame 數不變)。預設全程用 NVIDIA 顯卡 (cuvid 解碼時裁切、`scale_cuda` 縮放、NVENC 編碼，補黑邊在 CPU)，沒有顯卡就把 `GPU` 改成 `False` | `videos/keypoint_videos/` |
| 2 | `02keypoint_hand.py` | MediaPipe 雙手 21 點 (以手腕為原點、依手掌大小縮放) | `01hand/` |
| 2 | `02keypoint_pose.py` | YOLO 身體 9 點：鼻、肩、肘、腕、髖 | `01pose/` |
| 2 | `02keypoint_face.py` | MediaPipe 臉部：52 個表情分數、嘴唇 40 點、頭部轉向 | `01face/` |
| 3 | `03keypoint_combine.py` | 合併手 + 身體 + 臉 → **292 維** | `02combined/` |

只有第 1 步依路徑分成 `OCR/`、`STT/` 兩版；兩版都輸出到同一個 `keypoint_videos`，第 2、3 步兩條路徑共用。

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

字幕、語音出現的時間通常比手語早一點，所以不直接用它們的時間切，而是在句子邊界附近找手語員動作停下來的地方。
依切句錨點的來源分成兩版，演算法相同。兩版都輸出到同一個 `output/` (`Data Training V2` 只讀 `output/03labels`、`output/03segments`)；一個 part 只會走 OCR 或 STT 其中一條，所以不會互相蓋掉。STT 版只處理 STT 路徑的 part (`part_videos` 裡有範圍資料夾的 part)，輸入 `all` 也不會動到 OCR 的 part：

| 版本 | 錨點與標籤來源 | 程式 | 輸出 |
|---|---|---|---|
| `OCR/` | Phase3 的字幕 (`06GeminiOCR_merged`) | `01get_fps.py` → `02find_sign_pauses.py` → `03split_npy.py` | `output/` |
| `STT/` | Phase2 的語音轉文字 (`context_breeze`，`01voice.py` 的輸出) | `01get_fps.py` → `02find_sign_pauses.py` → `03split_npy.py` | `output/` |

OCR 版：

| 順序 | 程式 | 做什麼 | 輸出 (`output/`) |
|---|---|---|---|
| 1 | `OCR/01get_fps.py` | 讀每個片段的 fps | `01fps/fps_<part>.csv` |
| 2 | `OCR/02find_sign_pauses.py` | 依動作能量找停頓點，調整每句的起訖格 | `02pause/<part>/`、`02check_plots/` |
| 3 | `OCR/03split_npy.py` | 依切點把關鍵點 (`02combined`，292 維) 切成一句一個 `.npy` | `03segments/`、`03labels/` |

STT 版 (三支都可輸入 `3`、`1,2,5` 或 `all`)：

| 順序 | 程式 | 做什麼 | 輸出 (`output/`) |
|---|---|---|---|
| 1 | `STT/01get_fps.py` | 讀每個片段的 fps (和 OCR 版的輸出一樣) | `01fps/fps_<part>.csv` |
| 2 | `STT/02find_sign_pauses.py` | 同 OCR 版，錨點改用語音句子的時間：直接讀 `context_breeze` 的 `.csv` (秒)，用 `01fps` 換成格數；沒有語音句子的片段會跳過 | `02pause/<part>/`、`02check_plots/` |
| 3 | `STT/03split_npy.py` | 同 OCR 版，標籤是語音轉出來的文字 | `03segments/`、`03labels/` |

`03segments/` 裡每個 `.npy` 都是 292 維；`03labels/` 是對應的文字 (`分段檔名: 文字`)。
兩版都需要先用 Phase4 的 `03keypoint_combine.py` 產生 292 維的 `02combined`；STT 版另外需要 Phase2 的 `01voice.py`。
STT 版沿用 OCR 版的停頓搜尋參數 (`SEARCH_BEFORE`、`SEARCH_AFTER`、`EXPECTED_LAG`)，可以看 `02check_plots` 再調整。

### 小工具 (`tools/`)

| 程式 | 用途 |
|---|---|
| `face_pipeline_check.py` | 列出每個 part 在關鍵點與切句各步驟的檔案數、`02combined` 與 `03segments` 的維度，告訴你下一步要跑什麼 |
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
python keyword_predict.py "..\Data Preparation\Phase5-Pause\output\03segments\2\10\000641_001738_s00.npy"
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

輸入是 `03segments` 裡的 `.npy` (292 維)；只有手與身體的 156 維 `.npy` 也可以。

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
| `Data Training V1/sign_transformer_fixed.ipynb` | 整句翻譯模型的修正版 | 測試集 CER 1.251，仍無法做逐字翻譯 |

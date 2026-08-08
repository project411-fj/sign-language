# Video Subtitle Processing Pipeline

This project processes every MP4 video in the `target` directory through three sequential phases:

```text
Source MP4
    |
    v
Phase 1: Frame filtering
    |
    v
Phase 2: OCR and speech-to-text
    |
    v
Phase 3: JSON conversion and OCR/STT matching
    |
    v
Final subtitle JSON
```

## Project structure

```text
Formal/
|-- main.py
|-- target/                       # Source MP4 files
|-- cropvideos/                   # Phase 1 video output
|-- results/                      # Phase 2 and Phase 3 output
|-- Phase_1/
|   |-- cropping.py
|   |-- person.pt
|   `-- ties.pt
|-- Phase_2/
|   |-- OCR_full.py
|   `-- voice.py
`-- Phase_3/
    |-- ocr_txt_to_json.py
    |-- tts_txt_to_json.py
    `-- ocr_compare_stt.py
```

## Phase 1: Video-frame filtering

Program: `Phase_1/cropping.py`

Phase 1 reads the source MP4 frame by frame and applies two YOLO models:

- `person.pt` detects the required person class.
- `ties.pt` detects ties.
- A frame is written only when both detections are present.

Although the function is named `crop`, it filters complete frames; it does not crop a rectangular area from a frame.

### Input

```text
target/<video-name>.mp4
```

### Output

```text
cropvideos/<video-name>_output.avi
```

The AVI contains the selected video frames but does not contain the original audio track. For that reason, the Phase 1 result passed to Phase 2 contains both paths:

```python
{
    "source_video": Path("target/example.mp4"),
    "filtered_video": Path("cropvideos/example_output.avi")
}
```

## Phase 2: OCR and speech recognition

Programs:

- `Phase_2/OCR_full.py`
- `Phase_2/voice.py`

Phase 2 consumes the result returned by Phase 1 and performs two operations.

### OCR

OCR uses the filtered AVI from Phase 1. EasyOCR recognizes Traditional Chinese text from sampled frames and groups similar consecutive text into time ranges.

OCR text output:

```text
results/<video-name>/<video-name>_ocr.txt
```

OCR line format:

```text
<start>s --- <end>s ---文字: ['text 1', 'text 2']
```

Example:

```text
2.00s --- 5.20s ---文字: ['今日新聞', '即時報導']
```

### Speech-to-text

Speech recognition uses Faster Whisper. Because the filtered AVI has no audio, STT uses the original source MP4 reference carried in the Phase 1 result.

STT text output:

```text
results/<video-name>/<video-name>_stt.txt
```

STT line format:

```text
[<start>s -> <end>s] recognized text
```

Example:

```text
[2.10s -> 5.00s] 今日為您帶來最新新聞
```

### Phase 2 result passed to Phase 3

```python
{
    "source_video": Path(...),
    "filtered_video": Path(...),
    "ocr_text": Path("..._ocr.txt"),
    "stt_text": Path("..._stt.txt")
}
```

## Phase 3: Conversion and subtitle matching

Programs:

- `Phase_3/ocr_txt_to_json.py`
- `Phase_3/tts_txt_to_json.py`
- `Phase_3/ocr_compare_stt.py`

Phase 3 consumes the OCR and STT files returned by Phase 2.

### OCR JSON conversion

`ocr_txt_to_json.py` expands each OCR time range into one record per covered second.

Output:

```text
results/<video-name>/<video-name>_ocr.json
```

Format:

```json
[
    {
        "second": 2,
        "texts": ["今日新聞", "即時報導"]
    },
    {
        "second": 3,
        "texts": ["今日新聞", "即時報導"]
    }
]
```

### STT JSON conversion

`tts_txt_to_json.py` extracts the text from each Whisper timestamp line. Despite its filename, this program processes STT rather than text-to-speech.

Output:

```text
results/<video-name>/<video-name>_stt.json
```

Format:

```json
[
    "今日為您帶來最新新聞",
    "接下來是氣象消息"
]
```

### OCR/STT matching

`ocr_compare_stt.py` performs the final comparison:

1. Removes punctuation and whitespace.
2. Converts OCR and STT text to Pinyin.
3. Calculates their similarity with RapidFuzz.
4. Accepts matches whose score reaches the configured threshold.
5. Records the first and last matching seconds.

Final output:

```text
results/<video-name>/<video-name>_final_subtitles.json
```

Format:

```json
[
    {
        "ocr_ground_truth": "今日新聞",
        "stt_original": "今日為您帶來最新新聞",
        "start_second": 2,
        "end_second": 5,
        "duration": 4
    }
]
```

## How `main.py` connects the phases

For each source video, `main.py` uses this explicit data flow:

```python
phase_1_result = run_phase_1(source_video)
phase_2_result = run_phase_2(phase_1_result, output_dir, progress)
final_json = run_phase_3(phase_2_result, output_dir, threshold, progress)
```

The controller functions have these responsibilities:

- `find_videos()` finds and naturally sorts the MP4 files in `target`.
- `run_phase_1()` packages the original and filtered video paths.
- `run_phase_2()` receives the Phase 1 package and adds OCR/STT outputs.
- `run_phase_3()` receives the Phase 2 package and creates the JSON files.
- `run_pipeline()` runs all three phases for one video and displays phase progress.
- `run_all()` processes every video sequentially and displays overall progress.
- `main()` reads command-line options and prints each final JSON path.

Each video receives an independent result directory, so files from different videos do not overwrite one another.

## Running the pipeline

Place MP4 files in:

```text
E:\Formal\target
```

Open a terminal in `E:\Formal` and run:

```powershell
python main.py
```

Optional arguments:

```powershell
python main.py --target-dir target --output-dir results --threshold 80
```

- `--target-dir`: source-video directory; default is `target`.
- `--output-dir`: result directory; default is `results`.
- `--threshold`: minimum OCR/STT similarity; default is `80`.

---

# 影片字幕處理流程

本專案會將 `target` 目錄中的每一支 MP4 影片依序送入三個 Phase：

```text
來源 MP4
    |
    v
Phase 1：影片影格篩選
    |
    v
Phase 2：OCR 與語音辨識
    |
    v
Phase 3：JSON 轉換與 OCR/STT 比對
    |
    v
最終字幕 JSON
```

## 專案結構

```text
Formal/
|-- main.py
|-- target/                       # 來源 MP4
|-- cropvideos/                   # Phase 1 影片輸出
|-- results/                      # Phase 2、3 輸出
|-- Phase_1/
|   |-- cropping.py
|   |-- person.pt
|   `-- ties.pt
|-- Phase_2/
|   |-- OCR_full.py
|   `-- voice.py
`-- Phase_3/
    |-- ocr_txt_to_json.py
    |-- tts_txt_to_json.py
    `-- ocr_compare_stt.py
```

## Phase 1：影片影格篩選

程式：`Phase_1/cropping.py`

Phase 1 逐格讀取來源 MP4，並套用兩個 YOLO 模型：

- `person.pt` 偵測指定人物類別。
- `ties.pt` 偵測領帶。
- 只有兩種偵測結果同時存在時，才會寫入該影格。

雖然函式名稱是 `crop`，目前的行為是篩選完整影格，不是裁切畫面中的矩形區域。

### 輸入

```text
target/<影片名稱>.mp4
```

### 輸出

```text
cropvideos/<影片名稱>_output.avi
```

AVI 只包含篩選後的影片影格，不包含原始音軌。因此，傳給 Phase 2 的 Phase 1 結果會同時包含兩個路徑：

```python
{
    "source_video": Path("target/example.mp4"),
    "filtered_video": Path("cropvideos/example_output.avi")
}
```

## Phase 2：OCR 與語音辨識

程式：

- `Phase_2/OCR_full.py`
- `Phase_2/voice.py`

Phase 2 接收 Phase 1 回傳的結果，並執行兩項工作。

### OCR

OCR 使用 Phase 1 篩選後的 AVI。EasyOCR 會從取樣影格辨識繁體中文，並將連續且相似的文字整理成時間範圍。

OCR 輸出：

```text
results/<影片名稱>/<影片名稱>_ocr.txt
```

每行格式：

```text
<開始時間>s --- <結束時間>s ---文字: ['文字 1', '文字 2']
```

範例：

```text
2.00s --- 5.20s ---文字: ['今日新聞', '即時報導']
```

### 語音轉文字

語音辨識使用 Faster Whisper。由於篩選後的 AVI 沒有音軌，因此 STT 會使用 Phase 1 結果中一併傳遞的原始 MP4。

STT 輸出：

```text
results/<影片名稱>/<影片名稱>_stt.txt
```

每行格式：

```text
[<開始時間>s -> <結束時間>s] 辨識文字
```

範例：

```text
[2.10s -> 5.00s] 今日為您帶來最新新聞
```

### 傳給 Phase 3 的 Phase 2 結果

```python
{
    "source_video": Path(...),
    "filtered_video": Path(...),
    "ocr_text": Path("..._ocr.txt"),
    "stt_text": Path("..._stt.txt")
}
```

## Phase 3：資料轉換與字幕比對

程式：

- `Phase_3/ocr_txt_to_json.py`
- `Phase_3/tts_txt_to_json.py`
- `Phase_3/ocr_compare_stt.py`

Phase 3 接收 Phase 2 回傳的 OCR 和 STT 檔案。

### OCR JSON 轉換

`ocr_txt_to_json.py` 會將每個 OCR 時間範圍展開成涵蓋期間內每秒一筆的資料。

輸出：

```text
results/<影片名稱>/<影片名稱>_ocr.json
```

格式：

```json
[
    {
        "second": 2,
        "texts": ["今日新聞", "即時報導"]
    },
    {
        "second": 3,
        "texts": ["今日新聞", "即時報導"]
    }
]
```

### STT JSON 轉換

`tts_txt_to_json.py` 會擷取每個 Whisper 時間戳後面的文字。雖然檔名寫成 `tts`，但這支程式實際處理的是 STT，而不是文字轉語音。

輸出：

```text
results/<影片名稱>/<影片名稱>_stt.json
```

格式：

```json
[
    "今日為您帶來最新新聞",
    "接下來是氣象消息"
]
```

### OCR/STT 比對

`ocr_compare_stt.py` 負責最終比對：

1. 移除標點與空白。
2. 將 OCR 和 STT 文字轉成拼音。
3. 使用 RapidFuzz 計算相似度。
4. 接受分數達到設定門檻的結果。
5. 記錄第一次及最後一次匹配的秒數。

最終輸出：

```text
results/<影片名稱>/<影片名稱>_final_subtitles.json
```

格式：

```json
[
    {
        "ocr_ground_truth": "今日新聞",
        "stt_original": "今日為您帶來最新新聞",
        "start_second": 2,
        "end_second": 5,
        "duration": 4
    }
]
```

## `main.py` 如何串接三個 Phase

對每一支來源影片，`main.py` 使用以下明確資料流：

```python
phase_1_result = run_phase_1(source_video)
phase_2_result = run_phase_2(phase_1_result, output_dir, progress)
final_json = run_phase_3(phase_2_result, output_dir, threshold, progress)
```

各控制函式的責任：

- `find_videos()`：尋找 `target` 內的 MP4，並以自然檔名排序。
- `run_phase_1()`：包裝原始影片和篩選後影片路徑。
- `run_phase_2()`：接收 Phase 1 結果，並加入 OCR/STT 輸出。
- `run_phase_3()`：接收 Phase 2 結果，並產生所有 JSON。
- `run_pipeline()`：替單一影片執行三個 Phase，並顯示各 Phase 進度。
- `run_all()`：依序處理所有影片，並顯示影片總進度。
- `main()`：讀取命令列參數，並印出每支影片的最終 JSON 路徑。

每支影片都有獨立的結果目錄，因此不同影片的輸出不會互相覆蓋。

## 執行方式

將 MP4 放入：

```text
E:\Formal\target
```

在 `E:\Formal` 開啟終端並執行：

```powershell
python main.py
```

可選參數：

```powershell
python main.py --target-dir target --output-dir results --threshold 80
```

- `--target-dir`：來源影片目錄，預設為 `target`。
- `--output-dir`：結果目錄，預設為 `results`。
- `--threshold`：OCR/STT 最低相似度，預設為 `80`。

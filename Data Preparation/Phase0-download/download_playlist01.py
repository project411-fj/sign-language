import os
import time
import yt_dlp

# ===== 設定 =====
#PLAYLIST_URL = "https://youtube.com/playlist?list=PLHybHfSlbzTVGAfLGXA29jHcVql9PcRso"
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLGjeUzy1QRDCF5sL1V9drqDHL-quAOKmq"
SAVE_FOLDER = r"G:\videos01\20"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BATCH_SIZE = 50          # 每批最多下載幾部
BATCH_REST = 30 * 60     # 每批之間休息秒數（30 分鐘）
RATE_LIMIT = 8 * 1024 * 1024  # 下載頻寬上限 5 MB/s
# =================

os.makedirs(SAVE_FOLDER, exist_ok=True)

ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': os.path.join(SAVE_FOLDER, '%(playlist_title)s', '%(title)s [%(id)s].%(ext)s'),
    'merge_output_format': 'mp4',
    'download_archive': os.path.join(SCRIPT_DIR, 'downloaded.txt'),
    'js_runtimes': {'node': {}},
    'ignoreerrors': True,

    # --- 降低請求頻率（對應 yt-dlp 官方 `-t sleep` 預設值）---
    'sleep_interval': 10,             # 每部影片下載前隨機等待 10~20 秒
    'max_sleep_interval': 20,
    'sleep_interval_requests': 0.75,  # 解析網頁時每個請求間隔
    'sleep_interval_subtitles': 5,    # 下載字幕前等待

    # --- 降低流量特徵 ---
    'ratelimit': RATE_LIMIT,
    'concurrent_fragment_downloads': 1,
    'lazy_playlist': True,            # 邊解析邊下載，不一次抓完整個清單

    # --- 失敗時以指數退避重試，而不是立刻狂打 ---
    'retries': 10,
    'fragment_retries': 10,
    'extractor_retries': 3,
    'retry_sleep_functions': {
        'http': lambda n: min(2 ** n, 120),
        'fragment': lambda n: min(2 ** n, 120),
        'extractor': lambda n: min(5 * 2 ** n, 300),
    },

    # --- 分批下載 ---
    'max_downloads': BATCH_SIZE,
}

print("開始下載影片...")

while True:
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([PLAYLIST_URL])
        break
    except yt_dlp.utils.MaxDownloadsReached:
        print(f"本批已下載 {BATCH_SIZE} 部，休息 {BATCH_REST // 60} 分鐘後繼續...")
        time.sleep(BATCH_REST)

print("下載程序結束！")

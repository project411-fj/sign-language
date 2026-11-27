import yt_dlp
import os

# YouTube 播放清單
PLAYLIST_URL = "https://youtube.com/playlist?list=PLHybHfSlbzTVGAfLGXA29jHcVql9PcRso"

# Windows 儲存位置
SAVE_FOLDER = r"C:\Users\jimmy\Videos\YouTubeDownloads"

# 建立資料夾
os.makedirs(SAVE_FOLDER, exist_ok=True)

ydl_opts = {
    # 優先下載 MP4
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',

    # 依照播放清單名稱建立資料夾
    'outtmpl': os.path.join(
        SAVE_FOLDER,
        '%(playlist_title)s',
        '%(title)s [%(id)s].%(ext)s'
    ),

    # 合併為 MP4
    'merge_output_format': 'mp4',

    # 避免重複下載
   'download_archive': os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'downloaded.txt'
),

    # 遇到錯誤繼續
    'ignoreerrors': True,

    'js_runtimes': {'node': {}},

    # 讀取 YouTube Cookie
'cookiefile': os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'cookies.txt'
),


   
}

print("開始下載影片...")

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([PLAYLIST_URL])

print("下載程序結束！")
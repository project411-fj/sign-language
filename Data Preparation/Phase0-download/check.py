import yt_dlp
import os

PLAYLIST_URL = "https://youtube.com/playlist?list=PLHybHfSlbzTVGAfLGXA29jHcVql9PcRso"

ARCHIVE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "downloaded.txt"
)

with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
    downloaded = {
        parts[1]
        for line in f
        if len(parts := line.strip().split()) == 2
        and parts[0] == "youtube"
    }

opts = {
    "extract_flat": True,
    "ignoreerrors": True,
    "quiet": True,
}

with yt_dlp.YoutubeDL(opts) as ydl:
    info = ydl.extract_info(PLAYLIST_URL, download=False)

videos = [
    video["id"]
    for video in info.get("entries", [])
    if video and video.get("id")
]

remaining = [vid for vid in videos if vid not in downloaded]

print("\n========== 下載統計 ==========")
print(f"播放清單總數：{len(videos)} 部")
print(f"已記錄下載：{len(videos) - len(remaining)} 部")
print(f"剩餘未下載：{len(remaining)} 部")
print("==============================")
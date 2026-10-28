
import json

from pathlib import Path

INPUT_JSON = Path("rewrite_Phase3_train") / "Labels.json"
OUTPUT_JSON = Path("rewrite_Phase3_train") / "dataset" / "manifest.json"

NPY_DIR = Path("yolo_pose_mediapie") / "hand_keypoints"


def video_sort_key(item):
    parts = item["video_id"].split("_")
    return int(parts[1]), int(parts[2])


def convert_manifest():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = []

    for item in data:
        video_id = Path(item["npz_file"]).stem.replace(
            "_upper_body_coordinates", ""
        )

        manifest.append({
            "video_id": video_id,
            "npy_path": str(NPY_DIR / f"{video_id}.npy"),
            "text": item["text"]
        })

    manifest.sort(key=video_sort_key)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=4)

    print(f"轉換完成，共 {len(manifest)} 筆資料")


if __name__ == "__main__":
    convert_manifest()

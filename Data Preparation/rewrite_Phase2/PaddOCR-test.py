from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont


VIDEO_PATH = Path("part_videos") / "part_1_1.mp4"
OUTPUT_ROOT = Path("rewrite_Phase2") / "paddleocr"
OUTPUT_IMAGE_DIR = OUTPUT_ROOT / "part_1_1_frames"
OUTPUT_TEXT_PATH = OUTPUT_ROOT / "part_1_1.txt"

CROP_LEFT = 0.27
CROP_RIGHT = 1.00
CROP_TOP = 0.79
CROP_BOTTOM = 0.88
SCALE = 2
MIN_CONFIDENCE = 0.0

FONT_SIZE = 28
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msjh.ttc"),
    Path(r"C:\Windows\Fonts\msjhbd.ttc"),
    Path(r"C:\Windows\Fonts\mingliu.ttc"),
)


def load_chinese_font():
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), FONT_SIZE)

    raise FileNotFoundError(
        "找不到繁體中文字型，請修改 FONT_CANDIDATES，指定電腦中的中文字型。"
    )


def draw_ocr_annotations(image_bgr, result, font):
    """Draw OCR boxes, recognized text, and confidence on the enlarged crop."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image_pil)

    boxes = result["rec_boxes"]
    texts = result["rec_texts"]
    scores = result["rec_scores"]

    for box, text, score in zip(boxes, texts, scores):
        text = text.strip()
        score = float(score)

        if not text or score < MIN_CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(int, box)
        label = f"{text}  {score:.2f}"

        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)

        label_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        label_x = max(0, min(x1, image_pil.width - label_width - 8))
        label_y = max(0, y1 - label_height - 8)

        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + label_width + 8,
                label_y + label_height + 8,
            ),
            fill=(0, 128, 0),
        )
        draw.text(
            (label_x + 4, label_y + 2),
            label,
            font=font,
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 80, 0),
        )

    return cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)


OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
font = load_chinese_font()

ocr = PaddleOCR(
    lang="chinese_cht",
    ocr_version="PP-OCRv5",
    device="gpu:0",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

capture = cv2.VideoCapture(str(VIDEO_PATH))
if not capture.isOpened():
    raise FileNotFoundError(f"無法開啟影片：{VIDEO_PATH.resolve()}")

fps = capture.get(cv2.CAP_PROP_FPS)
frame_number = 0

with OUTPUT_TEXT_PATH.open("w", encoding="utf-8") as output_file:
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_number += 1
        height, width = frame.shape[:2]

        x1 = int(width * CROP_LEFT)
        x2 = int(width * CROP_RIGHT)
        y1 = int(height * CROP_TOP)
        y2 = int(height * CROP_BOTTOM)

        subtitle = frame[y1:y2, x1:x2]
        subtitle = cv2.resize(
            subtitle,
            None,
            fx=SCALE,
            fy=SCALE,
            interpolation=cv2.INTER_CUBIC,
        )

        results = ocr.predict(subtitle)
        annotated = subtitle.copy()
        frame_texts = []

        for result in results:
            annotated = draw_ocr_annotations(annotated, result, font)

            for text, score in zip(result["rec_texts"], result["rec_scores"]):
                text = text.strip()
                if text and float(score) >= MIN_CONFIDENCE:
                    frame_texts.append(text)

        image_name = f"frame_{frame_number:06d}.jpg"
        image_path = OUTPUT_IMAGE_DIR / image_name

        if not cv2.imwrite(str(image_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"圖片儲存失敗：{image_path.resolve()}")

        timestamp = (frame_number - 1) / fps if fps > 0 else 0.0
        line = " ".join(frame_texts)
        output_file.write(
            f"{frame_number:06d}\t{timestamp:.3f}\t{image_name}\t{line}\n"
        )
        print(f"[{frame_number:06d}] {timestamp:.3f}s | {line}")

capture.release()

print(f"共輸出 {frame_number} 張圖片：{OUTPUT_IMAGE_DIR.resolve()}")
print(f"逐影格文字紀錄：{OUTPUT_TEXT_PATH.resolve()}")

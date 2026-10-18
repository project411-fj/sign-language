import cv2
from paddleocr import PaddleOCR

video_path = r"part_videos\part_1_1.mp4"
output_path = r"rewrite_Phase2\paddleocr\part_1_1.txt"

ocr = PaddleOCR(
    lang="chinese_cht",
    ocr_version="PP-OCRv5",
    device="gpu:0",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

capture = cv2.VideoCapture(video_path)

with open(output_path, "w", encoding="utf-8") as output_file:
    while True:
        success, frame = capture.read()
        if not success:
            break

        height, width = frame.shape[:2]

        x1 = int(width * 0.27)
        x2 = int(width)
        y1 = int(height * 0.79)
        y2 = int(height * 0.88)

        subtitle = frame[y1:y2, x1:x2]

        subtitle = cv2.resize(
            subtitle,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

        results = ocr.predict(subtitle)

        for result in results:
            texts = result["rec_texts"]

            texts = [text.strip() for text in result["rec_texts"] if text.strip()]
            line = " ".join(texts)
            output_file.write(line + "\n")
            print(line)

capture.release()
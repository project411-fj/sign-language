from pathlib import Path

from paddleocr import PaddleOCR

 

image_path = Path(r"PaddOCR/ocr_test.png")

 

ocr = PaddleOCR(

    lang="chinese_cht",

    ocr_version="PP-OCRv5",

    device="gpu:0",

    use_doc_orientation_classify=False,

    use_doc_unwarping=False,

    use_textline_orientation=False,

)

 

results = ocr.predict(str(image_path))

for result in results:

    result.print()

    result.save_to_img("paddleocr_output")

    result.save_to_json("paddleocr_output")
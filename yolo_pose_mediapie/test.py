from ultralytics import YOLO
import cv2

model = YOLO("yolo_pose\hand.pt")

cap = cv2.VideoCapture(
    r"part_videos\part_1_1.mp4"
)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    "hand_detection_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        conf=0.5,
        verbose=False
    )

    # YOLO 自動畫框
    annotated_frame = results[0].plot()

    writer.write(annotated_frame)

cap.release()
writer.release()

print("完成")
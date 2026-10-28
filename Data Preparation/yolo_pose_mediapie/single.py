from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(r"part_videos\part_1_1.mp4")

with open("keypoints.txt", "w", encoding="utf-8") as file:

    frame_id = 0

    while True:
        ret, frame = cap.read()

        if not ret: break

        pose_results = model(frame)

        for result in pose_results:

            if result.keypoints is None: continue

            body_keypoints = result.keypoints.xy
            body_conf = result.keypoints.conf

            for person_id, person in enumerate(body_keypoints):

                file.write(f"Frame: {frame_id}, Person: {person_id}\n")

                for keypoint_id, point in enumerate(person):

                    x = point[0].item()
                    y = point[1].item()
                    confidence = body_conf[person_id][keypoint_id].item()

                    file.write(
                        f"{keypoint_id}: "
                        f"{x:.2f}, {y:.2f}, {confidence:.4f}\n")

                file.write("\n")

        frame_id += 1

cap.release()
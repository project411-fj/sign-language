import cv2
import mediapipe as mp

input_path = r"part_videos\part_1_1.mp4"
output_path = r"part_1_1_mediapipe.mp4"
model_path = r"yolo_pose\hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(input_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Width:", width)
print("Height:", height)
print("FPS:", fps)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(output_path,fourcc,fps,(width, height))

with open("mediapipe_keypoints.txt", "w", encoding="utf-8") as file:

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)

        time_stamp_ms = int(frame_id / fps * 1000)

        results = landmarker.detect_for_video(mp_image,time_stamp_ms)

        for hand_id, hand_landmarks in enumerate(results.hand_landmarks):

            file.write(f"Frame: {frame_id}, Hand: {hand_id}\n")

            for point_id, landmark in enumerate(hand_landmarks):

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                file.write(
                    f"{point_id}: "
                    f"{x}, {y}, "
                    f"{landmark.z:.4f}\n")

                cv2.circle(frame,(x, y),5,(0, 255, 0),-1)

                cv2.putText(frame,str(point_id),(x + 5, y - 5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 0, 255),1)

            file.write("\n")

        out.write(frame)

        print(f"\rProcessing Frame: {frame_id}",end="")

        frame_id += 1

cap.release()
out.release()
landmarker.close()
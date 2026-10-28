import numpy as np

data = np.load(r"yolo_pose_mediapie/hand_keypoints/part_1_1.npy")

print("資料形狀:", data.shape)

left = data[:, 126]
right = data[:, 127]

print("左手偵測影格數:", np.sum(left == 1))
print("右手偵測影格數:", np.sum(right == 1))

detected = np.where((left == 1) | (right == 1))[0]

print("至少有一隻手的影格數:", len(detected))

if len(detected) > 0:
    frame_idx = detected[0]
    print("第一個成功偵測的 Frame:", frame_idx)
    print("Keypoints:", data[frame_idx])
else:
    print("整支影片都沒有偵測到手！")
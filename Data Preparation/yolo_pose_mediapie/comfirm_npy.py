import numpy as np

data = np.load(r"yolo_pose_mediapie/hand_keypoints/part_1_1.npy")

left = data[:, 126] == 1
right = data[:, 127] == 1

print("只有左手:", np.sum(left & ~right))
print("只有右手:", np.sum(~left & right))
print("雙手都有:", np.sum(left & right))
print("雙手都沒有:", np.sum(~left & ~right))
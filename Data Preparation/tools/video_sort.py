import os

import sys
if len(sys.argv) < 2:
    print('Input ERROR')
    sys.exit(1)
folder_path = sys.argv[1]

target_extensions = [".mp4",".m4a"]

count = 1
for filename in sorted(os.listdir(folder_path)):
    old_path = os.path.join(folder_path, filename)

    if os.path.isfile(old_path):
        ext = os.path.splitext(filename)[1]
        if ext.lower() in target_extensions:
            new_path = os.path.join(folder_path, f"{count}{ext}")

            while os.path.exists(new_path):
                count += 1
                new_path = os.path.join(folder_path, f"{count}{ext}")

            os.rename(old_path, new_path)
            count += 1

print("更名完成！")
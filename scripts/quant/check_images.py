import cv2
import os
d='/mnt/g/WSL/calib_images'
for f in sorted(os.listdir(d)):
    if f.endswith('.jpg'):
        img = cv2.imread(os.path.join(d, f))
        print(f'{f}: {img.shape if img is not None else "Invalid"}')

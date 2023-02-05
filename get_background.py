# Code from https://debuggercafe.com/moving-object-detection-using-frame-differencing-with-opencv/

import numpy as np 
import cv2
import matplotlib.pyplot as plt

def get_background(file_path):

    # create Video object
    cap = cv2.VideoCapture(file_path)

    # we will randomly select 50 frames for the calculating the median
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)

    # we will store the frames in array
    frames = []
    for idx in frame_indices:
        # set the frame id to read that particular frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)

    # calculate the median
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame


if __name__ == "__main__":
    mframe = get_background("input/video_1.mp4")

    fig, ax = plt.subplots()
    ax.imshow(data, extent=[0, 1, 0, 1])

    plt.show()
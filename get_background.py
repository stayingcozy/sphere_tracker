# Code from https://debuggercafe.com/moving-object-detection-using-frame-differencing-with-opencv/

import numpy as np 
import cv2

def get_background(file_path):

    # get Video as cv2 object
    cap = cv2.VideoCapture(file_path)
    
    # randomly select 50 frames for calculating the median
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)

    # store frames in array
    frames = []
    for idx in frame_indices:
        # set frame id to read that particular frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)

    # Calculate the median
    median_frame = np.median(frames, axis=0).astype(np.uint8)

    return median_frame
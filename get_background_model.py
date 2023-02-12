# Code based off of paper "Moving Object Detection and Segmentation using Frame differencing and Summing Technique - Thapa, Sharma, Ghose"

import numpy as np 
import random
import cv2

from cv_helper import video_capture


def get_background_model(filename):
    # opted out of vidcap.set(CV_CAP_PROP_POS_FRAMES, index) as it takes too long and is unreliable

    vidcap,success = video_capture(filename)

    # Set random seed for consistent results
    random.seed(100)

    # Store frames in list of arrays
    frames = []
    while success:
        # read frame
        success, frame = vidcap.read()

        # Flip coin to record
        if bool(random.getrandbits(1)) and success:
            frames.append(frame)

    # Reduce number of frames for reasonable median calc time
    if len(frames) > 50:

        # Random sample without replacement
        idx = random.sample(range(len(frames)),50)
        
        # convert to np array for indexing
        frames = np.stack(frames,axis=0)
        frames = frames[idx,:,:,:]

    # calculate the median
    median_frame = np.median(frames, axis=0).astype(np.uint8)

    vidcap.release()

    return median_frame


if __name__ == "__main__":
    mframe = get_background_model("input/video_1.mp4")
    cv2.imshow('image',mframe)
    cv2.waitKey(0)
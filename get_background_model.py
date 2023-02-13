# Code based off of paper "Moving Object Detection and Segmentation using Frame differencing and Summing Technique - Thapa, Sharma, Ghose"

import numpy as np 
import random
import cv2

from cv_helper import video_capture
import camera_calibration as cc

def get_background_model(filename,undistort):
    # opted out of vidcap.set(CV_CAP_PROP_POS_FRAMES, index) as it takes too long and is unreliable

    vidcap,success = video_capture(filename)

    # Set random seed for consistent results
    random.seed(100)

    if undistort:
        # params for undistorting fisheye
        fe = cc.FisheyeCamera()
        DIM, K, D = fe.load_coefficients("chessboard_1080\\fisheye_calibration.yml")
        

    # Store frames in list of arrays
    frames = []
    while success:
        # read frame
        success, frame = vidcap.read()

        if success:
            # Flip coin to record
            if bool(random.getrandbits(1)):
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

    if undistort:
        # undistort fisheye
        median_frame = fe.undistort(median_frame,DIM,K,D)

    vidcap.release()

    return median_frame


if __name__ == "__main__":
    mframe = get_background_model("input/video_1.mp4")
    cv2.imshow('image',mframe)
    cv2.waitKey(0)
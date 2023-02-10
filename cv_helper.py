import cv2
import sys

def video_capture(filename):

    # Read video
    vidcap = cv2.VideoCapture(filename)

    # Exit if not opened
    if not vidcap.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame
    success, frame = vidcap.read()
    if not success:
        print('Cannot read video file')
        sys.exit()

    return vidcap, success
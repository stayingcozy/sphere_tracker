# Code based off of paper "Moving Object Detection and Segmentation using Frame differencing and Summing Technique - Thapa, Sharma, Ghose"


import cv2
import argparse
import os
import numpy as np

from cv_helper import video_capture
from get_background_model import get_background_model
import detect_contour_sphere as dcs

## Argument Description ##
# --input path to the input video file
# --consecutive-frames number of consecutive frames to consider for frame differencing and summing; choose any number of frames
#                       from 2 to 8; lower tends to produce smoother video; author of original paper choose 8
# --binary-threshold value from 0 to 255 that is the cutoff for 0's and 1's after difference
# --contour-threshold minimum area cutoff value for object contour acceptance
# default to start: "python detect.py --input input/video_1.mp4 -c 4 -b 50 -ct 500"

## Pros and Cons ##
# Light weight computation (CPU) and generally works
# Need objects to move
# Requires consecutive frames so not real time
# Camera must be stationary
# Background need to be distuishable with objects and good lighting 

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to the input video',
                    required=True)
parser.add_argument('-c', '--consecutive-frames', default=4, type=int,
                    dest='consecutive_frames', help='path to the input video')
parser.add_argument('-b','--binary-threshold', default=50, type=int,
                    dest='binary_threshold', help='0-255 binary threshold for image')
parser.add_argument('-ct','--contour-threshold',default=500,type=int,
                    dest='contour_threshold', help='Minimum area of contour threshold to process')
args = vars(parser.parse_args())


# get the background model
background = get_background_model(args['input'])

# convert the background model to grayscale format
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# Frames Setup
frame_count = -1
consecutive_frame = args['consecutive_frames']

# Read video
vidcap,success = video_capture(args['input'])

# get the video frame height and width
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

# Create save name
save_name = f"outputs/{args['input'].split('/')[-1]}"

# Create output directory if it doesn't exist
if not os.path.isdir("outputs"):
    os.makedirs("outputs")

# define codec and create VideoWriter object
out = cv2.VideoWriter(
    save_name,
    cv2.VideoWriter_fourcc(*'mp4v'), 10, 
    (frame_width, frame_height)
)

# loop through frames
while (vidcap.isOpened()):
    success, frame = vidcap.read()

    if success == True:
        frame_count += 1
        orig_frame = frame.copy()

        # IMPORTANT STEP: convert the frame to grayscale first
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_count % consecutive_frame == 0:
            frame_diff_list = []

        if frame_count == 200:
            print("check point")

        # find the difference between current frame and base frame
        frame_diff = cv2.absdiff(gray, background)

        # thresholding to convert the frame to binary
        success, thres = cv2.threshold(frame_diff, args['binary_threshold'], 255, cv2.THRESH_BINARY) # add var, argument for binary_threshold value

        # opening - erosion followed by dilation - remove dots | opposited is MORPH_CLOSE - remove holes in objects
        kernel = np.ones((3,3),np.uint8)
        opening_frame = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)

        # append the final result into the `frame_diff_list`
        frame_diff_list.append(opening_frame)

        # if we have reached `consecutive_frame` number of frames
        if len(frame_diff_list) == consecutive_frame:
            # add all the frames in the `frame_diff_list`
            sum_frames = sum(frame_diff_list)

            # find the contours around the white segmented areas
            contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #RETR_EXTERNAL

            # # draw all the contours
            # contours_image = cv2.drawContours(frame,contours,-1,(0,0,255),3)

            for contour in contours:

                # # draw individual contour
                # contour_image = cv2.drawContours(frame,[contour],0,(255,0,0),3)

                # Area of contours - to skip low area (noise)
                area = cv2.contourArea(contour)

                # Skip based on both conditions
                if (area > args['contour_threshold']): # add var, arg for contour_threshold value

                    # get the xmin, ymin, width, and height coordinates from the contours
                    (x, y, w, h) = cv2.boundingRect(contour)
            
                    # draw the bounding boxes
                    image = cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Detect if contour is a sphere or not
                    circle_bool = dcs.solidity(contour,area)

                    # if fits rc dog toy criteria label
                    if (circle_bool):
                        cv2.putText(image, 'RC Dog Toy', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
            cv2.imshow('Detected Objects', orig_frame)
            out.write(orig_frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    else:
        break

vidcap.release()
cv2.destroyAllWindows()


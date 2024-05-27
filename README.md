<!-- Insert logo here -->
<div align="center">
    <img src="images/SphereTrackerLogo.png" alt="Logo" width="200" height="200">
</div>

# sphere tracker
Track spherical objects in videos. Includes fish eye camera calibration and object tracking.

# Demo

<!-- Insert gif of output video  -->
<div align="center">
  <img src="images/demo1.gif" alt="animated" />
  <img src="images/demo2.gif" alt="animated" />
</div>


# Install
Required python libraries
- cv2, numpy, argparse, os, random, sys, glob

# Usage
```
python pixel_diff.py --i input/sphero1_1080.mp4
```
Check pixel_diff.py for more argument options

### Sources 
Code based on paper:
- "Moving Object Detection and Segmentation using Frame differencing and Summing Technique - Thapa, Sharma, Ghose”

Calibration based on:
- ![Python Calibration Tutorial](https://docs.opencv2.org/4.x/dc/dbb/tutorial_py_calibration.html)
- ![Calibrate Fisheye](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0)
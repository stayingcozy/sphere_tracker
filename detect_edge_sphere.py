import cv2
import numpy as np

def Hough_Circles(gray):

    # assume input is gray
    # image needs to be gray and blurred before Hough

    # Potential Improvements
    # increase blur, Increase mindist by screen (only one circle detected),
    #  method for implementing radius range, 

    # Detect sphere
    # gray_blurred = cv2.medianBlur(gray,5)
    gray_blurred = cv2.blur(gray, (3, 3))

    cimg = cv2.cvtColor(gray_blurred,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(gray_blurred,cv2.HOUGH_GRADIENT,1,minDist=1920,
                                param1=50,param2=30,minRadius=0,maxRadius=0) # minDist=20

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        circle_bool = True
        circle_coord = circles[0,0]
        x = circle_coord[0]
        y = circle_coord[1]

        image = cv2.circle(gray,(x,y),circle_coord[2],(255,0,0),2)

        if (circle_bool):
            cv2.putText(image, 'RC Dog Toy', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,36,12), 2)
        return image

    else:
        return gray


def Hough_Circles_Mask(gray,pixel_diff_mask,width_dis):
    ''' Apply pixel difference mask after dilation to gray image, blur, then apply Hough Transform'''

    # Inputs
    # gray - gray image from camera
    # summation of boolean pixel differences 
    #
    # Additional possible inputs for Hough to increase accuracy
    #  1. radius of circle (need to know distance to calculate)
    #  2. number of circles - distance from other circles (1 circle is an ok assumption for now) 

    # init 
    circle = None

    # Apply dilation on boolean mask
    kernel = np.ones((3,3),np.uint8)
    pixel_diff_dilate_mask = cv2.dilate(pixel_diff_mask,kernel,iterations=50) # High amount of dilation to make sure to capture the entire ball

    # Detect sphere
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply mask
    blurry_gray_mask = gray_blurred * pixel_diff_dilate_mask

    # Hough Transform
    circles = cv2.HoughCircles(blurry_gray_mask,cv2.HOUGH_GRADIENT,1,minDist=width_dis,
                                param1=50,param2=30,minRadius=0,maxRadius=0) # minDist=20

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle = circles[0,:]

        # draw circles
        # for i in circles[0,:]:
        #     # draw the outer circle
        #     cv2.circle(gray,(i[0],i[1]),i[2],(0,255,0),2)
        #     # draw the center of the circle
        #     cv2.circle(gray,(i[0],i[1]),2,(0,0,255),3)

    return circle

def draw_Hough_circle(img,circle):

    # draw the outer circle
    cv2.circle(img,(circle[0,0],circle[0,1]),circle[0,2],(255,0,0),2)
    # draw the center of the circle
    cv2.circle(img,(circle[0,0],circle[0,1]),2,(0,0,255),3)
    # put label text
    cv2.putText(img, 'RC Dog Toy', (circle[0,0], circle[0,1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,36,12), 2)

    return img

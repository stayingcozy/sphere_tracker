import cv2
import numpy as np

def contour_points(contour):
    # performs poorly - worst out of group

    # approxPolyDP for spherical shapes (high # of points more curvy)
    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)

    circle_bool = len(approx) > 8

    return circle_bool

def circle_ratio(contour,area):
    # better but fails when pixel diff is stretched out when moving

    # Find circle eqivalent diameter for area
    equi_diameter = np.sqrt(4*area/np.pi)
    #  Min enclosing circle
    (x,y),radius = cv2.minEnclosingCircle(contour)
    # calc ratio between equiv circle diameter / minimum enclosing circle diameter
    circle_ratio = equi_diameter / (radius*2)

    circle_bool = circle_ratio > 0.7

    return circle_bool

# def match_shapes(contour):
#     # will have same problem as circle_ratio

#     # read in binary image of circle
#     # calc contour
#     # matchShape with input contour
#     # ret = cv.matchShapes(cnt1,cnt2,1,0.0)
#     # lower the result better the match

#     return

def solidity(contour,area):
    # captures ball well even when moving fast - best

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area

    circle_bool = solidity > 0.9

    return circle_bool
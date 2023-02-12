import numpy as np
import cv2
import glob
import os
# Source: https://docs.opencv2.org/4.x/dc/dbb/tutorial_py_calibration.html

def calibrate_chessboard(dir_path,square_size,width,height):
    ''' Calibrate a camera using chessboard images '''

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((width*height,3), np.float32)
    objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    files = os.path.join(dir_path,"*.jpeg")
    images = glob.glob(files)

    for fname in images:

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width,height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (width,height), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''

    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)

    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''

    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

def calibrate_save(dir_path, square_size, width,height):
    ''' Calibrate off chessboard and save'''
    # setup and calibrate
    ret, mtx, dist, rvecs, tvecs = calibrate_chessboard(dir_path, square_size, width,height)

    # save off calibration values
    save_coefficients(mtx, dist, "calibration_chessboard.yml")

def undistort_params(mtx, dist):
    img = cv2.imread('left12.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    return newcameramtx, roi

def undistort(img, mtx, dist, newcameramtx, roi):
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)



if __name__ == "__main__":

    # init values
    dir_path = "chessboard"
    width = 9
    height = 6
    square_size =  2.188 # size of each block in cm

    calibrate_save(dir_path, square_size, width,height)

    # Load coefficients
    mtx, dist = load_coefficients('calibration_chessboard.yml')
    original = cv2.imread('chessboard\\chessboard_18.jpeg')
    dst = cv2.undistort(original, mtx, dist, None, None)
    cv2.imwrite('undist.jpg', dst)


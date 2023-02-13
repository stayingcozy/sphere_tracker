import numpy as np
import cv2
import glob
import os

# Source: https://docs.opencv2.org/4.x/dc/dbb/tutorial_py_calibration.html
# https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0

class FisheyeCamera():
    def __init__(self):
        self.DIM = []
        self.K = []
        self.D = []

    def fisheye_calibrate_chessboard(self,dir_path):

        CHECKERBOARD = (9,6) 
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        _img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        files = os.path.join(dir_path,"*.jpg")
        images = glob.glob(files)

        for fname in images:

            img = cv2.imread(fname)

            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."

            self.DIM = [_img_shape[1],_img_shape[0]]

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        N_OK = len(objpoints)

        K = np.zeros((3, 3))
        D = np.zeros((4, 1))

        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )

        self.K = K
        self.D = D

        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")

    def undistort_file(self,img_path):

        DIM= self.DIM
        K= self.K
        D= self.D

        img = cv2.imread(img_path)
        h,w = img.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow("distorted",img)
        cv2.imshow("undistorted", undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def undistort(self,img, dimension, camera_matrix, dist_matrix):
        DIM = dimension
        K = camera_matrix
        D = dist_matrix

        h,w = img.shape[:2]
        # assert np.all(np.array([[w],[h]]) == DIM), "Input image size must match calibration image size."

        if type(DIM) is np.ndarray:
            DIM = DIM.tolist()
            DIM = [DIM[0][0],DIM[1][0]]

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return undistorted_img


    def save_coefficients(self, path):
        '''Save the camera matrix and the distortion coefficients to given path/file.'''

        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write('DIM', np.array(self.DIM))
        cv_file.write('K', self.K)
        cv_file.write('D', self.D)

        # note you *release* you don't close() a FileStorage object
        cv_file.release()

    def load_coefficients(self,path):
        '''Loads camera matrix and distortion coefficients.'''

        # FILE_STORAGE_READ
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        # note we also have to specify the type to retrieve other wise we only get a
        # FileNode object back instead of a matrix
        dimension = cv_file.getNode('DIM').mat()
        camera_matrix = cv_file.getNode('K').mat()
        dist_matrix = cv_file.getNode('D').mat()

        cv_file.release()

        return dimension, camera_matrix, dist_matrix

class NormalCamera():
    def __init__(self):
        # width = 9
        # height = 6
        # square_size =  2.188 # size of each block in cm
        pass

    def calibrate_chessboard(self,dir_path,square_size,width,height):
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

    def save_coefficients(self,mtx, dist, path):
        '''Save the camera matrix and the distortion coefficients to given path/file.'''

        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write('K', mtx)
        cv_file.write('D', dist)

        # note you *release* you don't close() a FileStorage object
        cv_file.release()

    def load_coefficients(self,path):
        '''Loads camera matrix and distortion coefficients.'''

        # FILE_STORAGE_READ
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        # note we also have to specify the type to retrieve other wise we only get a
        # FileNode object back instead of a matrix
        camera_matrix = cv_file.getNode('K').mat()
        dist_matrix = cv_file.getNode('D').mat()

        cv_file.release()
        return [camera_matrix, dist_matrix]

    def calibrate_save(self,dir_path, square_size, width,height):
        ''' Calibrate off chessboard and save'''
        # setup and calibrate
        ret, mtx, dist, rvecs, tvecs = self.calibrate_chessboard(dir_path, square_size, width,height)

        # save off calibration values
        self.save_coefficients(mtx, dist, "calibration_chessboard.yml")




if __name__ == "__main__":

    # init values
    dir_path = "chessboard_1080"
    ex_img = os.path.join(dir_path,"chessboard_18.jpg")

    # # Normal Camera
    # nc = NormalCamera()
    # nc.calibrate_save(dir_path,2.188,9,6)

    # Fisheye / Wide angle lens Camera (160deg FOV)
    fe = FisheyeCamera()
    fe.fisheye_calibrate_chessboard(dir_path)
    fe.save_coefficients(os.path.join(dir_path,"fisheye_calibration.yml"))
    fe.undistort_file(ex_img)
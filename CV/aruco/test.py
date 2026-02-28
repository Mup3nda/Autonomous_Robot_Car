import numpy as np
import yaml
import glob
import cv2

CHESSBOARD_SIZE = (8,6)
SQUARE_SIZE = 21

objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3),np.float32)

objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)

objp = SQUARE_SIZE

objpoints = []
imgpoints = []

images = glob.glob('calib_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    found, corner = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    if found:
        object.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# print the calibration results
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
print("Reprojection error:", ret)

calib_data = {
    "camera_matrix": mtx.tolist(),
    "dis_coeff": dist.tolist(),
    "reprojection_error": float(ret)
}

with open('calibration.yaml') as f:
    yaml.dump(calib_data, f)
    
print("Saved calibration.yaml")



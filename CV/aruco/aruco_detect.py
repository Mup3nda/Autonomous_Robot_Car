import cv2
import numpy as np
import yaml
from picamera2 import Picamera2
 
# Load calibration data with Python's yaml
with open(r"calibration.yaml") as f:
    calib = yaml.safe_load(f)
 
""" 
load your camera’s intrinsic matrix and distortion coefficients from a YAML file
- camera_matrix is the 3×3 intrinsic parameters matrix.
- dist_coeffs holds lens distortion parameters.
- marker_length is the side length of the ArUco marker in meters.
"""
camera_matrix = np.array(calib["camera_matrix"])
dist_coeffs = np.array(calib["dist_coeff"])
marker_length = 0.026  
 
"""
DICT_4X4_50 is a predefined dictionary of ArUco markers.
4x4 markers with 50 unique IDs.(16 bits per marker)
parameters are the default detection parameters.
"""
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
 
# create a detector instance with default parameters.
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
 
 
picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(camera_config)
picam2.start()

#cap = cv2.VideoCapture(0)
 
num=0
 
while True:
    
    frame = picam2.capture_array()
    

    ## For webcam
    #ret, frame = cap.read()
    # if not ret:
    #     break
    
    if frame is None:
        break
 
    # Detect markers in the frame with detectMarkers method
    corners, ids, _ = detector.detectMarkers(frame)
    """
    Example output of corners:
    (array([[[ 15., 339.],
        [ 91., 334.],
        [ 98., 404.],
        [ 19., 414.]]], dtype=float32),)
    """
 
    if ids is not None and len(ids) > 0:
 
        # Draw detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
 
        # Define the 3D coordinates of the marker corners in the marker's coordinate system
        obj_points = np.array([
            [-marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float32)
 
         
        for marker_corners in corners:
            image_points = marker_corners[0].astype(np.float32)
 
            """
            solvePnP estimates the pose of a 3D object given its 3D points and corresponding 2D image points.
            It returns the rotation vector (rvec) and translation vector (tvec).
            """
            retval, rvec, tvec = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)
             
            if retval:
                # Draw the axis on the frame
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                 
                # Extract the translation vector and calculate the distance
                x, y, z = tvec.flatten()
                distance = np.linalg.norm(tvec)
                
                # Print to terminal
                print(f"X={x:.3f} m, Y={y:.3f} m, Z(depth)={z:.3f} m, Distance={distance:.3f} m")
            
                text_x = int(image_points[0][0])
                text_y = int(image_points[0][1]) - 10
                
                # Display on frame with different colors
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"X={x:.3f}", (text_x, text_y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Y={y:.3f}", (text_x, text_y+20), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Z(Depth)={z:.3f}", (text_x, text_y+40), font, 0.5, (255, 0, 0), 2,cv2.LINE_AA)
                cv2.putText(frame, f"Distance={distance:.3f}", (text_x, text_y+60), font, 0.5, (200, 255, 0), 2,cv2.LINE_AA)
 
    # Display the frame with detected markers and pose estimatio
    cv2.imshow('ArUco Pose Estimation', frame)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
#cap.release()
picam2.stop()
cv2.destroyAllWindows()
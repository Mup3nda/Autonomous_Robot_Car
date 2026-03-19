import cv2
import numpy as np
import yaml
from picamera2 import Picamera2

with open('calibration.yaml') as f:
    calib = yaml.safe_load(f)
    
camera_matrix = calib['camera_matrix']
dist_coeffs = calib['dist_coeff']
marker_length = 0.026
    
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

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
    
    if frame is None:
        break
    
    corners, ids, _ = detector.detectMarkers(frame)
    
    if corners is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        obj_points = np.array([
            [-marker_length/2, marker_length/2, 0],
            [marker_length/2, marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0],
            [marker_length/2, -marker_length/2, 0]
        ], dtype=np.float32)
        
        for marker_corners in corners:
            image_points = marker_corners[0].astype(np.float32)
            
            retval, rvec, tvec = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)
            
            if retval:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                
                x, y, z = tvec.flatten()
                distance = np.linalg.norm(tvec)
                
                print(f"X={x:.3f} m, Y={y:.3f} m, Z(depth)={z:.3f} m, Distance={distance:.3f} m")
                
                text_x = int(image_points[0])
                text_y = int(image_points[1]) - 10
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"X={x:.3f}", (text_x, text_y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Y={y:.3f}", (text_x, text_y+20), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Z(Depth)={z:.3f}", (text_x, text_y+40), font, 0.5, (255, 0, 0), 2,cv2.LINE_AA)
                cv2.putText(frame, f"Distance={distance:.3f}", (text_x, text_y+60), font, 0.5, (255, 0, 0), 2,cv2.LINE_AA)
        cv2.imshow('Aruco Pose', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q')
            break

picam2.stop()
cv2.destroyAllWindows()

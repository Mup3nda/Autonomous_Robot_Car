

from picamera2 import Picamera2
from collections import deque
from imutils.video import VideoStream  # Threaded video stream (used for optional video file)
from imutils.video import VideoStream  # Threaded video stream (used for optional video file)
import numpy as np                  # Numerical operations
import argparse                     # Command-line argument parsing
import cv2                          # OpenCV for image processing
import imutils                      # Convenience functions for OpenCV
import time                         # Sleep on startup
from picamera2 import Picamera2     # Raspberry Pi camera interface
import socket                       # For getting local IP address
from flask import Flask, Response   # Web server for MJPEG streaming
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

 
CUBE_MARKER_SIZE = 0.0350
PLATFORM_MARKER_SIZE = 0.0350
DROP_AREA_MARKER_SIZE = 0.100
STOP_AREA_MARKER_SIZE = 0.154
# TEST
PHONE_MARKER_SIZE = 0.026
PAD_MARKER_SIZE = 0.073  

MARKER_SIZES = {
    
    0: PHONE_MARKER_SIZE, #TEst
    99: PAD_MARKER_SIZE, #TEst
    #------------------------------
    5: PLATFORM_MARKER_SIZE, #Platform
    20: CUBE_MARKER_SIZE, #Cube 1
    53: CUBE_MARKER_SIZE, #CUBE 2
    #------------------------------
    10: DROP_AREA_MARKER_SIZE, #A
    11: DROP_AREA_MARKER_SIZE, #A
    12: DROP_AREA_MARKER_SIZE, #B
    13: DROP_AREA_MARKER_SIZE, #B
    14: DROP_AREA_MARKER_SIZE, #C
    15: DROP_AREA_MARKER_SIZE, #C
    16: DROP_AREA_MARKER_SIZE, #D
    17: DROP_AREA_MARKER_SIZE, #D
    #-------------------------------
    25: DROP_AREA_MARKER_SIZE, #Finish 
    
}


distance_buffer = deque(maxlen=5)
#camera_config = 'oliver_calibration.yaml'
camera_config = 'myraspi_calibration.yaml'
#ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_DICT = cv2.aruco.DICT_4X4_100

detected_markers = {}

def parse_arguments():
    """Set up command-line argument parser"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", metavar="PATH",  help="Path to video file (optional, default: Pi camera)")
    ap.add_argument("-s", "--stream", action="store_true", help="Serve MJPEG stream in browser instead of cv2.imshow")
    ap.add_argument("-H", "--host", default="0.0.0.0", metavar="IP",  help="Host/IP to bind the MJPEG server        (default: 0.0.0.0)")
    ap.add_argument("-p", "--port", type=int, default=5000,   metavar="PORT", help="Port to serve the MJPEG stream on       (default: 5000)")
    ap.add_argument("-q", "--jpeg-quality", type=int, default=70, metavar="1-100",  help="JPEG compression quality for streaming  (default: 70)")
    args = vars(ap.parse_args())
    return args
 
def load_camera_calibrations(file_path):
    """load camera calibration"""
    with open(file_path) as f:
        calib = yaml.safe_load(f)
        
    camera_matrix = np.array(calib["camera_matrix"])
    dist_coeffs = np.array(calib["dist_coeff"])

    return camera_matrix, dist_coeffs
 
def initialize_aruco_detector():
    """
    parameters are the default detection parameters.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    return aruco_dict, parameters, detector

def intiatialize_camera():
 
    picam2 = Picamera2()
    
    #cap = cv2.VideoCapture(0)

    camera_config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    
    time.sleep(2)
    
    return picam2    

def get_object_points(marker_size):
    # Define the 3D coordinates of the marker corners in the marker's coordinate system
    obj_points = np.array([
        [-marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)
    
    return obj_points

def display_text(frame, image_points, x , y, z, distance):
    
    #org = (20, 90)
    text_x = int(image_points[0][0])
    text_y = int(image_points[0][1]) - 10
    
    # Display on frame with different colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"X={x:.3f}", (text_x, text_y ), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Y={y:.3f}", (text_x, text_y +20), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Z={z:.3f}", (text_x, text_y +40), font, 0.5, (255, 0, 0), 1,cv2.LINE_AA)
    cv2.putText(frame, f"D={distance:.3f}", (text_x, text_y +60), font, 0.5, (200, 100, 0), 1,cv2.LINE_AA)
    # cv2.putText(frame, f"X={x:.3f}", (org[0], org[1]), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.putText(frame, f"Y={y:.3f}", (org[0], org[1]+20), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    # cv2.putText(frame, f"Z={z:.3f}", (org[0], org[1]+40), font, 0.7, (255, 0, 0), 1,cv2.LINE_AA)
    # cv2.putText(frame, f"Distance={distance:.3f}", (org[0], org[1]+60), font, 0.7, (200, 100, 0), 1,cv2.LINE_AA)

def detect_aruco(picam2, detector, camera_matrix, dist_coeffs):
    frame = picam2.capture_array()
    ## For webcam
    #ret, frame = cap.read()
    # if not ret:
    #     break
    if frame is None:
        return None

    detected_markers = {}
    # Detect markers in the frame with detectMarkers method
    corners, marker_ids, _ = detector.detectMarkers(frame)


    if marker_ids is not None and len(marker_ids) > 0:
        # Draw detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)

        
        
        for marker_corners, marker_id in zip(corners, marker_ids):
            
            _marker_id = marker_id[0]
            
            if _marker_id in MARKER_SIZES:
                _marker_size = MARKER_SIZES[_marker_id]
                
            else:
                logger.info("Did not find the marker id in he defined list")
                continue
        
        
            obj_points = get_object_points(_marker_size)
                
                
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
                x_cm = x*100
                y_cm = y*100
                z_cm = z*100
                distance_cm = distance*100
            
                
                detected_markers[int(_marker_id)] = {
                    "x": float(round(x_cm, 4)),
                    "y": float(round(y_cm, 4)),
                    "z": float(round(z_cm, 4)),
                    "distance": float(round(distance_cm, 4))
                }
                
                print(f"ID:{_marker_id}, x: {x_cm:.2f} cm, y: {y_cm:.2f} cm, z: {z_cm:.2f} cm, distance: {distance_cm:.2f} cm")
                #print(list(detected_markers.keys()))
                #print(detected_markers[0])
                #print(detected_markers[0]['distance'])
                
                display_text(frame, image_points, x, y, z, distance)
    return frame, detected_markers
 
def run_mjpeg_stream(picam2, detector, camera_matrix, dist_coeffs, args):
    app = Flask(__name__)
    
    def generate():
        """Generator that yields JPEG frames in multipart format"""
        while True:
            frame, _ = detect_aruco(picam2, detector, camera_matrix, dist_coeffs)
            if frame is None:
                break
            
            ok, buffer = cv2.imencode(
                '.jpg',
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), args['jpeg_quality']]
            )
            if not ok:
                continue
            
            yield(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )
    @app.route('/')
    def index():
        """Simple HTML page embedding the video stream"""
        return '<html><body><h3>Ollie Classified Stream</h3><img src="/video" /></body></html>'
    
    @app.route('/video')
    def video():
        """Endpoint that serves the raw MJPEG stream"""
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    try:
        app.run(host=args['host'],
                port=args['port'],
                threaded=True)    
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

def run_local_gui(picam2, detector, camera_matrix, dist_coeffs, args):
    try:
        while True:
            frame, _ = detect_aruco(picam2, detector, camera_matrix, dist_coeffs)
            
            if frame is None:
                break
            
            cv2.imshow("Aruco Detect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__=='__main__':
    
    args = parse_arguments()
    
    camera_matrix, dist_coeffs = load_camera_calibrations(camera_config)
    aruco_dict, parameters, detector = initialize_aruco_detector()
    picam2 = intiatialize_camera()
    
    logger.warning("Make sure you have included comand arguments")
    logger.info("Make sure you use correctg calibration file")
    
    if args.get("stream", False):
        run_mjpeg_stream(picam2, detector, camera_matrix, dist_coeffs, args)
    else:
        run_local_gui(picam2, detector, camera_matrix, dist_coeffs, args)
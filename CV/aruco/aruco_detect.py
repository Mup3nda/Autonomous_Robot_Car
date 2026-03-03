

from picamera2 import Picamera2
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

 
 
marker_length = 0.026   
camera_config = 'calibration.yaml'
ARUCO_4X4_50 = cv2.aruco.DICT_4X4_50

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
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
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
    
    return picam2    

def get_object_points():
    # Define the 3D coordinates of the marker corners in the marker's coordinate system
    obj_points = np.array([
        [-marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ], dtype=np.float32)
    
    return obj_points

def display_text(frame, image_points, x , y, distance):
    
    text_x = int(image_points[0][0])
    text_y = int(image_points[0][1]) - 10
    
    # Display on frame with different colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"X={x:.3f}", (text_x, text_y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Y={y:.3f}", (text_x, text_y+20), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Z(Depth)={z:.3f}", (text_x, text_y+40), font, 0.5, (255, 0, 0), 2,cv2.LINE_AA)
    cv2.putText(frame, f"Distance={distance:.3f}", (text_x, text_y+60), font, 0.5, (200, 255, 0), 2,cv2.LINE_AA)

def detect_markers(picam2, detector, camera_matrix, dist_coeffs):
    frame = picam2.capture_array()
    ## For webcam
    #ret, frame = cap.read()
    # if not ret:
    #     break
    if frame is None:
        return None

    # Detect markers in the frame with detectMarkers method
    corners, ids, _ = detector.detectMarkers(frame)


    if ids is not None and len(ids) > 0:
        # Draw detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        obj_points = get_object_points()
        
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
            
                display_text(frame, image_points, x, y, distance)
        return frame
 
def run_mjpeg_stream(picam2, detector, camera_matrix, dist_coeffs, args):
    app = Flask(__name__)
    
    def generate():
        """Generator that yields JPEG frames in multipart format"""
        while True:
            frame = detect_markers(picam2, detector, camera_matrix, dist_coeffs)
            if frame is None:
                break
            
            ok, buffer = cv2.imdecode(
                '.jpeg',
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
        return '<html><body><h3>MJPEG stream</h3><img src="/video" /></body></html>'
    
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
            frame = detect_markers(picam2, detector, camera_matrix, dist_coeffs)
            
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
    
    logger.info("Make sure you have included comand arguments")
    
    if args.get("stream", False):
        run_mjpeg_stream(picam2, detector, camera_matrix, dist_coeffs, args)
    else:
        run_local_gui(picam2, detector, camera_matrix, dist_coeffs, args)
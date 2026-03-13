
from collections import deque
from imutils.video import VideoStream  # Threaded video stream (used for optional video file)
import numpy as np                  # Numerical operations
import argparse                     # Command-line argument parsing
import cv2                          # OpenCV for image processing
import imutils                      # Convenience functions for OpenCV
import time                         # Sleep on startup
import socket                       # For getting local IP address
from flask import Flask, Response   # Web server for MJPEG streaming
import logging
import yaml
from scam_usb import cam_usb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArucoDetector:
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

    #ARUCO_DICT = cv2.aruco.DICT_4X4_50
    ARUCO_DICT = cv2.aruco.DICT_4X4_100
    
    def __init__(self, camera_config='myraspi_calibration.yaml'):
        self.detected_markers = {}
        self.aruco_dict = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.parameters = None
        self.detector = None
        self.distance_buffer = deque(maxlen=5)
        self.camera_config = camera_config
        
    def start(self):
        # Initialize any necessary resources for target detection
        cam_usb.setup()
        self.camera_matrix, self.dist_coeffs = self.load_camera_calibrations(self.camera_config)
        self.aruco_dict, self.parameters, self.detector = self.initialize_aruco_detector()
        
        logger.info("% ArucoDetector:: Setup complete")
    
    def get_target(self, target_id=None):
        # Placeholder for target detection logic
        # In a real implementation, this would return the detected target's position and confidence
        return self.detect_aruco(
            cam_usb, 
            self.detector, 
            self.camera_matrix, 
            self.dist_coeffs, target_id=target_id
        )
    
    def stop(self):
        # Clean up any resources if necessary
        cam_usb.terminate()
        logger.info("% ArucoDetector:: Setup stopped")
    
    def parse_arguments(self):
        """Set up command-line argument parser"""
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", metavar="PATH",  help="Path to video file (optional, default: Pi camera)")
        ap.add_argument("-s", "--stream", action="store_true", help="Serve MJPEG stream in browser instead of cv2.imshow")
        ap.add_argument("-t", "--target-id", type=int, default=None, metavar="ID", help="Detect only this ArUco marker ID (default: detect all)")
        ap.add_argument("-H", "--host", default="0.0.0.0", metavar="IP",  help="Host/IP to bind the MJPEG server        (default: 0.0.0.0)")
        ap.add_argument("-p", "--port", type=int, default=5000,   metavar="PORT", help="Port to serve the MJPEG stream on       (default: 5000)")
        ap.add_argument("-q", "--jpeg-quality", type=int, default=70, metavar="1-100",  help="JPEG compression quality for streaming  (default: 70)")
        args = vars(ap.parse_args())
        return args

        
    def load_camera_calibrations(self, file_path):
        """load camera calibration"""
        with open(file_path) as f:
            calib = yaml.safe_load(f)
            
        camera_matrix = np.array(calib["camera_matrix"])
        dist_coeffs = np.array(calib["dist_coeff"])

        return camera_matrix, dist_coeffs
    
    def initialize_aruco_detector(self):
        """
        parameters are the default detection parameters.
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        return aruco_dict, parameters, detector
    
    def get_object_points(self, marker_size):
        # Define the 3D coordinates of the marker corners in the marker's coordinate system
        obj_points = np.array([
            [-marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0]
        ], dtype=np.float32)
    
        return obj_points
    
    def detect_aruco(self, cam, detector, camera_matrix, dist_coeffs, target_id=None):
        ok, frame, timestamp = cam.getImage()
        if not ok or frame is None:
            return None, {}

        detected_markers = {}
        # Detect markers in the frame with detectMarkers method
        corners, marker_ids, _ = detector.detectMarkers(frame)


        if marker_ids is not None and len(marker_ids) > 0:
            # Draw detected markers on the frame
            
            cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)
            
            for marker_corners, marker_id in zip(corners, marker_ids):
                
                current_id = int(marker_id[0])

                if target_id is not None and current_id != target_id:
                    continue

                if current_id in self.MARKER_SIZES:
                    _marker_size = self.MARKER_SIZES[current_id]
                    
                else:
                    logger.info("Did not find the marker id in he defined list")
                    continue
            
            
                obj_points = self.get_object_points(_marker_size)
                    
                    
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
                
                    
                    detected_markers[current_id] = {
                        "x": float(round(x_cm, 4)),
                        "y": float(round(y_cm, 4)),
                        "z": float(round(z_cm, 4)),
                        "distance": float(round(distance_cm, 4))
                    }
                    
                    print(f"ID:{current_id}, x: {x_cm:.2f} cm, y: {y_cm:.2f} cm, z: {z_cm:.2f} cm, distance: {distance_cm:.2f} cm")
                    #print(list(detected_markers.keys()))
                    #print(detected_markers[0])
                    #print(detected_markers[0]['distance'])
                    #self.display_text(frame, image_points, x_cm, y_cm, z_cm, distance_cm)

        self.detected_markers = detected_markers
        return frame, detected_markers
    
    def run_mjpeg_stream(self, cam, detector, camera_matrix, dist_coeffs, args, target_id=None):
        app = Flask(__name__)
        
        def generate():
            """Generator that yields JPEG frames in multipart format"""
            while True:
                frame, _ = self.detect_aruco(cam, detector, camera_matrix, dist_coeffs, target_id=target_id)
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
            cam.terminate()
            cv2.destroyAllWindows()

    def run_local_gui(self, cam, detector, camera_matrix, dist_coeffs, args, target_id=None):
        try:
            while True:
                frame, _ = self.detect_aruco(cam, detector, camera_matrix, dist_coeffs, target_id=target_id)
                
                if frame is None:
                    break
                
                cv2.imshow("Aruco Detect", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        finally:
            cam.terminate()
            cv2.destroyAllWindows()

    def display_text(self, frame, image_points, x , y, z, distance):
        
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


aruco = ArucoDetector()


if __name__=='__main__':
    
    args = aruco.parse_arguments()
    
    aruco.start()
    
    logger.warning("Make sure you have included comand arguments")
    logger.info("Make sure you use correctg calibration file")

    target_id = args.get("target_id")
    if target_id is None:
        logger.info("% ArucoDetector:: Detecting all known markers")
    else:
        logger.info(f"% ArucoDetector:: Detecting only marker ID {target_id}")
    
    if args.get("stream", False):
        aruco.run_mjpeg_stream(cam_usb, aruco.detector, aruco.camera_matrix, aruco.dist_coeffs, args, target_id)
    else:
        aruco.run_local_gui(cam_usb, aruco.detector, aruco.camera_matrix, aruco.dist_coeffs, args, target_id)
        
        
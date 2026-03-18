
from collections import deque
import math
import numpy as np                  # Numerical operations
import argparse                     # Command-line argument parsing
import cv2                          # OpenCV for image processing
from flask import Flask, Response   # Web server for MJPEG streaming
import logging
import yaml
from scam import cam
from sgpio import gpio
from uservice import service
from target_detector import TargetDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArucoDetector(TargetDetector):
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
    
    def __init__(self,
                 cam, 
                 gpio, 
                 service, 
                 camera_config='/home/local/Autonomous_Robot_Car/CV/aruco/oliver_calibration.yaml', 
                 target_id=None, 
                 manage_camera=False
    ):
        super().__init__()
        self.detected_markers = {}
        self.aruco_dict = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.parameters = None
        self.detector = None
        self.distance_buffer = deque(maxlen=5)
        self.camera_config = camera_config
        self.target_id = target_id
        self.last_frame = None
        self.cam = cam
        self.gpio = gpio
        self.service = service
        
        
        # Contructor fields
        self.manage_camera = manage_camera
        self.camera_started_by_detector = False
        self.frame_fail_count = 0
        self.fail_log_iteration = 5
        
        
        
        
    def start(self):
        # Initialize any necessary resources for target detection
        if self.manage_camera:
            self.cam.setup()
            self.camera_started_by_detector = True
            
        if not self.cam.useCam:
            print("% ArucoDetector:: Camera not available")
            return
        
        self.camera_matrix, self.dist_coeffs = self.load_camera_calibrations(self.camera_config)
        self.aruco_dict, self.parameters, self.detector = self.initialize_aruco_detector()
        
        print("% ArucoDetector:: Setup complete")
    
    def stop(self):
        # Clean up any resources if necessary
        if self.manage_camera and self.camera_started_by_detector:
            self.cam.terminate()
            print("% ArucoDetector:: Stopped")
    
    def set_target_id(self, target_id):
        self.target_id = target_id
    
    def parse_arguments(self):
        """Set up command-line argument parser"""
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", metavar="PATH",  help="Path to video file (optional, default: Pi camera)")
        ap.add_argument("-s", "--stream", action="store_true", help="Serve MJPEG stream in browser")
        ap.add_argument("-l", "--local", action="store_true", help="Show detections in local OpenCV window")
        ap.add_argument("-t", "--target-id", type=int, default=None, metavar="ID", help="Detect only this ArUco marker ID (default: detect all)")
        ap.add_argument("-ch","--camera-host", default="localhost", metavar="IP", help="Host/IP of existing camera stream server (default: localhost) or pass in 10.197.218.199 if running on PC")
        ap.add_argument("-cp","--camera-port", type=int, default=7123, metavar="PORT", help="Port of existing camera stream server (default: 7123)")
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
    
    def detect_aruco(self, detector, camera_matrix, dist_coeffs, target_id=None):
        ok, frame, timestamp = self.cam.getImage()
        
        if not ok or frame is None:
            self.frame_fail_count +=1
            if self.frame_fail_count == 1 or self.frame_fail_count % self.fail_log_iteration == 0:
                print("% Unable to get frame from camera")
            return None, {}
        
        if self.frame_fail_count > 0:
            print("% Camera recovered after {self.frame_fail_count} failed frames")
            self.frame_fail_count = 0

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

                if current_id not in self.MARKER_SIZES:
                    print("Did not find the marker id in the defined list")
                    continue
            
                _marker_size = self.MARKER_SIZES[current_id]
                obj_points = self.get_object_points(_marker_size)
                image_points = marker_corners[0].astype(np.float32)

                """
                solvePnP estimates the pose of a 3D object given its 3D points and corresponding 2D image points.
                It returns the rotation vector (rvec) and translation vector (tvec).
                """
                retval, rvec, tvec = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)
                if not retval:
                    continue
                
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                
                # Extract the translation vector components and Euclidean distance
                x, y, z = tvec.flatten()
                distance = float(np.linalg.norm(tvec))
                #logger.info(f"% Marker {current_id}: distance={distance:.3f} m")

                # Pixel center of the marker corners in the image (used by Nav for rotation)
                pixel_x = float(np.mean(image_points[:, 0]))
                pixel_y = float(np.mean(image_points[:, 1]))

                # NOTE: SHOULD USE THIS FOR FINAL !!!!!!!!!!!!!!!!!!!
                detected_markers[current_id] = {
                    "x": float(pixel_x),        # for testing in pixel
                    "y": float(pixel_y),        # for testing in pixel
                    "z": float(z),        # tvec z — forward depth (meters)
                    "distance": distance, # Euclidean 3D distance (meters)
                    "pixel_x": pixel_x,  # pixel x center of marker (used by Nav)
                    "pixel_y": pixel_y,  # pixel y center of marker
                }
                
                # # NOTE: SHOULD USE THIS FOR THE FINAL !!!!!!!!!!!!!!!!
                # detected_markers[current_id] = {
                #     "x": float(x),        # tvec x — lateral offset (meters)
                #     "y": float(y),        # tvec y — vertical offset (meters)
                #     "z": float(z),        # tvec z — forward depth (meters)
                #     "distance": distance, # Euclidean 3D distance (meters)
                #     "pixel_x": pixel_x,  # pixel x center of marker (used by Nav)
                #     "pixel_y": pixel_y,  # pixel y center of marker
                # }
                
        self.detected_markers = detected_markers
        self.last_frame = frame
        return frame, detected_markers

    def get_target(self):
        if self.detector is None or self.camera_matrix is None or self.dist_coeffs is None:
            return None
        frame, self.detected_markers = self.detect_aruco(
            self.detector,
            self.camera_matrix,
            self.dist_coeffs,
            target_id=self.target_id,
        )
        if not self.detected_markers:
            return None
        
        if self.target_id is not None:
            selected_id = self.target_id
            target = self.detected_markers.get(self.target_id)
        else:
            selected_id = min(self.detected_markers, key=lambda mid: self.detected_markers[mid]["distance"])
            target = self.detected_markers[selected_id]
        if target is None:
            return None
        
        image_width = frame.shape[1] if frame is not None else 820

        # Bearing: horizontal angle from camera forward axis to marker.
        # tvec x is lateral (positive = right), tvec z is depth (forward).
        # Negate x so positive bearing = target is to the left, matching SWorldPoint convention.
        bearing = math.atan2(-target["x"], target["z"])

        return {
            "id":       selected_id,
            "x":        target["pixel_x"],  # pixel x center — required by Nav for rotation
            "y":        target["pixel_y"],  # pixel y center
            "tvec_x":   target["x"],        # lateral offset in meters (camera frame)
            "tvec_y":   target["y"],        # vertical offset in meters (camera frame)
            "z":        target["z"],        # forward depth in meters (camera frame)
            "distance": target["distance"], # Euclidean 3D distance in meters
            "bearing":  bearing,            # horizontal angle to marker (radians, positive = left)
            "image_width": image_width,     # required by Nav for pixel-to-angle conversion
            "valid":    True,
        }
    
    def get_all_targets(self):
        return self.detected_markers.copy()
    
    def run_mjpeg_stream(self, args):
        app = Flask(__name__)
        
        def generate():
            """Generator that yields JPEG frames in multipart format"""
            tick = 0
            while True:
                tick += 1
                
                target = self.get_target()
                frame = self.last_frame
                
                # frame, _ = self.detect_aruco(cam, detector, camera_matrix, dist_coeffs, target_id=target_id)
                if frame is None:
                    break
                
                if tick % 2 == 0:
                    if target is None:
                        print("% No target found")
                    else:
                        print(
                            f"ID: {target['id']}, "
                            f"dist:{target['distance']:.3f}m, "
                            f"(x={target['x']:.1f}, y={target['y']:.1f})"
                        )
                
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
            self.cam.terminate()
            cv2.destroyAllWindows()

    def run_local_gui(self):
        try:
            tick = 0
            while True:
                tick += 1

                target = self.get_target()
                frame = self.last_frame

                if frame is None:
                    break

                if target is None:
                    if tick % 2 == 0:
                        print("% No target found")
                else:
                    if tick % 2 == 0:
                        print(
                            f"ID: {target['id']}, "
                            f"dist:{target['distance']:.3f}m, "
                            f"(x={target['x']:.1f}, y={target['y']:.1f})"
                        )
                cv2.imshow("Aruco Detect", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        finally:
            self.cam.terminate()
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

#aruco = ArucoDetector()


if __name__=='__main__':
    
    aruco = ArucoDetector(cam=cam, gpio=gpio, service=service, manage_camera = True)
    
    args = aruco.parse_arguments()

    # Configure SCam input stream endpoint for standalone runs.
    # SCam builds URL as: http://{service.host}:7123/stream.mjpg
    
    service.host = args["camera_host"]

    if args.get("camera_port", 7123) != 7123:
        logger.warning("% camera_port is set to %s, but scam.py currently uses fixed port 7123", args["camera_port"])
        logger.warning("% Update scam.py if you need a non-7123 camera stream port")

    if aruco.target_id is None:
        aruco.target_id = args.get('target_id')
        
    aruco.set_target_id(aruco.target_id)
    
    aruco.start()
    
    logger.warning("Make sure you have included command arguments")
    print("Make sure you use correct calibration file")

    startup_target = aruco.get_target()
    if startup_target is None:
        print("% Startup get_target: None")
    else:
        print(
            f"% Startup get_target: id={startup_target['id']}, "
            f"dist={startup_target['distance']:.3f}m"
        )
    
    if args.get("stream", False):
        aruco.run_mjpeg_stream(args)
    elif args.get("local", False):
        aruco.run_local_gui()
    else:
        print("% Choose either -s for browser stream or -l for local view")
        
        
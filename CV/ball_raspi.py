from collections import deque       # For efficient queue operations to store tracked points
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
CIRCULARITY_THRESHOLD = 0.65    # Minimum circularity to accept a contour as a ball (0-1)
BUFFER_SIZE = 32                # Number of past positions to store for the tracking trail
args_global = {}                # Global copy of args (used inside create_red_mask)

# --- HSV color ranges for red detection ---
# Red wraps around the HSV hue wheel, so two ranges are needed
red_lower1 = (0, 245, 150)      # Lower red: H(0-10),   S(min 245), V(min 150)
red_upper1 = (10, 255, 255)     # Lower red: H(0-10),   S(max),     V(max)
red_lower2 = (170, 245, 150)    # Upper red: H(170-180), S(min 245), V(min 150)
red_upper2 = (180, 255, 255)    # Upper red: H(170-180), S(max),     V(max)


def parse_arguments():
    """Set up command-line argument parser"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", metavar="PATH",  help="Path to video file (optional, default: Pi camera)")
    ap.add_argument("-s", "--stream", action="store_true", help="Serve MJPEG stream in browser instead of cv2.imshow")
    ap.add_argument("-H", "--host", default="0.0.0.0", metavar="IP",  help="Host/IP to bind the MJPEG server        (default: 0.0.0.0)")
    ap.add_argument("-p", "--port", type=int, default=5000,   metavar="PORT", help="Port to serve the MJPEG stream on       (default: 5000)")
    ap.add_argument("-q", "--jpeg-quality", type=int, default=70, metavar="1-100",  help="JPEG compression quality for streaming  (default: 70)")
    ap.add_argument("-m", "--mask", action="store_true", help="Show red mask: side-by-side in browser, window in GUI")
    args = vars(ap.parse_args())
    args["buffer"] = BUFFER_SIZE    # Fixed buffer size, not exposed as CLI argument
    return args


def initialize_camera(args):
    """Initialize Picamera2 and tracking variables"""
    
    # Create a deque to store the last N ball positions for the trail
    pts = deque(maxlen=args["buffer"])
    
    # Configure and start the Pi camera
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    
    # Optional: use a video file instead of the live camera
    vs = None
    if args.get("video", False):
        vs = cv2.VideoCapture(args["video"])
    
    # Allow the camera sensor to warm up
    time.sleep(2.0)
    
    return picam2, vs, pts


def create_red_mask(hsv):
    """Create a binary mask isolating red pixels from an HSV frame"""
    
    # Create separate masks for the two red hue ranges
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)    # H: 0-10
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)    # H: 170-180
    mask = cv2.bitwise_or(mask1, mask2)                 # Combine both ranges
    
    # Morphological operations to clean up the mask
    mask = cv2.erode(mask, None, iterations=2)          # Remove small noise blobs
    mask = cv2.dilate(mask, None, iterations=2)         # Restore object size after erosion
    
    # Only show the mask window in local GUI mode (not when streaming)
    if not args_global.get("stream", False) and args_global.get("mask", False):
        cv2.imshow("Mask", mask)
    
    return mask


def find_ball_contour(mask):
    """Find the largest circular contour in the mask"""
    
    # Find all external contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # No contours found
    if len(cnts) == 0: 
        return None, None, None
    
    # Use the largest contour (most likely the ball)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    # Avoid division by zero
    if perimeter == 0: 
        return None, None, None
    
    # Circularity: 1.0 = perfect circle, lower = less circular
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Reject contours that are not circular enough
    if circularity < CIRCULARITY_THRESHOLD: 
        return None, None, None
    
    return c, area, circularity


def calculate_ball_position(contour):
    """Calculate ball center (via moments) and bounding circle (via minEnclosingCircle)"""
    
    # Get the minimum enclosing circle center and radius
    ((x, y), radius) = cv2.minEnclosingCircle(contour)

    # Use image moments for a more accurate centroid
    M = cv2.moments(contour)
    if M['m00'] == 0:
        center = (int(x), int(y))       # Fallback to circle center
    else:
        center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))  # Centroid
    
    return center, (int(x), int(y)), int(radius)


def draw_ball_detection(frame, contour_center, circle_center, radius, circularity):
    """Draw detection overlay: bounding circle, centroid dot and coordinates"""
    
    # Yellow circle around the ball
    cv2.circle(frame, circle_center, radius, (0, 255, 255), 2)
    
    # Red dot at the centroid
    cv2.circle(frame, contour_center, 3, (0, 0, 255), -1)
    
    # X/Y coordinates in the top-left corner
    cv2.putText(frame, f"x: {circle_center[0]:.2f}, y: {circle_center[1]:.2f}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    logger.info(f"Ball detected at (x: {circle_center[0]}, y: {circle_center[1]}) | Radius: {radius}")


def draw_tracking_trail(frame, pts, buffer_size):
    """Draw a fading trail of the ball's past positions"""
    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        # Thickness decreases as we go further back in history
        thickness = int(np.sqrt(buffer_size / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0, 255, 0), thickness)


def detect_ball(picam2, vs, pts, args):
    """Capture a frame, detect the ball, draw overlays and return the annotated frame"""
    global args_global
    args_global = args

    # --- Capture frame ---
    if args.get("video", False):
        ret, frame = vs.read()
        if not ret:
            return None     # End of video file
    else:
        frame = picam2.capture_array()  # Live camera frame
    
    if frame is None:
        return None

    # --- Preprocessing ---
    frame = imutils.resize(frame, width=600)            # Resize for consistent processing
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)     # Blur to reduce noise
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)     # Convert to HSV for color detection
    
    # --- Detection ---
    mask = create_red_mask(hsv)
    contour, area, circularity = find_ball_contour(mask)
    contour_center = None
    
    if contour is not None:
        contour_center, circle_center, radius = calculate_ball_position(contour)
        if radius > 10:     # Ignore very small detections
            draw_ball_detection(frame, contour_center, circle_center, radius, circularity)
    
    # --- Tracking trail ---
    pts.appendleft(contour_center)      # Add current position to trail history
    draw_tracking_trail(frame, pts, args['buffer'])

    # --- Stream mode: optionally show mask side by side ---
    if args.get("stream", False) and args.get("mask", False):
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)               # Mask needs 3 channels to stack
        mask_bgr = cv2.resize(mask_bgr, (frame.shape[1], frame.shape[0]))
        cv2.putText(frame,    "Camera", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(mask_bgr, "Mask",   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return np.hstack([frame, mask_bgr])     # Stack side by side

    return frame


def cleanup_resources(picam2, vs, args):
    """Release all camera and window resources"""
    if args.get("video", False) and vs is not None:
        vs.release()        # Release video file
    picam2.stop()           # Stop Pi camera
    cv2.destroyAllWindows() # Close all cv2 windows


def run_mjpeg_stream(picam2, vs, points, args):
    """Start a Flask web server and serve the video as an MJPEG stream"""
    app = Flask(__name__)

    def generate():
        """Generator that yields JPEG frames in multipart format"""
        while True:
            frame = detect_ball(picam2, vs, points, args)
            if frame is None:
                break

            # Encode frame as JPEG to reduce bandwidth
            ok, buffer = cv2.imencode(
                '.jpg',
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), args['jpeg_quality']]
            )
            if not ok:
                continue

            # Yield frame in MJPEG multipart format
            yield (
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
        # # Log both local and network URLs for easy access
        # local_ip = socket.gethostbyname(socket.gethostname())
        # logger.info(f"  → Local:   http://127.0.0.1:{args['port']}")
        # logger.info(f"  → Network: http://{local_ip}:{args['port']}\n")
        app.run(host=args['host'],
                port=args['port'],
                threaded=True)      # threaded=True allows multiple browser clients
    finally:
        cleanup_resources(picam2, vs, args)


def run_local_gui(picam2, vs, points, args):
    """Run the detector with a local cv2.imshow window"""
    try:
        while True:
            frame = detect_ball(picam2, vs, points, args)

            if frame is None:
                break

            cv2.imshow("Frame", frame)

            # Press 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cleanup_resources(picam2, vs, args)


if __name__ == '__main__':

    args = parse_arguments()
    picam2, video_stream, points = initialize_camera(args)

    # Choose between browser stream or local GUI
    if args.get("stream", False):
        run_mjpeg_stream(picam2, video_stream, points, args)
    else:
        run_local_gui(picam2, video_stream, points, args)






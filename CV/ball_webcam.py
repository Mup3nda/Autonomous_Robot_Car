from collections import deque  # For efficient queue operations to store tracked points
from imutils.video import VideoStream  # Threaded video stream for webcam
import numpy as np  
import argparse  # Command-line argument parsing
import cv2  
import imutils  # Convenience functions for OpenCV
import time  
from flask import Flask, Response

CIRCULARITY_THRESHOLD = 0.65  # Changed from 0.2 to be more strict
BUFFER_SIZE = 32
args_global = {}

# WORKING
red_lower1 = (0, 170, 40)      # Lower red: H(0-10), S(min 170), V(min 40)
red_upper1 = (10, 255, 255)     # H(0-10), S(max), V(max)
red_lower2 = (170, 170, 40)    # Upper red: H(170-180), S(min 170), V(min 40)
red_upper2 = (180, 255, 255)    # H(170-180), S(max), V(max)

def parse_arguments():
    """Set up command-line argument parser"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-v","--video", 
                    help="add path to video (optional)")
    ap.add_argument("--stream", action="store_true",
                    help="Serve a live MJPEG stream in a browser (no cv2.imshow)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="Host/IP to bind the MJPEG server (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=5000,
                    help="Port to serve the MJPEG stream on (default: 5000)")
    ap.add_argument("--jpeg-quality", type=int, default=70,
                    help="JPEG quality for MJPEG stream (1-100, default: 70)")
    ap.add_argument("--mask", action="store_true",
                    help="Show the red mask alongside the camera feed")
    args = vars(ap.parse_args())
    args["buffer"] = BUFFER_SIZE
    return args

def initialize_camera(args):
    """Initialize video stream and tracking variables"""
    
    # Initialize tracking variables
    pts = deque(maxlen=args["buffer"])  # Queue to store last N points (max 32)

    # Initialize video source
    if not args.get("video", False):
        vs = VideoStream(src=0).start()  # Use webcam (threaded)
    else:
        vs = cv2.VideoCapture(args["video"])  # Use video file
        
    time.sleep(2.0)
    
    return vs, pts

def create_red_mask(hsv):
    """Creating a mask for Red HSV"""
    
    # Create masks for red color detection
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)  # Detect lower red range
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)  # Detect upper red range
    mask = cv2.bitwise_or(mask1, mask2)  # Combine both masks
    
    # Morphological operations to clean mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)
    
    # Only show cv2 mask window in local GUI mode
    if not args_global.get("stream", False) and args_global.get("mask", False):
        cv2.imshow("Mask", mask)
    
    return mask
    
def find_ball_contour(mask):  # Fixed typo: counter -> contour
    """Find and validate ball contour"""
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0: 
        return None, None, None
    
    # Find largest contour
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    # To not divide by zero
    if perimeter == 0: 
        return None, None, None
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    if circularity < CIRCULARITY_THRESHOLD: 
        return None, None, None
    
    return c, area, circularity

def calculate_ball_position(contour):
    """Calculate ball center and radius"""
    ((x, y), radius) = cv2.minEnclosingCircle(contour)

    M = cv2.moments(contour)  # Pixel intensity
    # Avoid division by zero
    if M['m00'] == 0:
        center = (int(x), int(y))
    else:
        center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))  # (x, y) centroid
    
    return center, (int(x), int(y)), int(radius)

def draw_ball_detection(frame, contour_center, circle_center, radius, circularity):
    """Draw detection markers on frame"""
    # Draw yellow circle around detected ball
    cv2.circle(frame, circle_center, radius, (0, 255, 255), 2)  # Fixed: added comma
    # Draw red dot at centroid
    cv2.circle(frame, contour_center, 3, (0, 0, 255), -1)
    
    cv2.putText(frame, f"x: {circle_center[0]:.2f}, y: {circle_center[1]:.2f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2)

def draw_tracking_trail(frame, pts, buffer_size):  # Fixed typo: tail -> trail
    """Draw tracking trail"""
    for i in range(1, len(pts)):  # Fixed: was range(i, ...) should be range(1, ...)
        if pts[i-1] is None or pts[i] is None:
            continue
        
        # Calculate line thickness (thicker for recent points)
        thickness = int(np.sqrt(buffer_size/float(i+1)) * 2.5)  # Fixed: added closing parenthesis
        cv2.line(frame, pts[i-1], pts[i], (0, 255, 0), thickness)

def detect_ball(vs, pts, args):
    """Main ball detection function"""
    global args_global
    args_global = args

    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    
    if frame is None:
        return None

    # Preprocessing
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Create mask
    mask = create_red_mask(hsv)

    # Find contour
    contour, area, circularity = find_ball_contour(mask)
    contour_center = None
    
    # Process contour if found
    if contour is not None:
        contour_center, circle_center, radius = calculate_ball_position(contour)
        if radius > 10:
            draw_ball_detection(frame, contour_center, circle_center, radius, circularity)
    
    # Update tracking points
    pts.appendleft(contour_center)
    
    # Draw trail
    draw_tracking_trail(frame, pts, args['buffer'])

    # If streaming, optionally combine frame + mask side by side
    if args.get("stream", False):
        if args.get("mask", False):
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_bgr = cv2.resize(mask_bgr, (frame.shape[1], frame.shape[0]))

            # Add labels
            cv2.putText(frame,    "Camera", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(mask_bgr, "Mask",   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return np.hstack([frame, mask_bgr])

    return frame

def cleanup_resources(vs, args):
    """Release video resources"""
    if not args.get("video", False):
        vs.stop()  # Stop threaded video stream
    else:
        vs.release()  # Release video file
        
    cv2.destroyAllWindows()

def run_mjpeg_stream(video_stream, points, args):
    """Run MJPEG streaming mode - view in browser"""
    app = Flask(__name__)

    def generate():
        while True:
            frame = detect_ball(video_stream, points, args)
            if frame is None:
                break

            # Encode as JPEG (compression reduces bandwidth)
            ok, buffer = cv2.imencode(
                '.jpg',
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.get('jpeg_quality', 70))]
            )
            if not ok:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )

    @app.route('/')
    def index():
        return '<html><body><h3>MJPEG stream</h3><img src="/video" /></body></html>'

    @app.route('/video')
    def video():
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    try:
        app.run(host=str(args.get('host', '0.0.0.0')),
                port=int(args.get('port', 5000)),
                threaded=True)
    finally:
        cleanup_resources(video_stream, args)


def run_local_gui(video_stream, points, args):
    """Run local GUI mode - view with cv2.imshow"""
    try:
        while True:
            frame = detect_ball(video_stream, points, args)

            if frame is None:
                break

            cv2.imshow("Frame", frame)

            # Check for 'q' key press to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cleanup_resources(video_stream, args)


if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_arguments()

    # Initialize camera and tracking
    video_stream, points = initialize_camera(args)

    # MJPEG streaming mode (view in browser)
    if args.get("stream", False):
        run_mjpeg_stream(video_stream, points, args)
    # Local GUI mode (cv2.imshow)
    else:
        run_local_gui(video_stream, points, args)

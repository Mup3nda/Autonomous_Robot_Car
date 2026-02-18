from collections import deque  # For efficient queue operations to store tracked points
from imutils.video import VideoStream  # Threaded video stream for webcam
import numpy as np  
import argparse  # Command-line argument parsing
import cv2  
import imutils  # Convenience functions for OpenCV
import time  
from picamera2 import Picamera2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CIRCULARITY_THRESHOLD = 0.65  # Minimum circularity to accept as ball

# Define HSV color ranges for red detection
red_lower1 = (0, 245, 150)      # Lower red: H(0-10), S(min 245), V(min 150)
red_upper1 = (10, 255, 255)     # H(0-10), S(max), V(max)
red_lower2 = (170, 245, 150)    # Upper red: H(170-180), S(min 245), V(min 150)
red_upper2 = (180, 255, 255)    # H(170-180), S(max), V(max)


def parse_arguments():
    """Set up command-line argument parser"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", 
                    help="add path to video (optional)")
    ap.add_argument("-b", "--buffer", type=int, default=32,
                    help="add buffer size - max number of tracked points")
    args = vars(ap.parse_args())  # Convert to dictionary for easy access
    return args


def initialize_camera(args):
    """Initialize Picamera2 and tracking variables"""
    
    # Initialize tracking variables
    pts = deque(maxlen=args["buffer"])  # Queue to store last N points (max 32)
    
    # Initialize Picamera2
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    
    # Initialize video source (for optional video file)
    vs = None
    if args.get("video", False):
        vs = cv2.VideoCapture(args["video"])
    
    time.sleep(2.0)  # Allow camera to warm up
    
    return picam2, vs, pts


def create_red_mask(hsv):
    """Creating a mask for Red HSV"""
    
    # Create masks for red color detection
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)  # Detect lower red range
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)  # Detect upper red range
    mask = cv2.bitwise_or(mask1, mask2)  # Combine both masks
    
    # Morphological operations to clean mask
    mask = cv2.erode(mask, None, iterations=2)   # Remove small noise blobs
    mask = cv2.dilate(mask, None, iterations=2)  # Restore object size
    
    cv2.imshow("Mask", mask)
    
    return mask


def find_ball_contour(mask):
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
    cv2.circle(frame, circle_center, radius, (0, 255, 255), 2)
    # Draw red dot at centroid
    cv2.circle(frame, contour_center, 3, (0, 0, 255), -1)
    
    # Display circularity and radius
    cv2.putText(frame, f"C:{circularity:.2f} R:{radius}",
                (circle_center[0]-50, circle_center[1] - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Log detection
    logger.info(f"Ball detected at (x: {circle_center[0]}, y: {circle_center[1]}) - Radius: {radius}")


def draw_tracking_trail(frame, pts, buffer_size):
    """Draw tracking trail"""
    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        
        # Calculate line thickness (thicker for recent points)
        thickness = int(np.sqrt(buffer_size/float(i+1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0, 255, 0), thickness)


def detect_ball(picam2, vs, pts, args):
    """Main ball detection function"""
    # Read frame from video source
    if args.get("video", False):
        ret, frame = vs.read()
        if not ret:
            return None
    else:
        frame = picam2.capture_array()
    
    if frame is None:
        return None

    # Preprocessing
    frame = imutils.resize(frame, width=600)  # Resize for faster processing
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)  # Reduce noise
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    
    # Create mask
    mask = create_red_mask(hsv)
    
    # Find contour
    contour, area, circularity = find_ball_contour(mask)
    contour_center = None
    
    # Process contour if found
    if contour is not None:
        contour_center, circle_center, radius = calculate_ball_position(contour)
        
        # Only draw if radius is large enough
        if radius > 10:
            draw_ball_detection(frame, contour_center, circle_center, radius, circularity)
    
    # Update tracking points
    pts.appendleft(contour_center)
    
    # Draw trail
    draw_tracking_trail(frame, pts, args['buffer'])
        
    return frame


def cleanup_resources(picam2, vs, args):
    """Release video resources"""
    if args.get("video", False) and vs is not None:
        vs.release()  # Release video file
        
    picam2.stop()  # Stop Picamera2
    cv2.destroyAllWindows()


if __name__ == '__main__':
    
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize camera and tracking
    picam2, video_stream, points = initialize_camera(args)
    
    # Main loop
    while True:
        frame = detect_ball(picam2, video_stream, points, args)
        
        if frame is None:
            break
        
        cv2.imshow("Frame", frame)
        
        # Check for 'q' key press to exit
        key = cv2.waitKey(1) & 0xFF  # Wait 1ms for key press
        if key == ord('q'):
            break

    # Cleanup
    cleanup_resources(picam2, video_stream, args)






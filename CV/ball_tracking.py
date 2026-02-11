
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

# Set up command-line argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video", 
                help="add path to video (optional)")
ap.add_argument("-b", "--buffer", type=int, default=32,
                help="add buffer size - max number of tracked points")
args = vars(ap.parse_args())  # Convert to dictionary for easy access

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(camera_config)
picam2.start()

# Define HSV color ranges for red detection
# Red wraps around the HSV spectrum, so we need two ranges
red_lower1 = (0, 245, 150)      # Lower red: H(0-10), S(min), V(min)
red_upper1 = (10, 255, 255)     # H(0-10), S(max), V(max)
red_lower2 = (170, 245, 150)    # Upper red: H(170-180), S(min), V(min)
red_upper2 = (180, 255, 255)    # H(170-180), S(max), V(max)


# Initialize tracking variables
pts = deque(maxlen=args["buffer"])  # Queue to store last N points (max 32)
counter = 0  # Frame counter (currently unused)
(dx, dy) = (0,0)  # Direction deltas (currently unused)
direction = " "  # Movement direction (currently unused)

# Initialize video source
if not args.get("video", False):
    vs = VideoStream(src=0).start()  # Use webcam (threaded)
else:
    vs = cv2.VideoCapture(args["video"])  # Use video file
    
time.sleep(2.0) 

# Main processing loop
while True:
    # Read frame from video source
    if  args.get("video", False):
        ret, frame = vs.read()
        if not ret:
            break
    else:
        frame = picam2.capture_array()
    
    # Handle different return formats (VideoStream vs VideoCapture)
    frame = frame[1] if args.get("video", False) else frame
    
    # Preprocessing
    frame = imutils.resize(frame, width=600)  # Resize for faster processing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # NOT NECCASSARY TO CONVERT TO BRG - WON'T WORK
    blurred = cv2.GaussianBlur(frame, (11,11), 0)  # Reduce noise
    #blurred = cv2.GaussianBlur(frame_bgr, (11,11), 0)  # Reduce noise
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV color space
    
    # Create masks for red color detection
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)  # Detect lower red range
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)  # Detect upper red range
    mask = cv2.bitwise_or(mask1, mask2)  # Combine both masks
    mask = cv2.erode(mask, None, iterations=2)  # Remove small noise blobs
    mask = cv2.dilate(mask, None, iterations=2)  # Restore object size
    
    cv2.imshow("Mask",mask)
    
    
    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)  # Handle different OpenCV versions
    center = None  # Initialize center point
    
    # Process detected contours
    if len(cnts) > 0:
        # Find the largest contour (assume it's the ball)
        c = max(cnts, key=cv2.contourArea)
        
        # Calculate minimum enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        # Calculate centroid using image moments
        M = cv2.moments(c)
        center = (int(M['m10']/M['m00']), int(M['m01']/M["m00"]))  # (x, y) centroid
        
        # Only proceed if radius is large enough (filter small detections)
        if radius > 10:
            # Draw yellow circle around detected object
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
            # Draw red dot at centroid
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            logger.info(f" Ball detected at (x: {x:.2f}, y: {y:.2f}) - Radius: {radius:.2f}")
    
    # Update tracking queue (add most recent point at left)
    pts.appendleft(center)
    
    # Draw tracking trail
    for i in range(1, len(pts)):
        # Skip if either point is None
        if pts[i-1] is None or pts[i] is None:
            continue
        
        # Calculate line thickness (thicker for recent points)
        thickness = int(np.sqrt(args["buffer"]/float(i+1)) * 2.5)
        
        # Draw green line connecting consecutive points
        cv2.line(frame, pts[i-1], pts[i], (0, 255, 0), thickness)
    
    cv2.imshow("Frame", frame)
    
    # Check for 'q' key press to exit
    key = cv2.waitKey(1) & 0xFF  # Wait 1ms for key press
    if key == ord('q'):
        break

if not args.get("video", False):
    vs.stop()  # Stop threaded video stream
else:
    vs.release()  # Release video file

picam2.stop()
cv2.destroyAllWindows()  






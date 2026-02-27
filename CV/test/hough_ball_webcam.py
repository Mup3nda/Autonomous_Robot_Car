
from collections import deque  # For efficient queue operations to store tracked points
from imutils.video import VideoStream  # Threaded video stream for webcam
import numpy as np  
import argparse  # Command-line argument parsing
import cv2  
import imutils  # Convenience functions for OpenCV
import time  
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_ball(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    red_lower1 = np.array([0, 200, 120])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 200, 120])
    red_upper2 = np.array([180, 255, 255])
    
    
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up the mask with morphological operations
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    blur = cv2.GaussianBlur(mask, (11,11), sigmaX=10, sigmaY=10)
    
    cv2.imshow("BLur Frame", blur)
    
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100, #5000
        param1=50,
        param2=30, #50
        minRadius=20,
        maxRadius=200 #100
    )
    
    if circles is not None:
        circles = np.uint16(np.round(circles))
        for (x, y, r) in circles[0, : ]:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 10)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
    return frame
 
if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open the camera")

while True:
    ret, frame = cap.read()
    output_frame = detect_ball(frame)
    cv2.imshow("Detected", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    
cap.release()
cv2.destroyAllWindows()
    

        
    




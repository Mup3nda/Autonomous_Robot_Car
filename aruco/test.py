import cv2
import os
from picamera2 import Picamera2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAVE_DIR = ""
NUM_IMAGES = 15
CHESSBOARD_SIZE = (9,7)

os.makedirs(SAVE_DIR, exist_ok=True)

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
picam2.start()

count = 0

while True:
    frame = picam2.capture_array()
    display = frame.copy
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    if found:
        cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
        cv2.putText(display, "Press SPACE to save", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.imshow("Picamera2", display)
    key = cv2.await(1)
    
    if key==32 and found:
        filename = os.path.join(SAVE_DIR, f"{count+1:.02d}}.jpg")
        cv2.imwrite(filename, frame)
        logger.info("Image was saved")
        
        count +=1
        
        if count >= NUM_IMAGES
            break
        
    if key==27:
        break
    logger.info("Pressed ESC, quiting the program")
    
    picam2.stop()
    cv2.destroyAllWindows()
        
    


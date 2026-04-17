import cv2
import os

raspi = False

# if raspi:
#     from picamera2 import Picamera2

SAVE_DIR = "calib_images/usb_cam_820_616"
NUM_IMAGES = 30  
CHESSBOARD_SIZE = (8, 6)
# Chessboard settings
#CHESSBOARD_SIZE = (7, 7) #(8, 6)
os.makedirs(SAVE_DIR, exist_ok=True)

# picam2 = Picamera2()
# camera_config = picam2.create_preview_configuration(
#     main={"size": (640, 480), "format": "RGB888"}
# )
# picam2.configure(camera_config)
# picam2.start()  # ← lowercase 's'


#cap = cv2.VideoCapture(f'http://10.197.218.199:7123/stream.mjpg')

if not raspi:
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 820)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 616)
    #cap = cv2.VideoCapture(f'http://0.0.0.0:7124/usb_camera')

count = 0
while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape # (1080,1920)
    
    print(f"Frame size ({h, w})")

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
        cv2.putText(display, f"Press SPACE to save ({count}/{NUM_IMAGES})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No chessboard found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    if key == 32 and found:  # SPACE
        filename = os.path.join(SAVE_DIR, f"img_{count:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename} ({count+1}/{NUM_IMAGES})")
        count += 1
        if count >= NUM_IMAGES:
            break

    elif key == ord('q'):
        break

if raspi:
    cap.stop()
else:
    cap.release() 

cv2.destroyAllWindows()
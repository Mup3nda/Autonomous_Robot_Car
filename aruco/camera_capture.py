import cv2
import os
from picamera2 import Picamera2

SAVE_DIR = "calib_images"
NUM_IMAGES = 15  
CHESSBOARD_SIZE = (8, 6)

os.makedirs(SAVE_DIR, exist_ok=True)

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(camera_config)
picam2.start()  # ← lowercase 's'

count = 0

while True:
    frame = picam2.capture_array()  # ← Read from Pi camera, not cap.read()

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # ← RGB2GRAY (Picamera2 gives RGB)

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
        cv2.putText(display, f"Press SPACE to save ({count}/{NUM_IMAGES})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No chessboard found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Picamera2", display)
    key = cv2.waitKey(1) & 0xFF

    if key == 32 and found:  # SPACE
        filename = os.path.join(SAVE_DIR, f"img_{count:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename} ({count+1}/{NUM_IMAGES})")
        count += 1
        if count >= NUM_IMAGES:
            break

    elif key == 27:  # ESC
        break

picam2.stop()  # ← Stop Pi camera on exit
cv2.destroyAllWindows()
import cv2
import time

# ⭐ CHANGE THIS to your REAL stream endpoint
STREAM_URL = "http://10.197.218.11:7123/stream.mjpg" 

OUTPUT_FILE = "pi_recording_moving_ramp.mp4"
RECORD_SECONDS = 60

# Open stream
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("ERROR: Could not open camera stream")
    exit()

# Get frame size from stream
ret, frame = cap.read()
if not ret:
    print("ERROR: Could not read first frame")
    exit()

H, W = frame.shape[:2]

# Video writer

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
if fps == 0 or fps is None:
    fps = 10  

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 5, (W, H))

print("Recording started...")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame lost")
        break

    out.write(frame)

    cv2.imshow("Recording...", frame)

    # Stop after X seconds
    if time.time() - start_time > RECORD_SECONDS:
        break

    if cv2.waitKey(1) == 27:
        break

print("Recording finished. Saved as:", OUTPUT_FILE)

cap.release()
out.release()
cv2.destroyAllWindows()
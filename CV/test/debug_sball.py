import cv2
import numpy as np
import time
from collections import deque
from sball_saray import SBall   


# -----------------------------
# Fake service so robot is not required
# -----------------------------
class DummyService:
    stop = False
    def send(self, topic, msg):
        pass


# -----------------------------a
# Initialize
# -----------------------------
cap = cv2.VideoCapture(f'http://127.0.0.1:7124/usb_stream.mjpg')

pts = deque(maxlen=32)

# Create ball object WITHOUT robot
ball = SBall(cam=None, gpio=None, service=DummyService())

# Choose color here
ball.set_detection_color("red")
#ball.set_detection_color("orange")
#ball.set_detection_color("blue")
#ball.set_detection_color("white")
# ball.set_detection_color("all")


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    ret, frame = cap.read()
    #frame = imutils.resize(frame, width=600)

    result = ball.debug_detect_only(frame)

    # Draw center line
    H, W = frame.shape[:2]
    cv2.line(frame, (W//2, 0), (W//2, H), (255, 0, 0), 2)

    if result["valid"]:

        x = result["x"]
        y = result["y"]
        radius = result["radius"]
        color = result["color"]

        # Draw ball
        cv2.circle(frame, (x, y), radius, (0,255,255), 2)
        cv2.circle(frame, (x, y), 4, (0,0,255), -1)

        cv2.putText(frame, f"LOCKED: {color}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,255), 2)

        pts.appendleft((x, y))
    else:
        pts.appendleft(None)

    # Draw trail
    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0, 255, 0), thickness)

    cv2.imshow("SBall Debug", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

#picam2.stop()
cv2.destroyAllWindows()
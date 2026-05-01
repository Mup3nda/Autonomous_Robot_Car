import cv2
import numpy as np
import imutils
from collections import deque
import time

from sball_saray import SBall


VIDEO_PATH = r"C:\Users\saray\Documents\Master\2_SEM\34755_BDR\project\videos\pi_recording_balls.mp4"

START_COLOR = "red_orange"   # change to: blue / white / all


# ======================================
# Dummy service (robot not required)
# ======================================

class DummyService:
    stop = False
    def send(self, topic, msg):
        pass


# ======================================
# Initialize
# ======================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

ball = SBall(cam=None, gpio=None, service=DummyService())
ball.set_detection_color(START_COLOR)

pts = deque(maxlen=32)

print("Press:")
print("  r → red")
print("  b → blue")
print("  w → white")
print("  a → all")
print("  q → quit")

# ======================================
# MAIN LOOP
# ======================================

while True:

    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    frame = imutils.resize(frame, width=600)

    # Run detection only
    result = ball.debug_detect_only(frame)

    H, W = frame.shape[:2]
    center_x = W // 2

    # Draw center line
    cv2.line(frame, (center_x, 0), (center_x, H), (255, 0, 0), 2)

    forward = 0
    angular = 0

    if result["valid"]:

        x = result["x"]
        y = result["y"]
        radius = result["radius"]
        color = result["color"]

        # Draw detected ball
        cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # Compute control (same as robot would)
        err_x = x - center_x
        angular = -ball.Kp_turn * err_x
        forward = ball.Kp_fwd * (ball.r_target - radius)

        angular = max(min(angular, 1.0), -1.0)
        forward = max(min(forward, 0.5), 0)

        cv2.putText(frame, f"LOCKED: {color}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        pts.appendleft((x, y))

    else:
        pts.appendleft(None)

    # Draw tracking trail
    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0, 255, 0), thickness)

    # Show control values
    text = f"forward={forward:.2f}   angular={angular:.2f}"
    cv2.putText(frame, text,
                (20, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.imshow("SBall Debug Video", frame)

    key = cv2.waitKey(200) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        ball.set_detection_color("red_orange")
    elif key == ord('b'):
        ball.set_detection_color("blue")
    elif key == ord('w'):
        ball.set_detection_color("white")
    elif key == ord('a'):
        ball.set_detection_color("all")

cap.release()
cv2.destroyAllWindows()
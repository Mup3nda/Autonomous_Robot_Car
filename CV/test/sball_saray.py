#/***************************************************************************
#*   Copyright (C) 2026 by DTU
#*   Ball tracking and following module
#***************************************************************************/

from datetime import datetime
import time as t
from threading import Thread
import cv2 as cv
from matplotlib.pyplot import hsv
import numpy as np


class SBall:
    """Ball detection and tracking with selectable color."""

    ##########################################################
    # STATE VARIABLES
    ##########################################################

    ball_x = 0
    ball_y = 0
    ball_radius = 0
    ball_valid = False
    ball_update_cnt = 0
    ball_time = datetime.now()

    image_width = 640
    image_height = 480

    ##########################################################
    # CONTROL PARAMETERS
    ##########################################################

    Kp_turn = 0.004
    Kp_fwd = 0.01
    r_target = 80

    num_balls = 6
    LOCK_DISTANCE = 120

    locked_target = None
    valid_balls = []

    ballCtrl = False
    velocity = 0.2

    running = False
    thread = None
    update_interval = 0.033
    
    DETECTION_PARAMS = {
        "red": {
            "min_area": 350,
            "max_area": 9000,
            "min_circularity": 0.80,
        },
        "blue": {
            "min_area": 350,
            "max_area": 9000,
            "min_circularity": 0.80,
        },
        "white": {
            "min_area": 350,
            "max_area": 9000,
            "min_circularity": 0.80,
        },
        "orange": {
            "min_area": 200,
            "max_area": 9000,
            "min_circularity": 0.7,
        },
    }

    ##########################################################
    # COLOR SELECTION
    ##########################################################

    detection_color = "orange"   # default

    ##########################################################

    def __init__(self, cam, gpio, service):
        self.cam = cam
        self.gpio = gpio
        self.service = service

    ##########################################################
    # SELECT COLOR
    ##########################################################

    def set_detection_color(self, color_name):
        """
        Select which color to detect:
        'red', 'blue', 'white', 'orange', or 'all'
        """
        allowed = ["red", "blue", "white", "orange", "all"]
        if color_name in allowed:
            self.detection_color = color_name
            print(f"% Ball:: Detecting color = {color_name}")
        else:
            print("% Ball:: Invalid color selection")

    ##########################################################

    def setup(self):

        if not self.cam.useCam:
            print("% Ball:: Camera not available")
            return

        print("% Ball:: Starting ball tracking")

        self.running = True
        self.thread = Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()

    ##########################################################

    def _tracking_loop(self):

        while self.running and not self.service.stop:

            start_time = t.time()

            ok, frame, frame_time = self.cam.getImage()
            if not ok:
                t.sleep(self.update_interval)
                continue

            self.detect_ball(frame)

            if self.ballCtrl and self.ball_valid:
                self.followBall()

            elapsed = t.time() - start_time
            if elapsed < self.update_interval:
                t.sleep(self.update_interval - elapsed)

    ##########################################################
    # BALL DETECTION (MULTI COLOR + SELECTION)
    ##########################################################

    def detect_ball(self, frame):

        H, W = frame.shape[:2]
        self.image_width = W
        self.image_height = H

        blurred = cv.GaussianBlur(frame, (11, 11), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        # ---------- COLOR MASKS ----------

        masks = {}

        # RED
        lower_red1 = np.array([0, 120, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 80])
        upper_red2 = np.array([180, 255, 255])
        masks["red"] = cv.inRange(hsv, lower_red1, upper_red1) | \
                              cv.inRange(hsv, lower_red2, upper_red2)
                              
        # orange golf ball
        # ORANGE / YELLOWISH SHINY GOLF BALL

        # Main orange body
        lower_orange1 = np.array([5, 100, 80])
        upper_orange1 = np.array([22, 255, 255])

        # Yellow-orange shiny parts
        lower_orange2 = np.array([22, 60, 140])
        upper_orange2 = np.array([40, 255, 255])

        # Very bright highlight on the ball (low saturation because of glare)
        lower_orange3 = np.array([8, 30, 190])
        upper_orange3 = np.array([45, 170, 255])

        masks["orange"] = (
            cv.inRange(hsv, lower_orange1, upper_orange1) |
            cv.inRange(hsv, lower_orange2, upper_orange2) |
            cv.inRange(hsv, lower_orange3, upper_orange3)
        )

        # BLUE
        lower_blue = np.array([90, 60, 60])
        upper_blue = np.array([135, 255, 255])
        masks["blue"] = cv.inRange(hsv, lower_blue, upper_blue)

        # WHITE
        lower_white = np.array([0, 0, 90])
        upper_white = np.array([180, 60, 255])
        masks["white"] = cv.inRange(hsv, lower_white, upper_white)

        # Morphological cleanup
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13, 13))

        for key in masks:
            masks[key] = cv.morphologyEx(masks[key], cv.MORPH_OPEN, kernel)
            masks[key] = cv.morphologyEx(masks[key], cv.MORPH_CLOSE, kernel2)

        # ---------- Select which colors to process ----------

        if self.detection_color == "all":
            color_list = ["red", "blue", "white", "orange"]
        else:
            color_list = [self.detection_color]

        self.valid_balls = []

        for color_name in color_list:
            mask = masks[color_name]
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for c in contours:
                params = self.DETECTION_PARAMS[color_name]

                area = cv.contourArea(c)
                if area < params["min_area"] or area > params["max_area"]:
                    print(f"Rejected contour for color {color_name} with area {area:.1f}")
                    continue

                perimeter = cv.arcLength(c, True)
                if perimeter == 0:
                    print(f"Rejected contour for color {color_name} with zero perimeter")
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < params["min_circularity"]:
                    print(f"Rejected contour for color {color_name} with low circularity {circularity:.2f}")
                    continue

                (cx, cy), radius = cv.minEnclosingCircle(c)

                self.valid_balls.append((c, cx, cy, radius, color_name))

        self.valid_balls = sorted(
            self.valid_balls,
            key=lambda x: x[3],
            reverse=True
        )[:self.num_balls]

        # ---------- LOCK TARGET ----------

        if len(self.valid_balls) > 0:

            if self.locked_target is None:
                self.locked_target = self.valid_balls[0]
            else:
                _, prev_cx, prev_cy, _, _ = self.locked_target
                best_match = None
                best_dist = 99999

                for b in self.valid_balls:
                    _, cx, cy, _, _ = b
                    dist = np.hypot(cx - prev_cx, cy - prev_cy)

                    if dist < best_dist and dist < self.LOCK_DISTANCE:
                        best_dist = dist
                        best_match = b

                self.locked_target = best_match

        self.ball_valid = self.locked_target is not None

        if self.ball_valid:
            _, cx, cy, radius, _ = self.locked_target
            self.ball_x = int(cx)
            self.ball_y = int(cy)
            self.ball_radius = int(radius)
            self.ball_update_cnt += 1

    ##########################################################
    # FOLLOW CONTROL
    ##########################################################

    def followBall(self):

        center_x = self.image_width // 2
        _, cx, _, radius, color = self.locked_target

        err_x = cx - center_x

        angular = -self.Kp_turn * err_x
        forward = self.Kp_fwd * (self.r_target - radius)

        angular = max(min(angular, 1.0), -1.0)
        forward = max(min(forward, 0.5), 0)

        param = f"rc {forward:.3f} {angular:.3f} {t.time()}"
        self.service.send("robobot/cmd/ti", param)

        if self.ball_update_cnt % 10 == 0:
            print(f"% Ball: LOCKED {color} | fwd={forward:.2f} turn={angular:.2f}")

    ##########################################################

    def ballControl(self, velocity):
        self.velocity = velocity
        self.ballCtrl = velocity > 0.001

    ##########################################################
    
    def debug_detect_only(self, frame):
        """
        Runs detection but does NOT send robot commands.
        Returns detection info for visualization.
        """
        self.detect_ball(frame)

        if self.ball_valid:
            return {
                "valid": True,
                "x": self.ball_x,
                "y": self.ball_y,
                "radius": self.ball_radius,
                "color": self.locked_target[4]
            }
        else:
            return {"valid": False}
    
    
    ##########################################################

    def terminate(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


# Global instance
ball = SBall(None, None, None)
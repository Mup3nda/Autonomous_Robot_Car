#/***************************************************************************
#*   Copyright (C) 2026 by DTU
#*   Hole tracking and following module
#***************************************************************************/

from datetime import datetime
import time as t
from threading import Thread
import cv2 as cv
import numpy as np

from target_detector import TargetDetector


class SHole(TargetDetector):
    """Hole detection and tracking with same interface as SBall."""

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

    ##########################################################
    # DETECTION PARAMETERS
    ##########################################################
    DETECTION_PARAMS = {
        "min_area": 300,
        "max_area": 5000,
        "min_ratio": 0.35,
        "min_axis": 10,
        "max_axis": 200,
        "roi_y_start_ratio": 0.3,
    }

    ##########################################################

    def __init__(self, cam, gpio, service):
        super().__init__()
        self.cam = cam
        self.gpio = gpio
        self.service = service

    ##########################################################

    def setup(self):

        if not self.cam.useCam:
            print("% Hole:: Camera not available")
            return

        print("% Hole:: Starting hole tracking")

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

            # if self.ballCtrl and self.ball_valid:
            #     self.followBall()

            elapsed = t.time() - start_time
            if elapsed < self.update_interval:
                t.sleep(self.update_interval - elapsed)

    ##########################################################
    # HOLE DETECTION
    ##########################################################

    def detect_ball(self, frame):
        """
        Keeps same function name as old module for compatibility.
        Detects the hole as an ellipse and maps it to ball-style output.
        """

        H, W = frame.shape[:2]
        self.image_width = W
        self.image_height = H

        # ROI: lower part where the table/hole appears
        y_start = int(H * self.DETECTION_PARAMS["roi_y_start_ratio"])
        roi = frame[y_start:H, 0:W]

        blurred = cv.GaussianBlur(roi, (5, 5), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        # Brown / beige threshold for hole
        lower = np.array([5, 40, 40])
        upper = np.array([30, 255, 255])
        mask = cv.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Smooth mask
        mask = cv.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        self.valid_balls = []

        for c in contours:
            if len(c) < 5:
                continue

            area = cv.contourArea(c)
            if area < self.DETECTION_PARAMS["min_area"] or area > self.DETECTION_PARAMS["max_area"]:
                continue

            ellipse = cv.fitEllipse(c)
            (cx, cy), (axis1, axis2), angle = ellipse

            if axis1 <= 0 or axis2 <= 0:
                continue

            ratio = min(axis1, axis2) / max(axis1, axis2)
            if ratio < self.DETECTION_PARAMS["min_ratio"]:
                continue

            if (axis1 < self.DETECTION_PARAMS["min_axis"] or
                axis2 < self.DETECTION_PARAMS["min_axis"] or
                axis1 > self.DETECTION_PARAMS["max_axis"] or
                axis2 > self.DETECTION_PARAMS["max_axis"]):
                continue

            # Use half of major axis as radius for compatibility
            major_axis = max(axis1, axis2)
            radius = major_axis / 2.0

            # Convert ROI coordinates to full image coordinates
            full_cx = cx
            full_cy = cy + y_start

            # Score: prefer large and compact ellipses
            score = area * ratio

            self.valid_balls.append((c, full_cx, full_cy, radius, "hole", ellipse, score))

        self.valid_balls = sorted(
            self.valid_balls,
            key=lambda x: x[6],
            reverse=True
        )[:self.num_balls]

        ######################################################
        # LOCK TARGET
        ######################################################
        if len(self.valid_balls) > 0:

            if self.locked_target is None:
                self.locked_target = self.valid_balls[0]
            else:
                _, prev_cx, prev_cy, _, _, _, _ = self.locked_target
                best_match = None
                best_dist = 99999

                for b in self.valid_balls:
                    _, cx, cy, _, _, _, _ = b
                    dist = np.hypot(cx - prev_cx, cy - prev_cy)

                    if dist < best_dist and dist < self.LOCK_DISTANCE:
                        best_dist = dist
                        best_match = b

                if best_match is not None:
                    self.locked_target = best_match
                else:
                    self.locked_target = self.valid_balls[0]
        else:
            self.locked_target = None

        self.ball_valid = self.locked_target is not None

        if self.ball_valid:
            _, cx, cy, radius, _, _, _ = self.locked_target
            self.ball_x = int(cx)
            self.ball_y = int(cy)
            self.ball_radius = int(radius)
            self.ball_update_cnt += 1
            self.ball_time = datetime.now()
            return True
        else:
            self.ball_update_cnt = 0
            return False

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

    ###########################################################

    def get_target_info(self):
        """
        Get the position and size of the locked target.

        Returns:
            tuple: (x, y, radius) if a target is locked, None otherwise
        """
        if self.ball_valid:
            return self.ball_x, self.ball_y, self.ball_radius
        return None

    ##########################################################
    # TARGET DETECTOR INTERFACE IMPLEMENTATION
    ##########################################################

    def start(self):
        """Start the target detection system."""
        self.setup()

    def get_target(self):
        """
        Get the current target information.

        Returns:
            dict: Target information with keys:
                - 'valid': bool
                - 'x': int
                - 'y': int
                - 'radius': int
                - 'distance': float
                - 'color': str
                - 'confidence': int
            Returns None if no target is detected
        """
        if not self.ball_valid:
            return None

        confidence = min(self.ball_update_cnt, 20)

        # Approximate real radius of the hole in meters
        # Adjust this value to your real hole size
        hole_real_radius_m = 0.025

        # Same distance model as the ball code
        focal_length_px = 530.54

        if self.ball_radius > 0:
            distance_m = (hole_real_radius_m * focal_length_px) / self.ball_radius
        else:
            distance_m = 0.0

        return {
            'valid': True,
            'x': self.ball_x,
            'y': self.ball_y,
            'radius': self.ball_radius,
            'distance': distance_m,
            'color': self.locked_target[4] if self.locked_target else 'hole',
            'confidence': confidence
        }

    def stop(self):
        """Stop the target detection system."""
        self.terminate()

    def terminate(self):
        """Terminate the tracking thread and clean up resources."""
        if self.running:
            self.running = False
            if self.thread is not None and self.thread.is_alive():
                self.thread.join(timeout=1.0)
        print("% Hole:: Terminated")

    def is_target_visible(self, min_confidence=1):
        """
        Check if a target is currently visible with sufficient confidence.

        Args:
            min_confidence (int): Minimum confidence level required (0-20)

        Returns:
            bool: True if target is visible with required confidence
        """
        if not self.ball_valid:
            return False

        confidence = min(self.ball_update_cnt, 20)
        return confidence >= min_confidence

    def get_status(self):
        """
        Get detailed status information about the target detector.

        Returns:
            dict: Status information including target data and system state
        """
        target_info = self.get_target()

        return {
            'system_running': self.running,
            'target_detected': self.ball_valid,
            'image_size': (self.image_width, self.image_height),
            'target_info': target_info,
            'update_count': self.ball_update_cnt,
            'last_update_time': self.ball_time.isoformat() if self.ball_time else None
        }

    ##########################################################
    # OPTIONAL DEBUG DRAW
    ##########################################################

    def draw_debug(self, frame):
        output = frame.copy()

        if self.ball_valid and self.locked_target is not None:
            _, cx, cy, radius, _, ellipse, _ = self.locked_target
            (ex, ey), axes, angle = ellipse

            # ellipse was fitted in ROI coordinates, so recompute for full frame
            y_start = int(frame.shape[0] * self.DETECTION_PARAMS["roi_y_start_ratio"])
            full_ellipse = ((ex, ey + y_start), axes, angle)

            cv.ellipse(output, full_ellipse, (0, 255, 0), 2)
            cv.circle(output, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            cv.putText(
                output,
                f"Hole: ({int(cx)}, {int(cy)}) r={int(radius)}",
                (int(cx) + 10, int(cy) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        else:
            cv.putText(
                output,
                "No hole detected",
                (20, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        return output


# Global instance
ball = SHole(None, None, None)
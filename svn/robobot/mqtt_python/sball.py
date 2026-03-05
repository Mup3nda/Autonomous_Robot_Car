#/***************************************************************************
#*   Copyright (C) 2026 by DTU
#*   Ball tracking and following module
#*
#* The MIT License (MIT)  https://mit-license.org/
#*
#* Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#* and associated documentation files (the "Software"), to deal in the Software without restriction,
#* including without limitation the rights to use, copy, modify, merge, publish, distribute,
#* sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
#* is furnished to do so, subject to the following conditions:
#*
#* The above copyright notice and this permission notice shall be included in all copies
#* or substantial portions of the Software.
#*
#* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
#* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#* THE SOFTWARE. */

"""Ball tracking and following module using computer vision.

This module detects and tracks a colored ball in camera frames and provides
automatic ball-following control similar to line following in sedge.py.
"""

from datetime import datetime
import time as t
from threading import Thread
import cv2 as cv
import numpy as np
import imutils

class SBall:
    """Ball detection and tracking with automatic following control."""
    
    # Ball detection state
    ball_x = 0  # Ball X position in image (pixels)
    ball_y = 0  # Ball Y position in image (pixels)
    ball_radius = 0  # Ball radius in pixels (used for distance estimation)
    ball_valid = False  # Whether ball is currently detected
    ball_confidence = 0  # Detection confidence (0-20, similar to line detection)
    ball_update_cnt = 0  # Number of updates received
    ball_time = datetime.now()  # Timestamp of last detection
    
    # Image dimensions
    image_width = 640
    image_height = 480
    
    # Ball color detection parameters (RED ball by default)
    # Red wraps around HSV spectrum, so we need two ranges
    color_lower1 = (0, 245, 150)  # Lower red: H(0-10), S(min), V(min)
    color_upper1 = (10, 255, 255)
    color_lower2 = (170, 245, 150)  # Upper red: H(170-180), S(min), V(min)
    color_upper2 = (180, 255, 255)
    
    # Ball detection thresholds
    min_radius = 10  # Minimum radius to consider valid (pixels)
    min_circularity = 0.7  # Minimum circularity (0-1, 1=perfect circle)
    
    # Following control
    ballCtrl = False  # Whether ball following is active
    velocity = 0.2  # Forward velocity when following
    target_distance = 0.5  # Target distance to maintain (meters)
    
    # P-Lead controller parameters (similar to line following)
    ballKp = 0.002  # Proportional gain for steering
    ballTauZ = 0.8  # Lead time constant (seconds)
    ballTauP = 0.25  # Pole time constant (seconds)
    
    # Lead pre-calculated factors
    tauP2pT = 1.0
    tauP2mT = 0.0
    tauZ2pT = 1.0
    tauZ2mT = 0.0
    
    # Control values
    ballE1 = 0.0  # Old error * Kp (rad/s)
    ballY1 = 0.0  # Old control output (rad/s)
    ballY = 0.0  # Control output (rad/s)
    
    # Thread management
    running = False
    thread = None
    update_interval = 0.033  # ~30fps
    
    # Distance estimation (camera calibration)
    # Assuming known ball size: standard red ball ~7cm diameter
    ball_real_diameter = 0.07  # meters
    focal_length = 600  # Approximate focal length (pixels) - needs calibration
    
    ##########################################################
    
    def __init__(self, cam, gpio, service):
        """Initialize ball tracking module.
        
        Args:
            cam: Camera interface (scam.SCam instance)
            gpio: GPIO interface for environment checks
            service: MQTT service for sending commands
        """
        self.cam = cam
        self.gpio = gpio
        self.service = service
        pass
    
    ##########################################################
    
    def setup(self):
        """Start the ball tracking thread."""
        from uservice import service
        
        if not self.cam.useCam:
            print("% Ball (sball.py):: Camera not available, ball tracking disabled")
            return
        
        print("% Ball (sball.py):: Starting ball tracking")
        
        # Calculate PID factors for initial sample time
        self.PIDrecalculate(self.update_interval)
        
        # Start the tracking thread
        self.running = True
        self.thread = Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()
        
        # Wait for first detection
        loops = 0
        while not service.stop and self.ball_update_cnt == 0:
            t.sleep(0.05)
            loops += 1
            if loops > 20:
                print(f"% Ball (sball.py):: No ball detected after {loops} loops (continuing)")
                break
        
        if self.ball_update_cnt > 0:
            print(f"% Ball (sball.py):: Ball tracking active (after {loops} loops)")
    
    ##########################################################
    
    def _tracking_loop(self):
        """Continuous loop: grab frame → detect → control.
        
        This runs in a separate thread, continuously processing camera frames
        and sending control commands when ball following is enabled.
        """
        while self.running and not self.service.stop:
            start_time = t.time()
            
            # Grab frame from camera
            ok, frame, frame_time = self.cam.getImage()
            if not ok:
                t.sleep(self.update_interval)
                continue
            
            # Run ball detection
            self.detect_ball(frame)
            
            # If tracking enabled, calculate and send steering
            if self.ballCtrl and self.ball_valid:
                self.followBall()
            
            # Maintain consistent loop rate
            elapsed = t.time() - start_time
            if elapsed < self.update_interval:
                t.sleep(self.update_interval - elapsed)
    
    ##########################################################
    
    def detect_ball(self, frame):
        """Detect ball in the given frame using color and shape detection.
        
        Args:
            frame: BGR image from camera
        """
        # Preprocessing
        blurred = cv.GaussianBlur(frame, (11, 11), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        
        # Create masks for red color detection
        mask1 = cv.inRange(hsv, self.color_lower1, self.color_upper1)
        mask2 = cv.inRange(hsv, self.color_lower2, self.color_upper2)
        mask = cv.bitwise_or(mask1, mask2)
        
        # Morphological operations to remove noise
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)
        
        # Find contours
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # Process detected contours
        old_valid = self.ball_valid
        self.ball_valid = False
        
        if len(cnts) > 0:
            # Find the largest contour (assume it's the ball)
            c = max(cnts, key=cv.contourArea)
            area = cv.contourArea(c)
            
            if area > 100:  # Minimum area threshold
                # Calculate minimum enclosing circle
                ((x, y), radius) = cv.minEnclosingCircle(c)
                
                # Check circularity to filter non-ball objects
                perimeter = cv.arcLength(c, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > self.min_circularity and radius > self.min_radius:
                        # Valid ball detected!
                        self.ball_x = int(x)
                        self.ball_y = int(y)
                        self.ball_radius = int(radius)
                        self.ball_valid = True
                        self.ball_time = datetime.now()
                        self.ball_update_cnt += 1
                        
                        # Update confidence (similar to lineValidCnt)
                        if self.ball_confidence < 20:
                            self.ball_confidence += 1
        
        # Decrease confidence if ball lost
        if not self.ball_valid:
            if self.ball_confidence > 0:
                self.ball_confidence -= 1
    
    ##########################################################
    
    def ballControl(self, velocity, target_distance=0.5):
        """Enable or disable automatic ball following.
        
        Args:
            velocity: Forward speed (0.0 to 1.0). Set to 0 to disable tracking.
            target_distance: Target distance to maintain from ball (meters)
        """
        self.velocity = velocity
        self.target_distance = target_distance
        self.ballCtrl = velocity > 0.001
        
        if self.ballCtrl:
            print(f"% Ball (sball.py):: Ball following enabled (v={velocity:.2f}, target={target_distance:.2f}m)")
        else:
            print("% Ball (sball.py):: Ball following disabled")
    
    ##########################################################
    
    def followBall(self):
        """Calculate steering to center ball and approach target distance.
        
        Uses P-Lead controller similar to line following to generate smooth,
        stable steering commands.
        """
        if not self.ball_valid:
            return
        
        # Calculate error: how far from image center (pixels)
        center_x = self.image_width / 2
        e_pixels = center_x - self.ball_x  # Positive = ball is left, turn left
        
        # Normalize error to approximate angle (-1 to +1)
        # Assuming 60-degree FOV, each pixel = ~0.1 degrees
        e_normalized = e_pixels / center_x  # -1 to +1
        
        # Calculate distance error
        current_dist = self.estimated_distance()
        e_distance = current_dist - self.target_distance
        
        # Adjust forward velocity based on distance
        # If too far, speed up; if too close, slow down or stop
        forward = self.velocity + (0.3 * e_distance)
        forward = max(min(forward, 0.5), 0.0)  # Clamp to [0, 0.5]
        
        # P-Lead controller for steering
        u = self.ballKp * e_pixels  # Error times Kp
        
        # Lead filter
        self.ballY = (u * self.tauZ2pT - self.ballE1 * self.tauZ2mT + 
                      self.ballY1 * self.tauP2mT) / self.tauP2pT
        
        # Clamp turn rate
        if self.ballY > 2.0:
            self.ballY = 2.0
        elif self.ballY < -2.0:
            self.ballY = -2.0
        
        # Save old values
        self.ballE1 = u
        self.ballY1 = self.ballY
        
        # Send command to robot
        param = f"rc {forward:.3f} {self.ballY:.3f} {t.time()}"
        self.service.send("robobot/cmd/ti", param)
        
        # Debug print
        if self.ball_update_cnt % 10 == 0:
            print(f"% Ball::followBall: e={e_pixels:.1f}px, dist={current_dist:.2f}m, "
                  f"fwd={forward:.2f}, turn={self.ballY:.2f} -> {param}")
    
    ##########################################################
    
    def estimated_distance(self):
        """Estimate distance to ball based on its apparent size.
        
        Uses pinhole camera model: distance = (real_size * focal_length) / pixel_size
        
        Returns:
            float: Estimated distance to ball in meters
        """
        if self.ball_radius <= 0:
            return 10.0  # Default large distance
        
        # Distance = (real_diameter * focal_length) / (2 * radius_pixels)
        distance = (self.ball_real_diameter * self.focal_length) / (2 * self.ball_radius)
        return distance
    
    ##########################################################
    
    def PIDrecalculate(self, sample_time):
        """Recalculate PID controller factors based on sample time.
        
        Args:
            sample_time: Sample time in seconds
        """
        print(f"% Ball::PIDrecalculate: T={sample_time:.3f} sec")
        self.tauP2pT = self.ballTauP * 2.0 + sample_time
        self.tauP2mT = self.ballTauP * 2.0 - sample_time
        self.tauZ2pT = self.ballTauZ * 2.0 + sample_time
        self.tauZ2mT = self.ballTauZ * 2.0 - sample_time
        
        print(f"%%   Lead: tauZ={self.ballTauZ:.3f} sec, tauP={self.ballTauP:.3f} sec")
        print(f"%%   tauZ2pT={self.tauZ2pT:.4f}, tauZ2mT={self.tauZ2mT:.4f}, "
              f"tauP2pT={self.tauP2pT:.4f}, tauP2mT={self.tauP2mT:.4f}")
    
    ##########################################################
    
    def terminate(self):
        """Stop the tracking thread and clean up."""
        print("% Ball (sball.py):: Stopping ball tracking")
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("% Ball (sball.py):: terminated")
    
    ##########################################################
    
    def print_status(self):
        """Print current ball detection status."""
        if self.ball_valid:
            print(f"% Ball: detected at ({self.ball_x}, {self.ball_y}) "
                  f"radius={self.ball_radius}px, dist={self.estimated_distance():.2f}m, "
                  f"confidence={self.ball_confidence}")
        else:
            print(f"% Ball: not detected (confidence={self.ball_confidence})")


# Global instance (similar to sedge pattern)
ball = SBall(None, None, None)

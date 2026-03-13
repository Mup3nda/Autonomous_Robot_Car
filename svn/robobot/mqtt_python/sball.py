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
    # Multiple colors supported: red, blue, white
    color_detect_mode = "white"  # Which color to track
    
    # RED color detection (wraps around HSV spectrum)
    color_lower1 = (0, 120, 80)   # Lower red: H(0-10), S(min), V(min)
    color_upper1 = (10, 255, 255)
    color_lower2 = (170, 120, 80)  # Upper red: H(170-180), S(min), V(min)
    color_upper2 = (180, 255, 255)
    
    # BLUE color detection
    blue_lower = (90, 60, 60)
    blue_upper = (135, 255, 255)
    
    # WHITE color detection
    white_lower = (0, 0, 90)       # V down (shadow tolerant)
    white_upper = (180, 60, 255)   # Keep S low so it stays "white"
    
    # Ball detection thresholds
    min_area = 350  # Minimum contour area (pixels)
    max_area = 9000  # Maximum contour area
    min_fill_ratio = 0.5  # Minimum (area / bounding_rect_area)
    min_radius = 10  # Minimum radius to consider valid (pixels)
    min_circularity = 0.8  # Minimum circularity (0-1, 1=perfect circle)
    
    # Multi-ball tracking
    locked_target = None  # Currently tracked ball
    lock_distance = 120  # Max pixel distance for target continuity
    
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
    
    # Focal length: can be specified in pixels or converted from mm
    # For Raspberry Pi Camera v2 with 3.68mm sensor width and 640px image width:
    #   focal_length_pixels = (focal_length_mm * 640) / 3.68
    # Example: 3.6mm lens -> focal_length = (3.6 * 640) / 3.68 = 627px
    focal_length = 600  # Approximate focal length (pixels) - needs calibration
    
    ##########################################################
    
    def __init__(self):
        """Initialize ball tracking module.
        
        Module dependencies (cam, gpio, service) are set during setup().
        """
        pass
    
    ##########################################################
    
    def setup(self):
        """Start the ball tracking thread."""
        from uservice import service
        from scam import cam
        from sgpio import gpio
        
        # Set module dependencies
        self.cam = cam
        self.gpio = gpio
        self.service = service
        
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
        
        Supports multiple colors (red, blue, white) and uses advanced filtering:
        - Area thresholding
        - Fill ratio checking
        - Circularity validation
        - Multi-ball tracking with target locking
        
        Args:
            frame: BGR image from camera
        """
        # Preprocessing
        blurred = cv.GaussianBlur(frame, (11, 11), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        
        h, w = frame.shape[:2]
        center_x = w // 2
        
        # Create color masks based on detection mode
        if self.color_detect_mode == "red":
            mask1 = cv.inRange(hsv, self.color_lower1, self.color_upper1)
            mask2 = cv.inRange(hsv, self.color_lower2, self.color_upper2)
            mask = cv.bitwise_or(mask1, mask2)
        elif self.color_detect_mode == "blue":
            mask = cv.inRange(hsv, self.blue_lower, self.blue_upper)
        elif self.color_detect_mode == "white":
            mask = cv.inRange(hsv, self.white_lower, self.white_upper)
        else:
            # Default to red
            mask1 = cv.inRange(hsv, self.color_lower1, self.color_upper1)
            mask2 = cv.inRange(hsv, self.color_lower2, self.color_upper2)
            mask = cv.bitwise_or(mask1, mask2)
        
        # Morphological operations to remove noise
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13, 13))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel2)
        
        # Find contours
        # Handle different OpenCV versions: older returns (img, cnts, hier), newer returns (cnts, hier)
        result = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = result[-2] if len(result) == 3 else result[0]
        
        # Process all detected contours and filter candidates
        valid_balls = []
        old_valid = self.ball_valid
        self.ball_valid = False
        
        if len(cnts) > 0:
            for c in cnts:
                area = cv.contourArea(c)
                
                # Area filtering
                if area < self.min_area or area > self.max_area:
                    continue
                
                # Get bounding rect for fill ratio check
                (xr, yr, wr, hr) = cv.boundingRect(c)
                rect_area = wr * hr
                if rect_area == 0:
                    continue
                    
                fill_ratio = area / rect_area
                if fill_ratio < self.min_fill_ratio:
                    continue
                
                # For white balls, check aspect ratio (should be roughly square)
                if self.color_detect_mode == "white":
                    aspect = wr / float(hr) if hr > 0 else 0
                    if aspect < 0.7 or aspect > 1.5:
                        continue
                
                # Circularity check
                perimeter = cv.arcLength(c, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < self.min_circularity:
                    continue
                
                # Calculate minimum enclosing circle
                ((x, y), radius) = cv.minEnclosingCircle(c)
                
                # Check radius
                if radius < self.min_radius:
                    continue
                
                # Check bounds
                if int(y) >= h or int(x) >= w:
                    continue
                
                # Valid ball candidate!
                valid_balls.append({
                    'x': x,
                    'y': y,
                    'radius': radius,
                    'area': area,
                    'circularity': circularity
                })
        
        # Ball prioritization: sort by radius (size = distance proxy)
        valid_balls.sort(key=lambda b: b['radius'], reverse=True)
        
        # Target locking: maintain continuity across frames
        if len(valid_balls) > 0:
            if self.locked_target is None:
                # No locked target, take the largest (closest) ball
                self.locked_target = valid_balls[0]
            else:
                # Try to match with tracked target
                prev_x = self.locked_target['x']
                prev_y = self.locked_target['y']
                best_match = None
                best_dist = 999999
                
                for ball in valid_balls:
                    dist = np.sqrt((ball['x'] - prev_x)**2 + (ball['y'] - prev_y)**2)
                    
                    if dist < best_dist and dist < self.lock_distance:
                        best_dist = dist
                        best_match = ball
                
                # Update lock to best match, or re-pick if no continuity
                if best_match is not None:
                    self.locked_target = best_match
                else:
                    # Lost tracked target, relock to closest
                    self.locked_target = valid_balls[0]
        else:
            # No balls detected
            self.locked_target = None
        
        # Update ball detection state based on locked target
        if self.locked_target is not None:
            self.ball_x = int(self.locked_target['x'])
            self.ball_y = int(self.locked_target['y'])
            self.ball_radius = int(self.locked_target['radius'])
            self.ball_valid = True
            self.ball_time = datetime.now()
            self.ball_update_cnt += 1
            
            # Update confidence
            if self.ball_confidence < 20:
                self.ball_confidence += 1
        else:
            # Decrease confidence if no ball
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
    
    def set_color(self, color_name):
        """Set the color to track.
        
        Args:
            color_name: 'red', 'blue', or 'white'
        """
        if color_name in ['red', 'blue', 'white']:
            self.color_detect_mode = color_name
            print(f"% Ball (sball.py):: Color mode set to {color_name}")
        else:
            print(f"% Ball (sball.py):: Unknown color '{color_name}', keeping {self.color_detect_mode}")
    
    ##########################################################
    
    def set_focal_length_mm(self, focal_length_mm, sensor_width_mm=3.68):
        """Convert focal length from mm to pixels and set it.
        
        Uses the formula: focal_length_pixels = (focal_length_mm * image_width) / sensor_width
        
        Args:
            focal_length_mm: Focal length in millimeters (e.g., 3.6mm for Pi Camera)
            sensor_width_mm: Camera sensor width (default 3.68mm for Pi Camera v2)
                           For other cameras, measure or look up specifications
        """
        # Calculate focal length in pixels based on sensor geometry
        self.focal_length = (focal_length_mm * self.image_width) / sensor_width_mm
        print(f"% Ball (sball.py):: Focal length set to {focal_length_mm}mm = {self.focal_length:.0f}px "
              f"(sensor_width={sensor_width_mm}mm)")
    
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
ball = SBall()

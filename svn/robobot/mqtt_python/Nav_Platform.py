import threading
import time
import scam as cam
from collections import deque
import math


class Nav:
    """Navigation controller to center a detected target and move towards it."""

    def __init__(self):
        self.detector = None
        self.desired_distance = None
        self.target = None
        self.is_running = False
        self.nav_thread = None

    def setup(self, detector, desired_distance_to_target, ctx):

        self.detector = detector
        self.desired_distance = desired_distance_to_target
        self.ctx = ctx

        # navigation state
        self.rotation_phase = True
        self.forward_phase = False

        # robot limits
        self.MAX_LINEAR_SPEED = 0.08
        self.MAX_ANGULAR_SPEED = 0.3
        
        
        #NOTE: Maybe change to logitech 1.36 (78 degrees)
        # camera parameters
        self.CAMERA_FOV = 1.047 

        # visual servoing rotation gain
        self.K_ROT = 5.0

        # steering gain while moving
        self.K_STEER = 1.0

        # forward controller using Y position
        #self.K_FORWARD = 0.0015
        self.K_FORWARD = 0.4
        #angle
        self.K_BEARING = 0.5
        # desired vertical position of the ball
        self.DESIRED_DISTANCE = 0.2
        self.DOCK_DISTANCE = 0.35
        self.BEARING_TOL = 2.0
        self.PLATFORM_VEL_EPS = 0.03

        # tolerances
        self.ROTATION_TOLERANCE = 0.015
        self.DISTANCE_TOLERANCE = 0.010
        
        # Platform variable
        self.history = deque(maxlen=8)
        self.prev_velocity = 0
        self.turnaround_detected = False
        self.state = 'FOLLOW'

        # timing
        self.last_time = time.time()

        # platform detector
        self.platform_direction = 0

        # debug
        self.print_every_n_ticks = 20
        self.debug_tick = 0

    def start(self):

        if not self.detector:
            print("Detector not initialized")
            return

        self.is_running = True
        self.hasReachedTarget = False
        self.nav_thread = threading.Thread(target=self.go_to_target, daemon=True)
        self.nav_thread.start()

    def go_to_target(self):

        print("% Stating tracking")
        while self.is_running:
            try:

                self.debug_tick += 1
                should_log = (self.debug_tick % self.print_every_n_ticks) == 0

                self.target = self.detector.get_target()

                if self.target is None:

                    if should_log:
                        print("No target detected")

                    time.sleep(0.05)
                    continue

                now = time.time()
                
                img_width = self.target.get("image_width", 820) 
                img_center = img_width / 2.0
                
                #---------------------
                
                result = self.update_tracking_history(self.target)
                
                self.follow_platform(self.target)
                
                if result is not None and self.debug_tick % 5 == 0:
                    vx, dt = result
                    # print(f"x: {self.target['tvec_x']:.3f}, time: {self.target['time']:.3f}")
                    # print(f"z: {self.target['tvec_z']:.3f}, dist: {self.target['distance']:.3f}")
                    
                    #print(f"velocity= {vx:.4f}, time= {dt:.4f}")
                    print(f"velocity= {vx:.4f}, bearing= {self.target['bearing']:.4f}}")
                    
                time.sleep(0.034)
                    
                

            except Exception as e:
                pass
                # print(f"Navigation error: {e}")
                # time.sleep(0.1)

    def stop(self):

        self.is_running = False
        self.target = None
        self.ctx.actions.drive.stop()

        if self.nav_thread and self.nav_thread.is_alive():
            self.nav_thread.join(timeout=1.0)

    def update_tracking_history(self, target):

        sample = {
            't': target['time'],
            'x': target['tvec_x'],
            'z': target['tvec_z'],
        }
        
        self.history.append(sample)
        if len(self.history) < 2:
            return None
        
        old_sample = self.history[0]
        new_sample = self.history[-1]
        
        dt = new_sample['t'] - old_sample['t']
        
        if dt <= 1e-3:
            return None
        
        vx = -(new_sample['x'] - old_sample['x']) / dt
        
        return vx, dt
        
    def follow_platform(self, target):
        bearing_error = target['bearing']
        distance_error = target['tvec_z'] - self.DESIRED_DISTANCE
        
        angular_cmd = self.K_BEARING * bearing_error
        linear_cmd = self.K_FORWARD * distance_error
        
        angular_cmd = max(-self.MAX_ANGULAR_SPEED, min(self.MAX_ANGULAR_SPEED, angular_cmd))
        linear_cmd = max(0, min(self.MAX_LINEAR_SPEED, linear_cmd))
        
        if abs(bearing_error) > self.BEARING_TOL:
            linear_cmd *= 0.4
        
        
        self.ctx.actions.drive.rc(linear_cmd, angular_cmd)
        #print("% Driving..")

    def hasReachedTarget(self, target):

        return self.hasReachedTarget

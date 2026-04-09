import threading
import time
import scam as cam
from collections import deque
import math
import numpy as np

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

        # Navigation state
        self.rotation_phase = True
        self.forward_phase = False

        # Robot limits
        self.MAX_LINEAR_SPEED = 0.6
        self.MAX_ANGULAR_SPEED = 0.4
        
        # Camera parameters
        self.CAMERA_FOV = 1.047 

        # Visual servoing gains
        self.K_FORWARD = 2.0
        self.K_BEARING = 0.75
        self.DESIRED_DISTANCE = 0.25
        self.DOCK_DISTANCE = 0.35
        self.BEARING_TOL = 2.0

        # Tolerances
        self.ROTATION_TOLERANCE = 0.015
        self.DISTANCE_TOLERANCE = 0.010
        
        # Platform tracking variables
        self.history = deque(maxlen=8)
        self.prev_velocity = 0
        self.turnaround_detected = False
        self.state = 'FOLLOW'

        # Timing and debug
        self.last_time = time.time()
        self.platform_direction = 0
        self.print_every_n_ticks = 20
        self.debug_tick = 0

        # --- KALMAN FILTER STATE MACHINE ---
        self.kf_state_machine = 'WAIT_FIRST'
        self.first_pos = None
        self.first_time = 0
        self.last_kf_time = 0

        # --- KALMAN FILTER MATRICES ---
        self.kf_X = None  # State vector: [x, z, vx, vz]^T
        self.kf_P = None  # Covariance Matrix (uncertainty)
        
        self.kf_H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        self.kf_R = np.eye(2) * 0.1
        self.kf_Q = np.eye(4) * 0.01

        # --- BLENDING & TIMEOUT PARAMETERS ---
        self.BLEND_ALPHA = 0.8
        self.PREDICTION_HORIZON = 2.0 
        
        # How long to drive blind before giving up
        self.LOST_TIMEOUT = 0.5 
        self.last_seen_time = 0

    # --- INTERNAL KALMAN FILTER FUNCTIONS ---
    def _init_kalman(self, init_x, init_z, init_vx, init_vz):
        self.kf_X = np.array([[init_x], [init_z], [init_vx], [init_vz]], dtype=float)
        self.kf_P = np.eye(4) * 1.0

    def _predict_kalman(self, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        self.kf_X = np.dot(F, self.kf_X)
        self.kf_P = np.dot(F, np.dot(self.kf_P, F.T)) + self.kf_Q

    def _update_kalman(self, meas_x, meas_z):
        Z = np.array([[meas_x], [meas_z]])
        Y = Z - np.dot(self.kf_H, self.kf_X)
        S = np.dot(self.kf_H, np.dot(self.kf_P, self.kf_H.T)) + self.kf_R
        K = np.dot(self.kf_P, np.dot(self.kf_H.T, np.linalg.inv(S)))

        self.kf_X = self.kf_X + np.dot(K, Y)
        I = np.eye(4)
        self.kf_P = np.dot((I - np.dot(K, self.kf_H)), self.kf_P)

    def _predict_future_kalman(self, dt_future):
        future_x = self.kf_X[0, 0] + self.kf_X[2, 0] * dt_future
        future_z = self.kf_X[1, 0] + self.kf_X[3, 0] * dt_future
        return future_x, future_z
    # ----------------------------------------

    def start(self):
        if not self.detector:
            print("Detector not initialized")
            return

        self.is_running = True
        self.hasReachedTarget = False
        self.nav_thread = threading.Thread(target=self.go_to_target, daemon=True)
        self.nav_thread.start()

    def go_to_target(self):
        print("% Starting tracking")
        while self.is_running:
            try:
                system_now = time.time()
                self.debug_tick += 1
                should_log = (self.debug_tick % self.print_every_n_ticks) == 0

                self.target = self.detector.get_target()

                # --- TARGET LOST LOGIC (COAST MODE) ---
                if self.target is None:
                    time_since_lost = system_now - self.last_seen_time
                    
                    if time_since_lost < self.LOST_TIMEOUT and self.kf_state_machine == 'TRACKING':
                        dt = system_now - self.last_kf_time
                        if dt > 0:
                            self._predict_kalman(dt)
                            self.last_kf_time = system_now
                            
                        kf_current_x = self.kf_X[0, 0]
                        kf_current_z = self.kf_X[1, 0]
                        
                        future_x, future_z = self._predict_future_kalman(self.PREDICTION_HORIZON)
                        
                        blended_x = (1.0 - self.BLEND_ALPHA) * kf_current_x + (self.BLEND_ALPHA * future_x)
                        blended_z = (1.0 - self.BLEND_ALPHA) * kf_current_z + (self.BLEND_ALPHA * future_z)
                        
                        drive_target_z = blended_z
                        
                        # FIX: Calculate bearing in RADIANS using math.atan2 directly
                        drive_bearing = math.atan2(blended_x, blended_z)

                        if should_log:
                            print(f"[BLIND COASTING] Target Z: {drive_target_z:.2f}, Bearing: {drive_bearing:.2f} rad")
                            
                        self.follow_platform(drive_target_z, drive_bearing)
                        time.sleep(0.034)
                        continue

                    else:
                        self.ctx.actions.drive.rc(0, 0)
                        if self.kf_state_machine != 'WAIT_FIRST':
                            self.kf_state_machine = 'WAIT_FIRST'
                            if should_log:
                                print("Timeout reached. Robot stopped. Waiting for ArUco...")
                        time.sleep(0.05)
                        continue

                # --- TARGET VISIBLE LOGIC ---
                self.last_seen_time = system_now 
                
                target_time = self.target['time']
                meas_x = self.target['tvec_x']
                meas_z = self.target['tvec_z']
                
                if self.kf_state_machine == 'WAIT_FIRST':
                    self.ctx.actions.drive.rc(0, 0)
                    self.first_pos = (meas_x, meas_z)
                    self.first_time = target_time
                    self.kf_state_machine = 'WAIT_HALF_SEC'
                    print("[KF] Target found! Stopping for 0.5s to read real velocity...")

                elif self.kf_state_machine == 'WAIT_HALF_SEC':
                    self.ctx.actions.drive.rc(0, 0)
                    dt = target_time - self.first_time
                    if dt >= 0.5:
                        vx = (meas_x - self.first_pos[0]) / dt
                        vz = (meas_z - self.first_pos[1]) / dt

                        self._init_kalman(meas_x, meas_z, vx, vz)
                        self.last_kf_time = system_now 
                        self.kf_state_machine = 'TRACKING'
                        print(f"[KF] Initialized! vx={vx:.3f}, vz={vz:.3f}. Resuming drive.")

                elif self.kf_state_machine == 'TRACKING':
                    dt = system_now - self.last_kf_time
                    if dt > 0:
                        self._predict_kalman(dt)
                        self._update_kalman(meas_x, meas_z)
                        self.last_kf_time = system_now

                    future_x, future_z = self._predict_future_kalman(self.PREDICTION_HORIZON)

                    blended_x = (1.0 - self.BLEND_ALPHA) * meas_x + (self.BLEND_ALPHA * future_x)
                    blended_z = (1.0 - self.BLEND_ALPHA) * meas_z + (self.BLEND_ALPHA * future_z)
                    
                    drive_target_z = blended_z
                    
                    # FIX: Calculate bearing in RADIANS using math.atan2 directly
                    drive_bearing = math.atan2(blended_x, blended_z)

                    if should_log:
                        print(f"Tracking -> Target Z: {drive_target_z:.2f}, Bearing: {drive_bearing:.2f} rad")
                
                    self.follow_platform(drive_target_z, drive_bearing) 
                
                time.sleep(0.034)

            except Exception as e:
                pass

    def stop(self):
        self.is_running = False
        self.target = None
        self.ctx.actions.drive.stop()

        if self.nav_thread and self.nav_thread.is_alive():
            self.nav_thread.join(timeout=1.0)

    def follow_platform(self, target_z, target_bearing):
        bearing_error = target_bearing
        distance_error = target_z - self.DESIRED_DISTANCE
        
        angular_cmd = self.K_BEARING * bearing_error
        linear_cmd = self.K_FORWARD * distance_error
        
        angular_cmd = max(-self.MAX_ANGULAR_SPEED, min(self.MAX_ANGULAR_SPEED, angular_cmd))
        linear_cmd = max(0, min(self.MAX_LINEAR_SPEED, linear_cmd))
        
        if abs(bearing_error) > self.BEARING_TOL:
            linear_cmd *= 0.4
        
        self.ctx.actions.drive.rc(linear_cmd, angular_cmd)

    def hasReachedTarget(self, target):
        return self.hasReachedTarget
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

        self.rotation_phase = True
        self.forward_phase = False

        self.MAX_LINEAR_SPEED = 0.6
        self.MAX_ANGULAR_SPEED = 0.5
        
        self.CAMERA_FOV = 1.047 

        self.K_FORWARD = 0.75
        self.K_BEARING = 0.75
        
        # 1. Ajuste de distancias para evitar choques
        self.DESIRED_DISTANCE = 0.33 # Distancia del punto fantasma (Carrot point)
        self.SAFE_STOP_DISTANCE = 0.33 # Freno de emergencia absoluto (30cm de la plataforma)
        
        # FIX: Tolerancia en Radianes (0.2 rad ~= 11.5 grados)
        self.BEARING_TOL = 0.2 
        
        self.history = deque(maxlen=8)
        self.prev_velocity = 0
        self.turnaround_detected = False
        self.state = 'FOLLOW'

        self.last_time = time.time()
        self.platform_direction = 0
        self.print_every_n_ticks = 20
        self.debug_tick = 0

        # --- PSEUDO-ODOMETRY (ROBOT POSE) ---
        self.robot_x = 0.0
        self.robot_z = 0.0
        self.robot_theta = 0.0
        self.current_linear_v = 0.0
        self.current_angular_v = 0.0
        self.last_loop_time = time.time()

        # --- KALMAN FILTER STATE MACHINE ---
        self.kf_state_machine = 'WAIT_FIRST'
        self.first_pos = None
        self.first_time = 0
        self.last_kf_time = 0

        self.kf_X = None  
        self.kf_P = None  
        
        self.kf_H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        self.kf_R = np.eye(2) * 0.1
        
        # --- FIX: DEPTH NOISE CANCELLATION ---
        self.kf_Q = np.array([
            [0.05, 0, 0, 0],       
            [0, 0.0003, 0, 0],    
            [0, 0, 0.05, 0],       
            [0, 0, 0, 0.0001]     
        ], dtype=float)

        self.BLEND_ALPHA = 0.9
        self.PREDICTION_HORIZON = 1.5
        self.LOST_TIMEOUT = 2 #1
        self.last_seen_time = 0
        
        # --- ANGLE APPROACH CONFIG ---
        self.APPROACH_ANGLE_RAD = math.radians(0)

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

    def start(self):
        if not self.detector:
            print("Detector not initialized")
            return
        self.is_running = True
        self.hasReachedTarget = False
        self.last_loop_time = time.time()
        self.nav_thread = threading.Thread(target=self.go_to_target, daemon=True)
        self.nav_thread.start()

    def go_to_target(self):
        print("% Starting tracking with 45-deg approach & collision avoidance")
        while self.is_running:
            try:
                system_now = time.time()
                dt_loop = system_now - self.last_loop_time
                self.last_loop_time = system_now
                
                self.debug_tick += 1
                should_log = (self.debug_tick % self.print_every_n_ticks) == 0

                # 1. Update Robot Pose
                self.robot_theta += self.current_angular_v * dt_loop
                self.robot_x += self.current_linear_v * math.sin(self.robot_theta) * dt_loop
                self.robot_z += self.current_linear_v * math.cos(self.robot_theta) * dt_loop

                self.target = self.detector.get_target()

                # --- TARGET LOST LOGIC ---
                if self.target is None:
                    time_since_lost = system_now - self.last_seen_time
                    
                    if time_since_lost < self.LOST_TIMEOUT and self.kf_state_machine == 'TRACKING':
                        dt = system_now - self.last_kf_time
                        if dt > 0:
                            self._predict_kalman(dt)
                            self.last_kf_time = system_now
                            
                        plat_g_x = self.kf_X[0, 0]
                        plat_g_z = self.kf_X[1, 0]
                        
                        # Calculamos la distancia REAL a la plataforma (estimada a ciegas)
                        dx_plat = plat_g_x - self.robot_x
                        dz_plat = plat_g_z - self.robot_z
                        plat_local_x = dx_plat * math.cos(self.robot_theta) - dz_plat * math.sin(self.robot_theta)
                        plat_local_z = dx_plat * math.sin(self.robot_theta) + dz_plat * math.cos(self.robot_theta)
                        real_dist_to_platform = math.hypot(plat_local_x, plat_local_z)
                        
                        approach_angle = self.last_known_approach_angle
                        
                        dest_g_x = plat_g_x + self.DESIRED_DISTANCE * math.sin(approach_angle)
                        dest_g_z = plat_g_z - self.DESIRED_DISTANCE * math.cos(approach_angle)
                        
                        dx = dest_g_x - self.robot_x
                        dz = dest_g_z - self.robot_z
                        local_x = dx * math.cos(self.robot_theta) - dz * math.sin(self.robot_theta)
                        local_z = dx * math.sin(self.robot_theta) + dz * math.cos(self.robot_theta)

                        drive_distance = math.hypot(local_x, local_z)
                        drive_bearing = math.atan2(local_x, local_z)
                        
                        # Le pasamos también la distancia real para el freno de emergencia
                        self.follow_platform(drive_distance, drive_bearing, real_dist_to_platform)
                        time.sleep(0.034)
                        continue
                    else:
                        self.ctx.actions.drive.rc(0, 0)
                        self.current_linear_v = 0.0
                        self.current_angular_v = 0.0
                        if self.kf_state_machine != 'WAIT_FIRST':
                            self.kf_state_machine = 'WAIT_FIRST'
                        time.sleep(0.05)
                        continue

                # --- TARGET VISIBLE LOGIC ---
                self.last_seen_time = system_now 
                
                target_time = self.target['time']
                meas_local_x = self.target['tvec_x']
                meas_local_z = self.target['tvec_z'] 
                target_yaw = self.target.get('tilt', 0.0) 
                
                # Distancia física real a la plataforma basada en cámara
                real_dist_to_platform = math.hypot(meas_local_x, meas_local_z)
                
                meas_global_x = self.robot_x + meas_local_x * math.cos(self.robot_theta) + meas_local_z * math.sin(self.robot_theta)
                meas_global_z = self.robot_z - meas_local_x * math.sin(self.robot_theta) + meas_local_z * math.cos(self.robot_theta)
                
                if self.kf_state_machine == 'WAIT_FIRST':
                    self.ctx.actions.drive.rc(0, 0)
                    self.current_linear_v = 0.0
                    self.current_angular_v = 0.0
                    
                    self.robot_x = 0.0
                    self.robot_z = 0.0
                    self.robot_theta = 0.0
                    meas_global_x = meas_local_x
                    meas_global_z = meas_local_z
                    
                    self.first_pos = (meas_global_x, meas_global_z)
                    self.first_time = target_time
                    self.kf_state_machine = 'WAIT_HALF_SEC'

                elif self.kf_state_machine == 'WAIT_HALF_SEC':
                    self.ctx.actions.drive.rc(0, 0)
                    self.current_linear_v = 0.0
                    self.current_angular_v = 0.0
                    dt = target_time - self.first_time
                    if dt >= 0.5:
                        vx = (meas_global_x - self.first_pos[0]) / dt
                        vz = 0.0 

                        self._init_kalman(meas_global_x, meas_global_z, vx, vz)
                        self.last_kf_time = system_now 
                        self.kf_state_machine = 'TRACKING'

                elif self.kf_state_machine == 'TRACKING':
                    dt = system_now - self.last_kf_time
                    if dt > 0:
                        self._predict_kalman(dt)
                        self._update_kalman(meas_global_x, meas_global_z)
                        self.last_kf_time = system_now

                    # Extract velocity in x
                    vx = self.kf_X[2, 0]
                    
                    if vx < 0:
                        # Don't proceed with navigation if velocity in x is negative
                        self.ctx.actions.drive.rc(0, 0)
                        self.current_linear_v = 0.0
                        self.current_angular_v = 0.0
                        time.sleep(0.034)
                        continue

                    future_global_x, future_global_z = self._predict_future_kalman(self.PREDICTION_HORIZON)

                    blended_global_x = (1.0 - self.BLEND_ALPHA) * meas_global_x + (self.BLEND_ALPHA * future_global_x)
                    blended_global_z = (1.0 - self.BLEND_ALPHA) * meas_global_z + (self.BLEND_ALPHA * future_global_z)
                    
                    plat_facing_angle = self.robot_theta + target_yaw
                    approach_angle = plat_facing_angle + self.APPROACH_ANGLE_RAD
                    self.last_known_approach_angle = approach_angle 
                    
                    dest_global_x = blended_global_x + self.DESIRED_DISTANCE * math.sin(approach_angle)
                    dest_global_z = blended_global_z - self.DESIRED_DISTANCE * math.cos(approach_angle)
                    
                    dx = dest_global_x - self.robot_x
                    dz = dest_global_z - self.robot_z
                    
                    local_x = dx * math.cos(self.robot_theta) - dz * math.sin(self.robot_theta)
                    local_z = dx * math.sin(self.robot_theta) + dz * math.cos(self.robot_theta)
                    
                    drive_distance = math.hypot(local_x, local_z)
                    drive_bearing = math.atan2(local_x, local_z)

                    if should_log:
                        print(f"Tracking 45° -> Dist Punto: {drive_distance:.2f}m | Dist Real: {real_dist_to_platform:.2f}m")
                
                    self.follow_platform(drive_distance, drive_bearing, real_dist_to_platform) 
                
                time.sleep(0.034)

            except Exception as e:
                pass

    def stop(self):
        self.is_running = False
        self.target = None
        self.current_linear_v = 0.0
        self.current_angular_v = 0.0
        self.ctx.actions.drive.stop()

        if self.nav_thread and self.nav_thread.is_alive():
            self.nav_thread.join(timeout=1.0)

    # --- MODIFIED: Added real_distance_to_platform parameter ---
    def follow_platform(self, distance_to_point, target_bearing, real_distance_to_platform):
        
        bearing_error = target_bearing
        
        # 1. CAPA DE SEGURIDAD ABSOLUTA (Anti-Choque físico)
        if (real_distance_to_platform < self.SAFE_STOP_DISTANCE):
            # Clavamos frenos de avance. Le permitimos seguir rotando para no perderla de vista.
            self.ctx.actions.drive.stop()
            self.hasReachedTarget = True
            self.is_running = False
            
        else:
            # 2. CAPA DE ZONA MUERTA (Evitar que el robot baile alrededor del punto)
            if distance_to_point < 0.05:
                linear_cmd = 0.0
                angular_cmd = 0.0 
            else:
                # 3. CONDUCCIÓN NORMAL
                linear_cmd = self.K_FORWARD * distance_to_point
                angular_cmd = self.K_BEARING * bearing_error
                
                # Ralentizar si hay que hacer un giro brusco
                if abs(bearing_error) > self.BEARING_TOL:
                    linear_cmd *= 0.4

        # Aplicar límites máximos del robot
        angular_cmd = max(-self.MAX_ANGULAR_SPEED, min(self.MAX_ANGULAR_SPEED, angular_cmd))
        
        # Solo aplicamos límites si no hemos forzado el freno a 0.0
        if linear_cmd > 0.0:
            linear_cmd = max(0.0, min(self.MAX_LINEAR_SPEED, linear_cmd))
        
        # Guardar para la Odometría
        self.current_linear_v = linear_cmd
        self.current_angular_v = angular_cmd
        
        self.ctx.actions.drive.rc(linear_cmd, angular_cmd)

    def hasReachedTarget(self, target):
        return self.hasReachedTarget
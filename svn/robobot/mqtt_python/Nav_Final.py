import threading
import time
#import scam as cam
import scam_usb as cam


class Nav:
    """Navigation controller to center a detected target and move towards it."""

    def __init__(self):
        self.detector = None
        self.desired_distance = 0.32
        self.target = None
        self.is_running = False
        self.nav_thread = None
        self.COMPENSATE_PARAMETER = 30

    def setup(self, detector, desired_distance_to_target, COMPENSATE_PARAMETER, ctx):

        self.detector = detector
        self.desired_distance = desired_distance_to_target
        self.ctx = ctx
        self.COMPENSATE_PARAMETER = COMPENSATE_PARAMETER


        # navigation state
        self.rotation_phase = True
        self.forward_phase = False

        # robot limits
        self.MAX_LINEAR_SPEED = 0.25
        self.MAX_ANGULAR_SPEED = 0.4


        #NOTE: Maybe change to logitech 1.36 (78 degrees)
        # camera parameters
        # camera parameters
        self.CAMERA_FOV = 1.047

        # visual servoing rotation gain
        self.K_ROT = 7.5

        # steering gain while moving
        self.K_STEER = 1.2

        # forward controller using Y position
        #self.K_FORWARD = 0.0015
        self.K_FORWARD = 0.75
        self.I_FORWARD = 0.02
        self.D_FORWARD = 1.25

        self.ACC_ERROR = 0.0
        self.LAST_ERROR = 0.0
        
        # desired vertical position of the ball

        self.DESIRED_DISTANCE = self.desired_distance
        #self.DESIRED_DISTANCE = 0.28
        #self.DESIRED_DISTANCE = 0.41

        # tolerances
        self.ROTATION_TOLERANCE = 0.03
        self.DISTANCE_TOLERANCE = 0.010
        self.Y_TOLERANCE = 5

        # timing
        self.last_time = time.time()

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

        while self.is_running:
            try:
                
                self.debug_tick += 1
                should_log = (self.debug_tick % self.print_every_n_ticks) == 0

                self.target = self.detector.get_target()

                if self.target is None:

                    if should_log:
                        print("No target detected")
                        self.ACC_ERROR = 0.0
                    time.sleep(0.05)
                    continue

                now = time.time()
                dt = now - self.last_time

                if dt <= 0.0:
                    dt = 0.05

                img_width = self.target.get("image_width", 820)
                img_center = (img_width) / 2.0

                # ---------- rotation error ----------
                pixel_error = img_center - self.target["x"] - self.COMPENSATE_PARAMETER #To compensate the arm offset
                rotation_error = pixel_error * (self.CAMERA_FOV / img_width)

                # ---------- y error ----------
                #ball_y = self.target["y"]
                #y_error = self.DESIRED_Y - ball_y
                distance = self.target["distance"]
                distance_error = - self.DESIRED_DISTANCE + distance

                print(f"Distance: {distance}, Error: {distance_error}")

                # ---------------------------------------------------
                # ROTATION PHASE
                # ---------------------------------------------------

                if self.rotation_phase:

                    if abs(rotation_error) > self.ROTATION_TOLERANCE:
                        angular_speed = self.K_ROT * rotation_error * abs(rotation_error)

                        MIN_W = 0.15

                        if abs(angular_speed) < MIN_W:
                            angular_speed = MIN_W * (1 if angular_speed > 0 else -1)

                        if angular_speed > self.MAX_ANGULAR_SPEED:
                            angular_speed = self.MAX_ANGULAR_SPEED
                        if angular_speed < -self.MAX_ANGULAR_SPEED:
                            angular_speed = -self.MAX_ANGULAR_SPEED

                        self.ctx.actions.drive.rc(0, angular_speed)

                        if should_log:
                            print(
                                f"Rotating | rot_error={rotation_error:.3f}  w={angular_speed:.3f}"
                            )

                        time.sleep(0.001)
                        continue

                    else:

                        print("Ball centered, starting forward motion")

                        self.rotation_phase = False
                        self.forward_phase = True

                # ---------------------------------------------------
                # FORWARD PHASE
                # ---------------------------------------------------

                if self.forward_phase:

                    #print(f"Y error: {y_error}, Rotation error: {rotation_error}")
                    #print(f"Target info: {self.target}")
                    
                    if self.target is None:
                        print("Lost target during forward motion")
                        self.ctx.actions.drive.rc(-0.2, 0)
                        time.sleep(0.100)
                        continue
                   
                    #if y_error <= self.Y_TOLERANCE:
                    if distance_error <= self.DISTANCE_TOLERANCE:
                        
                        print("Target reached")
                        #time.sleep(1.5)
                        self.ctx.actions.drive.stop()
                        self.hasReachedTarget = True
                        self.is_running = False

                        time.sleep(0.05)
                        continue

                    # PID forward controller
                    p_term = self.K_FORWARD * distance_error
                    self.ACC_ERROR += distance_error * dt
                    i_term = self.I_FORWARD * self.ACC_ERROR
                    d_term = self.D_FORWARD * (distance_error - self.LAST_ERROR) / dt

                    linear_speed = p_term + i_term + d_term


                    # clamp forward speed
                    if linear_speed > self.MAX_LINEAR_SPEED:
                        linear_speed = self.MAX_LINEAR_SPEED
                    if linear_speed < 0:
                        linear_speed = 0

                    # steering correction
                    angular_speed = self.K_STEER * rotation_error

                    if angular_speed > self.MAX_ANGULAR_SPEED:
                        angular_speed = self.MAX_ANGULAR_SPEED
                    if angular_speed < -self.MAX_ANGULAR_SPEED:
                        angular_speed = -self.MAX_ANGULAR_SPEED

                    self.ctx.actions.drive.rc(linear_speed, angular_speed)

                    #if should_log:
                        #print(
                            #f"Forward | y_err={y_error:.1f} rot_err={rotation_error:.3f} v={linear_speed:.3f} w={angular_speed:.3f}"
                        #)
                self.LAST_ERROR = distance_error
                self.last_time = now
                time.sleep(0.01)

            except Exception as e:

                print(f"Navigation error: {e}")
                time.sleep(0.1)

    def stop(self):

        self.is_running = False
        self.target = None
        self.ctx.actions.drive.stop()

        if self.nav_thread and self.nav_thread.is_alive():
            self.nav_thread.join(timeout=1.0)

    def hasReachedTarget(self):

        return self.hasReachedTarget
import threading
import time
import scam as cam


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
        self.MAX_LINEAR_SPEED = 0.3
        self.MAX_ANGULAR_SPEED = 0.4

        # camera parameters
        self.CAMERA_FOV = 1.047

        # distance PID
        self.KP_DIST = 0.4
        self.KI_DIST = 0.02
        self.KD_DIST = -0.1

        # steering gain while moving
        self.K_STEER = 1.5

        # visual servoing gain
        self.K_ROT = 2.0

        # tolerances
        self.ROTATION_TOLERANCE = 0.005
        self.DISTANCE_TOLERANCE = 0.05

        # PID state
        self.distance_error_integral = 0.0
        self.last_distance_error = 0.0

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

                    time.sleep(0.05)
                    continue

                now = time.time()
                dt = now - self.last_time

                if dt <= 0:
                    dt = 0.05

                img_width = self.target.get("image_width", 820)
                img_center = img_width / 2.0

                pixel_error = img_center - self.target["x"]

                rotation_error = pixel_error * (self.CAMERA_FOV / img_width) # convert pixel error to radians

                distance_error = self.target["distance"] - self.desired_distance

                if should_log:
                    print(f"Distance to ball: {self.target['distance']}")

                # ---------------------------------------------------
                # ROTATION PHASE
                # ---------------------------------------------------

                if self.rotation_phase:

                    if abs(rotation_error) > self.ROTATION_TOLERANCE:
                        
                        # visual servoing control (angular speed proportional to error, with a gain and saturation)
                        angular_speed = self.K_ROT * rotation_error * abs(rotation_error)

                        #a voing very small angular speeds that might not overcome friction
                        if abs(angular_speed) < 0.02:
                            angular_speed = 0

                        if angular_speed > self.MAX_ANGULAR_SPEED:
                            angular_speed = self.MAX_ANGULAR_SPEED
                        if angular_speed < -self.MAX_ANGULAR_SPEED:
                            angular_speed = -self.MAX_ANGULAR_SPEED

                        self.ctx.actions.drive.rc(0, angular_speed)

                        if should_log:
                            print(
                                f"Rotating | error={rotation_error:.3f}  w={angular_speed:.3f}"
                            )

                        self.last_time = now
                        time.sleep(0.01)
                        continue

                    else:

                        print("Ball centered, starting forward motion")

                        self.rotation_phase = False
                        self.forward_phase = True

                        self.distance_error_integral = 0
                        self.last_distance_error = 0

                # ---------------------------------------------------
                # FORWARD PHASE
                # ---------------------------------------------------

                if self.forward_phase:

                    if abs(distance_error) <= self.DISTANCE_TOLERANCE:

                        print("Target reached")

                        self.ctx.actions.drive.stop()
                        time.sleep(0.05)
                        continue

                    # distance PID
                    p_term = self.KP_DIST * distance_error

                    self.distance_error_integral += distance_error * dt
                    i_term = self.KI_DIST * self.distance_error_integral

                    d_term = self.KD_DIST * ((distance_error - self.last_distance_error) / dt)

                    linear_speed = p_term + i_term + d_term

                    if linear_speed > self.MAX_LINEAR_SPEED:
                        linear_speed = self.MAX_LINEAR_SPEED
                    if linear_speed < -self.MAX_LINEAR_SPEED:
                        linear_speed = -self.MAX_LINEAR_SPEED

                    # steering correction while moving
                    angular_speed = self.K_STEER * rotation_error

                    if angular_speed > self.MAX_ANGULAR_SPEED:
                        angular_speed = self.MAX_ANGULAR_SPEED
                    if angular_speed < -self.MAX_ANGULAR_SPEED:
                        angular_speed = -self.MAX_ANGULAR_SPEED

                    self.ctx.actions.drive.rc(linear_speed, angular_speed)

                    if should_log:
                        print(
                            f"Forward | dist_err={distance_error:.3f}  rot_err={rotation_error:.3f}  v={linear_speed:.3f}  w={angular_speed:.3f}"
                        )

                    self.last_distance_error = distance_error
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
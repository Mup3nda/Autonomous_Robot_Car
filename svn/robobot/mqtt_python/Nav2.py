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
        self.MAX_LINEAR_SPEED = 0.25
        self.MAX_ANGULAR_SPEED = 0.4

        # camera parameters
        self.CAMERA_FOV = 1.047

        # visual servoing rotation gain
        self.K_ROT = 5.0

        # steering gain while moving
        self.K_STEER = 1.5

        # forward controller using Y position
        self.K_FORWARD = 0.0015

        # desired vertical position of the ball
        self.DESIRED_Y = 545

        # tolerances
        self.ROTATION_TOLERANCE = 0.015
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

                    time.sleep(0.05)
                    continue

                now = time.time()

                img_width = self.target.get("image_width", 820)
                img_center = img_width / 2.0

                # ---------- rotation error ----------
                pixel_error = img_center - self.target["x"]
                rotation_error = pixel_error * (self.CAMERA_FOV / img_width)

                # ---------- y error ----------
                ball_y = self.target["y"]
                y_error = self.DESIRED_Y - ball_y


                print(f"Ball y: {ball_y}, y_error: {y_error}")

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

                        time.sleep(0.01)
                        continue

                    else:

                        print("Ball centered, starting forward motion")

                        self.rotation_phase = False
                        self.forward_phase = True

                # ---------------------------------------------------
                # FORWARD PHASE
                # ---------------------------------------------------

                if self.forward_phase:

                    print(f"Y error: {y_error}, Rotation error: {rotation_error}")
                    print(f"Target info: {self.target}")
                    
                    if self.target is None:
                        print("Lost target during forward motion")
                        self.ctx.actions.drive.rc(-0.2, 0)
                        time.sleep(0.100)
                        continue
                   
                    if y_error <= self.Y_TOLERANCE:

                        print("Target reached")
                        time.sleep(1.5)
                        self.ctx.actions.drive.stop()
                        self.hasReachedTarget = True
                        self.is_running = False

                        time.sleep(0.05)
                        continue

                    # proportional forward controller
                    linear_speed = self.K_FORWARD * y_error

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

                    if should_log:
                        print(
                            f"Forward | y_err={y_error:.1f} rot_err={rotation_error:.3f} v={linear_speed:.3f} w={angular_speed:.3f}"
                        )

                time.sleep(0.03)

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
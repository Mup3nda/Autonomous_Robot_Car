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
        #self.K_FORWARD = 0.0015
        self.K_FORWARD = 0.5
        
        # desired vertical position of the ball
        self.DESIRED_DISTANCE = 0.41

        # tolerances
        self.ROTATION_TOLERANCE = 0.015
        self.DISTANCE_TOLERANCE = 0.010

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

                x_1 = self.target["x"] # x at time 1

                time.sleep(0.1)

                x_2 = self.target["x"] # x at time 2

                if x_2 - x_1 > 0:
                    self.platform_direction = 1 # 1 = moving to the right of the image
                else:
                    self.platform_direction = 0 # 0 = moving to the left of the image

                if self.platform_direction:
                    self.ctx.actions.drive.rc(0, 0.2) # rotate to right
                    time.sleep(0.1)
                    self.ctx.actions.drive.rc(0.2, 0) # move forward
                    time.sleep(0.1)
                    self.ctx.actions.drive.rc(0, -0.2) # rotate to left
                    time.sleep(0.1)
                    self.ctx.actions.drive.rc(0.2, 0) # move forward
                    time.sleep(0.1)
                else:
                    self.ctx.actions.drive.rc(0, -0.2) # rotate to left
                    time.sleep(0.1)
                    self.ctx.actions.drive.rc(0.2, 0) # move forward
                    time.sleep(0.1)
                    self.ctx.actions.drive.rc(0, 0.2) # rotate to right
                    time.sleep(0.1)
                    self.ctx.actions.drive.rc(0.2, 0) # move forward
                    time.sleep(0.1)

                print("Target reached")
                
                self.ctx.actions.drive.stop()
                self.hasReachedTarget = True
                self.is_running = False

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
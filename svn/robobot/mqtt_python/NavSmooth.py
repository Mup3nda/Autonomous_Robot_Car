import threading
import time


class NavSmooth:
    """Smooth navigation controller that drives and turns simultaneously."""

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

        self.MAX_SPEED = 0.7
        self.MAX_W_SPEED = 0.45

        self.KP = 0.8
        self.KI = 0.02
        self.KD = 0.1

        self.KP_X = 0.75
        self.KI_X = 0.02
        self.KD_X = 0.04

        self.TOLERANCIA_R = 5
        self.TOLERANCIA_R_RAD = 0.02
        self.TOLERANCIA_D = 0.02

        # Tuning constants for smooth-mode behavior
        self.ROT_INTEGRAL_ACTIVE_BAND = 0.6
        self.ROT_INTEGRAL_DECAY = 0.98
        self.ROT_INTEGRAL_CLAMP = 0.8

        self.HEADING_SCALE_THRESH_1 = 2.0
        self.HEADING_SCALE_THRESH_2 = 4.0
        self.HEADING_SCALE_1 = 0.5
        self.HEADING_SCALE_2 = 0.2

        # (bearing threshold in rad, max linear speed in m/s)
        self.HEADING_LINEAR_CAPS = [
            (1.00, 0.02),
            (0.70, 0.04),
            (0.45, 0.08),
            (0.25, 0.16),
        ]

        self.CLOSE_DIST_W_LIMIT_1 = 0.20
        self.CLOSE_DIST_W_SCALE_1 = 0.65
        self.CLOSE_DIST_W_LIMIT_2 = 0.10
        self.CLOSE_DIST_W_SCALE_2 = 0.50

        self.W_CMD_PREV_WEIGHT = 0.75
        self.W_CMD_NEW_WEIGHT = 0.25

        self.error_acumulado = 0.0
        self.ultimo_error = 0.0
        self.error_rot_acumulado = 0.0
        self.ultimo_rot_error = 0.0
        self.ultima_vez = time.time()
        self.w_cmd = 0.0

        self.print_every_n_ticks = 1
        self.debug_tick = 0

    def start(self):
        if not self.detector:
            print("Error: Detector not initialized. Call setup() first.")
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
                    time.sleep(0.1)
                    continue

                ahora = time.time()
                dt = ahora - self.ultima_vez
                if dt <= 0.0:
                    dt = 0.05

                if "bearing" in self.target:
                    error_rot = self.target["bearing"]
                    tol_rot = self.TOLERANCIA_R_RAD
                else:
                    img_center = self.target.get("image_width", 820) / 2.0
                    error_rot = img_center - self.target["x"]
                    tol_rot = self.TOLERANCIA_R

                error = self.target["distance"] - self.desired_distance
                target_bearing = self.target.get("bearing")
                target_id = self.target.get("id")

                if abs(error) <= self.TOLERANCIA_D:
                    print("Target reached.")
                    self.ctx.actions.drive.stop()
                    self.hasReachedTarget = True
                    time.sleep(0.05)
                    continue

                p_term = self.KP * error
                self.error_acumulado += error * dt
                i_term = self.KI * self.error_acumulado
                d_term = self.KD * (error - self.ultimo_error) / dt
                velocidad = p_term + i_term + d_term

                # Apply a small deadband around zero bearing to avoid chatter.
                rot_for_control = 0.0 if abs(error_rot) < tol_rot else error_rot

                p_rot = self.KP_X * rot_for_control

                # Anti-windup: integrate mainly when reasonably aligned.
                if abs(rot_for_control) < self.ROT_INTEGRAL_ACTIVE_BAND:
                    self.error_rot_acumulado += rot_for_control * dt
                else:
                    self.error_rot_acumulado *= self.ROT_INTEGRAL_DECAY
                if self.error_rot_acumulado > self.ROT_INTEGRAL_CLAMP:
                    self.error_rot_acumulado = self.ROT_INTEGRAL_CLAMP
                if self.error_rot_acumulado < -self.ROT_INTEGRAL_CLAMP:
                    self.error_rot_acumulado = -self.ROT_INTEGRAL_CLAMP
                i_rot = self.KI_X * self.error_rot_acumulado
                d_rot = self.KD_X * (rot_for_control - self.ultimo_rot_error) / dt
                w_speed = p_rot + i_rot + d_rot

                if abs(error_rot) > (self.HEADING_SCALE_THRESH_1 * tol_rot):
                    velocidad *= self.HEADING_SCALE_1
                if abs(error_rot) > (self.HEADING_SCALE_THRESH_2 * tol_rot):
                    velocidad *= self.HEADING_SCALE_2

                # Heading-gated linear speed: when orientation error is large,
                # behave almost like turn-in-place even in smooth mode.
                if "bearing" in self.target:
                    abs_rot = abs(rot_for_control)
                    max_v_from_heading = self.MAX_SPEED
                    for rot_thresh, v_cap in self.HEADING_LINEAR_CAPS:
                        if abs_rot > rot_thresh:
                            max_v_from_heading = v_cap
                            break
                    if velocidad > max_v_from_heading:
                        velocidad = max_v_from_heading
                    if velocidad < -max_v_from_heading:
                        velocidad = -max_v_from_heading

                if velocidad > self.MAX_SPEED:
                    velocidad = self.MAX_SPEED
                if velocidad < -self.MAX_SPEED:
                    velocidad = -self.MAX_SPEED

                # Limit turning harder when close to target to avoid oversteer.
                max_w = self.MAX_W_SPEED
                if abs(error) < self.CLOSE_DIST_W_LIMIT_1:
                    max_w *= self.CLOSE_DIST_W_SCALE_1
                if abs(error) < self.CLOSE_DIST_W_LIMIT_2:
                    max_w *= self.CLOSE_DIST_W_SCALE_2

                if w_speed > max_w:
                    w_speed = max_w
                if w_speed < -max_w:
                    w_speed = -max_w

                # First-order smoothing on angular command to reduce jitter/sign flips.
                self.w_cmd = self.W_CMD_PREV_WEIGHT * self.w_cmd + self.W_CMD_NEW_WEIGHT * w_speed

                self.ctx.actions.drive.rc(velocidad, self.w_cmd)
                if should_log:
                    print(
                        f"[NAVSMOOTH] id={target_id} dist={self.target['distance']:.3f}m "
                        f"dist_err={error:.3f}m bearing={target_bearing} rot_err={error_rot:.3f} "
                        f"tol_rot={tol_rot:.3f} v={velocidad:.3f} w={self.w_cmd:.3f}"
                    )

                self.ultimo_error = error
                self.ultimo_rot_error = rot_for_control
                self.ultima_vez = ahora
                time.sleep(0.01)

            except Exception as e:
                print(f"Error in go_to_target (smooth): {e}")
                time.sleep(0.1)

    def stop(self):
        self.is_running = False
        self.target = None
        self.ctx.actions.drive.stop()
        if self.nav_thread and self.nav_thread.is_alive():
            self.nav_thread.join(timeout=1.0)

    def hasReachedTarget(self):
        return self.hasReachedTarget

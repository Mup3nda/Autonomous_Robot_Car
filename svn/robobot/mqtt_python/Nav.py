import threading
import time


class Nav:
    """Simple navigation controller for moving towards a detected target."""
    
    def __init__(self):
        self.detector = None
        self.desired_distance = None
        self.target = None
        self.is_running = False
        self.nav_thread = None
    
    def setup(self, detector, desired_distance_to_target, ctx):
        """
        Initialize the navigation controller.
        
        Args:
            detector: TargetDetector instance
            desired_distance_to_target: Target distance to maintain from target
        """
        self.detector = detector
        self.desired_distance = desired_distance_to_target

        self.ctx = ctx

        #self.DISTANCIA_DESEADA = 0.41  # meters
        self.MAX_SPEED = 0.6  # m/s

        # PID constants (tune later with real sensor input)
        self.KP = 0.8
        self.KI = 0.02
        self.KD = 0.1

        self.error_acumulado = 0.0
        self.ultimo_error = 0.0
        self.ultima_vez = time.time()

        # Consider objective complete after stable hold near setpoint
        self.stable_since = None
        self.stable_band = 0.03  # +/- 3 cm
        self.stable_time_required = 2.0  # seconds
    
    def start(self):
        """Start moving towards the target."""
        if not self.detector:
            print("Error: Detector not initialized. Call setup() first.")
            return
        
        self.is_running = True
        self.nav_thread = threading.Thread(target=self.go_to_target, daemon=True)
        self.hasReachedTarget = False
        self.nav_thread.start()
    
    def go_to_target(self):
        """
        Continuous loop that tracks and moves towards the detected target.
        Runs in a separate thread.
        """
        while self.is_running:
            try:
                # Get current target from detector
                self.target = self.detector.get_target() #target is a list with X Y distance
                
                if self.target is None:
                    # No target detected, stop or wait
                    print("No target detected")
                    time.sleep(0.1)
                    continue
                
                ahora = time.time()
                dt = ahora - self.ultima_vez
                if dt <= 0.0:
                    dt = 0.05

                # PID error: positive means we are too far and should move forward
                error = self.target - self.DISTANCIA_DESEADA

                p_term = self.KP * error
                self.error_acumulado += error * dt
                i_term = self.KI * self.error_acumulado
                d_term = self.KD * (error - self.ultimo_error) / dt

                velocidad = p_term + i_term + d_term

                # Saturation
                if velocidad > self.MAX_SPEED:
                    velocidad = self.MAX_SPEED
                if velocidad < -self.MAX_SPEED:
                    velocidad = -self.MAX_SPEED

                # Send command to robot interface
                self.ctx.actions.drive.rc(velocidad, 0.0)


        # Completion condition: stable around target for a while
                if abs(error) < self.stable_band:
                    if self.stable_since is None:
                        self.stable_since = ahora
                    elif ahora - self.stable_since >= self.stable_time_required:
                        self.ctx.actions.drive.stop()
                        print(f"Distancia estabilizada en {self.distancia_actual:.3f}m (simulada).")
                        self.hasReachedTarget = True
                else:
                    self.stable_since = None

                self.ultimo_error = error
                self.ultima_vez = ahora
                
                time.sleep(0.05)  # Control loop frequency
                
            except Exception as e:
                print(f"Error in go_to_target: {e}")
                time.sleep(0.1)
    
    def stop(self):
        """Stop movement and clear target."""
        self.is_running = False
        self.target = None
        self.ctx.actions.drive.stop()
        # Wait for thread to finish
        if self.nav_thread and self.nav_thread.is_alive():
            self.nav_thread.join(timeout=1.0)
            
    def hasReachedTarget(self):
        """Check if the robot has reached the target within a certain threshold."""
        return self.hasReachedTarget
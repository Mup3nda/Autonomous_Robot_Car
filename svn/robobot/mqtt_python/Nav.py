import threading
import time
import scam as cam

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
        self.rot_flag = True
        self.forward_flag = False

        #self.DISTANCIA_DESEADA = 0.41  # meters
        self.MAX_SPEED = 0.4  # m/s
        self.MAX_W_SPEED = 0.4

        # PID constants (tune later with real sensor input)
        self.KP = 0.8
        self.KI = 0.02
        self.KD = 0.1

        self.KP_X = 0.8    # Ganancia Proporcional: velocidad de reacción inicial
        self.KI_X = 0.1   # Ganancia Integral: corrige errores acumulados o fricción
        self.KD_X = 0.1    # Ganancia Derivativa: evita que el robot oscile (frena antes de llegar)

        self.TOLERANCIA_R = 5          # pixels (fallback for camera detectors)
        self.TOLERANCIA_R_RAD = 0.05      # radians (~8.6 deg, for bearing-based detectors)
        self.TOLERANCIA_D = 0.05

        self.error_acumulado = 0.0
        self.ultimo_error = 0.0
        self.error_rot_acumulado = 0.0
        self.ultimo_rot_error = 0.0
        self.ultima_vez = time.time()
        self.print_every_n_ticks = 20
        self.debug_tick = 0
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
                self.debug_tick += 1
                should_log = (self.debug_tick % self.print_every_n_ticks) == 0

                # Get current target from detector
                self.target = self.detector.get_target() #target is a list with valid x y radius color conficence
                
                if self.target is None:
                    # No target detected, stop or wait
                    if should_log:
                        print("No target detected")
                    time.sleep(0.1)
                    continue

                ahora = time.time()
                dt = ahora - self.ultima_vez

                if dt <= 0.0:
                    dt = 0.05

                # Use bearing (radians) from world-point detectors if available,
                # otherwise fall back to pixel-based offset for camera detectors.
                if "bearing" in self.target:
                    error_rot = self.target["bearing"]  # radians; positive = target is left
                    tol_rot = self.TOLERANCIA_R_RAD
                else:
                    img_center = self.target.get("image_width", 820) / 2.0
                    error_rot = img_center - self.target["x"]  # pixels
                    tol_rot = self.TOLERANCIA_R
                error = self.target["distance"] - self.desired_distance
                
                if (self.TOLERANCIA_D < abs(error) and self.rot_flag == False and self.forward_flag == True):

                    # If heading drift grows while driving forward, re-enter
                    # the rotation phase before continuing.
                    if abs(error_rot) > (2.0 * tol_rot):
                        self.ctx.actions.drive.stop()
                        self.forward_flag = False
                        self.rot_flag = True
                        self.ultimo_rot_error = 0.0
                        self.error_rot_acumulado = 0.0
                        if should_log:
                            print(f"Heading drift {error_rot:.3f} too large, rotating to re-align")
                        self.ultima_vez = ahora
                        time.sleep(0.05)
                        continue

                    # Forward PID only - no turning while driving
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

                    if should_log:
                        print(f"Error: {error:.3f} m, Velocity: {velocidad:.3f} m/s")
                    self.ctx.actions.drive.rc(velocidad, 0.0)

                
            # Completion condition: stable around target for a while
                    '''
                    if abs(error) < self.stable_band:
                        if self.stable_since is None:
                            self.stable_since = ahora
                        elif ahora - self.stable_since >= self.stable_time_required:
                            print(f"Target reached and stable for {self.stable_time_required} seconds. Stopping.")
                            self.ctx.actions.drive.stop()
                            self.hasReachedTarget = True
                    else:
                        self.stable_since = None
                    '''
                    
                    self.ultimo_error = error
                    self.ultima_vez = ahora
                    
                elif abs(error) <= self.TOLERANCIA_D and self.rot_flag == False and self.forward_flag == True:
                    print("Target reached.")
                    self.ctx.actions.drive.stop()
                    self.hasReachedTarget = True
                    
                if(abs(error_rot) >= tol_rot and self.rot_flag == True):
                    self.rot_flag = True
                    P = self.KP_X * error_rot
            
                    # Integral (evita que el robot se quede "atascado" por fricción cerca del 0)
                    self.error_rot_acumulado += error_rot * dt
                    I = self.KI_X * self.error_rot_acumulado
                    
                    # Derivativo (amortigua el giro para no pasarse de largo)
                    D = self.KD_X * (error_rot - self.ultimo_rot_error) / dt
                    
                    # Salida: Velocidad angular W
                    # Nota: Multiplicamos por -1 si el sistema de coordenadas del robot es inverso
                    w_speed = P + I + D
                    
                    # Saturación por seguridad
                    if w_speed > self.MAX_W_SPEED: w_speed = self.MAX_W_SPEED
                    if w_speed < -self.MAX_W_SPEED: w_speed = -self.MAX_W_SPEED
                    if should_log:
                        print(f"Rot Error: {error_rot:.3f}, W Speed: {w_speed:.3f}")
                    # Enviar comando: lineal 0, angular w_speed
                    self.ctx.service.send("robobot/cmd/ti", f"rc 0 {w_speed:.3f}")
                    
                    # Actualizar variables
                    self.ultimo_rot_error = error_rot
                    self.ultima_vez = ahora
                    
                elif(self.forward_flag == False):
                    # Si el error de rotación es pequeño, no giramos
                    self.ctx.service.send("robobot/cmd/ti", f"rc 0 0")
                    self.ultimo_rot_error = 0.0
                    self.error_rot_acumulado = 0.0
                    self.rot_flag = False
                    self.forward_flag = True
                    print("Rotation aligned, moving forward")
                    time.sleep(3)  # Control loop frequency
                    
                time.sleep(0.01)  # Control loop frequency
                
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
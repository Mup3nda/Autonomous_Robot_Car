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
        self.MAX_W_SPEED = 0.2

        # PID constants (tune later with real sensor input)
        self.KP = 0.8
        self.KI = 0.02
        self.KD = 0.1

        self.KP_X = 0.8    # Ganancia Proporcional: velocidad de reacción inicial
        self.KI_X = 0.1   # Ganancia Integral: corrige errores acumulados o fricción
        self.KD_X = 0.1    # Ganancia Derivativa: evita que el robot oscile (frena antes de llegar)

        self.TOLERANCIA_R = 5
        self.TOLERANCIA_D = 0.05

        self.error_acumulado = 0.0
        self.ultimo_error = 0.0
        self.error_rot_acumulado = 0.0
        self.ultimo_rot_error = 0.0
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
                self.target = self.detector.get_target() #target is a list with valid x y radius color conficence
                
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
                error_rot =  820/2 - self.target["x"]
                
                if (self.TOLERANCIA_D < abs(error) and self.rot_flag == False and self.forward_flag == True):
                    #self.ctx.actions.drive.stop()
                    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                    
                    error = self.target["distance"] - self.desired_distance

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
                    #print error and velocity for debugging
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
                    if abs(error_rot) <= self.TOLERANCIA_R:
                        print(f"Target reached and stable for {self.stable_time_required} seconds. Stopping.")
                        self.ctx.actions.drive.stop()
                        self.ultimo_error = 0.0
                        self.error_acumulado = 0.0
                        self.forward_flag = False
                        print("Distance aligned, rotating to target")
                        self.hasReachedTarget = True
                        time.sleep(2)  # Control loop frequency
                    else:
                        self.rot_flag = True
                    
                if(abs(error_rot) >= self.TOLERANCIA_R and self.rot_flag == True):
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
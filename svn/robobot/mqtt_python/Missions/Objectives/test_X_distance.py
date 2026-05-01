import time
from enum import IntEnum

from objective import Objective


class TestXDState(IntEnum):
    START = 0
    CONTROL = 1
    DONE = 99


class TestXDObjective(Objective):
    name = "test"

    def start(self, ctx):
        self.state = TestXDState.START

        # Objective target and controller limits
        self.DISTANCIA_DESEADA = 0.41  # meters
        self.MAX_SPEED = 0.6  # m/s

        # PID constants (tune later with real sensor input)
        self.KP = 0.8
        self.KI = 0.02
        self.KD = 0.1

        # Fake sensor model state (until vision target distance exists)
        self.distancia_actual = 2.0
        self.error_acumulado = 0.0
        self.ultimo_error = 0.0
        self.ultima_vez = time.time()
        # Inject a one-time overshoot near setpoint to test reverse recovery.
        self.overshoot_injected = True
        self.overshoot_trigger_band = 0.1  # trigger when within +20cm above setpoint
        self.overshoot_amount = 0.5  # jump 18cm too close to force backing up

        # Consider objective complete after stable hold near setpoint
        self.stable_since = None
        self.stable_band = 0.03  # +/- 3 cm
        self.stable_time_required = 2.0  # seconds

    def tick(self, ctx):
        if self.state == TestXDState.START:
            print(f"Manteniendo distancia de {self.DISTANCIA_DESEADA}m (modo simulado)...")
            self.state = TestXDState.CONTROL
            return

        if self.state != TestXDState.CONTROL:
            return

        ahora = time.time()
        dt = ahora - self.ultima_vez
        if dt <= 0.0:
            dt = 0.05

        # PID error: positive means we are too far and should move forward
        error = self.distancia_actual - self.DISTANCIA_DESEADA

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
        ctx.actions.drive.rc(velocidad, 0.0)

        # Fake distance update each tick (simple kinematic model)
        # Positive forward velocity reduces target distance.
        self.distancia_actual -= velocidad * dt
        if (not self.overshoot_injected and
                self.distancia_actual <= self.DISTANCIA_DESEADA + self.overshoot_trigger_band):
            self.distancia_actual -= self.overshoot_amount
            self.overshoot_injected = True
            print("% Sim: overshoot injected, now too close -> controller should back up")
        if self.distancia_actual < 0.1:
            self.distancia_actual = 0.1

        # Completion condition: stable around target for a while
        if abs(error) < self.stable_band:
            if self.stable_since is None:
                self.stable_since = ahora
            elif ahora - self.stable_since >= self.stable_time_required:
                ctx.actions.drive.stop()
                print(f"Distancia estabilizada en {self.distancia_actual:.3f}m (simulada).")
                self.state = TestXDState.DONE
                self._done = True
        else:
            self.stable_since = None

        self.ultimo_error = error
        self.ultima_vez = ahora

    def stop(self, ctx):
        ctx.actions.drive.stop()

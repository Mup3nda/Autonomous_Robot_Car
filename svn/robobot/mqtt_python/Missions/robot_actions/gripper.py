#!/usr/bin/env python3
"""
gripper.py
----------
Controls a servo gripper connected directly to a Raspberry Pi GPIO pin using
hardware PWM via the RPi.GPIO library.

Wiring (standard hobby servo — 3 wires):
  VCC (red)    → RPi 5V  (pin 2 or 4)
  GND (brown)  → RPi GND (pin 6, 9, 14, …)
  Signal (orange/yellow) → RPi GPIO pin (default: GPIO 18, pin 12)
                           GPIO 18 supports hardware PWM on all RPi models.

Servo PWM basics:
  Frequency : 50 Hz  (20 ms period)
  Duty cycle: 2.5 %  → ~0.5 ms pulse  → typically full CW  (~0°)
              7.5 %  → ~1.5 ms pulse  → centre             (~90°)
             12.5 %  → ~2.5 ms pulse  → full CCW           (~180°)

Tune GRIPPER_OPEN_DC / GRIPPER_CLOSE_DC for your physical servo.
"""

import RPi.GPIO as GPIO
import time
import sys

# ── Configuration ─────────────────────────────────────────────────────────────

SERVO_PIN       = 18       # BCM GPIO pin number (GPIO 18 = physical pin 12)
PWM_FREQUENCY   = 50       # Hz — standard for hobby servos

# Duty-cycle values (percentage) — tune to match your physical gripper travel.
# Typical range: 2.5 (fully CW) … 12.5 (fully CCW) for a 180° servo.
GRIPPER_OPEN_DC  = 2.5     # Fully open
GRIPPER_CLOSE_DC = 12.5    # Fully closed / gripping
GRIPPER_MID_DC   = 7.5     # Safe transit / middle position

# How long (seconds) to hold the PWM signal after a move command before
# going idle (prevents servo jitter while holding position).
MOVE_HOLD_TIME  = 0.5      # seconds

# ── Gripper class ─────────────────────────────────────────────────────────────

class Gripper:
    """Thin wrapper around RPi.GPIO PWM for a single-servo gripper."""

    def __init__(self, pin: int = SERVO_PIN, freq: int = PWM_FREQUENCY):
        self.pin  = pin
        self.freq = freq
        self._pwm = None
        self._setup()

    def _setup(self):
        """Initialise GPIO and start PWM at 0 % duty (idle / no pulse)."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.pin, GPIO.OUT)
        self._pwm = GPIO.PWM(self.pin, self.freq)
        self._pwm.start(0)  # start with no pulse — servo stays still
        print(f"[GRIPPER] Initialised on GPIO {self.pin} @ {self.freq} Hz")

    # ── public API ────────────────────────────────────────────────────────────

    def set_duty(self, duty_cycle: float):
        """
        Send a raw duty-cycle value (%) to the servo and hold briefly.
        Use this when you want fine-grained control beyond the three presets.
        """
        duty_cycle = max(0.0, min(100.0, duty_cycle))  # clamp to valid range
        self._pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(MOVE_HOLD_TIME)
        self._pwm.ChangeDutyCycle(0)  # silence signal — reduces jitter

    def open(self):
        """Move gripper to the fully OPEN position."""
        print(f"[GRIPPER] Opening  (duty={GRIPPER_OPEN_DC} %)")
        self.set_duty(GRIPPER_OPEN_DC)

    def close(self):
        """Move gripper to the fully CLOSED / gripping position."""
        print(f"[GRIPPER] Closing  (duty={GRIPPER_CLOSE_DC} %)")
        self.set_duty(GRIPPER_CLOSE_DC)

    def middle(self):
        """Move gripper to the MID / safe transit position."""
        print(f"[GRIPPER] Mid pos  (duty={GRIPPER_MID_DC} %)")
        self.set_duty(GRIPPER_MID_DC)

    def command(self, action: str):
        """
        High-level command interface — mirrors the C++ commandGripper(char).
        action: 'o' = open | 'c' = close | 'm' = middle
        """
        action = action.lower()
        if   action == 'o': self.open()
        elif action == 'c': self.close()
        elif action == 'm': self.middle()
        else:
            print(f"[GRIPPER] Unknown action '{action}'. Use o / c / m.")

    def cleanup(self):
        """Release GPIO resources. Always call this on exit."""
        self._pwm.stop()
        GPIO.cleanup()
        print("[GRIPPER] GPIO cleaned up.")

# ── Interactive test loop ─────────────────────────────────────────────────────

def test_gripper_loop(gripper: Gripper):
    """
    Manual keyboard test — equivalent to the C++ testGripperLoop().
    Replace this loop with your ball-detection trigger when ready:

        ball_detected = (ball_distance < THRESHOLD)
        gripper.command('c' if ball_detected else 'o')
    """
    print("\n=============================")
    print(" Servo Gripper Python Test")
    print(f" GPIO pin : {gripper.pin}")
    print("=============================")
    print(" [o] - Open gripper")
    print(" [c] - Close gripper (grip)")
    print(" [m] - Mid / transit position")
    print(" [q] - Quit")
    print("=============================\n")

    # Default: open on start
    gripper.open()

    while True:
        try:
            key = input("Enter command: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if key == 'q':
            print("[TEST] Quitting — returning gripper to OPEN")
            gripper.open()
            break
        elif key in ('o', 'c', 'm'):
            gripper.command(key)
        elif key == '':
            continue
        else:
            print("[TEST] Unknown command. Use o, c, m, or q.")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gripper = Gripper(pin=SERVO_PIN, freq=PWM_FREQUENCY)
    try:
        test_gripper_loop(gripper)
    finally:
        gripper.cleanup()
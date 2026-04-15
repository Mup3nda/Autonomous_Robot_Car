"""Drive action commands: motor control, LED, and servo control."""

import time

class DriveActions:
    """Provides high-level drive control commands.
    
    Args:
        service: MQTT service for sending commands to Teensy
        gpio: GPIO interface for direct pin control
    """
    def __init__(self, service, gpio):
        self.service = service
        self.gpio = gpio
        self.current_v = 0.0
        self.current_w = 0.0

    def rc(self, v, w):
        """Send motor command to robot.
        
        Args:
            v: Forward/backward throttle (-1.0 to 1.0, where 0.2 = 20% forward)
            w: Rotation rate (-1.0 to 1.0, where 0.5 = rotation direction)
        """
        self.current_v = v
        self.current_w = w
        self.service.send("robobot/cmd/ti", f"rc {v} {w}")
    
    def ramp_to(self, target_v, target_w, steps=10):
        start_v = self.current_v
        start_w = self.current_w
        
        for i in range(1, steps + 1):
            factor = i / steps
            v = start_v + factor * (target_v - start_v)
            w = start_w + factor * (target_w - start_w)
            self.rc(v, w)
            time.sleep(0.05)

    def stop(self,instant = True):
        """Stop all robot movement (throttle and rotation to 0)."""
        # Ramp down for smoothness
        start_v = self.current_v
        start_w = self.current_w
        steps = 10 #5
        if not instant:
            for i in range(steps):
                factor = (steps - i - 1) / steps
                self.rc(start_v * factor, start_w * factor)
                time.sleep(0.05)
        self.rc(0.0, 0.0)
        

    def leds(self, r, g, b, led=16):
        """Control LED color.
        
        Args:
            r: Red intensity (0-255)
            g: Green intensity (0-255)
            b: Blue intensity (0-255)
            led: LED number (default 16)
        """
        self.service.send("robobot/cmd/T0", f"leds {led} {r} {g} {b}")

    def servo(self, idx, pos, speed):
        """Move servo to position.
        
        Args:
            idx: Servo index (1-based)
            pos: Target position (e.g., -800 to 500 range)
            speed: Movement speed
        """
        self.service.send("robobot/cmd/T0", f"servo {idx} {pos} {speed}")

    def lognow(self, level=3):
        """Trigger logging at specified level.
        
        Args:
            level: Log level (default 3)
        """
        self.service.send("robobot/cmd/T0/", f"lognow {level}")

    def set_gpio(self, pin, value):
        """Set GPIO pin output.
        
        Args:
            pin: GPIO pin number
            value: 0 (low) or 1 (high)
        """
        self.gpio.set_value(pin, value)

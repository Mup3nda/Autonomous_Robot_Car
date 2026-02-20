class DriveActions:
    def __init__(self, service, gpio):
        self.service = service
        self.gpio = gpio

    def rc(self, v, w):
        self.service.send("robobot/cmd/ti", f"rc {v} {w}")

    def stop(self):
        self.rc(0.0, 0.0)

    def leds(self, r, g, b, led=16):
        self.service.send("robobot/cmd/T0", f"leds {led} {r} {g} {b}")

    def servo(self, idx, pos, speed):
        self.service.send("robobot/cmd/T0", f"servo {idx} {pos} {speed}")

    def lognow(self, level=3):
        self.service.send("robobot/cmd/T0/", f"lognow {level}")

    def set_gpio(self, pin, value):
        self.gpio.set_value(pin, value)

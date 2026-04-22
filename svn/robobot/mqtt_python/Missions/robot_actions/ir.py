"""IR sensor actions: high-level access to the shared IR distance stream.

This wraps the global ``sir.ir`` object in the same mission-facing style as
the arm and gripper helpers. It does not start hardware on its own; the IR
stream is already initialized by ``uservice``.
"""

import time


class IRSensorActions:
    """Mission-friendly access to the shared IR distance readings."""

    def __init__(self, ir_sensor, memory=None):
        self.ir_sensor = ir_sensor
        self.memory = memory

    def bind_memory(self, memory):
        self.memory = memory

    def has_data(self):
        """Return True once at least one IR update has been received."""
        return int(self.ir_sensor.irUpdCnt) > 0

    def latest(self):
        """Return both IR readings and timing info as a small dict."""
        return {
            "left": float(self.ir_sensor.ir[0]),
            "right": float(self.ir_sensor.ir[1]),
            "update_count": int(self.ir_sensor.irUpdCnt),
            "interval_s": float(self.ir_sensor.irInterval),
            "time": self.ir_sensor.irTime,
        }

    def left(self):
        return float(self.ir_sensor.ir[0])

    def right(self):
        return float(self.ir_sensor.ir[1])

    def remember_latest(self, key="_ir_latest"):
        """Store the most recent IR readings in mission memory."""
        data = self.latest()
        if self.memory is not None:
            self.memory[str(key)] = data
        return data

    def wait_until_ready(self, timeout_s=1.0, poll_s=0.02):
        """Block briefly until the first IR reading arrives."""
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            if self.has_data():
                return True
            time.sleep(float(poll_s))
        return self.has_data()

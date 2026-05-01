from spose import pose
from sir import ir
from scam_usb import cam_usb as cam
#from scam import cam
from sedge import edge
from sgpio import gpio
from uservice import service
import time as t
from datetime import datetime
from robot_actions import RobotActions

class MissionContext:
    def __init__(self, service):
        actions = RobotActions(service, gpio, cam, edge)
        self.actions = actions
        self.pose = pose
        self.ir = ir
        self.edge = edge
        self.cam = cam
        self.gpio = gpio
        self.service = service
        self.state_time = datetime.now()
        self.memory = {} # a place to store arbitrary data across objectives
        self.actions.arm.bind_memory(self.memory)
        self.actions.ir.bind_memory(self.memory)

    def reset_state_time(self):
        self.state_time = datetime.now()

    def state_time_passed(self):
        return (datetime.now() - self.state_time).total_seconds()

    def start_local_progress(self, name="default"):
        """Store a local progress baseline using non-resettable odometry counters."""
        progress_map = self.memory.setdefault("_local_progress", {})
        progress_map[str(name)] = {
            "tripA": float(self.pose.tripA),
            "tripAh": float(self.pose.tripAh),
            "time_s": t.time(),
        }

    def distance_since_start(self, name="default"):
        """Return signed distance in meters since start_local_progress(name)."""
        key = str(name)
        progress_map = self.memory.setdefault("_local_progress", {})
        marker = progress_map.get(key)
        if marker is None:
            self.start_local_progress(key)
            return 0.0
        return float(self.pose.tripA) - marker["tripA"]

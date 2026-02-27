from spose import pose
from sir import ir
from scam import cam
from sedge import edge
from sgpio import gpio
from uservice import service
import time as t
from datetime import datetime

class MissionContext:
    def __init__(self, actions):
        self.actions = actions
        self.pose = pose
        self.ir = ir
        self.edge = edge
        self.cam = cam
        self.gpio = gpio
        self.service = service
        self.state_time = datetime.now()
        self.memory = {} # a place to store arbitrary data across objectives

    def reset_state_time(self):
        self.state_time = datetime.now()

    def state_time_passed(self):
        return (datetime.now() - self.state_time).total_seconds()
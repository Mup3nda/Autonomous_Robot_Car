from objective import Objective
from robot_actions import RobotActions
from mission_context import MissionContext
import time as t
from datetime import datetime
import numpy as np

class LineTurnImageObjective(Objective):
    name = "line_turn_image"

    def start(self, ctx):
        self.state = 0
        self.images = 0
        self.ledon = True
        self.state_time = datetime.now()
        ctx.edge.lineControl(0, True)
        ctx.actions.drive.leds(30, 30, 0)
        print("% Starting line/turn/image objective")

    def _state_time_passed(self):
        return (datetime.now() - self.state_time).total_seconds()

    def _set_state(self, state):
        self.state = state
        self.state_time = datetime.now()

    def tick(self, ctx):
        if self.state == 0:
            start = True
            if start:
                ctx.actions.drive.leds(0, 0, 30)
                ctx.actions.drive.rc(0.25, 0.0)
                ctx.actions.drive.servo(1, 100, 300)
                ctx.pose.tripBreset()
                self._set_state(12)
        elif self.state == 12:
            if ctx.pose.tripB > 0.5 or ctx.pose.tripBtimePassed() > 10:
                ctx.edge.lineControl(0, True)
                ctx.pose.tripBreset()
                ctx.actions.drive.rc(0.1, 0.5)
                ctx.actions.drive.servo(1, -800, 1000)
                self._set_state(14)
        elif self.state == 14:
            if ctx.pose.tripBh > np.pi / 2 or ctx.pose.tripBtimePassed() > 10:
                ctx.actions.drive.stop()
                ctx.actions.drive.servo(1, 0, 1000)
                self._set_state(20)
        elif self.state == 20:
            ctx.actions.vision.image_analysis(self.images == 2)
            self.images += 1
            if self.ledon:
                ctx.actions.drive.leds(0, 64, 0)
                ctx.actions.drive.set_gpio(20, 1)
            else:
                ctx.actions.drive.leds(0, 30, 30)
                ctx.actions.drive.set_gpio(20, 0)
            self.ledon = not self.ledon
            if self.images >= 10 or (not ctx.cam.useCam) or self._state_time_passed() > 20:
                self._done = True
        t.sleep(0.1)

    def stop(self, ctx):
        ctx.actions.drive.leds(0, 0, 0)
        ctx.actions.drive.set_gpio(20, 0)
        ctx.edge.lineControl(0, True)
        ctx.actions.drive.stop()
        ctx.actions.drive.servo(1, 0, 0)
        print("% Line/turn/image objective end")
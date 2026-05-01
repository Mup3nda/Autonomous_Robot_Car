"""Move arm to down position and finish immediately."""

from objective import Objective


class ArmDownObjective(Objective):
    name = "arm_down"
    def __init__(self, wait_after_s = 0.0):
        super().__init__()
        self.wait_after_s = float(wait_after_s)
    def start(self, ctx):
        ctx.actions.arm.move_down()
        self.tick_count = 0
        if(self.wait_after_s <= 0.0):
            self._done = True

    def tick(self, ctx):
        self.tick_count += 1
        if self.wait_after_s > 0:
            if self.tick_count >= int(self.wait_after_s / 0.01):
                self._done = True
                
        pass

    def stop(self, ctx):
        pass

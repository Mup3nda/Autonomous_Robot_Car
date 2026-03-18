"""Move arm to down position and finish immediately."""

from objective import Objective


class ArmDownObjective(Objective):
    name = "arm_down"

    def start(self, ctx):
        ctx.actions.arm.move_down()
        self._done = True

    def tick(self, ctx):
        pass

    def stop(self, ctx):
        pass

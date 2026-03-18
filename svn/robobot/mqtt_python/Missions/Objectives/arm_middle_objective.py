"""Move arm to middle position and finish immediately."""

from objective import Objective


class ArmMiddleObjective(Objective):
    name = "arm_middle"

    def start(self, ctx):
        ctx.actions.arm.move_mid()
        self._done = True

    def tick(self, ctx):
        pass

    def stop(self, ctx):
        pass

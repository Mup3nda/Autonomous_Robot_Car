"""Move arm to up position and finish immediately."""

from objective import Objective


class ArmUpObjective(Objective):
    name = "arm_up"

    def start(self, ctx):
        ctx.actions.arm.move_up()
        self._done = True

    def tick(self, ctx):
        pass

    def stop(self, ctx):
        pass

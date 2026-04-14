"""Move gripper to open position and finish immediately."""

from objective import Objective


class GripperOpenObjective(Objective):
    name = "gripper_open"

    def start(self, ctx):
        ctx.actions.gripper.open()
        self._done = True

    def tick(self, ctx):
        pass

    def stop(self, ctx):
        pass
"""Move gripper to close position and finish immediately."""

from objective import Objective


class GripperCloseObjective(Objective):
    name = "gripper_close"

    def start(self, ctx):
        ctx.actions.gripper.close()
        self._done = True

    def tick(self, ctx):
        pass

    def stop(self, ctx):
        pass
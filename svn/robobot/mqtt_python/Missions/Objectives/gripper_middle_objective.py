"""Move gripper to middle position and finish immediately."""

from objective import Objective


class GripperMiddleObjective(Objective):
    name = "gripper_middle"

    def start(self, ctx):
        ctx.actions.gripper.middle()
        self._done = True

    def tick(self, ctx):
        pass

    def stop(self, ctx):
        pass
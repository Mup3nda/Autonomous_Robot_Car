"""Composite objective for dropping a target and clearing the arm."""

from objective import CompositeObjective

from Objectives.arm_down_objective import ArmDownObjective
from Objectives.arm_up_objective import ArmUpObjective
from Objectives.delay_objective import DelayObjective
from Objectives.gripper_close_objective import GripperCloseObjective
from Objectives.gripper_open_objective import GripperOpenObjective


class DropTargetObjective(CompositeObjective):
    """Lower arm, open/close gripper, then lift arm with delays between steps."""

    name = "drop_target"

    def __init__(self, delay_s=1.0):
        objectives = [
            ArmDownObjective(),
            DelayObjective(delay_s),
            GripperOpenObjective(),
            DelayObjective(delay_s),
            GripperCloseObjective(),
            ArmUpObjective(),
            DelayObjective(delay_s),
        ]
        super().__init__(objectives)
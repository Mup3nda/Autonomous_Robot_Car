"""Composite objective for grabbing a target in front of the robot."""

from objective import CompositeObjective

from Objectives.arm_down_objective import ArmDownObjective
from Objectives.arm_up_objective import ArmUpObjective
from Objectives.delay_objective import DelayObjective
from Objectives.drive_to_waypoint_objective import DriveToWaypointObjective
from Objectives.gripper_close_objective import GripperCloseObjective
from Objectives.gripper_open_objective import GripperOpenObjective


class GrabTargetObjective(CompositeObjective):
    """Open, lower, approach, close, and lift to grab a nearby target."""

    name = "grab_target"

    def __init__(
        self,
        nav_mode,
        approach_waypoint=(0.1, 0.0),
        approach_is_local=True,
        approach_print_interval=20,
        approach_heading_deg=0.0,
        arm_down_wait_s=2.0,
        pre_arm_delay_s=1.0,
        pre_drive_delay_s=1.0,
        post_grip_delay_s=1.0,
    ):
        objectives = [
            GripperOpenObjective(),
            DelayObjective(pre_arm_delay_s),
            ArmDownObjective(wait_after_s=arm_down_wait_s),
            DelayObjective(pre_drive_delay_s),
            DriveToWaypointObjective(
                waypoint=approach_waypoint,
                is_local=approach_is_local,
                print_interval=approach_print_interval,
                relative_heading_deg=approach_heading_deg,
                nav_mode=nav_mode,
            ),
            GripperCloseObjective(),
            DelayObjective(post_grip_delay_s),
            ArmUpObjective(),
        ]
        super().__init__(objectives)
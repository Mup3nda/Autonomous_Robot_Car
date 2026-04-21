from objective import CompositeObjective

from Objectives.arm_down_objective import ArmDownObjective
from Objectives.arm_middle_objective import ArmMiddleObjective
from Objectives.arm_up_objective import ArmUpObjective
from Objectives.drive_backward_until_line_stop_objective import DriveBackwardUntilLineStopObjective
from Objectives.drive_circle_objective import DriveCircleObjective
from Objectives.drive_to_line_objective import DriveToLineObjective
from Objectives.drive_to_line_zone_switch_objective import DriveToLineZoneSwitchObjective
from Objectives.drive_to_timer_and_back_objective import DriveToTimerAndBackObjective
from Objectives.drive_to_waypoint_objective import DriveToWaypointObjective
from Objectives.drive_to_waypoint_until_line_count_objective import DriveToWaypointUntilLineCountObjective
from Objectives.drive_turn_angle_objective import DriveTurnAngleObjective
from Objectives.delay_objective import DelayObjective
from Objectives.gripper_open_objective import GripperOpenObjective
from Objectives.search_and_navigate_to_aruco_objective import SearchAndNavigateToAruco
from Objectives.reset_origin_objective import ResetOriginObjective


DUMMY_ARUCO_MARKER_ID = 0
class FollowLineApproachRoundaboutComposite(CompositeObjective):
    name = "follow_line_approach_roundabout"

    def __init__(self):
        super().__init__([
            DriveToLineObjective(
                follow_left=True,
                follow_speed=0.4,
                search_speed=0.25,
                centering_speed=0.2,
                lost_line_timeout_s=0,
                max_line_distance_m=1.90,
                max_duration=0.0,
            )
        ])


class RoundaboutComposite(CompositeObjective):
    name = "roundabout"

    def __init__(self, nav_mode, radius_m, revolutions, forward_cmd, turn_cmd, turn_rate_scale, clockwise, timeout_s):
        super().__init__([
            DriveToWaypointObjective(
                waypoint=(0.40, 0.0),
                is_local=True,
                print_interval=20,
                nav_mode=nav_mode,
            ),
            DriveTurnAngleObjective(angle_deg=90, linear_cmd=0.0, timeout_s=6.0),
            DriveCircleObjective(
                radius_m=radius_m,
                revolutions=revolutions,
                forward_cmd=forward_cmd,
                turn_cmd=turn_cmd,
                turn_rate_scale=turn_rate_scale,
                clockwise=clockwise,
                timeout_s=timeout_s,
            ),
            DriveTurnAngleObjective(angle_deg=90.0, linear_cmd=0.0, timeout_s=6.0),
        ])


class PostRoundaboutLineFollowAndExitComposite(CompositeObjective):
    name = "post_roundabout_line_follow_and_exit"

    def __init__(self, follow_left, search_speed, switch_zones, lost_line_timeout_s):
        super().__init__([
            DriveToLineZoneSwitchObjective(
                follow_left=follow_left,
                follow_speed=0.7,
                search_speed=search_speed,
                lost_line_timeout_s=lost_line_timeout_s,
                switch_zones=switch_zones,
                ordered_switches=True,
                instant_stop=False,
            ),
            DriveBackwardUntilLineStopObjective(
                reverse_speed=-0.2,
                line_found_confidence=2,
                timeout_s=8.0,
                max_distance_m=1.5,
            ),
            ResetOriginObjective(),
            DelayObjective(1.0),
        ])


class GetExtraTimeReturnToLineComposite(CompositeObjective):
    name = "get_extra_time_return_to_line"

    def __init__(self):
        super().__init__([
            ArmUpObjective(),
            DriveToTimerAndBackObjective(),
            DelayObjective(2.0),
        ])


class FollowLineBeforeKnockCupComposite(CompositeObjective):
    name = "follow_line_before_knock_cup"

    def __init__(self):
        super().__init__([
            DriveToLineObjective(
                follow_left=True,
                follow_speed=0.4,
                search_speed=0.25,
                centering_speed=0.2,
                lost_line_timeout_s=0,
                max_duration=0.0,
                max_line_distance_m=0.6,
            ),
            DelayObjective(2.0),
        ])


class KnockDownCupComposite(CompositeObjective):
    name = "knock_down_cup"

    def __init__(self):
        super().__init__([
            ArmMiddleObjective(),
            DelayObjective(2.0),
            DriveTurnAngleObjective(angle_deg=-100, linear_cmd=0.0, timeout_s=6.0),
            DelayObjective(2.0),
            ArmUpObjective(),
            DriveTurnAngleObjective(angle_deg=100, linear_cmd=0.0, timeout_s=6.0),
        ])


class GoToBallPickupLocationAComposite(CompositeObjective):
    name = "go_to_ball_pickup_location_a"

    def __init__(self, nav_mode):
        super().__init__([
            DriveToWaypointObjective(
                waypoint=(0.2, 0.0),
                is_local=True,
                print_interval=20,
                relative_heading_deg=-0.0,
                nav_mode=nav_mode,
            ),
            DriveToWaypointObjective(
                waypoint=(0.6, 1.2),
                is_local=False,
                print_interval=20,
                relative_heading_deg=-100.0,
                nav_mode=nav_mode,
            ),
        ])


class GoToBlueBallDropoffComposite(CompositeObjective):
    name = "go_to_blue_ball_dropoff"

    def __init__(self, nav_mode):
        marker_id = DUMMY_ARUCO_MARKER_ID
        objectives: list[Objective] = [
            DelayObjective(2.0),
            DriveToWaypointObjective(
                waypoint=(0.9, 1.3),
                is_local=False,
                print_interval=20,
                relative_heading_deg=0.0,
                nav_mode=nav_mode,
            ),
            SearchAndNavigateToAruco(marker_id=marker_id, desired_distance=0.35),
            ArmDownObjective(),
            GripperOpenObjective(),
            DelayObjective(2.0),
            ArmUpObjective(),
            DriveTurnAngleObjective(angle_deg=170.0, linear_cmd=0.0, timeout_s=6.0),
        ]
        super().__init__(objectives)


class GoToBallPickupLocationBComposite(CompositeObjective):
    name = "go_to_ball_pickup_location_b"

    def __init__(self, nav_mode):
        super().__init__([
            DriveToWaypointObjective(
                waypoint=(0.5, 1.0),
                is_local=False,
                print_interval=20,
                relative_heading_deg=-100.0,
                nav_mode=nav_mode,
            ),
            DelayObjective(2.0),
        ])


class GoToRedBallDropoffComposite(CompositeObjective):
    name = "go_to_red_ball_dropoff"

    def __init__(self, nav_mode):
        marker_id = DUMMY_ARUCO_MARKER_ID
        objectives: list[Objective] = [
            DriveToWaypointObjective(
                waypoint=(1.35, 0.8),
                is_local=False,
                print_interval=20,
                relative_heading_deg=90.0,
                nav_mode=nav_mode,
            ),
            SearchAndNavigateToAruco(marker_id=marker_id, desired_distance=0.35),
            ArmDownObjective(),
            GripperOpenObjective(),
            DelayObjective(2.0),
            ArmUpObjective(),
            DriveTurnAngleObjective(angle_deg=-90.0, linear_cmd=0.0, timeout_s=6.0),
        ]
        super().__init__(objectives)


class GoToPlatformPickupComposite(CompositeObjective):
    name = "go_to_platform_pickup"

    def __init__(self, nav_mode):
        super().__init__([
            DelayObjective(2.0),
            DriveToWaypointObjective(
                waypoint=(1.94, 0.7),
                is_local=False,
                print_interval=20,
                relative_heading_deg=0.0,
                nav_mode=nav_mode,
            ),
            DriveToWaypointObjective(
                waypoint=(0.2, 0.0),
                is_local=True,
                print_interval=20,
                relative_heading_deg=0.0,
                nav_mode=nav_mode,
            ),
        ])


class GoToPlatformDropoffDComposite(CompositeObjective):
    name = "go_to_platform_dropoff_d"

    def __init__(self, nav_mode):
        marker_id = DUMMY_ARUCO_MARKER_ID
        objectives: list[Objective] = [
            DriveToWaypointObjective(
                waypoint=(1.89, 1.1),
                is_local=False,
                print_interval=20,
                relative_heading_deg=165.0,
                nav_mode=nav_mode,
            ),
            SearchAndNavigateToAruco(marker_id=marker_id, desired_distance=0.35),
            ArmDownObjective(),
            GripperOpenObjective(),
            DelayObjective(2.0),
            ArmUpObjective(),
            DriveTurnAngleObjective(angle_deg=180.0, linear_cmd=0.0, timeout_s=6.0),
        ]
        super().__init__(objectives)


class GoToPlatformPickupAgainComposite(CompositeObjective):
    name = "go_to_platform_pickup_again"

    def __init__(self, nav_mode):
        super().__init__([
            DriveToWaypointObjective(
                waypoint=(1.94, 0.7),
                is_local=False,
                print_interval=20,
                relative_heading_deg=0.0,
                nav_mode=nav_mode,
            ),
        ])


class GoToPlatformDropoffAComposite(CompositeObjective):
    name = "go_to_platform_dropoff_a"

    def __init__(self, nav_mode):
        marker_id = DUMMY_ARUCO_MARKER_ID
        objectives: list[Objective] = [
            DriveToWaypointObjective(
                waypoint=(1.94, 1.6),
                is_local=False,
                print_interval=20,
                relative_heading_deg=110.0,
                nav_mode=nav_mode,
            ),
            DriveToWaypointObjective(
                waypoint=(1.5, 1.82),
                is_local=False,
                print_interval=20,
                relative_heading_deg=-90.0,
                nav_mode=nav_mode,
            ),
            SearchAndNavigateToAruco(marker_id=marker_id, desired_distance=0.35),
            ArmDownObjective(),
            GripperOpenObjective(),
            DelayObjective(2.0),
            ArmUpObjective(),
        ]
        super().__init__(objectives)


class ReturnToFinishLineComposite(CompositeObjective):
    name = "return_to_finish_line"

    def __init__(self, nav_mode):
        super().__init__([
            DelayObjective(2.0),
            DriveToWaypointUntilLineCountObjective(
                waypoint=(1.3, 5.0),
                is_local=False,
                print_interval=20,
                nav_mode=nav_mode,
                line_detect_confidence=4,
                line_clear_confidence=1,
                stop_line_count=2,
            ),
        ])


class GoToLineAndFollowToEndComposite(CompositeObjective):
    name = "go_to_line_and_follow_to_end"

    def __init__(self):
        super().__init__([
            DriveTurnAngleObjective(angle_deg=90.0, linear_cmd=0.0, timeout_s=6.0),
            DriveToLineObjective(
                follow_left=False,
                follow_speed=0.7,
                search_speed=0.25,
                centering_speed=0.2,
                lost_line_timeout_s=0,
                max_duration=0.0,
            ),
        ])

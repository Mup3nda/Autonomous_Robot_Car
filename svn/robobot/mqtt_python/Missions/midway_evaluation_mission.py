#!/usr/bin/env python3

import os
import sys
import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from uservice import service
from mission_runner import MissionRunner
from robot_actions import RobotActions
from mission_context import MissionContext
from objective import Objective
from Objectives.drive_circle_objective import DriveCircleObjective
from Objectives.drive_to_waypoint_objective import DriveToWaypointObjective
from Objectives.drive_turn_angle_objective import DriveTurnAngleObjective
from Objectives.align_to_circle_tangent_objective import AlignToCircleTangentObjective
from Objectives.search_and_navigate_to_blue_ball_objective import SearchAndNavigateToBlueBall
from Objectives.arm_up_objective import ArmUpObjective
from Objectives.arm_down_objective import ArmDownObjective
from Objectives.arm_middle_objective import ArmMiddleObjective
from Objectives.gripper_open_objective import GripperOpenObjective
from Objectives.gripper_middle_objective import GripperMiddleObjective
from Objectives.gripper_close_objective import GripperCloseObjective
from Objectives.drive_to_line_objective import DriveToLineObjective
from Objectives.drive_to_line_zone_switch_objective import DriveToLineZoneSwitchObjective
from Objectives.search_and_navigate_to_aruco_objective import SearchAndNavigateToAruco
from Objectives.search_and_navigate_to_platform_objective import SearchAndNavigateToPlatform
from Objectives.drive_to_timer_and_back_objective import DriveToTimerAndBackObjective
from Objectives.reset_origin_objective import ResetOriginObjective
from Objectives.drive_backward_until_line_stop_objective import DriveBackwardUntilLineStopObjective
from sodom import odom

# Roundabout three-step tuning parameters.
# Step 1: Entry line follow
LINE_ENTRY_FOLLOW_LEFT = False
LINE_ENTRY_FOLLOW_SPEED = 0.45
LINE_ENTRY_SEARCH_SPEED = 0.35
LINE_ENTRY_TIMEOUT_S = 0.35  # How long to wait after line disappears before handing to waypoint

# Step 2: Waypoint alignment (position robot at circle entry point)
WAYPOINT_FOR_CIRCLE_M = (0.3, 0.0)  # Distance (forward, sideways) from line end to circle entry
WAYPOINT_NAV_MODE = "smooth"  # "smooth" (drive+turn together) or "sequential" (rotate-then-drive)

# Step 3: Circle roundabout
CIRCLE_RADIUS_M = 0.36
CIRCLE_REVOLUTIONS = 1.5
CIRCLE_FORWARD_CMD = 0.28
CIRCLE_TURN_CMD = None  # Set e.g. 0.24 to override auto radius-based turning.
CIRCLE_TURN_RATE_SCALE = 1.0
CIRCLE_CLOCKWISE = True
CIRCLE_TIMEOUT_S = 40.0

# Entry alignment before starting circle drive.
ENTRY_TURN_1_DEG = 68.0
ENTRY_ADVANCE_AFTER_TURN_1_M = 0.20
ENTRY_LEG_1_WAYPOINT_M = (0.25, 0.0)
ENTRY_LEG_2_WAYPOINT_M = (
    ENTRY_LEG_1_WAYPOINT_M[0] + ENTRY_ADVANCE_AFTER_TURN_1_M * math.cos(math.radians(ENTRY_TURN_1_DEG)),
    ENTRY_LEG_1_WAYPOINT_M[1] + ENTRY_ADVANCE_AFTER_TURN_1_M * math.sin(math.radians(ENTRY_TURN_1_DEG)),
)

# Compute second in-place turn so heading is tangent to the circle.
# Assumption: in the local frame, the circle center lies on the original
# entry-line axis (y=0), and the second turn is done in place so robot center
# stays on that line point while aligning tangent heading.
_entry_offset_y_m = ENTRY_ADVANCE_AFTER_TURN_1_M * math.sin(math.radians(ENTRY_TURN_1_DEG))
_tangent_cos = (_entry_offset_y_m / CIRCLE_RADIUS_M) if CIRCLE_CLOCKWISE else (-_entry_offset_y_m / CIRCLE_RADIUS_M)
_tangent_cos = max(-1.0, min(1.0, _tangent_cos))
CIRCLE_ENTRY_TANGENT_HEADING_DEG = math.degrees(math.acos(_tangent_cos))
ENTRY_TURN_2_DEG = CIRCLE_ENTRY_TANGENT_HEADING_DEG - ENTRY_TURN_1_DEG
# Compute second in-place turn so heading is tangent to the circle.
# Assumption: in the local frame, the circle center lies on the original
# entry-line axis (y=0), and the second turn is done in place so robot center
# stays on that line point while aligning tangent heading.
_entry_offset_y_m = ENTRY_ADVANCE_AFTER_TURN_1_M * math.sin(math.radians(ENTRY_TURN_1_DEG))
_tangent_cos = (_entry_offset_y_m / CIRCLE_RADIUS_M) if CIRCLE_CLOCKWISE else (-_entry_offset_y_m / CIRCLE_RADIUS_M)
_tangent_cos = max(-1.0, min(1.0, _tangent_cos))
CIRCLE_ENTRY_TANGENT_HEADING_DEG = math.degrees(math.acos(_tangent_cos))
ENTRY_TURN_2_DEG = CIRCLE_ENTRY_TANGENT_HEADING_DEG - ENTRY_TURN_1_DEG

# Step 4: Exit line follow
LINE_EXIT_FOLLOW_LEFT = False
LINE_EXIT_FOLLOW_SPEED = 0.75

# Zone-based side/speed switches for the post-roundabout line follow.
# Tune x/y/radius in world frame using MissionLogs plot output.
POST_ROUNDABOUT_SWITCH_ZONES = [
    {
        "name": "slow_down_zone",
        "trigger_dist_m": 13.5,
        "follow_left": False,
        "follow_speed": 0.30,
        "lost_line_timeout_s": 0.0
    }
]

class DelayObjective(Objective):
    """Wait for a fixed amount of time, then finish."""

    def __init__(self, duration_s):
        super().__init__()
        self.duration_s = float(duration_s)

    def start(self, ctx):
        self._done = self.duration_s <= 0.0

    def tick(self, ctx):
        if ctx.state_time_passed() >= self.duration_s:
            self._done = True

    def stop(self, ctx):
        pass


# Add objectives in the list below in the exact order they should execute.
def build_objectives():
    objectives=[
    ## region Following line
     #   DriveToLineObjective(
     #       follow_left=True,
     #       follow_speed=0.4,
     #       search_speed=0.25,
     #       centering_speed=0.2,
     #       lost_line_timeout_s=0.3,
     #       max_line_distance_m=1.90,
     #       max_duration=0.0,
     #       ),
    ## endregion
    ##region Roundabout
     #   DriveToWaypointObjective(
     #       waypoint=(0.40,0.0),
     #       is_local=True,
     #       print_interval=20,
     #       nav_mode=WAYPOINT_NAV_MODE,
     #       ),
     #   DriveTurnAngleObjective(
     #       angle_deg=90,
     #       linear_cmd=0.0,
     #       timeout_s=6.0,
     #   ),
     #   DriveCircleObjective(
     #       radius_m=CIRCLE_RADIUS_M,
     #       revolutions=1.61, # one full circle + half circle
     #       forward_cmd=CIRCLE_FORWARD_CMD,
     #       turn_cmd=CIRCLE_TURN_CMD,
     #       turn_rate_scale=CIRCLE_TURN_RATE_SCALE,
     #       clockwise=CIRCLE_CLOCKWISE,
     #       timeout_s=CIRCLE_TIMEOUT_S,
     #   ),
     #   DriveTurnAngleObjective(
     #       angle_deg=90.0,
     #       linear_cmd=0.0,
     #       timeout_s=6.0,
     #   ),
    # endregion
    # region follow line and reduce speed before hitting 90 degree left turn, which should stop the robot at that junction 
        DriveToLineZoneSwitchObjective(
            follow_left=LINE_ENTRY_FOLLOW_LEFT,
            follow_speed=0.7,
            search_speed=LINE_ENTRY_SEARCH_SPEED,
            lost_line_timeout_s=LINE_ENTRY_TIMEOUT_S,
            switch_zones=POST_ROUNDABOUT_SWITCH_ZONES,
            ordered_switches=True,
        ),
    #    DriveToLineObjective(
    #        follow_left=False,
    #        follow_speed=0.7,
    #        search_speed=0.25,
    #        centering_speed=0.2,
    #        lost_line_timeout_s=0.01,
    #        max_line_distance_m=1.91,
    #        max_duration=0.0,
    #        ),
        DriveBackwardUntilLineStopObjective(
            reverse_speed=-0.25,
            line_found_confidence=2,
            timeout_s=8.0,
            max_distance_m=1.5,
        ), 
        ResetOriginObjective(), # new 0,0,0 origin after following line end
    # endregion
    #region Get extra time return to line
        ArmUpObjective(),
        DriveToTimerAndBackObjective(),
        # region Following line
        DelayObjective(2.0),
    # endregion
    # region Following line
        DriveToLineObjective(
            follow_left=True,
            follow_speed=0.4,
            search_speed=0.25,
            centering_speed=0.2,
            lost_line_timeout_s=0.0,
            max_duration = 0.0,
            max_line_distance_m=0.6,
            ),
        DelayObjective(2.0),
    # endregion
    # region arm out and knock down cup by turning left 90 degree
        ArmMiddleObjective(),
        DelayObjective(2.0),
        DriveTurnAngleObjective(
            angle_deg=-100,
            linear_cmd=0.0,
            timeout_s=6.0,
        ),
        DelayObjective(2.0),
        ArmUpObjective(),
        DriveTurnAngleObjective(
            angle_deg=100,
            linear_cmd=0.0,
            timeout_s=6.0,
        ),
    #end region

    # region go to ball pick up location
        DriveToWaypointObjective(
            waypoint=(0.6,1.2),
            is_local=False,
            print_interval=20,
            relative_heading_deg=-100.0,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
        #SearchAndNavigateToBlueBall(),

    # end region
        DelayObjective(2.0),
    # region go to blue ball drop off location
        DriveToWaypointObjective(
            waypoint=(0.9,1.3),
            is_local=False,
            print_interval=20,
            relative_heading_deg=0.0,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
        ArmDownObjective(),
        DelayObjective(2.0),
        ArmUpObjective(),
        DriveTurnAngleObjective(
            angle_deg=170.0,
            linear_cmd=0.0,
            timeout_s=6.0,
        ),
    # end region
    # region go to ball pick up location
        DriveToWaypointObjective(
            waypoint=(0.5,1.0),
            is_local=False,
            print_interval=20,
            relative_heading_deg=-100.0,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
        DelayObjective(2.0),
    # end region
    # region go to red ball drop off location
        DriveToWaypointObjective(
            waypoint=(1.35,0.8),
            is_local=False,
            print_interval=20,
            relative_heading_deg=90.0,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
        ArmDownObjective(),
        DelayObjective(2.0),
        ArmUpObjective(),
        DriveTurnAngleObjective(
            angle_deg=-90.0,
            linear_cmd=0.0,
            timeout_s=6.0,
        ),
    # end region
    # region go to platform pick up location
        DelayObjective(2.0),
        DriveToWaypointObjective(
            waypoint=(1.94,0.7),
            is_local=False,
            print_interval=20,
            relative_heading_deg=0.0,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
        #SearchAndNavigateToPlatform(marker_id=5),
        #drive forward 0.5m to simulate going to the platform for now since the platform detection is not working reliably
        DriveToWaypointObjective(
            waypoint=(0.2,0.0),
            is_local=True,
            print_interval=20,
            relative_heading_deg=0.0,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
    # end region
    # region go to platform drop off location D
        DriveToWaypointObjective(
            waypoint=(1.92,1.1),
            is_local=False,
            print_interval=20,
            relative_heading_deg=165.0,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
            ArmDownObjective(),
            DelayObjective(2.0),
            ArmUpObjective(),
            DriveTurnAngleObjective(
                angle_deg=180.0,
                linear_cmd=0.0,
                timeout_s=6.0,            
                ),
    # end region
    # region go to platform pick up location again
        DriveToWaypointObjective(
            waypoint=(1.94,0.7),
            is_local=False,
            print_interval=20,
            relative_heading_deg=0.0,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
    # end region
    # region go to platform drop off location A
        DriveToWaypointObjective(
            waypoint=(1.94,1.6),
            is_local=False,
            print_interval=20,
            relative_heading_deg=110.0,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
        DriveToWaypointObjective(
            waypoint=(1.5,1.85),
            is_local=False,
            print_interval=20,
            relative_heading_deg=180.0,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
            ArmDownObjective(),
            DelayObjective(2.0),
            ArmUpObjective(),
    #end region
    # region go to line and follow to end
        #SearchAndNavigateToBlueBall(),
        #ArmDownObjective(wait_after_s=2.0),

        #ArmUpObjective(),
        #SearchAndNavigateToAruco(marker_id=53,turn_rate=0.3),
        #ArmDownObjective(wait_after_s=2.0),

        #ArmUpObjective(),
        # SearchAndNavigateToPlatform(marker_id=5),
        # ArmDownObjective(wait_after_s=2.0),

        # SearchAndNavigateToBlueBall(turn_rate=0.8),
        # ArmDownObjective(wait_after_s=2.0),
        # # ArmUpObjective(),

        # SearchAndNavigateToAruco(marker_id=20,turn_rate=0.4),
        # ArmDownObjective(wait_after_s=2.0),
        #ArmUpObjective(),
    ]
    

    return objectives

def store_robot_position(ctx):
    """Store global odometry pose samples for mission route plotting."""
    log_dir = os.path.join(THIS_DIR, "MissionLogs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "robot_position_log.txt")
    with open(log_file, "a") as f:
        x_world, y_world, h_world = odom.get_world_pose()
        sample_ts = odom.pose_source.poseTime.timestamp()
        obj_name = str(ctx.memory.get("_mission_current_objective") or "unknown").replace(",", ";")
        obj_idx = ctx.memory.get("_mission_current_objective_index")
        obj_idx_str = "" if obj_idx is None else str(int(obj_idx))
        # CSV: pose_timestamp_s, x_world_m, y_world_m, heading_rad, objective_name, objective_index
        f.write(f"{sample_ts:.3f},{x_world:.3f},{y_world:.3f},{h_world:.4f},{obj_name},{obj_idx_str}\n")

if __name__ == "__main__":
    if service.process_running("midway-evaluation-mission"):
        print("% midway-evaluation-mission is already running - terminating")
        print("%   if it is partially crashed in the background, then try:")
        print("%     pkill midway-evaluation-mission")
        print("%   or, if that fails use the most brutal kill")
        print("%     pkill -9 midway-evaluation-mission")
    else:
        print("% Starting midway evaluation mission")
        service.setup("localhost")
        if service.connected:
            log_dir = os.path.join(THIS_DIR, "MissionLogs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "robot_position_log.txt")
            open(log_file, "w").close()
            print("% Cleared MissionLogs/robot_position_log.txt for new run")

            ctx = MissionContext(service)
            ctx.actions.arm.bind_memory(ctx.memory)
            ctx.actions.arm.move_up()
            objectives = build_objectives()
            runner = MissionRunner(objectives, ctx, tick_hook=store_robot_position)
            runner.run()
        service.terminate()
    print("% Main Terminated")

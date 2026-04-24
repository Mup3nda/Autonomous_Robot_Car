#!/usr/bin/env python3

import os
import sys
import time
import math
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from Objectives.search_and_navigate_to_red_ball import SearchAndNavigateToRedBall
from Objectives.look_for_aruco_objective import LookForArucoObjective
from uservice import service
from mission_runner import MissionRunner
from robot_actions import RobotActions
from mission_context import MissionContext
from Objectives.drive_circle_objective import DriveCircleObjective
from Objectives.drive_to_waypoint_objective import DriveToWaypointObjective
from Objectives.drive_turn_angle_objective import DriveTurnAngleObjective
from Objectives.align_to_circle_tangent_objective import AlignToCircleTangentObjective
from Objectives.search_and_navigate_to_blue_ball_objective import SearchAndNavigateToBlueBall
from Objectives.arm_up_objective import ArmUpObjective
from Objectives.arm_middle_objective import ArmMiddleObjective
from Objectives.arm_down_objective import ArmDownObjective
from Objectives.arm_middle_objective import ArmMiddleObjective
from Objectives.gripper_open_objective import GripperOpenObjective
from Objectives.gripper_middle_objective import GripperMiddleObjective
from Objectives.gripper_close_objective import GripperCloseObjective
from Objectives.drive_to_line_objective import DriveToLineObjective
from Objectives.drive_to_line_until_curve_objective import DriveToLineUntilCurveObjective
from Objectives.drive_to_line_zone_switch_objective import DriveToLineZoneSwitchObjective
from Objectives.drive_to_line_lock_straight_heading_objective import DriveToLineLockStraightHeadingObjective
from Objectives.drive_distance_objective import DriveDistanceObjective
from Objectives.search_and_navigate_to_aruco_objective import SearchAndNavigateToAruco
from Objectives.search_and_navigate_to_platform_objective import SearchAndNavigateToPlatform
from Objectives.drive_to_timer_and_back_objective import DriveToTimerAndBackObjective
from Objectives.reset_origin_objective import ResetOriginObjective
from Objectives.drive_backward_until_line_stop_objective import DriveBackwardUntilLineStopObjective
from Objectives.drive_to_waypoint_until_line_count_objective import DriveToWaypointUntilLineCountObjective
from Objectives.delay_objective import DelayObjective
from Objectives.grab_target_objective import GrabTargetObjective
from Objectives.drop_target_objective import DropTargetObjective
from Objectives.line_recovery_objective import LineRecovery
from sodom import odom
from Objectives.search_and_navigate_to_golf_ball import SearchAndNavigateToGolfBall
from Objectives.drive_until_end_ramp import DriveUntilEndRamp
from Objectives.drive_to_line_objective_ramp_imu import DriveToLineObjectiveIMU
from Objectives.search_and_navigate_to_hole_objective import SearchAndNavigateToHole

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
CIRCLE_RADIUS_M = 0.349
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
        "follow_speed": 0.45,
        "lost_line_timeout_s": 0.0,
    }
]

#
# Add objectives in the list below in the exact order they should execute.
def build_objectives():
    objectives = [
    GripperCloseObjective(),
        DelayObjective(2.0),
        GripperOpenObjective(),
        ArmUpObjective(),
    # region Following line approach roundabout
        DriveToLineObjective(
            follow_left=True,
            follow_speed=0.40,
            search_speed=0.25,
            centering_speed=0.2,
            lost_line_timeout_s=0.3,
            max_line_distance_m=1.77,
            max_duration=0.0,
            ),
        DelayObjective(0.5),
        DriveDistanceObjective(
            target_distance_m=0.52,
            throttle=0.30,
            timeout_s=8.0,
            instant_stop=True,
        ),

        # region entry turn and align to tangent and drive circle
        DriveTurnAngleObjective(
            angle_deg=82.0,
            linear_cmd=0.0,
            timeout_s=6.0,
        ),
        
        DriveCircleObjective(
            radius_m=CIRCLE_RADIUS_M,
            revolutions=1.6, # one full circle + half circle
            forward_cmd=0.28,
            turn_cmd=CIRCLE_TURN_CMD,
            turn_rate_scale=CIRCLE_TURN_RATE_SCALE,
            clockwise=CIRCLE_CLOCKWISE,
            timeout_s=CIRCLE_TIMEOUT_S,
        ),
        DriveTurnAngleObjective(
            angle_deg=93.0,
            linear_cmd=0.0,
            timeout_s=6.0,
        ),
       #endregion
       #region Follow line until extra time
        DriveToLineObjective(follow_left=LINE_ENTRY_FOLLOW_LEFT,
                            follow_speed=0.45,
                            search_speed=LINE_ENTRY_SEARCH_SPEED,
                            lost_line_timeout_s=LINE_ENTRY_TIMEOUT_S,
                            max_line_distance_m=4.0,
                            instant_stop=False),
        LineRecovery(max_line_distance_m=3.2),
        DelayObjective(0.3),
        ResetOriginObjective(), # new 0,0,0 origin after following line end
        DriveToWaypointObjective(
             waypoint=(0.4,-1.4),
             is_local=False,
             print_interval=20,
             relative_heading_deg=None,
             nav_mode=WAYPOINT_NAV_MODE,
             ),
        DriveToWaypointObjective(
             waypoint=(0.0,0.0),
             is_local=False,
             print_interval=20,
             relative_heading_deg=None,
             nav_mode=WAYPOINT_NAV_MODE,
             ),
        #endregion
        DriveToLineObjective(follow_speed=0.5, stop_after_centering=False, max_line_distance_m=5.9),
        DriveToLineObjective(follow_speed=0.5, stop_after_centering=True, max_duration=2.0),
        DriveToLineUntilCurveObjective(curve_detection_delay_s=1.0, follow_speed=0.3),
        SearchAndNavigateToGolfBall(desired_distance=0.35, COMPENSATE_PARAMETER = 30),
        GripperOpenObjective(),
        DelayObjective(1.0),
        ArmDownObjective(wait_after_s=2.0),
        DelayObjective(1.0),
        DriveToWaypointObjective(
              waypoint=(0.03,0),
              is_local=True,
              print_interval=20,
              relative_heading_deg=0.0,
              nav_mode=WAYPOINT_NAV_MODE,
              ),
        GripperCloseObjective(),
        DelayObjective(2.0),
        ArmUpObjective(),
        DelayObjective(1.0), 
        DriveTurnAngleObjective(-150.0, linear_cmd=0.01, turn_cmd=0.8, timeout_s=5.0),
        DriveToWaypointObjective(
              waypoint=(0.3,0),
              is_local=True,
              print_interval=20,
              relative_heading_deg=0,
              nav_mode=WAYPOINT_NAV_MODE,
              ), 
        DriveTurnAngleObjective(40.0, linear_cmd=0.0, turn_cmd=0.8, timeout_s=5.0),
        DriveToLineObjective(follow_speed=0.5,max_duration=2.0, stop_after_centering=True),
        DriveToLineUntilCurveObjective(curve_detection_delay_s=1.0, follow_speed=0.3),
        DelayObjective(1.0),
        SearchAndNavigateToHole(desired_distance=0.35,COMPENSATE_PARAMETER = 35),
        ArmDownObjective(wait_after_s=2.0),
        DelayObjective(1.0), 
        GripperOpenObjective(),
        DelayObjective(1.0), 
        DriveToWaypointObjective(
              waypoint=(0.1,0),
              is_local=True,
              print_interval=20,
              relative_heading_deg=0,
              nav_mode=WAYPOINT_NAV_MODE,
              ), 
        DelayObjective(1.0), 
        ArmUpObjective(),
        DriveTurnAngleObjective(100.0, linear_cmd=0.01, turn_cmd=0.8, timeout_s=5.0),
        #region go back to line after ramp and get extra time
        DriveToLineLockStraightHeadingObjective(
            follow_left=LINE_ENTRY_FOLLOW_LEFT,
            follow_speed=0.5,
            search_speed=LINE_ENTRY_SEARCH_SPEED,
            lost_line_timeout_s=LINE_ENTRY_TIMEOUT_S,
            heading_track_window_s=4.0,
            heading_tolerance_deg=0.5,
            instant_stop=False,
        ),
        LineRecovery(),
        DriveBackwardUntilLineStopObjective(
             reverse_speed=-0.2,
             line_found_confidence=2,
             timeout_s=8.0,
             max_distance_m=1.5,
         ), 

         ResetOriginObjective(), # new 0,0,0 origin after following line end
         DelayObjective(0.5),
    #endregion
     #region Get extra time return to line
         ArmUpObjective(),
         DriveToTimerAndBackObjective(),
         # region Following line
         DelayObjective(2.0),
     # endregion
     # region Following line before knocking down cup
         DriveToLineObjective(
             follow_left=True,
             follow_speed=0.4,
             search_speed=0.25,
             centering_speed=0.2,
             lost_line_timeout_s=0.0,
             max_duration = 0.0,
             max_line_distance_m=0.5,
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
             waypoint=(0.2,0.0),
             is_local=True,
             print_interval=20,
             relative_heading_deg=-0.0,
             nav_mode=WAYPOINT_NAV_MODE,
             ),
         DriveToWaypointObjective(
             waypoint=(0.6,1.2),
             is_local=False,
             print_interval=20,
             relative_heading_deg=-100.0,
             nav_mode=WAYPOINT_NAV_MODE,
             ),
         SearchAndNavigateToBlueBall(turn_rate=0.3),
         GrabTargetObjective(nav_mode=WAYPOINT_NAV_MODE),
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
         SearchAndNavigateToAruco(marker_id=12,desired_distance=0.35,scan_mode=LookForArucoObjective.SCAN_MODE_SWEEP_90),
         DropTargetObjective(delay_s=1.0),
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
        SearchAndNavigateToRedBall(),
        GrabTargetObjective(nav_mode=WAYPOINT_NAV_MODE),
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
        SearchAndNavigateToAruco(marker_id=15,desired_distance=0.35,scan_mode=LookForArucoObjective.SCAN_MODE_SWEEP_90),
         DropTargetObjective(
             delay_s=1.0),
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
             waypoint=(1.94,0.7),
             is_local=False,
             print_interval=20,
             relative_heading_deg=165.0,
             nav_mode=WAYPOINT_NAV_MODE,
             ),
            SearchAndNavigateToAruco(marker_id=17,desired_distance=0.35,scan_mode=LookForArucoObjective.SCAN_MODE_SWEEP_90),
            DropTargetObjective(delay_s=1.0),
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
             waypoint=(1.5,1.82),
             is_local=False,
             print_interval=20,
             relative_heading_deg=-90.0,
             nav_mode=WAYPOINT_NAV_MODE,
             ),
             SearchAndNavigateToAruco(marker_id=11,desired_distance=0.35,scan_mode=LookForArucoObjective.SCAN_MODE_SWEEP_90),
             DropTargetObjective(delay_s=1.0),
     #end region
     # region return to finish line
         DelayObjective(2.0),
         DriveToWaypointUntilLineCountObjective(
             waypoint=(1.6,5.0),
             is_local=False,
             print_interval=20,
             nav_mode=WAYPOINT_NAV_MODE,
             line_detect_confidence=4,
             line_clear_confidence=1,
             stop_line_count=2,
             ),
     # end region
     # region go to line and follow to end
         DriveTurnAngleObjective(
             angle_deg=90.0,
             linear_cmd=0.0,
             timeout_s=6.0,
         ),
         DriveToLineObjective(
             follow_left=False,
             follow_speed=0.5,
             search_speed=0.25,
             centering_speed=0.2,
             lost_line_timeout_s=0.0,
             max_duration = 0.0,
             max_line_distance_m=2.0,
             instant_stop=True,
             ),
        
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

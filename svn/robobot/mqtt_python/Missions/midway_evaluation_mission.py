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
from Objectives.arm_middle_objective import ArmMiddleObjective
from Objectives.arm_down_objective import ArmDownObjective
from Objectives.gripper_open_objective import GripperOpenObjective
from Objectives.gripper_middle_objective import GripperMiddleObjective
from Objectives.gripper_close_objective import GripperCloseObjective
from Objectives.drive_to_line_objective import DriveToLineObjective
from Objectives.search_and_navigate_to_aruco_objective import SearchAndNavigateToAruco
from Objectives.search_and_navigate_to_platform_objective import SearchAndNavigateToPlatform
from Objectives.drive_distance_objective import DriveDistanceObjective

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
    objectives = [
        # ## GRIPPER TEST
        # ArmUpObjective(),
        # GripperOpenObjective(),
        # DelayObjective(2.0),
        # ArmDownObjective(),
        # DelayObjective(2.0),
        # GripperCloseObjective(),
        # DelayObjective(1.0),
        # GripperCloseObjective(),
        # DelayObjective(2.0),
        # ArmUpObjective(),

        # ## TEST GRABBING BLUE BALL HAVE NOT TESTED
        # ArmUpObjective(),
        # GripperOpenObjective(),
        # SearchAndNavigateToBlueBall(turn_rate=0.8, desired_distance=0.35),
        # ArmDownObjective(),
        # DelayObjective(2.0),
        # DriveDistanceObjective(target_distance_m=0.2),
        # DelayObjective(1.0),
        # GripperCloseObjective(),
        # DelayObjective(1.0),
        # GripperCloseObjective(),
        # DelayObjective(1.0),
        # ArmUpObjective(),      
        
        # ## TEST GRABBING ARUCO
        # ArmUpObjective(),
        # GripperOpenObjective(),
        # SearchAndNavigateToAruco(marker_id=53,turn_rate=0.4, desired_distance=0.35),
        # ArmDownObjective(),
        # DelayObjective(2.0),
        # DriveDistanceObjective(target_distance_m=0.2),
        # DelayObjective(1.0),
        # GripperCloseObjective(),
        # DelayObjective(1.0),
        # GripperCloseObjective(),
        # DelayObjective(1.0),
        # ArmUpObjective(),

        ArmUpObjective(),
        GripperOpenObjective(),
        SearchAndNavigateToPlatform(marker_id=5),
        ArmMiddleObjective(),

        # SearchAndNavigateToAruco(marker_id=20,turn_rate=0.4),
        # ArmDownObjective(wait_after_s=2.0),
        # ArmUpObjective(),
        
        ]
    return objectives


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
            ctx = MissionContext(service)
            ctx.actions.arm.bind_memory(ctx.memory)
            ctx.actions.arm.move_up()
            objectives = build_objectives()
            runner = MissionRunner(objectives, ctx)
            runner.run()
        service.terminate()
    print("% Main Terminated")

#!/usr/bin/env python3

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scam import cam
from sedge import edge
from sgpio import gpio
from uservice import service

from mission_runner import MissionRunner
from robot_actions import RobotActions
from mission_context import MissionContext
from Objectives.drive_circle_objective import DriveCircleObjective
from Objectives.drive_to_waypoint_objective import DriveToWaypointObjective
from Objectives.drive_turn_angle_objective import DriveTurnAngleObjective
from Objectives.search_and_navigate_to_blue_ball_objective import SearchAndNavigateToBlueBall
from Objectives.arm_up_objective import ArmUpObjective
from Objectives.arm_down_objective import ArmDownObjective
from Objectives.drive_to_line_objective import DriveToLineObjective

# Roundabout three-step tuning parameters.
# Step 1: Entry line follow
LINE_ENTRY_FOLLOW_LEFT = True
LINE_ENTRY_FOLLOW_SPEED = 0.8
LINE_ENTRY_SEARCH_SPEED = 0.35
LINE_ENTRY_TIMEOUT_S = 0.3  # How long to wait after line disappears before handing to waypoint

# Step 2: Waypoint alignment (position robot at circle entry point)
WAYPOINT_FOR_CIRCLE_M = (0.3, 0.0)  # Distance (forward, sideways) from line end to circle entry
WAYPOINT_NAV_MODE = "smooth"  # "smooth" (drive+turn together) or "sequential" (rotate-then-drive)

# Step 3: Circle roundabout
CIRCLE_RADIUS_M = 0.35
CIRCLE_REVOLUTIONS = 1.5
CIRCLE_FORWARD_CMD = 0.28
CIRCLE_TURN_CMD = None  # Set e.g. 0.24 to override auto radius-based turning.
CIRCLE_TURN_RATE_SCALE = 1.0
CIRCLE_CLOCKWISE = True
CIRCLE_TIMEOUT_S = 40.0

# Step 4: Exit line follow
LINE_EXIT_FOLLOW_LEFT = False
LINE_EXIT_FOLLOW_SPEED = 0.75


# Add objectives in the list below in the exact order they should execute.
def build_objectives():
    objectives = [
        ArmUpObjective(),
        DriveToLineObjective(
            follow_left=True,
            follow_speed=0.8,
            search_speed=0.35,
            centering_speed=0.3,
            lost_line_timeout_s=0.3,
            ),
        DriveToWaypointObjective(
            waypoint=(0.20, 0.0),
            reset_origin=True,
            print_interval=20,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
        DriveTurnAngleObjective(
            angle_deg=60.0,
            linear_cmd=0.0,
            timeout_s=6.0,
        ),
        DriveToWaypointObjective(
            waypoint=(0.05,0.0),
            reset_origin=True,
            print_interval=20,
            nav_mode=WAYPOINT_NAV_MODE,
            ),
        # DriveToWaypointObjective(
        #     waypoint=(0.1, 0.4),
        #     reset_origin=True,
        #     print_interval=20,
        #     nav_mode=WAYPOINT_NAV_MODE,
        #     ),
        # DriveToWaypointObjective(
        #     waypoint=(0.09, -0.2),
        #     reset_origin=True,
        #     print_interval=20,
        #     nav_mode=WAYPOINT_NAV_MODE,
        #     ),  

        # We are tamhemtial to the circle
        # DriveToWaypointObjective(
        #     waypoint=(0.3, 0.35),
        #     reset_origin=True,
        #     print_interval=20,
        #     nav_mode=WAYPOINT_NAV_MODE,
        #     ),
                    
        # Step 3: Execute roundabout
        DriveCircleObjective(
            radius_m=0.35,   # 35 cm from robot center to circle center
            revolutions=1.5, # one full circle + half circle
            forward_cmd=CIRCLE_FORWARD_CMD,
            turn_cmd=CIRCLE_TURN_CMD,
            turn_rate_scale=CIRCLE_TURN_RATE_SCALE,
            clockwise=CIRCLE_CLOCKWISE,
            timeout_s=CIRCLE_TIMEOUT_S,
        ),
        # Removed roundabout waypoint chain (kept as comment for reference):
        # DriveToWaypointObjective(
        #     waypoint=(0.2, 0.5),
        #     reset_origin=True,
        #     print_interval=20,
        #     nav_mode="smooth",
        # ),
        # DriveToWaypointObjective(
        #     waypoint=(0.25, -0.1),
        #     reset_origin=True,
        #     print_interval=20,
        #     nav_mode="smooth",
        # ),
        # DriveToWaypointObjective(
        #     waypoint=(0.2, -0.2),
        #     reset_origin=True,
        #     print_interval=20,
        #     nav_mode="smooth",
        # ),
        # DriveToWaypointObjective(
        #     waypoint=(0.4, -0.35),
        #     reset_origin=True,
        #     print_interval=20,
        #     nav_mode="smooth",
        # ),
        # DriveToWaypointObjective(
        #     waypoint=(0.15, 0.0),
        #     reset_origin=True,
        #     print_interval=20,
        #     nav_mode="smooth",
        # ),
        # DriveToLineObjective(
        #     follow_left=False,
        #     follow_speed=0.75,
        #     search_speed=0.35,
        #     centering_speed=0.3,
        #     lost_line_timeout_s=3.0,
        #     ),
        #  SearchAndNavigateToBlueBall(),
        #  ArmDownObjective(),
        # Next objectives for midpoint demo can be appended here.
        # DriveToLineObjective(),
        # FollowLineOpenSpaceObjective(),
        # ExtraShowcaseObjective(),
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
            actions = RobotActions(service, gpio, cam, edge)
            ctx = MissionContext(actions)
            ctx.actions.arm.bind_memory(ctx.memory)
            ctx.actions.arm.move_up()
            objectives = build_objectives()
            runner = MissionRunner(objectives, ctx)
            runner.run()
        service.terminate()
    print("% Main Terminated")

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
from Objectives.search_and_navigate_to_blue_ball_objective import SearchAndNavigateToBlueBall
from Objectives.arm_up_objective import ArmUpObjective
from Objectives.arm_down_objective import ArmDownObjective
from Objectives.drive_to_line_objective import DriveToLineObjective
from Objectives.search_and_navigate_to_aruco_objective import SearchAndNavigateToAruco

# Roundabout tuning parameters.
CIRCLE_RADIUS_M = 0.8
CIRCLE_REVOLUTIONS = 1.0
CIRCLE_FORWARD_CMD = 0.18
CIRCLE_TURN_CMD = None  # Set e.g. 0.24 to override auto radius-based turning.
CIRCLE_TURN_RATE_SCALE = 1.0
CIRCLE_CLOCKWISE = False
CIRCLE_TIMEOUT_S = 40.0


# Add objectives in the list below in the exact order they should execute.
def build_objectives():
    objectives = [
        ArmUpObjective(),
        # DriveToLineObjective(
        #     follow_left=True,
        #     follow_speed=0.8,
        #     search_speed=0.35,
        #     centering_speed=0.3,
        #     lost_line_timeout_s=0.3,
        #     ),
        # DriveToWaypointObjective(
        #     waypoint=(0.2, 0.5), #10 cm forward from current position
        #     reset_origin=True,
        #     print_interval=20,
        #     nav_mode="smooth",
        #     ),
        # DriveToWaypointObjective(
        # waypoint=(0.25, -0.1), #10 cm forward from current position
        # reset_origin=True,
        # print_interval=20,
        # nav_mode="smooth",
        # ),
        # DriveToWaypointObjective(
        # waypoint=(0.2, -0.2), #10 cm forward from current position
        # reset_origin=True,
        # print_interval=20,
        # nav_mode="smooth",
        # ),
        #  DriveToWaypointObjective(
        # waypoint=(0.4, -0.35), #10 cm forward from current position
        # reset_origin=True,
        # print_interval=20,
        # nav_mode="smooth",
        # ),
        # DriveToWaypointObjective(
        # waypoint=(0.15, 0.0), #10 cm forward from current position
        # reset_origin=True,
        # print_interval=20,
        # nav_mode="smooth",
        # ),
        # DriveToLineObjective(
        #     follow_left=False,
        #     follow_speed=0.75,
        #     search_speed=0.35,
        #     centering_speed=0.3,
        #     lost_line_timeout_s=3.0,
        #     ),
        # DriveCircleObjective(
        #     radius_m=CIRCLE_RADIUS_M,
        #     revolutions=CIRCLE_REVOLUTIONS,
        #     forward_cmd=CIRCLE_FORWARD_CMD,
        #     turn_cmd=CIRCLE_TURN_CMD,
        #     turn_rate_scale=CIRCLE_TURN_RATE_SCALE,
        #     clockwise=CIRCLE_CLOCKWISE,
        #     timeout_s=CIRCLE_TIMEOUT_S,
        # ),
        # DriveToWaypointObjective(
        #     waypoint=(0.0, 0.0),
        #     reset_origin=False,
        #     print_interval=20,
        #     nav_mode="smooth",
        # ),
        #SearchAndNavigateToBlueBall(),
        #NavigateToBlueBall(),
        
        
        SearchAndNavigateToAruco(marker_id=53),
        ArmDownObjective(),
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

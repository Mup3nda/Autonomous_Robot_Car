#!/usr/bin/env python3

import os
import sys

from svn.robobot.mqtt_python.Missions.Objectives import drive_to_waypoint_objective
from svn.robobot.mqtt_python.Missions.Objectives.drive_to_line_objective import DriveToLineObjective

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
        DriveToLineObjective(
            follow_left=True,
            follow_speed=0.8,
            search_speed=0.35,
            centering_speed=0.3,
            ),
        DriveCircleObjective(
            radius_m=CIRCLE_RADIUS_M,
            revolutions=CIRCLE_REVOLUTIONS,
            forward_cmd=CIRCLE_FORWARD_CMD,
            turn_cmd=CIRCLE_TURN_CMD,
            turn_rate_scale=CIRCLE_TURN_RATE_SCALE,
            clockwise=CIRCLE_CLOCKWISE,
            timeout_s=CIRCLE_TIMEOUT_S,
        ),
        DriveToWaypointObjective(
            waypoint=(0.0, 0.0),
            reset_origin=False,
            print_interval=20,
            nav_mode="smooth",
        ),
        SearchAndNavigateToBlueBall(),
        # Next objectives for midpoint demo can be appended here.
        # DriveToLineObjective(),
        # FollowLineOpenSpaceObjective(),
        # ExtraShowcaseObjective(),
    ]
    return objectives

#function that makes sure to tell the Arm to stay at its current position, so it doesn't flop around during the mission
def keep_arm_position():
    # Implementation for keeping arm position
    # lets add a arm state to ctx.memory that tracks the current arm position, and then in each tick of the mission, we can send a command to hold that position
    # when we call ctx.arm.
    pass


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
            objectives = build_objectives()
            runner = MissionRunner(objectives, ctx)
            runner.run()
        service.terminate()
    print("% Main Terminated")

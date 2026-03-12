#!/usr/bin/env python3

import os
import sys
import time as t
from datetime import datetime
import numpy as np
from setproctitle import setproctitle

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from spose import pose
from sir import ir
from scam import cam
from sedge import edge

from sgpio import gpio
from uservice import service

from mission_runner import MissionRunner
from objective import Objective
from robot_actions import RobotActions
from mission_context import MissionContext

from Objectives.drive_turn_pi_objective import DriveTurnPiObjective
from Objectives.line_turn_image_objective import LineTurnImageObjective
from Objectives.drive_to_line_objective import DriveToLineObjective
from Objectives.drive_one_meter_objective import DriveOneMeterObjective
from Objectives.navigate_to_blue_ball_objective import NavigateToBallObjective
from Objectives.look_for_blue_ball_objective import LookForBlueBallObjective
from Objectives.search_and_navigate_to_blue_ball_objective import Search_And_Navigate_To_Blue_Ball


# This is a demo mission that can be used to test the robot and MQTT connection.
def build_objectives():
    if service.args.meter:
        return [DriveOneMeterObjective()]
    if service.args.pi:
        return [DriveTurnPiObjective()]
    if service.args.edge:
        return [DriveToLineObjective()]
    if service.args.SearchAndNavBlueball:
        return [Search_And_Navigate_To_Blue_Ball()]
    if service.args.look_ball:
        return [LookForBlueBallObjective()]
    if service.args.nav_ball:
        return [NavigateToBallObjective()]
    return [LineTurnImageObjective()]


if __name__ == "__main__":
    if service.process_running("mqtt-client-mission"):
        print("% mqtt-client-mission is already running - terminating")
        print("%   if it is partially crashed in the background, then try:")
        print("%     pkill mqtt-client-mission")
        print("%   or, if that fails use the most brutal kill")
        print("%     pkill -9 mqtt-client-mission")
    else:
        setproctitle("mqtt-client-mission")
        print("% Starting")
        service.setup("localhost")
        if service.connected:
            actions = RobotActions(service, gpio, cam, edge)
            ctx = MissionContext(actions)
            objectives = build_objectives()
            runner = MissionRunner(objectives, ctx)
            runner.run()
        service.terminate()
    print("% Main Terminated")
